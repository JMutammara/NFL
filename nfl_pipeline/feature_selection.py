"""Feature filters: missingness, variance, collinearity clusters, null importance.

``select_features`` must be run on *training* data only (rows before the first
validation block) so that selection itself cannot leak validation outcomes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .utils import N_JOBS, log

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None


def _quick_gain(X: pd.DataFrame, y: np.ndarray, task: str, seed: int, n_estimators: int = 300) -> pd.Series:
    params = dict(n_estimators=n_estimators, learning_rate=0.05, num_leaves=15, min_child_samples=30,
                  colsample_bytree=0.5, subsample=0.8, subsample_freq=1, reg_lambda=5.0, random_state=seed,
                  verbose=-1, n_jobs=N_JOBS, importance_type="gain")
    model = lgb.LGBMClassifier(**params) if task == "classification" else lgb.LGBMRegressor(**params)
    model.fit(X, y)
    return pd.Series(model.feature_importances_, index=X.columns, dtype=float)


def collinearity_clusters(X: pd.DataFrame, threshold: float = 0.92, max_rows: int = 6000, seed: int = 0) -> list[list[str]]:
    """Group features whose |Spearman rho| exceeds ``threshold`` (average linkage)."""
    xs = X.sample(n=min(max_rows, len(X)), random_state=seed) if len(X) > max_rows else X
    xs = xs.fillna(xs.median(numeric_only=True))
    corr_df = xs.corr(method="spearman").fillna(0.0).abs().clip(0, 1)
    corr = corr_df.to_numpy(dtype=float).copy()
    np.fill_diagonal(corr, 1.0)
    dist = 1.0 - corr
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method="average")
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")
    groups: dict[int, list[str]] = {}
    for col, lab in zip(corr_df.columns, labels):
        groups.setdefault(int(lab), []).append(col)
    return [g for g in groups.values()]


def select_features(
    X: pd.DataFrame, y: pd.Series | np.ndarray, task: str, cfg, seed: int = 42, always_keep: list[str] | None = None,
) -> dict:
    """Return {'selected': [...], 'report': DataFrame, 'dropped': {feature: reason}}."""
    fs = cfg.get("feature_selection", {})
    always_keep = [c for c in (always_keep or []) if c in X.columns]
    y = np.asarray(y, dtype=float)
    dropped: dict[str, str] = {}
    cols = list(X.columns)

    # 1) missingness & variance
    miss = X.isna().mean()
    for c in cols:
        if c in always_keep:
            continue
        if miss[c] > fs.get("max_missing_frac", 0.7):
            dropped[c] = f"missing {miss[c]:.0%}"
        elif np.nanvar(X[c].to_numpy(dtype=float)) < fs.get("min_variance", 1e-8):
            dropped[c] = "near-constant"
    cols = [c for c in cols if c not in dropped]
    log.info("selection: %s -> %s after missing/variance filters", X.shape[1], len(cols))

    # 2) preliminary gain to rank within collinearity clusters
    gain0 = _quick_gain(X[cols], y, task, seed)
    clusters = collinearity_clusters(X[cols], threshold=fs.get("corr_threshold", 0.92), seed=seed)
    keep = []
    for grp in clusters:
        forced = [c for c in grp if c in always_keep]
        best = max(grp, key=lambda c: gain0[c])
        keep += list(dict.fromkeys(forced + [best]))
        for c in grp:
            if c not in keep:
                dropped[c] = f"collinear with {best}"
    cols = [c for c in cols if c in set(keep)]
    log.info("selection: %s after collinearity clustering (%s clusters)", len(cols), len(clusters))

    # 3) null importance: real gain must beat the q-quantile of permuted-target gains
    n_shuf = int(fs.get("null_importance_shuffles", 8))
    q = float(fs.get("null_importance_quantile", 0.9))
    real = _quick_gain(X[cols], y, task, seed)
    rng = np.random.default_rng(seed)
    nulls = []
    for i in range(n_shuf):
        nulls.append(_quick_gain(X[cols], rng.permutation(y), task, seed + 100 + i))
    null_df = pd.concat(nulls, axis=1) if nulls else pd.DataFrame(index=cols)
    null_q = null_df.quantile(q, axis=1) if nulls else pd.Series(0.0, index=cols)
    ratio = (real + 1e-9) / (null_q + 1e-9)
    report = pd.DataFrame({"gain": real, "null_q": null_q, "ratio": ratio, "missing": miss[cols]}).sort_values("gain", ascending=False)
    passed = [c for c in cols if (real[c] > null_q[c]) and real[c] > 0]
    min_keep, max_keep = int(fs.get("min_keep", 25)), int(fs.get("max_keep", 160))
    ranked = report.index.tolist()
    if len(passed) < min_keep:
        passed = ranked[:min_keep]
    if len(passed) > max_keep:
        passed = [c for c in ranked if c in set(passed)][:max_keep]
    selected = list(dict.fromkeys(always_keep + passed))
    for c in cols:
        if c not in selected:
            dropped[c] = "failed null-importance test"
    report["selected"] = report.index.isin(selected)
    log.info("selection: %s features selected (%s failed null-importance)", len(selected), len(cols) - len(passed))
    return {"selected": selected, "report": report, "dropped": dropped}
