#!/usr/bin/env python
"""Stage 3 — Model training with rolling-origin validation.

Refuses to run until the engineered features have been approved
(``python build_features.py --approve`` or ``--approve-features`` here).

    python train_models.py                               # all targets
    python train_models.py --targets margin total win    # game models only
    python train_models.py --targets players             # player models only
    python train_models.py --smoke                       # fast end-to-end check (tiny models, 2 folds)
    python train_models.py --no-market                   # market-free power-rating variant

Artifacts (data/models/):
    game/<target>.joblib, game/<target>_selection.csv, game/<target>_metrics.json
    player/<target>.joblib, player/<target>_metrics.json
    metrics_summary.json
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
try:
    import pandas as _pd

    warnings.filterwarnings("ignore", category=_pd.errors.PerformanceWarning)
except Exception:  # pragma: no cover
    pass

import argparse
import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from nfl_pipeline.config import APPROVAL_FILE, FEATURES_DIR, MODELS_DIR, load_config
from nfl_pipeline.feature_selection import select_features
from nfl_pipeline.models.edge_model import FEATURES, EdgeModel, breakeven, build_edge_frame, probability_scores, roi_table
from nfl_pipeline.models.game_ensemble import (TASK_KIND, GameEnsemble, classification_metrics, regression_metrics)
from nfl_pipeline.models.player_model import (DEFAULT_TRANSFORMS, TARGET_POSITIONS, PlayerModel, dist_mean, dist_prob_over,
                                              dist_quantile, player_metrics, prop_probability_report, target_kind)
from nfl_pipeline.utils import log, timed
from nfl_pipeline.validation import assert_folds_are_causal, rolling_origin_folds

GAME_TARGET_COL = {"margin": "home_margin", "total": "total_points", "win": "home_win"}
GAME_MARKET_COL = {"margin": "spread_line", "total": "total_line", "win": "home_ml_prob_novig"}


def _fmt(d: dict) -> str:
    return "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items())


def _select(rows: pd.DataFrame, y_rows: np.ndarray, feats: list[str], kind: str, cfg, seed: int,
            always_keep: list[str], args) -> tuple[list[str], pd.DataFrame]:
    """Feature selection restricted to ``rows``.

    Called once per CV fold with that fold's *training* rows (so selection can
    never see the fold's validation outcomes) and once more with every played
    row for the final fit.
    """
    if args.skip_selection:
        return feats, pd.DataFrame()
    res = select_features(rows[feats], y_rows, kind, cfg, seed=seed, always_keep=always_keep)
    return res["selected"], res["report"]


def _clip_p(p) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def _fit_win_stack(game_ids: np.ndarray, p_cls: np.ndarray, y: np.ndarray, mk: np.ndarray, out_dir) -> tuple[dict | None, dict | None]:
    """Stack the win classifier with the margin model's implied win probability.

    Both inputs are out-of-fold, so the two logit weights are an honest estimate
    of how much each source deserves. Returns (None, None) when no margin OOF
    file exists, in which case predict_week.py uses a fixed 50/50 blend.
    """
    path, mpath = out_dir / "margin_oof.parquet", out_dir / "margin_metrics.json"
    if not path.exists() or not mpath.exists():
        log.warning("[win] margin OOF predictions not found; train the margin target first to enable stacking")
        return None, None
    mo = pd.read_parquet(path).set_index("game_id")["oof"]
    sigma = float(json.load(open(mpath))["sigma"])
    pm = pd.Series(game_ids).map(mo).to_numpy(dtype=float)
    ok = ~np.isnan(pm)
    if ok.sum() < 200:
        log.warning("[win] only %s games overlap the margin OOF; skipping stacking", int(ok.sum()))
        return None, None
    p_margin = norm.sf(-pm[ok] / sigma)
    Z = np.column_stack([logit(_clip_p(p_cls[ok])), logit(_clip_p(p_margin))])
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(Z, y[ok].astype(int))
    stack = {"coef_cls": float(lr.coef_[0][0]), "coef_margin": float(lr.coef_[0][1]),
             "intercept": float(lr.intercept_[0]), "margin_sigma": sigma}
    p_stack = expit(Z @ lr.coef_[0] + lr.intercept_[0])
    report = {"n": int(ok.sum()), "stacked": classification_metrics(y[ok], p_stack, mk[ok]),
              "margin_only": classification_metrics(y[ok], p_margin, mk[ok]),
              "classifier_only": classification_metrics(y[ok], p_cls[ok], mk[ok]), "weights": stack}
    log.info("[win] stack weights: classifier %.3f, margin %.3f, intercept %.3f | stacked logloss %.4f (classifier %.4f, "
             "margin-only %.4f, market %.4f)", stack["coef_cls"], stack["coef_margin"], stack["intercept"],
             report["stacked"]["logloss"], report["classifier_only"]["logloss"], report["margin_only"]["logloss"],
             report["stacked"]["market_logloss"])
    return stack, report


# ---------------------------------------------------------------------------
# Game models
# ---------------------------------------------------------------------------
def train_game_target(target: str, gf: pd.DataFrame, man: dict, cfg, args, use_market: bool | None = None,
                      name: str | None = None, min_train_seasons: int | None = None) -> dict:
    """Gradient-boosting ensemble for one game target. ``name`` prefixes the saved artifacts (e.g. ``pf_margin``)."""
    kind = TASK_KIND[target]
    name = name or target
    ycol = GAME_TARGET_COL[target]
    mcfg = cfg.get("models.game", {})
    vcfg = cfg.get("validation", {})
    df = gf[(gf["played"] == 1) & gf[ycol].notna()].reset_index(drop=True)
    feats = [c for c in man["features"] if c in df.columns]
    fam = man["families"]
    market_all = fam.get("market", []) + fam.get("market_ratings", []) + fam.get("lines", [])
    market = fam.get("market", [])
    if use_market is None:
        use_market = mcfg.get("use_market_features", True) and not args.no_market
    if not use_market:
        feats = [c for c in feats if c not in market_all]
    y = df[ycol].to_numpy(dtype=float)
    members = tuple(mcfg.get("ensemble_members", ["lgbm", "xgb", "cat"])) + (("ridge",) if not args.no_ridge else ())
    n_est = 120 if args.smoke else int(mcfg.get("n_estimators", 1500))
    lr = 0.1 if args.smoke else float(mcfg.get("learning_rate", 0.02))
    seed = int(mcfg.get("seed", 42))

    folds = rolling_origin_folds(df, min_train_seasons=min_train_seasons or args.min_train_seasons or int(vcfg.get("min_train_seasons", 5)),
                                 fold_unit=args.fold_unit or vcfg.get("fold_unit", "season"),
                                 es_tail_frac=float(vcfg.get("early_stop_tail_frac", 0.12)),
                                 max_folds=args.max_folds)
    assert_folds_are_causal(df, folds)
    log.info("[%s] %s rows, %s candidate features (market features %s), %s folds (%s .. %s)", name, len(df), len(feats),
             "on" if use_market else "off", len(folds), folds[0].name, folds[-1].name)

    if args.smoke:
        cfg.raw.setdefault("feature_selection", {})["null_importance_shuffles"] = 2
    always_keep = (market if use_market else []) + ["is_neutral", "rest_diff", "week"]
    out_dir = MODELS_DIR / "game"
    out_dir.mkdir(parents=True, exist_ok=True)

    # rolling-origin CV; every fold selects features on its own training rows only
    oof = pd.DataFrame(np.nan, index=df.index, columns=list(members))
    best_iters = defaultdict(list)
    fold_rows = []
    fold_features: dict[str, list[str]] = {}
    for f in folds:
        t0 = time.time()
        sel_f, _ = _select(df.iloc[f.train_idx], y[f.train_idx], feats, kind, cfg, seed, always_keep, args)
        fold_features[f.name] = sel_f
        Xf = df[sel_f]
        ens = GameEnsemble(target, members, n_est, lr, seed)
        ens.fit(Xf.iloc[f.fit_idx], y[f.fit_idx], Xf.iloc[f.es_idx], y[f.es_idx])
        preds = ens.predict_members(Xf.iloc[f.val_idx])
        oof.iloc[f.val_idx] = preds.to_numpy()
        for m, it in ens.best_iters_.items():
            best_iters[m].append(it)
        mk = df.iloc[f.val_idx][GAME_MARKET_COL[target]].to_numpy()
        row = {"fold": f.name, "n_train": len(f.train_idx), "n_val": len(f.val_idx), "n_features": len(sel_f),
               "secs": round(time.time() - t0, 1)}
        for m in members:
            met = (classification_metrics(y[f.val_idx], preds[m], mk) if kind == "classification"
                   else regression_metrics(y[f.val_idx], preds[m], mk))
            row[f"{m}_" + ("logloss" if kind == "classification" else "mae")] = met["logloss" if kind == "classification" else "mae"]
        fold_rows.append(row)
        log.info("[%s] fold %s: %s", target, f.name, _fmt({k: v for k, v in row.items() if k not in ("fold",)}))

    valid = oof.notna().all(axis=1)
    final = GameEnsemble(target, members, n_est, lr, seed)
    final.fit_blend(oof[valid], y[valid.to_numpy()])
    blended = final.blend(oof[valid])
    mk = df.loc[valid, GAME_MARKET_COL[target]].to_numpy()
    overall = (classification_metrics(y[valid.to_numpy()], blended, mk) if kind == "classification"
               else regression_metrics(y[valid.to_numpy()], blended, mk))
    per_member = {}
    for m in members:
        per_member[m] = (classification_metrics(y[valid.to_numpy()], oof.loc[valid, m], mk) if kind == "classification"
                         else regression_metrics(y[valid.to_numpy()], oof.loc[valid, m], mk))
    # edge-bucketed accuracy vs the line (regression targets)
    edge_table = {}
    if kind == "regression":
        yv = y[valid.to_numpy()]
        for thr in (0.0, 1.0, 2.0, 3.0, 4.0):
            d = blended - mk
            pick = np.abs(d) >= thr
            real = np.sign(yv - mk)
            live = pick & (real != 0)
            edge_table[f"edge>={thr:.0f}"] = {"n": int(live.sum()), "acc": float(np.mean(np.sign(d[live]) == real[live])) if live.any() else np.nan}
    # per-season blended metric
    seasons = df.loc[valid, "season"].to_numpy()
    per_season = {}
    for s in np.unique(seasons):
        m_ = seasons == s
        yv = y[valid.to_numpy()][m_]
        per_season[int(s)] = (classification_metrics(yv, blended[m_], mk[m_]) if kind == "classification"
                              else regression_metrics(yv, blended[m_], mk[m_]))
    log.info("[%s] OOF blended: %s", target, _fmt(overall))
    if edge_table:
        log.info("[%s] accuracy vs line by edge: %s", target, {k: (v["n"], round(v["acc"], 3)) for k, v in edge_table.items()})

    stack_report = None
    pd.DataFrame({"game_id": df.loc[valid, "game_id"].to_numpy(), "oof": blended}).to_parquet(out_dir / f"{name}_oof.parquet", index=False)
    if target == "win":
        final.stack_, stack_report = _fit_win_stack(df.loc[valid, "game_id"].to_numpy(), blended, y[valid.to_numpy()], mk, out_dir)

    # final fit: select on all played rows, fixed iterations from CV
    with timed(f"[{target}] final selection + fit"):
        selected, report = _select(df, y, feats, kind, cfg, seed, always_keep, args)
        X = df[selected]
        fixed = {m: int(np.median(v) * 1.1) for m, v in best_iters.items()} if best_iters else None
        final.fit(X, y, fixed_iters=fixed)
    final.save(out_dir / f"{name}.joblib")
    if len(report):
        report.to_csv(out_dir / f"{name}_selection.csv")
    metrics = {"target": target, "name": name, "kind": kind, "n_rows": int(len(df)), "n_features_candidate": len(feats),
               "n_features_selected": len(selected), "features": selected, "use_market_features": use_market,
               "n_features_by_fold": {k: len(v) for k, v in fold_features.items()}, "fold_features": fold_features,
               "members": list(members), "weights": final.weights_, "sigma": final.sigma_ if kind == "regression" else None,
               "platt": final.calib_ if kind == "classification" else None, "fixed_iters": fixed,
               "oof_overall": overall, "oof_by_member": per_member, "oof_by_season": per_season, "edge_buckets": edge_table,
               "win_stack": stack_report, "folds": fold_rows}
    with open(out_dir / f"{name}_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    return metrics


# ---------------------------------------------------------------------------
# Stage 2: market-relative edge models (spread / total) against the opening line
# ---------------------------------------------------------------------------
def _calibration_bins(p_cover: np.ndarray, cover: np.ndarray) -> list[dict]:
    conf = np.maximum(p_cover, 1 - p_cover)
    side = np.where(p_cover >= 0.5, 1.0, -1.0)
    edges = [0.5, 0.52, 0.54, 0.56, 0.58, 0.60, 1.01]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi) & (cover != 0)
        rows.append({"bin": f"{lo:.2f}-{min(hi, 1.0):.2f}", "n": int(m.sum()),
                     "hit_rate": float(np.mean(side[m] == cover[m])) if m.any() else np.nan,
                     "mean_p": float(conf[m].mean()) if m.any() else np.nan})
    return rows


def train_edge_target(task: str, gf: pd.DataFrame, cfg, args) -> dict:
    out_dir = MODELS_DIR / "game"
    pf = {}
    for t in ("margin", "total"):
        p = out_dir / f"pf_{t}_oof.parquet"
        if not p.exists():
            raise SystemExit(f"missing {p}: run the stage-1 power models first (python train_models.py --targets games)")
        pf[t] = pd.read_parquet(p).set_index("game_id")["oof"]
    df = gf[gf["played"] == 1].reset_index(drop=True)
    e = build_edge_frame(df, df["game_id"].map(pf["margin"]).to_numpy(dtype=float),
                         df["game_id"].map(pf["total"]).to_numpy(dtype=float), ref="open")
    proto = EdgeModel(task)
    ycol, ccol, rcol = proto.target_col, proto.cover_col, proto.ref_col
    outcome_col = "home_margin" if task == "spread" else "total_points"
    close_col = "spread_line" if task == "spread" else "total_line"
    clv_col = "clv_spread" if task == "spread" else "clv_total"
    e = e[e[ycol].notna() & e["pf_margin"].notna() & e["pf_total"].notna()].reset_index(drop=True)
    drop = set(args.edge_drop or [])
    feats = [c for c in FEATURES[task] if c in e.columns and c not in drop]
    tag = f"_{args.tag}" if getattr(args, "tag", None) else ""
    y = e[ycol].to_numpy(dtype=float)
    cover = e[ccol].to_numpy(dtype=float)
    vcfg = cfg.get("validation", {})
    ecfg = cfg.get("models.edge", {})
    mts = 1 if args.smoke else int(args.edge_min_train_seasons or vcfg.get("edge_min_train_seasons", 3))
    folds = rolling_origin_folds(e, min_train_seasons=mts, es_tail_frac=float(vcfg.get("early_stop_tail_frac", 0.12)),
                                 max_folds=args.max_folds)
    assert_folds_are_causal(e, folds)
    n_open = int((e["ref_is_open"] == 1).sum())
    log.info("[edge_%s] %s rows (%s with true opening lines), %s features, %s folds (%s .. %s)", task, len(e), n_open,
             len(feats), len(folds), folds[0].name, folds[-1].name)
    X = e[feats]
    n_bags = 1 if args.smoke else int(ecfg.get("n_bags", 3))
    iterations = 100 if args.smoke else int(ecfg.get("iterations", 800))
    use_cls = not args.edge_no_cls and bool(ecfg.get("use_cls", True))
    recency = args.edge_recency_halflife if args.edge_recency_halflife is not None else ecfg.get("recency_halflife")
    recency = float(recency) if recency else None
    seasons_all = e["season"].to_numpy()
    log.info("[edge_%s] classifier %s, recency halflife %s seasons", task, "on" if use_cls else "off", recency or "none")
    oof = pd.DataFrame(np.nan, index=e.index, columns=["ridge", "cat", "p_cls"])
    iters: list[dict] = []
    fold_rows = []
    for f in folds:
        t0 = time.time()
        m = EdgeModel(task, n_bags=n_bags, iterations=iterations, use_cls=use_cls, recency_halflife=recency)
        m.fit(X.iloc[f.fit_idx], y[f.fit_idx], cover[f.fit_idx], X.iloc[f.es_idx], y[f.es_idx], cover[f.es_idx], seasons=seasons_all[f.fit_idx])
        pm = m.predict_members(X.iloc[f.val_idx])
        oof.iloc[f.val_idx] = pm.to_numpy()
        iters.append(m.best_iters_)
        yv, cv = y[f.val_idx], cover[f.val_idx]
        live = cv != 0
        row = {"fold": f.name, "n_train": len(f.train_idx), "n_val": len(f.val_idx), "secs": round(time.time() - t0, 1),
               "ridge_mae": float(np.mean(np.abs(yv - pm["ridge"]))), "cat_mae": float(np.mean(np.abs(yv - pm["cat"]))),
               "ref_mae": float(np.mean(np.abs(yv))),
               "cls_hit": float(np.mean(np.sign(pm["p_cls"].to_numpy()[live] - 0.5) == cv[live])) if live.any() else np.nan}
        fold_rows.append(row)
        log.info("[edge_%s] fold %s: %s", task, f.name, _fmt({k: v for k, v in row.items() if k != "fold"}))

    valid = oof.notna().all(axis=1).to_numpy()
    ev = e.loc[valid].reset_index(drop=True)
    final = EdgeModel(task, n_bags=n_bags, iterations=iterations, use_cls=use_cls, recency_halflife=recency)
    final.fit_blend_and_calibration(oof[valid].reset_index(drop=True), y[valid], cover[valid],
                                    outcome=ev[outcome_col].to_numpy() if task == "spread" else None,
                                    ref_line=ev[rcol].to_numpy() if task == "spread" else None)
    resid_pred = final.blend_resid(oof[valid].reset_index(drop=True))
    p_cover = final.cover_prob(resid_pred, oof.loc[valid, "p_cls"].to_numpy())
    outcome = ev[outcome_col].to_numpy(dtype=float)
    ref = ev[rcol].to_numpy(dtype=float)
    close = ev[close_col].to_numpy(dtype=float)
    clv = ev[clv_col].to_numpy(dtype=float)
    cov = cover[valid]
    is_open = ev["ref_is_open"].to_numpy() == 1
    mae = {"model": float(np.mean(np.abs(outcome - (ref + resid_pred)))), "ref_line": float(np.mean(np.abs(outcome - ref))),
           "close_line": float(np.nanmean(np.abs(outcome - close))),
           "model_open_rows": float(np.mean(np.abs(outcome[is_open] - (ref[is_open] + resid_pred[is_open])))) if is_open.any() else np.nan,
           "open_line_open_rows": float(np.mean(np.abs(outcome[is_open] - ref[is_open]))) if is_open.any() else np.nan,
           "close_line_open_rows": float(np.nanmean(np.abs(outcome[is_open] - close[is_open]))) if is_open.any() else np.nan}
    from scipy.stats import spearmanr
    scores = {"all_rows": probability_scores(p_cover, cov), "open_rows": probability_scores(p_cover[is_open], cov[is_open]) if is_open.any() else {},
              "spearman_resid": float(spearmanr(resid_pred, y[valid]).correlation), "spearman_resid_open": float(spearmanr(resid_pred[is_open], y[valid][is_open]).correlation) if is_open.sum() > 10 else np.nan}
    log.info("[edge_%s] cover-probability log loss %.4f (coin flip %.4f, skill %+.2f%%) | Brier skill %+.2f%% | spearman(resid) %.3f",
             task, scores["all_rows"]["logloss"], np.log(2), 100 * scores["all_rows"]["logloss_skill"], 100 * scores["all_rows"]["brier_skill"], scores["spearman_resid"])
    report = {"all_rows": roi_table(p_cover, cov, clv),
              "open_rows": roi_table(p_cover[is_open], cov[is_open], clv[is_open]) if is_open.any() else {},
              "close_ref_rows": roi_table(p_cover[~is_open], cov[~is_open], None) if (~is_open).any() else {}}
    calib_bins = _calibration_bins(p_cover, cov)
    per_season = {}
    for s in np.unique(ev["season"]):
        sm = (ev["season"] == s).to_numpy()
        rt = roi_table(p_cover[sm], cov[sm], clv[sm], thresholds=(0.50, 0.54), n_boot=200)
        per_season[int(s)] = {"n": int(sm.sum()), "open_rows": int(is_open[sm].sum()), "flat": rt.get("p>=0.50", {}), "p54": rt.get("p>=0.54", {})}
    win_report = None
    if task == "spread":
        mu = ref + resid_pred
        p_norm = np.clip(norm.sf(-mu / final.sigma_), 1e-6, 1 - 1e-6)
        p_win = expit(final.win_calib_["a"] * logit(p_norm) + final.win_calib_["b"]) if final.win_calib_ else p_norm
        yw = (outcome > 0).astype(float)
        mk = ev["home_ml_prob_novig"].to_numpy(dtype=float) if "home_ml_prob_novig" in ev else None
        win_report = classification_metrics(yw, p_win, mk)
    log.info("[edge_%s] OOF MAE model %.3f | ref line %.3f | close line %.3f  (open-line rows: model %.3f open %.3f close %.3f)",
             task, mae["model"], mae["ref_line"], mae["close_line"], mae["model_open_rows"], mae["open_line_open_rows"], mae["close_line_open_rows"])
    for label, key in (("OPEN", "open_rows"), ("ALL", "all_rows")):
        for thr, r in report[key].items():
            if r.get("n", 0) >= 20:
                log.info("[edge_%s] vs %s line %s: n=%s hit=%.3f roi=%+.1f%% [%+.1f, %+.1f] P(roi>0)=%.2f clv=%+.2f", task, label, thr,
                         r["n"], r["hit_rate"], 100 * r["roi"], 100 * r["roi_ci_low"], 100 * r["roi_ci_high"], r["p_roi_positive"], r.get("clv_pts", np.nan))

    with timed(f"[edge_{task}] final fit"):
        fixed = {k: int(np.median([d[k] for d in iters if k in d]) * 1.1) for k in ("cat", "cls") if any(k in d for d in iters)} or None
        final.fit(X, y, cover, fixed_iters=fixed, seasons=seasons_all)
    final.save(out_dir / f"edge_{task}{tag}.joblib")
    oof_out = ev[["game_id", "season", "week", "home_team", "away_team", rcol, close_col, outcome_col, "ref_is_open"]].copy()
    oof_out["resid_pred"] = resid_pred
    oof_out["p_cover"] = p_cover
    oof_out["p_cls"] = oof.loc[valid, "p_cls"].to_numpy()
    oof_out["cover"] = cov
    oof_out["clv"] = clv
    oof_out.to_parquet(out_dir / f"edge_{task}{tag}_oof.parquet", index=False)
    metrics = {"task": task, "n_rows": int(len(e)), "n_open_rows": n_open, "n_oof": int(valid.sum()), "n_oof_open": int(is_open.sum()),
               "features": feats, "weights": final.weights_, "sigma": final.sigma_, "calib": final.calib_, "win_calib": final.win_calib_,
               "fixed_iters": fixed, "mae": mae, "roi": report, "calibration_bins": calib_bins, "by_season": per_season,
               "win_prob": win_report, "breakeven_110": breakeven(-110.0), "folds": fold_rows, "probability_scores": scores,
               "settings": {"use_cls": use_cls, "recency_halflife": recency, "n_bags": n_bags, "iterations": iterations}}
    metrics["dropped_features"] = sorted(drop)
    with open(out_dir / f"edge_{task}{tag}_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    return metrics


# ---------------------------------------------------------------------------
# Player models
# ---------------------------------------------------------------------------
def train_player_target(target: str, pf: pd.DataFrame, pman: dict, cfg, args) -> dict:
    kind = target_kind(target)
    pcfg = cfg.get("models.player", {})
    vcfg = cfg.get("validation", {})
    min_prior = int(cfg.get("features.player_min_prior_games", 1))
    transform = "none" if kind == "poisson" else str((pcfg.get("transforms") or {}).get(target, DEFAULT_TRANSFORMS.get(target, "none")))
    if args.player_transform:
        transform = "none" if kind == "poisson" else args.player_transform
    df = pf[(pf["is_projection"] == 0) & pf["position"].isin(TARGET_POSITIONS[target]) & (pf["career_games"] >= min_prior)
            & pf[target].notna()].reset_index(drop=True)
    feats = [c for c in pman["features"] if c in df.columns]
    y = df[target].to_numpy(dtype=float)
    seed = int(pcfg.get("nn", {}).get("seed", 42))
    nn_params = dict(pcfg.get("nn", {}))
    gbm_params = dict(pcfg.get("gbm", {}))
    model_type = args.player_model or pcfg.get("model_type", "blend")
    if args.smoke:
        nn_params.update(max_epochs=6, patience=3)
        gbm_params.update(n_estimators=150, learning_rate=0.1)
    nn_params.pop("seed", None)

    folds = rolling_origin_folds(df, min_train_seasons=args.min_train_seasons or int(vcfg.get("min_train_seasons", 5)),
                                 fold_unit="season", es_tail_frac=float(vcfg.get("early_stop_tail_frac", 0.12)),
                                 max_folds=args.max_folds or args.player_max_folds)
    assert_folds_are_causal(df, folds)
    log.info("[%s] %s rows (%s), %s features, transform %s, %s folds (%s .. %s)", target, len(df), "/".join(TARGET_POSITIONS[target]),
             len(feats), transform, len(folds), folds[0].name, folds[-1].name)

    if args.smoke:
        cfg.raw.setdefault("feature_selection", {})["null_importance_shuffles"] = 2
    always_keep = ["is_home", "team_spread", "total_line", "team_implied_pts", "inj_status", "depth_rank",
                   "pos_QB", "pos_RB", "pos_WR", "pos_TE", f"{target}_ewm", f"{target}_r3"]

    oof_mu = np.full(len(df), np.nan)
    oof_sigma = np.full(len(df), np.nan)
    fold_rows = []
    fold_features: dict[str, list[str]] = {}
    for f in folds:
        t0 = time.time()
        sel_f, _ = _select(df.iloc[f.train_idx], y[f.train_idx], feats, "regression", cfg, seed, always_keep, args)
        fold_features[f.name] = sel_f
        Xf = df[sel_f]
        model = PlayerModel(target, model_type, nn_params, gbm_params, seed, transform=transform)
        model.fit(Xf.iloc[f.fit_idx], y[f.fit_idx], Xf.iloc[f.es_idx], y[f.es_idx])
        mu_t, sg_t = model.raw_params(Xf.iloc[f.val_idx])
        oof_mu[f.val_idx], oof_sigma[f.val_idx] = mu_t, sg_t
        met = player_metrics(y[f.val_idx], mu_t, sg_t, kind, transform)
        met.pop("td_reliability", None)
        base = df.iloc[f.val_idx][f"{target}_ewm"].fillna(df.iloc[f.val_idx][target].mean()).to_numpy()
        met["baseline_ewm_mae"] = float(np.mean(np.abs(y[f.val_idx] - base)))
        met.update(fold=f.name, n_features=len(sel_f), secs=round(time.time() - t0, 1))
        fold_rows.append(met)
        log.info("[%s] fold %s: %s", target, f.name, _fmt({k: v for k, v in met.items() if k in ("n", "mae", "medae", "bias", "cover80", "crps", "td_brier", "baseline_ewm_mae", "secs")}))

    valid = ~np.isnan(oof_mu)
    raw = player_metrics(y[valid], oof_mu[valid], oof_sigma[valid], kind, transform)
    final = PlayerModel(target, model_type, nn_params, gbm_params, seed, transform=transform)
    final.calibrate(y[valid], oof_mu[valid], oof_sigma[valid])
    cal_mu, cal_sigma = final.calibrated_params(oof_mu[valid], oof_sigma[valid])
    overall = player_metrics(y[valid], cal_mu, cal_sigma, kind, transform)
    base = df.loc[valid, f"{target}_ewm"].fillna(df[target].mean()).to_numpy()
    overall["baseline_ewm_mae"] = float(np.mean(np.abs(y[valid] - base)))
    overall["baseline_ewm_bias"] = float(np.mean(base - y[valid]))
    overall["baseline_ewm_rmse"] = float(np.sqrt(np.mean((base - y[valid]) ** 2)))
    overall["mase"] = float(overall["mae"] / overall["baseline_ewm_mae"]) if overall["baseline_ewm_mae"] > 0 else np.nan
    # synthetic prop test: line at the recent-average projection rounded to the half; bet the calibrated distribution's side
    line = np.round(base * 2) / 2
    if kind == "poisson":
        line = np.maximum(np.round(base) - 0.5, 0.5)
    prop = prop_probability_report(y[valid], cal_mu, cal_sigma, line, transform, kind)
    overall["synthetic_prop_n"] = prop["n"]
    overall["synthetic_prop_hit"] = prop["hit_rate"]
    overall["synthetic_prop_hit_conf60"] = prop["hit_rate_conf60"]
    overall["synthetic_prop_n_conf60"] = prop["n_conf60"]
    overall["synthetic_prop_logloss_skill"] = prop["logloss_skill"]
    overall["synthetic_prop_brier_skill"] = prop["brier_skill"]
    log.info("[%s] OOF (calibrated): %s", target, _fmt({k: overall[k] for k in ("n", "mae", "medae", "bias", "rmse", "baseline_ewm_mae", "mase") if k in overall}))
    if kind != "poisson":
        log.info("[%s] coverage 50/80/95 raw %.3f/%.3f/%.3f -> calibrated %.3f/%.3f/%.3f | PIT sd %.3f -> %.3f | CRPS %.3f -> %.3f",
                 target, raw["cover50"], raw["cover80"], raw["cover95"], overall["cover50"], overall["cover80"], overall["cover95"],
                 raw["pit_sd"], overall["pit_sd"], raw["crps"], overall["crps"])
    log.info("[%s] synthetic prop: n=%s hit %.3f (conf>=0.60: %.3f on %s) | P(over) log-loss skill %+.2f%% Brier skill %+.2f%%",
             target, prop["n"], prop["hit_rate"], prop["hit_rate_conf60"], prop["n_conf60"], 100 * prop["logloss_skill"], 100 * prop["brier_skill"])

    with timed(f"[{target}] final selection + fit"):
        selected, report = _select(df, y, feats, "regression", cfg, seed, always_keep, args)
        X = df[selected]
        rng = np.random.default_rng(seed)
        recent = df["season"] >= df["season"].max() - 2
        es_mask = recent & (rng.random(len(df)) < 0.10)
        final.fit(X[~es_mask], y[~es_mask.to_numpy()], X[es_mask], y[es_mask.to_numpy()])
    out_dir = MODELS_DIR / "player"
    final.save(out_dir / f"{target}.joblib")
    if len(report):
        report.to_csv(out_dir / f"{target}_selection.csv")
    oof_out = df.loc[valid, ["player_id", "player_name", "position", "team", "opponent", "season", "week", "game_id", "is_home"]].copy()
    oof_out["actual"] = y[valid]
    oof_out["mu_t_raw"], oof_out["sigma_t_raw"] = oof_mu[valid], oof_sigma[valid]
    oof_out["mu_t"], oof_out["sigma_t"] = cal_mu, cal_sigma
    oof_out["mean"] = dist_mean(cal_mu, cal_sigma, transform) if kind != "poisson" else cal_mu
    oof_out["median"] = dist_quantile(cal_mu, cal_sigma, 0.5, transform) if kind != "poisson" else np.floor(cal_mu)
    oof_out["q10"] = dist_quantile(cal_mu, cal_sigma, 0.1, transform) if kind != "poisson" else np.nan
    oof_out["q90"] = dist_quantile(cal_mu, cal_sigma, 0.9, transform) if kind != "poisson" else np.nan
    oof_out["baseline_ewm"] = base
    oof_out["synthetic_line"] = line
    oof_out["p_over_synthetic"] = dist_prob_over(cal_mu, cal_sigma, line, transform, kind)
    oof_out["transform"] = transform
    oof_out.to_parquet(out_dir / f"{target}_oof.parquet", index=False)
    metrics = {"target": target, "kind": kind, "transform": transform, "positions": TARGET_POSITIONS[target], "n_rows": int(len(df)),
               "n_features_selected": len(selected), "features": selected, "model_type": model_type,
               "n_features_by_fold": {k: len(v) for k, v in fold_features.items()}, "fold_features": fold_features,
               "sigma_scale": final.sigma_scale_, "mu_calib": list(final.mu_calib_), "oof_overall": overall, "oof_raw": raw,
               "synthetic_prop": prop, "folds": fold_rows}
    with open(out_dir / f"{target}_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", nargs="*", default=None,
                    help="subset of: games (stage 1 + edge), edge (stage 2 only), players, <player_target>, "
                         "or the legacy market-aware targets margin total win")
    ap.add_argument("--approve-features", action="store_true", help="approve the engineered features and proceed")
    ap.add_argument("--smoke", action="store_true", help="tiny models, 2 folds: verifies the pipeline end-to-end")
    ap.add_argument("--no-market", action="store_true", help="legacy targets only: exclude market features")
    ap.add_argument("--no-ridge", action="store_true", help="drop the linear stabiliser from the stage-1 ensemble")
    ap.add_argument("--skip-selection", action="store_true")
    ap.add_argument("--selection-shuffles", type=int, default=None, help="null-importance shuffles for stage 1 (default 4)")
    ap.add_argument("--fold-unit", default=None, choices=[None, "season", "half", "week"])
    ap.add_argument("--min-train-seasons", type=int, default=None)
    ap.add_argument("--stage1-min-train-seasons", type=int, default=None, help="stage-1 power models (default 3)")
    ap.add_argument("--edge-min-train-seasons", type=int, default=None, help="stage-2 edge models (default 3)")
    ap.add_argument("--edge-drop", nargs="*", default=None, help="edge models: feature names to exclude (ablation)")
    ap.add_argument("--edge-no-cls", action="store_true", help="edge models: drop the cover classifier from the calibration")
    ap.add_argument("--edge-recency-halflife", type=float, default=None, help="edge models: sample-weight halflife in seasons")
    ap.add_argument("--tag", default=None, help="suffix for edge-model artifacts, e.g. --tag noweather (keeps the main models intact)")
    ap.add_argument("--max-folds", type=int, default=None, help="evaluate only the most recent N folds")
    ap.add_argument("--player-max-folds", type=int, default=4, help="player models: most recent N season folds")
    ap.add_argument("--player-model", default=None, choices=[None, "nn", "gbm", "blend"])
    ap.add_argument("--player-transform", default=None, choices=[None, "none", "log1p", "sqrt"], help="override the target transform for all yardage models")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.approve_features:
        APPROVAL_FILE.write_text(pd.Timestamp.now().isoformat())
    if not APPROVAL_FILE.exists():
        raise SystemExit("Features not approved. Review `python build_features.py --summary`, then run\n"
                         "  python build_features.py --approve   (or)   python train_models.py --approve-features")
    if args.smoke:
        args.max_folds = args.max_folds or 2
        args.min_train_seasons = args.min_train_seasons or 10
        args.player_max_folds = min(args.player_max_folds, 2)

    targets = args.targets or ["games", "players"]
    stage1, edge_tasks, legacy, player_targets = [], [], [], []
    for t in targets:
        if t == "games":
            stage1 += ["margin", "total"]
            edge_tasks += ["spread", "total"]
        elif t == "edge":
            edge_tasks += ["spread", "total"]
        elif t in ("edge_spread", "edge_total"):
            edge_tasks.append(t.split("_")[1])
        elif t in GAME_TARGET_COL:
            legacy.append(t)
        elif t == "players":
            player_targets += list(cfg.get("models.player.targets", []))
        elif t in TARGET_POSITIONS:
            player_targets.append(t)
        else:
            raise SystemExit(f"unknown target {t}")
    edge_tasks = list(dict.fromkeys(edge_tasks))

    summary = {"stage1": {}, "edge": {}, "game": {}, "player": {}, "smoke": args.smoke}
    if stage1 or edge_tasks or legacy:
        gf = pd.read_parquet(FEATURES_DIR / "game_features.parquet")
        man = json.load(open(FEATURES_DIR / "game_manifest.json"))
    if stage1:
        shuffles_saved = cfg.get("feature_selection.null_importance_shuffles", 8)
        cfg.raw.setdefault("feature_selection", {})["null_importance_shuffles"] = int(args.selection_shuffles or 4)
        s1 = args.stage1_min_train_seasons or int(cfg.get("validation.stage1_min_train_seasons", 3))
        for t in stage1:
            with timed(f"STAGE 1 (market-free) MODEL: pf_{t}"):
                m = train_game_target(t, gf, man, cfg, args, use_market=False, name=f"pf_{t}",
                                      min_train_seasons=None if args.smoke else s1)
            summary["stage1"][f"pf_{t}"] = {"oof": m["oof_overall"], "weights": m["weights"], "n_features": m["n_features_selected"]}
        cfg.raw["feature_selection"]["null_importance_shuffles"] = shuffles_saved
    for task in edge_tasks:
        with timed(f"EDGE MODEL: {task}"):
            m = train_edge_target(task, gf, cfg, args)
        summary["edge"][task] = {"mae": m["mae"], "roi_open": m["roi"]["open_rows"], "roi_all": m["roi"]["all_rows"],
                                 "n_oof": m["n_oof"], "n_oof_open": m["n_oof_open"], "weights": m["weights"], "win_prob": m["win_prob"]}
    for t in legacy:
        with timed(f"LEGACY GAME MODEL: {t}"):
            m = train_game_target(t, gf, man, cfg, args)
        summary["game"][t] = {"oof": m["oof_overall"], "weights": m["weights"], "n_features": m["n_features_selected"],
                              "edge_buckets": m.get("edge_buckets")}
    if player_targets:
        pf = pd.read_parquet(FEATURES_DIR / "player_features.parquet")
        pman = json.load(open(FEATURES_DIR / "player_manifest.json"))
        for t in player_targets:
            with timed(f"PLAYER MODEL: {t}"):
                m = train_player_target(t, pf, pman, cfg, args)
            summary["player"][t] = {"oof": m["oof_overall"], "n_features": m["n_features_selected"], "sigma_scale": m["sigma_scale"]}
    with open(MODELS_DIR / "metrics_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    print("\n" + "=" * 78 + "\nTRAINING SUMMARY (out-of-fold, rolling origin)\n" + "=" * 78)
    for t, m in summary["stage1"].items():
        print(f"[stage1/{t}] features={m['n_features']}  weights={ {k: round(v, 2) for k, v in m['weights'].items()} }")
        print("   " + _fmt(m["oof"]))
    for t, m in summary["edge"].items():
        mae = m["mae"]
        print(f"[edge/{t}] OOF rows={m['n_oof']} (true-open rows {m['n_oof_open']})  blend={ {k: round(v, 2) for k, v in m['weights'].items()} }")
        print(f"   MAE: model {mae['model']:.3f} | reference line {mae['ref_line']:.3f} | closing line {mae['close_line']:.3f}")
        for label, key in (("vs OPENING line", "roi_open"), ("vs all reference lines", "roi_all")):
            for thr, r in m[key].items():
                if r.get("n", 0) >= 20:
                    print(f"   {label:<24} {thr}: n={r['n']:<5} hit={r['hit_rate']:.3f}  ROI={100 * r['roi']:+.1f}% "
                          f"[{100 * r['roi_ci_low']:+.1f}, {100 * r['roi_ci_high']:+.1f}]  P(ROI>0)={r['p_roi_positive']:.2f}  CLV={r.get('clv_pts', float('nan')):+.2f} pts")
        if m.get("win_prob"):
            w = m["win_prob"]
            print(f"   win prob: logloss {w['logloss']:.4f} vs market {w.get('market_logloss', float('nan')):.4f}")
    for t, m in summary["game"].items():
        print(f"[game/{t}] features={m['n_features']}  weights={ {k: round(v, 2) for k, v in m['weights'].items()} }")
        print("   " + _fmt(m["oof"]))
        if m.get("edge_buckets"):
            print("   vs line: " + ", ".join(f"{k}: {v['acc']:.3f} (n={v['n']})" for k, v in m["edge_buckets"].items()))
    for t, m in summary["player"].items():
        print(f"[player/{t}] features={m['n_features']}  sigma_scale={m['sigma_scale']:.2f}")
        print("   " + _fmt(m["oof"]))
    print(f"\nartifacts -> {MODELS_DIR}")


if __name__ == "__main__":
    main()
