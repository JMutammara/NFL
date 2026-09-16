"""Player-level distributional models.

Two engines share one interface (``fit`` / ``predict_dist`` -> mu, sigma):

* ``PlayerVolatilityNet`` — PyTorch MLP with a heteroscedastic head. Yardage
  targets use a Gaussian negative log-likelihood with a learned per-row
  sigma (volatility); touchdown targets use a Poisson rate head.
* ``HeteroscedasticGBM`` — LightGBM mean model (L2 for yards, Poisson for
  TDs) plus a second LightGBM fitted on log squared residuals for sigma.

``PlayerModel`` wraps preprocessing (median imputation + missing indicators +
standardisation for the net; raw features for the GBM) and optional blending.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..config import BEST_PARAMS_DIR
from ..utils import N_JOBS, log

try:
    import torch
    from torch import nn

    TORCH_OK = True
except Exception:  # pragma: no cover
    TORCH_OK = False

TD_TARGETS = {"passing_tds", "rushing_tds", "receiving_tds"}
DEFAULT_TRANSFORMS = {"passing_yards": "none", "rushing_yards": "log1p", "receiving_yards": "log1p",
                      "passing_tds": "none", "rushing_tds": "none", "receiving_tds": "none"}
TARGET_POSITIONS = {
    "passing_yards": ["QB"], "passing_tds": ["QB"],
    "rushing_yards": ["RB", "QB"], "rushing_tds": ["RB", "QB"],
    "receiving_yards": ["WR", "TE", "RB"], "receiving_tds": ["WR", "TE", "RB"],
    "targets": ["WR", "TE", "RB"], "receptions": ["WR", "TE", "RB"], "carries": ["RB", "QB", "WR"], "attempts": ["QB"],
}


def target_kind(target: str) -> str:
    return "poisson" if target in TD_TARGETS else "gaussian"


# ---------------------------------------------------------------------------
# Preprocessing for the network
# ---------------------------------------------------------------------------
@dataclass
class NNPreprocessor:
    medians_: pd.Series = None
    ind_cols_: list[str] = field(default_factory=list)
    mean_: np.ndarray = None
    std_: np.ndarray = None
    columns_: list[str] = field(default_factory=list)

    def fit(self, X: pd.DataFrame) -> "NNPreprocessor":
        self.columns_ = list(X.columns)
        self.medians_ = X.median(numeric_only=True).fillna(0.0)
        miss = X.isna().mean()
        self.ind_cols_ = [c for c in X.columns if miss[c] > 0.01]
        Z = self._assemble(X)
        self.mean_ = Z.mean(axis=0)
        self.std_ = Z.std(axis=0) + 1e-6
        return self

    def _assemble(self, X: pd.DataFrame) -> np.ndarray:
        Xf = X[self.columns_].astype(float)
        ind = Xf[self.ind_cols_].isna().astype(np.float32).to_numpy() if self.ind_cols_ else np.zeros((len(X), 0), np.float32)
        base = Xf.fillna(self.medians_).to_numpy(dtype=np.float32)
        return np.hstack([base, ind])

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Z = self._assemble(X)
        return ((Z - self.mean_) / self.std_).astype(np.float32)


# ---------------------------------------------------------------------------
# PyTorch heteroscedastic network
# ---------------------------------------------------------------------------
if TORCH_OK:

    class _HeteroMLP(nn.Module):
        def __init__(self, n_in: int, hidden: list[int], dropout: float, kind: str):
            super().__init__()
            layers: list[nn.Module] = []
            d = n_in
            for h in hidden:
                layers += [nn.Linear(d, h), nn.SiLU(), nn.Dropout(dropout)]
                d = h
            self.body = nn.Sequential(*layers)
            self.kind = kind
            self.head = nn.Linear(d, 1 if kind == "poisson" else 2)

        def forward(self, x):
            h = self.body(x)
            out = self.head(h)
            if self.kind == "poisson":
                return nn.functional.softplus(out[:, 0]) + 1e-4, None
            mu = out[:, 0]
            sigma = nn.functional.softplus(out[:, 1]) + 0.05
            return mu, sigma


@dataclass
class PlayerVolatilityNet:
    kind: str = "gaussian"
    hidden: list[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.15
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    max_epochs: int = 200
    patience: int = 15
    seed: int = 42
    y_scale_: float = 1.0
    model_: Any = None
    history_: list[float] = field(default_factory=list)

    def _nll(self, mu, sigma, y):
        if self.kind == "poisson":
            return (mu - y * torch.log(mu)).mean()
        var = sigma ** 2
        return (0.5 * torch.log(var) + (y - mu) ** 2 / (2 * var)).mean()

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> "PlayerVolatilityNet":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.set_num_threads(N_JOBS)
        self.y_scale_ = 1.0 if self.kind == "poisson" else float(np.std(y) + 1e-6)
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y / self.y_scale_, dtype=torch.float32)
        Xv = torch.tensor(X_val, dtype=torch.float32)
        yv = torch.tensor(y_val / self.y_scale_, dtype=torch.float32)
        self.model_ = _HeteroMLP(X.shape[1], self.hidden, self.dropout, self.kind)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(2, self.patience // 3))
        best, best_state, bad = np.inf, None, 0
        n = len(Xt)
        for epoch in range(self.max_epochs):
            self.model_.train()
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                mu, sigma = self.model_(Xt[idx])
                loss = self._nll(mu, sigma, yt[idx])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 5.0)
                opt.step()
            self.model_.eval()
            with torch.no_grad():
                mu, sigma = self.model_(Xv)
                vl = float(self._nll(mu, sigma, yv))
            self.history_.append(vl)
            sched.step(vl)
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        log.info("net (%s): stopped after %s epochs, best val NLL %.4f", self.kind, len(self.history_), best)
        return self

    def predict_dist(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.model_.eval()
        with torch.no_grad():
            mu, sigma = self.model_(torch.tensor(X, dtype=torch.float32))
        if self.kind == "poisson":
            rate = mu.numpy()
            return rate, np.sqrt(rate)
        return mu.numpy() * self.y_scale_, sigma.numpy() * self.y_scale_

    # torch modules pickle fine via joblib, but keep state dict explicit for portability
    def __getstate__(self):
        d = self.__dict__.copy()
        if self.model_ is not None:
            d["model_state"] = {k: v.cpu() for k, v in self.model_.state_dict().items()}
            d["n_in"] = self.model_.head.in_features if not self.hidden else self.model_.body[0].in_features
            d["model_"] = None
        return d

    def __setstate__(self, d):
        state = d.pop("model_state", None)
        n_in = d.pop("n_in", None)
        self.__dict__.update(d)
        if state is not None and TORCH_OK:
            self.model_ = _HeteroMLP(n_in, self.hidden, self.dropout, self.kind)
            self.model_.load_state_dict(state)
            self.model_.eval()


# ---------------------------------------------------------------------------
# GBM heteroscedastic fallback / blend partner
# ---------------------------------------------------------------------------
@dataclass
class HeteroscedasticGBM:
    kind: str = "gaussian"
    n_estimators: int = 1200
    learning_rate: float = 0.02
    num_leaves: int = 31
    seed: int = 42
    es_rounds: int = 80
    mean_: Any = None
    var_: Any = None
    best_iter_: int = 0

    def _params(self, objective: str, n_est: int) -> dict:
        return dict(n_estimators=n_est, learning_rate=self.learning_rate, num_leaves=self.num_leaves, min_child_samples=50,
                    colsample_bytree=0.6, subsample=0.8, subsample_freq=1, reg_lambda=5.0, objective=objective,
                    random_state=self.seed, verbose=-1, n_jobs=N_JOBS)

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray) -> "HeteroscedasticGBM":
        import lightgbm as lgb

        obj = "poisson" if self.kind == "poisson" else "regression"
        self.mean_ = lgb.LGBMRegressor(**self._params(obj, self.n_estimators))
        self.mean_.fit(X, y, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(self.es_rounds, verbose=False)])
        self.best_iter_ = int(self.mean_.best_iteration_ or self.n_estimators)
        if self.kind == "gaussian":
            resid2 = np.log((y - self.mean_.predict(X)) ** 2 + 1.0)
            resid2_val = np.log((y_val - self.mean_.predict(X_val)) ** 2 + 1.0)
            self.var_ = lgb.LGBMRegressor(**self._params("regression", max(200, self.n_estimators // 3)))
            self.var_.fit(X, resid2, eval_set=[(X_val, resid2_val)], callbacks=[lgb.early_stopping(self.es_rounds, verbose=False)])
        return self

    def predict_dist(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        mu = self.mean_.predict(X)
        if self.kind == "poisson":
            mu = np.maximum(mu, 1e-4)
            return mu, np.sqrt(mu)
        sigma = np.sqrt(np.maximum(np.exp(self.var_.predict(X)) - 1.0, 1e-4))
        return mu, sigma


def load_best_player_params(target: str, engine: str) -> dict | None:
    p = BEST_PARAMS_DIR / f"player_{target}_{engine}.json"
    if p.exists():
        with open(p) as fh:
            return json.load(fh)
    return None


# ---------------------------------------------------------------------------
# Target transforms (Gaussian head is fitted in the transformed space)
# ---------------------------------------------------------------------------
def t_fwd(y, transform: str):
    y = np.asarray(y, dtype=float)
    if transform == "log1p":
        return np.log1p(np.maximum(y, 0.0))
    if transform == "sqrt":
        return np.sqrt(np.maximum(y, 0.0))
    return y


def t_inv(z, transform: str):
    z = np.asarray(z, dtype=float)
    if transform == "log1p":
        return np.expm1(z)
    if transform == "sqrt":
        return np.square(np.maximum(z, 0.0))
    return z


def dist_quantile(mu_t, sigma_t, q, transform: str):
    from scipy.stats import norm

    z = norm.ppf(q)
    return np.maximum(t_inv(np.asarray(mu_t, dtype=float) + z * np.asarray(sigma_t, dtype=float), transform), 0.0)


def dist_mean(mu_t, sigma_t, transform: str):
    """Mean on the original scale (exact for none / log1p, close for sqrt)."""
    mu_t, sigma_t = np.asarray(mu_t, dtype=float), np.asarray(sigma_t, dtype=float)
    if transform == "log1p":
        return np.maximum(np.exp(mu_t + sigma_t ** 2 / 2.0) - 1.0, 0.0)
    if transform == "sqrt":
        return np.maximum(mu_t ** 2 + sigma_t ** 2, 0.0)
    return mu_t


def dist_prob_over(mu_t, sigma_t, line, transform: str, kind: str = "gaussian"):
    """P(outcome > line). Poisson targets use the rate in mu_t."""
    from scipy.stats import norm, poisson

    if kind == "poisson":
        rate = np.maximum(np.asarray(mu_t, dtype=float), 1e-6)
        k = np.ceil(np.asarray(line, dtype=float) + 1e-9)
        return 1.0 - poisson.cdf(k - 1, rate)
    lt = t_fwd(line, transform)
    return norm.sf((lt - np.asarray(mu_t, dtype=float)) / np.maximum(np.asarray(sigma_t, dtype=float), 1e-6))


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
@dataclass
class PlayerModel:
    target: str
    model_type: str = "blend"  # nn | gbm | blend
    nn_params: dict = field(default_factory=dict)
    gbm_params: dict = field(default_factory=dict)
    seed: int = 42
    transform: str = "none"
    features_: list[str] = field(default_factory=list)
    prep_: NNPreprocessor | None = None
    net_: PlayerVolatilityNet | None = None
    gbm_: HeteroscedasticGBM | None = None
    sigma_scale_: float = 1.0                 # coverage-matched width scale (from OOF)
    mu_calib_: tuple[float, float] = (0.0, 1.0)  # mu_t -> a + b * mu_t (from OOF)

    @property
    def kind(self) -> str:
        return target_kind(self.target)

    def _use_nn(self) -> bool:
        return self.model_type in ("nn", "blend") and TORCH_OK

    def _use_gbm(self) -> bool:
        return self.model_type in ("gbm", "blend") or not TORCH_OK

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_es: pd.DataFrame, y_es: np.ndarray) -> "PlayerModel":
        self.features_ = list(X.columns)
        if self.kind == "poisson":
            self.transform = "none"
        y_t = t_fwd(y, self.transform)
        y_es_t = t_fwd(y_es, self.transform)
        if self._use_nn():
            self.prep_ = NNPreprocessor().fit(X)
            p = dict(kind=self.kind, seed=self.seed)
            p.update(load_best_player_params(self.target, "nn") or {})
            p.update(self.nn_params)
            self.net_ = PlayerVolatilityNet(**p).fit(self.prep_.transform(X), y_t, self.prep_.transform(X_es), y_es_t)
        if self._use_gbm():
            p = dict(kind=self.kind, seed=self.seed)
            p.update(load_best_player_params(self.target, "gbm") or {})
            p.update(self.gbm_params)
            self.gbm_ = HeteroscedasticGBM(**p).fit(X, y_t, X_es, y_es_t)
        return self

    # ---- raw (uncalibrated) parameters in the transformed space ----------
    def raw_params(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = X[self.features_]
        mus, sigmas = [], []
        if self.net_ is not None:
            m, s = self.net_.predict_dist(self.prep_.transform(X))
            mus.append(m); sigmas.append(s)
        if self.gbm_ is not None:
            m, s = self.gbm_.predict_dist(X)
            mus.append(m); sigmas.append(s)
        mu = np.mean(mus, axis=0)
        sigma = np.mean(sigmas, axis=0)
        if self.kind == "poisson":
            mu = np.maximum(mu, 1e-4)
            sigma = np.sqrt(mu)
        return mu, sigma

    def calibrated_params(self, mu_t: np.ndarray, sigma_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.kind == "poisson":
            return mu_t, sigma_t
        a, b = self.mu_calib_
        return a + b * np.asarray(mu_t, dtype=float), np.asarray(sigma_t, dtype=float) * self.sigma_scale_

    def predict_dist(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calibrated distribution per row: mean, median, sigma (original scale) plus the transformed-space parameters."""
        mu_t, sigma_t = self.calibrated_params(*self.raw_params(X))
        if self.kind == "poisson":
            return pd.DataFrame({"mu": mu_t, "median": np.floor(mu_t), "sigma": np.sqrt(mu_t), "mu_t": mu_t, "sigma_t": sigma_t}, index=X.index)
        mean = dist_mean(mu_t, sigma_t, self.transform)
        med = dist_quantile(mu_t, sigma_t, 0.5, self.transform)
        q16, q84 = dist_quantile(mu_t, sigma_t, 0.1587, self.transform), dist_quantile(mu_t, sigma_t, 0.8413, self.transform)
        return pd.DataFrame({"mu": mean, "median": med, "sigma": (q84 - q16) / 2.0, "mu_t": mu_t, "sigma_t": sigma_t}, index=X.index)

    def quantile(self, X: pd.DataFrame, q: float) -> np.ndarray:
        mu_t, sigma_t = self.calibrated_params(*self.raw_params(X))
        if self.kind == "poisson":
            from scipy.stats import poisson

            return poisson.ppf(q, np.maximum(mu_t, 1e-6))
        return dist_quantile(mu_t, sigma_t, q, self.transform)

    def prob_over(self, X: pd.DataFrame, line) -> np.ndarray:
        mu_t, sigma_t = self.calibrated_params(*self.raw_params(X))
        return dist_prob_over(mu_t, sigma_t, line, self.transform, self.kind)

    def calibrate(self, y: np.ndarray, mu_t: np.ndarray, sigma_t: np.ndarray) -> None:
        """Fit the mean recalibration (a, b) and the coverage-matched width scale on out-of-fold rows.

        The width scale is the factor that puts exactly 80% of results inside the
        stated 80% range, which (unlike a standard-deviation match) is not
        dominated by the tails of a skewed target.
        """
        from scipy.stats import norm

        if self.kind == "poisson":
            return
        y_t = t_fwd(y, self.transform)
        mu_t, sigma_t = np.asarray(mu_t, dtype=float), np.asarray(sigma_t, dtype=float)
        vx = np.var(mu_t)
        b = float(np.cov(mu_t, y_t)[0, 1] / vx) if vx > 0 else 1.0
        b = float(np.clip(b, 0.5, 1.5))
        a = float(np.mean(y_t) - b * np.mean(mu_t))
        self.mu_calib_ = (a, b)
        z = np.abs(y_t - (a + b * mu_t)) / np.maximum(sigma_t, 1e-6)
        self.sigma_scale_ = float(np.clip(np.quantile(z, 0.8) / norm.ppf(0.9), 0.4, 3.0))
        log.info("%s calibration: mu = %.3f + %.3f * mu_t, sigma scale %.3f (%s scale)", self.target, a, b, self.sigma_scale_, self.transform)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "PlayerModel":
        return joblib.load(path)


def player_metrics(y: np.ndarray, mu_t: np.ndarray, sigma_t: np.ndarray, kind: str, transform: str = "none") -> dict[str, float]:
    """Point, distribution and probabilistic accuracy of a player forecast, on the ORIGINAL scale.

    Point (median forecast for absolute errors, mean forecast for bias): mae, rmse, medae,
    bias, error percentiles, share within +-10 / +-25, top-quartile error, non-zero-game error.
    Distribution: coverage of the central 50 / 80 / 95% intervals, PIT mean/sd, pinball loss
    at q10/q50/q90 and CRPS (quantile-grid approximation). Touchdowns: Poisson NLL, Brier of
    P(>=1) and P(>=2), reliability by decile, calibration slope.
    """
    from scipy.stats import norm, poisson

    y, mu_t, sigma_t = (np.asarray(v, dtype=float) for v in (y, mu_t, sigma_t))
    out: dict = {"n": int(len(y))}
    if kind == "poisson":
        rate = np.maximum(mu_t, 1e-6)
        err = rate - y
        out.update({"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))), "medae": float(np.median(np.abs(err))),
                    "bias": float(err.mean()), "mean_pred": float(rate.mean()), "mean_actual": float(y.mean()),
                    "err_p05": float(np.percentile(err, 5)), "err_p25": float(np.percentile(err, 25)), "err_p50": float(np.percentile(err, 50)),
                    "err_p75": float(np.percentile(err, 75)), "err_p95": float(np.percentile(err, 95)),
                    "share_over": float(np.mean(err > 0)), "share_under": float(np.mean(err < 0))})
        out["poisson_nll"] = float(np.mean(rate - y * np.log(rate)))
        p1 = 1 - poisson.cdf(0, rate)
        hit = (y >= 1).astype(float)
        out["td_brier"] = float(np.mean((hit - p1) ** 2))
        pc = np.clip(p1, 1e-6, 1 - 1e-6)
        out["td_logloss"] = float(-np.mean(hit * np.log(pc) + (1 - hit) * np.log(1 - pc)))
        base = hit.mean()
        out["td_logloss_constant"] = float(-(base * np.log(base) + (1 - base) * np.log(1 - base)))
        out["td_brier_constant"] = float(base * (1 - base))
        out["td_rate_pred"] = float(p1.mean())
        out["td_rate_actual"] = float(hit.mean())
        order = np.argsort(p1)
        out["td_reliability"] = [{"pred": float(p1[b].mean()), "actual": float(hit[b].mean()), "n": int(len(b))} for b in np.array_split(order, 10) if len(b)]
        vx = np.var(p1)
        out["td_calib_slope"] = float(np.cov(p1, hit)[0, 1] / vx) if vx > 0 else np.nan
        out["p2_brier"] = float(np.mean(((y >= 2).astype(float) - (1 - poisson.cdf(1, rate))) ** 2))
        return out

    s = np.maximum(sigma_t, 1e-6)
    med = dist_quantile(mu_t, s, 0.5, transform)
    mean = dist_mean(mu_t, s, transform)
    err = med - y
    out.update({"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean((mean - y) ** 2))), "medae": float(np.median(np.abs(err))),
                "bias": float((mean - y).mean()), "median_bias": float(err.mean()), "mean_pred": float(mean.mean()), "mean_actual": float(y.mean()),
                "err_p05": float(np.percentile(err, 5)), "err_p25": float(np.percentile(err, 25)), "err_p50": float(np.percentile(err, 50)),
                "err_p75": float(np.percentile(err, 75)), "err_p95": float(np.percentile(err, 95)),
                "share_over": float(np.mean(err > 0)), "share_under": float(np.mean(err < 0))})
    y_t = t_fwd(y, transform)
    z = (y_t - mu_t) / s
    out["gauss_nll_t"] = float(np.mean(0.5 * np.log(2 * np.pi * s ** 2) + z ** 2 / 2))
    for lvl in (50, 80, 95):
        out[f"cover{lvl}"] = float(np.mean(np.abs(z) <= norm.ppf(0.5 + lvl / 200)))
    pit = norm.cdf(z)
    out["pit_mean"], out["pit_sd"] = float(pit.mean()), float(pit.std())
    for q in (0.1, 0.5, 0.9):
        d = y - dist_quantile(mu_t, s, q, transform)
        out[f"pinball_q{int(q * 100)}"] = float(np.mean(np.maximum(q * d, (q - 1) * d)))
    qs = (np.arange(1, 40) - 0.5) / 39
    crps = 0.0
    for q in qs:
        d = y - dist_quantile(mu_t, s, q, transform)
        crps += np.mean(np.maximum(q * d, (q - 1) * d))
    out["crps"] = float(2 * crps / len(qs))
    out["within_10"] = float(np.mean(np.abs(err) <= 10))
    out["within_25"] = float(np.mean(np.abs(err) <= 25))
    top = y >= np.percentile(y, 75)
    out["mae_top_quartile"] = float(np.mean(np.abs(err[top]))) if top.any() else np.nan
    out["bias_top_quartile"] = float((mean - y)[top].mean()) if top.any() else np.nan
    zero = y == 0
    out["share_actual_zero"] = float(zero.mean())
    out["mae_nonzero"] = float(np.mean(np.abs(err[~zero]))) if (~zero).any() else np.nan
    return out


def prop_probability_report(y: np.ndarray, mu_t: np.ndarray, sigma_t: np.ndarray, line: np.ndarray, transform: str, kind: str) -> dict:
    """Scoring of P(over line) for a synthetic prop line: log loss / Brier vs coin flip, reliability, hit rate by confidence."""
    y, line = np.asarray(y, dtype=float), np.asarray(line, dtype=float)
    p = np.clip(dist_prob_over(mu_t, sigma_t, line, transform, kind), 1e-6, 1 - 1e-6)
    live = y != line
    yb = (y[live] > line[live]).astype(float)
    pl = p[live]
    ll = float(-np.mean(yb * np.log(pl) + (1 - yb) * np.log(1 - pl)))
    conf = np.maximum(pl, 1 - pl)
    side_ok = ((pl >= 0.5) == (yb == 1)).astype(float)
    bins = [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.01]
    rel = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi)
        rel.append({"bin": f"{lo:.2f}-{min(hi, 1.0):.2f}", "n": int(m.sum()), "stated": float(conf[m].mean()) if m.any() else np.nan,
                    "actual": float(side_ok[m].mean()) if m.any() else np.nan})
    return {"n": int(live.sum()), "logloss": ll, "logloss_coinflip": float(np.log(2)), "logloss_skill": float(1 - ll / np.log(2)),
            "brier": float(np.mean((yb - pl) ** 2)), "brier_skill": float(1 - np.mean((yb - pl) ** 2) / 0.25),
            "hit_rate": float(side_ok.mean()), "hit_rate_conf60": float(side_ok[conf >= 0.6].mean()) if (conf >= 0.6).any() else np.nan,
            "n_conf60": int((conf >= 0.6).sum()), "reliability": rel}
