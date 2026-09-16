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
TARGET_POSITIONS = {
    "passing_yards": ["QB"], "passing_tds": ["QB"],
    "rushing_yards": ["RB", "QB", "WR"], "rushing_tds": ["RB", "QB", "WR"],
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
# Wrapper
# ---------------------------------------------------------------------------
@dataclass
class PlayerModel:
    target: str
    model_type: str = "blend"  # nn | gbm | blend
    nn_params: dict = field(default_factory=dict)
    gbm_params: dict = field(default_factory=dict)
    seed: int = 42
    features_: list[str] = field(default_factory=list)
    prep_: NNPreprocessor | None = None
    net_: PlayerVolatilityNet | None = None
    gbm_: HeteroscedasticGBM | None = None
    sigma_scale_: float = 1.0  # post-hoc calibration of sigma from OOF residuals

    @property
    def kind(self) -> str:
        return target_kind(self.target)

    def _use_nn(self) -> bool:
        return self.model_type in ("nn", "blend") and TORCH_OK

    def _use_gbm(self) -> bool:
        return self.model_type in ("gbm", "blend") or not TORCH_OK

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_es: pd.DataFrame, y_es: np.ndarray) -> "PlayerModel":
        self.features_ = list(X.columns)
        y = np.asarray(y, dtype=float)
        y_es = np.asarray(y_es, dtype=float)
        if self._use_nn():
            self.prep_ = NNPreprocessor().fit(X)
            p = dict(kind=self.kind, seed=self.seed)
            p.update(load_best_player_params(self.target, "nn") or {})
            p.update(self.nn_params)
            self.net_ = PlayerVolatilityNet(**p).fit(self.prep_.transform(X), y, self.prep_.transform(X_es), y_es)
        if self._use_gbm():
            p = dict(kind=self.kind, seed=self.seed)
            p.update(load_best_player_params(self.target, "gbm") or {})
            p.update(self.gbm_params)
            self.gbm_ = HeteroscedasticGBM(**p).fit(X, y, X_es, y_es)
        return self

    def predict_dist(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.features_]
        mus, sigmas = [], []
        if self.net_ is not None:
            m, s = self.net_.predict_dist(self.prep_.transform(X))
            mus.append(m); sigmas.append(s)
        if self.gbm_ is not None:
            m, s = self.gbm_.predict_dist(X)
            mus.append(m); sigmas.append(s)
        mu = np.mean(mus, axis=0)
        sigma = np.mean(sigmas, axis=0) * self.sigma_scale_
        if self.kind == "poisson":
            mu = np.maximum(mu, 1e-4)
            sigma = np.sqrt(mu)
        else:
            mu = np.maximum(mu, 0.0)
        return pd.DataFrame({"mu": mu, "sigma": sigma}, index=X.index)

    def calibrate_sigma(self, y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> None:
        """Scale sigma so that standardised OOF residuals have unit variance."""
        if self.kind == "poisson":
            return
        z = (np.asarray(y) - np.asarray(mu)) / np.maximum(np.asarray(sigma), 1e-6)
        self.sigma_scale_ = float(np.clip(np.std(z), 0.5, 2.0))
        log.info("%s sigma calibration scale = %.3f", self.target, self.sigma_scale_)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "PlayerModel":
        return joblib.load(path)


def player_metrics(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, kind: str) -> dict[str, float]:
    """Point, distribution and probabilistic accuracy of a player forecast.

    Point:        mae, rmse, medae, bias (mean signed error, + = over-projecting), error percentiles,
                  share of games within +-10 / +-25 units, and the mean absolute error on the
                  games that mattered most (top quartile of actual production).
    Distribution: coverage of the central 50 / 80 / 95% intervals, PIT mean/sd (uniform(0,1)
                  when the distribution is right), pinball loss at q10/q50/q90 and CRPS.
    Touchdowns:   Poisson NLL, Brier of P(>=1), its calibration slope, and P(>=1) reliability.
    """
    from scipy.stats import norm, poisson

    y, mu, sigma = (np.asarray(v, dtype=float) for v in (y, mu, sigma))
    err = mu - y
    out = {"n": int(len(y)), "mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
           "medae": float(np.median(np.abs(err))), "bias": float(err.mean()), "mean_pred": float(mu.mean()), "mean_actual": float(y.mean()),
           "err_p05": float(np.percentile(err, 5)), "err_p25": float(np.percentile(err, 25)), "err_p50": float(np.percentile(err, 50)),
           "err_p75": float(np.percentile(err, 75)), "err_p95": float(np.percentile(err, 95)),
           "share_over": float(np.mean(err > 0)), "share_under": float(np.mean(err < 0))}
    if kind == "poisson":
        rate = np.maximum(mu, 1e-6)
        out["poisson_nll"] = float(np.mean(rate - y * np.log(rate)))
        p1 = 1 - poisson.cdf(0, rate)
        hit = (y >= 1).astype(float)
        out["td_brier"] = float(np.mean((hit - p1) ** 2))
        out["td_rate_pred"] = float(p1.mean())
        out["td_rate_actual"] = float(hit.mean())
        # reliability: actual rate inside deciles of predicted P(>=1)
        order = np.argsort(p1)
        bins = np.array_split(order, 10)
        out["td_reliability"] = [{"pred": float(p1[b].mean()), "actual": float(hit[b].mean()), "n": int(len(b))} for b in bins if len(b)]
        # calibration slope of actual on predicted probability (1 = perfect)
        vx = np.var(p1)
        out["td_calib_slope"] = float(np.cov(p1, hit)[0, 1] / vx) if vx > 0 else np.nan
        out["p2_brier"] = float(np.mean(((y >= 2).astype(float) - (1 - poisson.cdf(1, rate))) ** 2))
    else:
        s = np.maximum(sigma, 1e-6)
        z = (y - mu) / s
        out["gauss_nll"] = float(np.mean(0.5 * np.log(2 * np.pi * s ** 2) + z ** 2 / 2))
        for lvl in (50, 80, 95):
            out[f"cover{lvl}"] = float(np.mean(np.abs(z) <= norm.ppf(0.5 + lvl / 200)))
        pit = norm.cdf(z)
        out["pit_mean"], out["pit_sd"] = float(pit.mean()), float(pit.std())
        for q in (0.1, 0.5, 0.9):
            qv = mu + norm.ppf(q) * s
            d = y - qv
            out[f"pinball_q{int(q * 100)}"] = float(np.mean(np.maximum(q * d, (q - 1) * d)))
        out["crps"] = float(np.mean(s * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))))
        scale = 10.0 if y.mean() < 60 else 25.0
        out["within_10"] = float(np.mean(np.abs(err) <= 10))
        out["within_25"] = float(np.mean(np.abs(err) <= 25))
        top = y >= np.percentile(y, 75)
        out["mae_top_quartile"] = float(np.mean(np.abs(err[top]))) if top.any() else np.nan
        out["bias_top_quartile"] = float(err[top].mean()) if top.any() else np.nan
        zero = y == 0
        out["share_actual_zero"] = float(zero.mean())
        out["mae_nonzero"] = float(np.mean(np.abs(err[~zero]))) if (~zero).any() else np.nan
    return out
