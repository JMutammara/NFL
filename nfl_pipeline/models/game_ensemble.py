"""Gradient-boosting ensemble for game outcomes (margin, total, win).

Members: LightGBM, XGBoost, CatBoost (+ a regularised linear model as a
stabiliser). Each member early-stops on a chronological tail of the training
rows. Blend weights are non-negative least squares fitted on out-of-fold
predictions from rolling-origin CV; classification blends are re-calibrated
with a Platt (logit-linear) map.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.special import expit, logit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import BEST_PARAMS_DIR
from ..utils import log

TASK_KIND = {"margin": "regression", "total": "regression", "win": "classification"}


def default_params(member: str, kind: str, n_estimators: int, lr: float, seed: int) -> dict[str, Any]:
    if member == "lgbm":
        p = dict(n_estimators=n_estimators, learning_rate=lr, num_leaves=15, min_child_samples=40, colsample_bytree=0.5,
                 subsample=0.8, subsample_freq=1, reg_lambda=5.0, reg_alpha=0.0, random_state=seed, verbose=-1, n_jobs=4)
        if kind == "classification":
            p["objective"] = "binary"
        return p
    if member == "xgb":
        p = dict(n_estimators=n_estimators, learning_rate=lr, max_depth=4, min_child_weight=10, colsample_bytree=0.5,
                 subsample=0.8, reg_lambda=5.0, gamma=0.0, random_state=seed, n_jobs=4, tree_method="hist")
        p["objective"] = "binary:logistic" if kind == "classification" else "reg:squarederror"
        if kind == "classification":
            p["eval_metric"] = "logloss"
        return p
    if member == "cat":
        p = dict(iterations=n_estimators, learning_rate=lr * 1.5, depth=5, l2_leaf_reg=6.0, rsm=0.5, random_seed=seed,
                 verbose=0, thread_count=4, allow_writing_files=False)
        p["loss_function"] = "Logloss" if kind == "classification" else "RMSE"
        return p
    if member == "ridge":
        return dict(alpha=30.0) if kind == "regression" else dict(C=0.05, max_iter=2000)
    raise ValueError(member)


def load_best_params(target: str, member: str) -> dict[str, Any] | None:
    p = BEST_PARAMS_DIR / f"game_{target}_{member}.json"
    if p.exists():
        with open(p) as fh:
            return json.load(fh)
    return None


@dataclass
class GameEnsemble:
    task: str
    members: tuple[str, ...] = ("lgbm", "xgb", "cat", "ridge")
    n_estimators: int = 1500
    learning_rate: float = 0.02
    seed: int = 42
    es_rounds: int = 100
    param_overrides: dict[str, dict] = field(default_factory=dict)
    models_: dict[str, Any] = field(default_factory=dict)
    best_iters_: dict[str, int] = field(default_factory=dict)
    weights_: dict[str, float] = field(default_factory=dict)
    calib_: tuple[float, float] = (1.0, 0.0)  # Platt (a, b) on blended logit
    sigma_: float = np.nan
    features_: list[str] = field(default_factory=list)

    @property
    def kind(self) -> str:
        return TASK_KIND[self.task]

    def _params(self, member: str, fixed_iters: dict[str, int] | None) -> dict[str, Any]:
        p = default_params(member, self.kind, self.n_estimators, self.learning_rate, self.seed)
        p.update(load_best_params(self.task, member) or {})
        p.update(self.param_overrides.get(member, {}))
        if fixed_iters and member in fixed_iters:
            if member == "cat":
                p["iterations"] = fixed_iters[member]
            elif member in ("lgbm", "xgb"):
                p["n_estimators"] = fixed_iters[member]
        return p

    def _build(self, member: str, params: dict[str, Any], use_es: bool):
        if member == "lgbm":
            import lightgbm as lgb
            return lgb.LGBMClassifier(**params) if self.kind == "classification" else lgb.LGBMRegressor(**params)
        if member == "xgb":
            import xgboost as xgb
            if use_es:
                params = {**params, "early_stopping_rounds": self.es_rounds}
            return xgb.XGBClassifier(**params) if self.kind == "classification" else xgb.XGBRegressor(**params)
        if member == "cat":
            from catboost import CatBoostClassifier, CatBoostRegressor
            if use_es:
                params = {**params, "od_type": "Iter", "od_wait": self.es_rounds}
            return CatBoostClassifier(**params) if self.kind == "classification" else CatBoostRegressor(**params)
        if member == "ridge":
            est = Ridge(**params) if self.kind == "regression" else LogisticRegression(**params)
            return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)
        raise ValueError(member)

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_es: pd.DataFrame | None = None, y_es: np.ndarray | None = None,
            fixed_iters: dict[str, int] | None = None) -> "GameEnsemble":
        self.features_ = list(X.columns)
        use_es = X_es is not None and len(X_es) > 0 and not fixed_iters
        for m in self.members:
            params = self._params(m, fixed_iters)
            model = self._build(m, params, use_es)
            if m == "lgbm":
                if use_es:
                    import lightgbm as lgb
                    model.fit(X, y, eval_set=[(X_es, y_es)], callbacks=[lgb.early_stopping(self.es_rounds, verbose=False)])
                    self.best_iters_[m] = int(model.best_iteration_ or params["n_estimators"])
                else:
                    model.fit(X, y)
            elif m == "xgb":
                if use_es:
                    model.fit(X, y, eval_set=[(X_es, y_es)], verbose=False)
                    self.best_iters_[m] = int(model.best_iteration + 1)
                else:
                    model.fit(X, y, verbose=False)
            elif m == "cat":
                if use_es:
                    model.fit(X, y, eval_set=(X_es, y_es), use_best_model=True)
                    self.best_iters_[m] = int(model.get_best_iteration() + 1)
                else:
                    model.fit(X, y)
            else:
                model.fit(X, y)
            self.models_[m] = model
        if not self.weights_:
            self.weights_ = {m: 1.0 / len(self.members) for m in self.members}
        return self

    def predict_members(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.features_]
        out = {}
        for m, model in self.models_.items():
            if self.kind == "classification":
                out[m] = model.predict_proba(X)[:, 1]
            else:
                out[m] = model.predict(X)
        return pd.DataFrame(out, index=X.index)

    def blend(self, members: pd.DataFrame) -> np.ndarray:
        w = np.array([self.weights_.get(m, 0.0) for m in members.columns])
        raw = members.to_numpy() @ w
        if self.kind == "classification":
            a, b = self.calib_
            raw = expit(a * logit(np.clip(raw, 1e-6, 1 - 1e-6)) + b)
        return raw

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.blend(self.predict_members(X))

    # ---- fitted on out-of-fold predictions ---------------------------------
    def fit_blend(self, oof: pd.DataFrame, y: np.ndarray) -> None:
        cols = [m for m in self.members if m in oof.columns]
        A = oof[cols].to_numpy()
        w, _ = nnls(A, np.asarray(y, dtype=float))
        if w.sum() <= 0:
            w = np.ones(len(cols))
        w = w / w.sum()
        self.weights_ = {m: float(v) for m, v in zip(cols, w)}
        blended = A @ w
        if self.kind == "classification":
            lr = LogisticRegression(C=1e6, max_iter=1000)
            lr.fit(logit(np.clip(blended, 1e-6, 1 - 1e-6)).reshape(-1, 1), y.astype(int))
            self.calib_ = (float(lr.coef_[0][0]), float(lr.intercept_[0]))
        else:
            self.sigma_ = float(np.std(np.asarray(y, dtype=float) - blended, ddof=1))
        log.info("%s blend weights: %s%s", self.task, {k: round(v, 3) for k, v in self.weights_.items()},
                 f" | sigma={self.sigma_:.2f}" if self.kind == "regression" else f" | platt={tuple(round(v, 3) for v in self.calib_)}")

    # ---- persistence ------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "GameEnsemble":
        return joblib.load(path)


def regression_metrics(y, pred, market=None) -> dict[str, float]:
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    out = {"mae": float(np.mean(np.abs(y - pred))), "rmse": float(np.sqrt(np.mean((y - pred) ** 2))), "n": int(len(y))}
    if market is not None:
        mk = np.asarray(market, dtype=float)
        ok = ~np.isnan(mk)
        out["market_mae"] = float(np.mean(np.abs(y[ok] - mk[ok])))
        # directional accuracy vs the market line (pushes excluded)
        side = np.sign(pred[ok] - mk[ok])
        real = np.sign(y[ok] - mk[ok])
        live = real != 0
        out["beat_line_acc"] = float(np.mean(side[live] == real[live])) if live.any() else np.nan
        out["beat_line_n"] = int(live.sum())
    return out


def classification_metrics(y, p, market_p=None) -> dict[str, float]:
    y, p = np.asarray(y, dtype=float), np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    out = {"logloss": float(ll), "brier": float(np.mean((y - p) ** 2)), "acc": float(np.mean((p > 0.5) == (y == 1))), "n": int(len(y))}
    if market_p is not None:
        mp = np.clip(np.asarray(market_p, dtype=float), 1e-6, 1 - 1e-6)
        ok = ~np.isnan(mp)
        out["market_logloss"] = float(-np.mean(y[ok] * np.log(mp[ok]) + (1 - y[ok]) * np.log(1 - mp[ok])))
        out["market_brier"] = float(np.mean((y[ok] - mp[ok]) ** 2))
    return out
