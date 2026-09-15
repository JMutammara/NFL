#!/usr/bin/env python
"""Optuna study for one game-model ensemble member (run locally, outside the training loop).

    python optuna_studies/tune_game_models.py --target margin --member lgbm --n-trials 100 --timeout 3600
    python optuna_studies/tune_game_models.py --target win --member cat --n-trials 60

Objective: mean out-of-fold MAE (margin/total) or log-loss (win) over the most
recent ``--max-folds`` rolling-origin folds, using the feature set selected by
train_models.py if present (data/models/game/<target>_metrics.json).
Best params are written to config/best_params/game_<target>_<member>.json and
picked up automatically by the next train_models.py run.
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
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nfl_pipeline.config import BEST_PARAMS_DIR, FEATURES_DIR, MODELS_DIR, load_config  # noqa: E402
from nfl_pipeline.models.game_ensemble import TASK_KIND, GameEnsemble, classification_metrics, regression_metrics  # noqa: E402
from nfl_pipeline.validation import rolling_origin_folds  # noqa: E402
from train_models import GAME_TARGET_COL  # noqa: E402


def space(trial: optuna.Trial, member: str, kind: str) -> dict:
    if member == "lgbm":
        return dict(num_leaves=trial.suggest_int("num_leaves", 7, 63, log=True),
                    min_child_samples=trial.suggest_int("min_child_samples", 10, 120, log=True),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.25, 0.9),
                    subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
                    reg_lambda=trial.suggest_float("reg_lambda", 0.1, 50, log=True),
                    reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5, log=True),
                    learning_rate=trial.suggest_float("learning_rate", 0.005, 0.08, log=True))
    if member == "xgb":
        return dict(max_depth=trial.suggest_int("max_depth", 2, 7),
                    min_child_weight=trial.suggest_float("min_child_weight", 1, 50, log=True),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.25, 0.9),
                    subsample=trial.suggest_float("subsample", 0.5, 1.0),
                    reg_lambda=trial.suggest_float("reg_lambda", 0.1, 50, log=True),
                    gamma=trial.suggest_float("gamma", 1e-3, 5, log=True),
                    learning_rate=trial.suggest_float("learning_rate", 0.005, 0.08, log=True))
    if member == "cat":
        return dict(depth=trial.suggest_int("depth", 3, 8),
                    l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 30, log=True),
                    rsm=trial.suggest_float("rsm", 0.25, 0.9),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
                    bagging_temperature=trial.suggest_float("bagging_temperature", 0, 1))
    if member == "ridge":
        return dict(alpha=trial.suggest_float("alpha", 0.1, 1000, log=True)) if kind == "regression" else \
            dict(C=trial.suggest_float("C", 1e-3, 10, log=True))
    raise ValueError(member)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=list(GAME_TARGET_COL))
    ap.add_argument("--member", required=True, choices=["lgbm", "xgb", "cat", "ridge"])
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--timeout", type=int, default=None, help="seconds")
    ap.add_argument("--max-folds", type=int, default=4)
    ap.add_argument("--min-train-seasons", type=int, default=6)
    ap.add_argument("--n-estimators", type=int, default=2000)
    ap.add_argument("--storage", default=None, help="optuna storage url, e.g. sqlite:///optuna_studies/game.db")
    args = ap.parse_args()
    cfg = load_config()
    kind = TASK_KIND[args.target]
    gf = pd.read_parquet(FEATURES_DIR / "game_features.parquet")
    man = json.load(open(FEATURES_DIR / "game_manifest.json"))
    ycol = GAME_TARGET_COL[args.target]
    df = gf[(gf["played"] == 1) & gf[ycol].notna()].reset_index(drop=True)
    mpath = MODELS_DIR / "game" / f"{args.target}_metrics.json"
    feats = json.load(open(mpath))["features"] if mpath.exists() else man["features"]
    feats = [c for c in feats if c in df.columns]
    y = df[ycol].to_numpy(dtype=float)
    X = df[feats]
    folds = rolling_origin_folds(df, min_train_seasons=args.min_train_seasons, max_folds=args.max_folds,
                                 es_tail_frac=float(cfg.get("validation.early_stop_tail_frac", 0.12)))
    print(f"{args.target}/{args.member}: {len(df)} rows, {len(feats)} features, folds {[f.name for f in folds]}")

    def objective(trial: optuna.Trial) -> float:
        params = space(trial, args.member, kind)
        scores = []
        for f in folds:
            ens = GameEnsemble(args.target, (args.member,), args.n_estimators, 0.02, 42, param_overrides={args.member: params})
            ens.fit(X.iloc[f.fit_idx], y[f.fit_idx], X.iloc[f.es_idx], y[f.es_idx])
            pred = ens.predict_members(X.iloc[f.val_idx])[args.member].to_numpy()
            met = classification_metrics(y[f.val_idx], pred) if kind == "classification" else regression_metrics(y[f.val_idx], pred)
            scores.append(met["logloss" if kind == "classification" else "mae"])
            trial.report(float(np.mean(scores)), len(scores))
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize", study_name=f"game_{args.target}_{args.member}", storage=args.storage,
                                load_if_exists=True, pruner=optuna.pruners.MedianPruner(n_startup_trials=8))
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, show_progress_bar=True)
    best = dict(study.best_params)
    if args.member == "lgbm":
        best["subsample_freq"] = 1
    out = BEST_PARAMS_DIR / f"game_{args.target}_{args.member}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(best, fh, indent=2)
    print(f"best value {study.best_value:.4f}; params -> {out}")


if __name__ == "__main__":
    main()
