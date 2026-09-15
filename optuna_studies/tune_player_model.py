#!/usr/bin/env python
"""Optuna study for a player-level model (network or GBM engine).

    python optuna_studies/tune_player_model.py --target receiving_yards --engine nn --n-trials 40
    python optuna_studies/tune_player_model.py --target rushing_tds --engine gbm --n-trials 60

Objective: mean OOF negative log-likelihood (Gaussian for yards, Poisson for
TDs) over the most recent ``--max-folds`` season folds. Best params are
written to config/best_params/player_<target>_<engine>.json.
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
from nfl_pipeline.models.player_model import TARGET_POSITIONS, PlayerModel, player_metrics, target_kind  # noqa: E402
from nfl_pipeline.validation import rolling_origin_folds  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=list(TARGET_POSITIONS))
    ap.add_argument("--engine", default="nn", choices=["nn", "gbm"])
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=None)
    ap.add_argument("--max-folds", type=int, default=2)
    ap.add_argument("--min-train-seasons", type=int, default=8)
    ap.add_argument("--storage", default=None)
    args = ap.parse_args()
    cfg = load_config()
    kind = target_kind(args.target)
    pf = pd.read_parquet(FEATURES_DIR / "player_features.parquet")
    pman = json.load(open(FEATURES_DIR / "player_manifest.json"))
    df = pf[(pf["is_projection"] == 0) & pf["position"].isin(TARGET_POSITIONS[args.target]) & (pf["career_games"] >= 1)].reset_index(drop=True)
    mpath = MODELS_DIR / "player" / f"{args.target}_metrics.json"
    feats = json.load(open(mpath))["features"] if mpath.exists() else pman["features"]
    feats = [c for c in feats if c in df.columns]
    X, y = df[feats], df[args.target].to_numpy(dtype=float)
    folds = rolling_origin_folds(df, min_train_seasons=args.min_train_seasons, max_folds=args.max_folds)
    print(f"{args.target}/{args.engine}: {len(df)} rows, {len(feats)} features, folds {[f.name for f in folds]}")
    metric = "poisson_nll" if kind == "poisson" else "gauss_nll"

    def objective(trial: optuna.Trial) -> float:
        if args.engine == "nn":
            n_layers = trial.suggest_int("n_layers", 1, 3)
            width = trial.suggest_categorical("width", [64, 128, 256, 512])
            hidden = [max(32, width // (2 ** i)) for i in range(n_layers)]
            nn_params = dict(hidden=hidden, dropout=trial.suggest_float("dropout", 0.0, 0.4),
                             lr=trial.suggest_float("lr", 1e-4, 5e-3, log=True),
                             weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
                             batch_size=trial.suggest_categorical("batch_size", [256, 512, 1024]), max_epochs=120, patience=12)
            model_type, gbm_params = "nn", {}
        else:
            gbm_params = dict(num_leaves=trial.suggest_int("num_leaves", 7, 127, log=True),
                              learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                              n_estimators=2000)
            model_type, nn_params = "gbm", {}
        scores = []
        for f in folds:
            m = PlayerModel(args.target, model_type, nn_params, gbm_params, 42).fit(X.iloc[f.fit_idx], y[f.fit_idx], X.iloc[f.es_idx], y[f.es_idx])
            d = m.predict_dist(X.iloc[f.val_idx])
            scores.append(player_metrics(y[f.val_idx], d["mu"], d["sigma"], kind)[metric])
            trial.report(float(np.mean(scores)), len(scores))
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize", study_name=f"player_{args.target}_{args.engine}", storage=args.storage,
                                load_if_exists=True, pruner=optuna.pruners.MedianPruner(n_startup_trials=6))
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, show_progress_bar=True)
    best = dict(study.best_params)
    if args.engine == "nn":
        n_layers, width = best.pop("n_layers"), best.pop("width")
        best["hidden"] = [max(32, width // (2 ** i)) for i in range(n_layers)]
    out = BEST_PARAMS_DIR / f"player_{args.target}_{args.engine}.json"
    with open(out, "w") as fh:
        json.dump(best, fh, indent=2)
    print(f"best {metric} {study.best_value:.4f}; params -> {out}")


if __name__ == "__main__":
    main()
