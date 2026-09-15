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
from nfl_pipeline.models.game_ensemble import (TASK_KIND, GameEnsemble, classification_metrics, regression_metrics)
from nfl_pipeline.models.player_model import (TARGET_POSITIONS, PlayerModel, player_metrics, target_kind)
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
    mo = pd.read_parquet(path).set_index("game_id")["oof_margin"]
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
def train_game_target(target: str, gf: pd.DataFrame, man: dict, cfg, args) -> dict:
    kind = TASK_KIND[target]
    ycol = GAME_TARGET_COL[target]
    mcfg = cfg.get("models.game", {})
    vcfg = cfg.get("validation", {})
    df = gf[(gf["played"] == 1) & gf[ycol].notna()].reset_index(drop=True)
    feats = [c for c in man["features"] if c in df.columns]
    market = man["families"].get("market", [])
    use_market = mcfg.get("use_market_features", True) and not args.no_market
    if not use_market:
        feats = [c for c in feats if c not in market]
    y = df[ycol].to_numpy(dtype=float)
    members = tuple(mcfg.get("ensemble_members", ["lgbm", "xgb", "cat"])) + (("ridge",) if not args.no_ridge else ())
    n_est = 120 if args.smoke else int(mcfg.get("n_estimators", 1500))
    lr = 0.1 if args.smoke else float(mcfg.get("learning_rate", 0.02))
    seed = int(mcfg.get("seed", 42))

    folds = rolling_origin_folds(df, min_train_seasons=args.min_train_seasons or int(vcfg.get("min_train_seasons", 5)),
                                 fold_unit=args.fold_unit or vcfg.get("fold_unit", "season"),
                                 es_tail_frac=float(vcfg.get("early_stop_tail_frac", 0.12)),
                                 max_folds=args.max_folds)
    assert_folds_are_causal(df, folds)
    log.info("[%s] %s rows, %s candidate features, %s folds (%s .. %s)", target, len(df), len(feats), len(folds), folds[0].name, folds[-1].name)

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
    if target == "margin":
        pd.DataFrame({"game_id": df.loc[valid, "game_id"].to_numpy(), "oof_margin": blended}).to_parquet(
            out_dir / "margin_oof.parquet", index=False)
    elif target == "win":
        final.stack_, stack_report = _fit_win_stack(df.loc[valid, "game_id"].to_numpy(), blended, y[valid.to_numpy()], mk, out_dir)

    # final fit: select on all played rows, fixed iterations from CV
    with timed(f"[{target}] final selection + fit"):
        selected, report = _select(df, y, feats, kind, cfg, seed, always_keep, args)
        X = df[selected]
        fixed = {m: int(np.median(v) * 1.1) for m, v in best_iters.items()} if best_iters else None
        final.fit(X, y, fixed_iters=fixed)
    final.save(out_dir / f"{target}.joblib")
    if len(report):
        report.to_csv(out_dir / f"{target}_selection.csv")
    metrics = {"target": target, "kind": kind, "n_rows": int(len(df)), "n_features_candidate": len(feats),
               "n_features_selected": len(selected), "features": selected, "use_market_features": use_market,
               "n_features_by_fold": {k: len(v) for k, v in fold_features.items()}, "fold_features": fold_features,
               "members": list(members), "weights": final.weights_, "sigma": final.sigma_ if kind == "regression" else None,
               "platt": final.calib_ if kind == "classification" else None, "fixed_iters": fixed,
               "oof_overall": overall, "oof_by_member": per_member, "oof_by_season": per_season, "edge_buckets": edge_table,
               "win_stack": stack_report, "folds": fold_rows}
    with open(out_dir / f"{target}_metrics.json", "w") as fh:
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
    log.info("[%s] %s rows (%s), %s features, %s folds (%s .. %s)", target, len(df), "/".join(TARGET_POSITIONS[target]),
             len(feats), len(folds), folds[0].name, folds[-1].name)

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
        model = PlayerModel(target, model_type, nn_params, gbm_params, seed)
        model.fit(Xf.iloc[f.fit_idx], y[f.fit_idx], Xf.iloc[f.es_idx], y[f.es_idx])
        d = model.predict_dist(Xf.iloc[f.val_idx])
        oof_mu[f.val_idx], oof_sigma[f.val_idx] = d["mu"].to_numpy(), d["sigma"].to_numpy()
        met = player_metrics(y[f.val_idx], oof_mu[f.val_idx], oof_sigma[f.val_idx], kind)
        # naive baseline: player's EWM of the target
        base = df.iloc[f.val_idx][f"{target}_ewm"].fillna(df.iloc[f.val_idx][target].mean()).to_numpy()
        met["baseline_ewm_mae"] = float(np.mean(np.abs(y[f.val_idx] - base)))
        met.update(fold=f.name, n_features=len(sel_f), secs=round(time.time() - t0, 1))
        fold_rows.append(met)
        log.info("[%s] fold %s: %s", target, f.name, _fmt({k: v for k, v in met.items() if k != "fold"}))

    valid = ~np.isnan(oof_mu)
    overall = player_metrics(y[valid], oof_mu[valid], oof_sigma[valid], kind)
    base = df.loc[valid, f"{target}_ewm"].fillna(df[target].mean()).to_numpy()
    overall["baseline_ewm_mae"] = float(np.mean(np.abs(y[valid] - base)))
    log.info("[%s] OOF: %s", target, _fmt(overall))

    with timed(f"[{target}] final selection + fit"):
        selected, report = _select(df, y, feats, "regression", cfg, seed, always_keep, args)
        X = df[selected]
        final = PlayerModel(target, model_type, nn_params, gbm_params, seed)
        final.calibrate_sigma(y[valid], oof_mu[valid], oof_sigma[valid])
        # early-stopping set: random 10% of the most recent three seasons (keeps recency in training)
        rng = np.random.default_rng(seed)
        recent = df["season"] >= df["season"].max() - 2
        es_mask = recent & (rng.random(len(df)) < 0.10)
        final.fit(X[~es_mask], y[~es_mask.to_numpy()], X[es_mask], y[es_mask.to_numpy()])
    out_dir = MODELS_DIR / "player"
    final.save(out_dir / f"{target}.joblib")
    if len(report):
        report.to_csv(out_dir / f"{target}_selection.csv")
    metrics = {"target": target, "kind": kind, "positions": TARGET_POSITIONS[target], "n_rows": int(len(df)),
               "n_features_selected": len(selected), "features": selected, "model_type": model_type,
               "n_features_by_fold": {k: len(v) for k, v in fold_features.items()}, "fold_features": fold_features,
               "sigma_scale": final.sigma_scale_, "oof_overall": overall, "folds": fold_rows}
    with open(out_dir / f"{target}_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", nargs="*", default=None, help="subset of: margin total win players <player_target>")
    ap.add_argument("--approve-features", action="store_true", help="approve the engineered features and proceed")
    ap.add_argument("--smoke", action="store_true", help="tiny models, 2 folds: verifies the pipeline end-to-end")
    ap.add_argument("--no-market", action="store_true", help="exclude closing-line features (pure power-rating model)")
    ap.add_argument("--no-ridge", action="store_true", help="drop the linear stabiliser from the game ensemble")
    ap.add_argument("--skip-selection", action="store_true")
    ap.add_argument("--fold-unit", default=None, choices=[None, "season", "half", "week"])
    ap.add_argument("--min-train-seasons", type=int, default=None)
    ap.add_argument("--max-folds", type=int, default=None, help="evaluate only the most recent N folds")
    ap.add_argument("--player-max-folds", type=int, default=4, help="player models: most recent N season folds")
    ap.add_argument("--player-model", default=None, choices=[None, "nn", "gbm", "blend"])
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

    targets = args.targets or (list(cfg.get("models.game.targets", [])) + ["players"])
    game_targets = [t for t in targets if t in GAME_TARGET_COL]
    player_targets = []
    for t in targets:
        if t == "players":
            player_targets += list(cfg.get("models.player.targets", []))
        elif t in TARGET_POSITIONS:
            player_targets.append(t)

    summary = {"game": {}, "player": {}, "smoke": args.smoke, "no_market": args.no_market}
    if game_targets:
        gf = pd.read_parquet(FEATURES_DIR / "game_features.parquet")
        man = json.load(open(FEATURES_DIR / "game_manifest.json"))
        for t in game_targets:
            with timed(f"GAME MODEL: {t}"):
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
