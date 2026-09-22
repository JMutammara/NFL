#!/usr/bin/env python
"""Generate predictions for one NFL week: games (spread / moneyline / total) and player props.

    python predict_week.py --season 2026 --week 2
    python predict_week.py --season 2026 --week 2 --refresh        # pull latest nflverse data, lines, forecasts; rebuild features
    python predict_week.py --season 2026 --week 2 --props my_props.csv
    python predict_week.py --season 2026 --week 2 --bet-line open  # price against the opening line instead of the current one

Outputs (predictions/):
    <season>_week<ww>_games.csv       one row per game: market-free power numbers, opening / current lines,
                                      edge-model means, calibrated cover / over / win probabilities, EV, Kelly stakes
    <season>_week<ww>_players.csv     one row per player with mu / sigma / quantiles / TD probabilities
    <season>_week<ww>_props.csv       (if --props) P(over) and EV for a user-supplied prop sheet
    <season>_week<ww>_dashboard.html  self-contained interactive dashboard (open in a browser)
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
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm

from nfl_pipeline.config import FEATURES_DIR, MODELS_DIR, PREDICTIONS_DIR, load_config
from nfl_pipeline.ingest import load_schedules
from nfl_pipeline.lines import fetch_lines
from nfl_pipeline.models.betting import (expected_value, kelly_fraction, market_edge_pct, poisson_prob_at_least,
                                         prob_cover_home, prob_over)
from nfl_pipeline.models.edge_model import EdgeModel, build_edge_frame
from nfl_pipeline.models.game_ensemble import GameEnsemble
from nfl_pipeline.models.player_model import TARGET_POSITIONS, PlayerModel, dist_prob_over, target_kind
from nfl_pipeline.utils import american_to_prob, log
from nfl_pipeline.weather import apply_forecasts, fetch_forecasts

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _load(path: Path, cls):
    return cls.load(path) if path.exists() else None


def load_game_models() -> dict:
    d = MODELS_DIR / "game"
    models = {
        "pf_margin": _load(d / "pf_margin.joblib", GameEnsemble), "pf_total": _load(d / "pf_total.joblib", GameEnsemble),
        "edge_spread": _load(d / "edge_spread.joblib", EdgeModel), "edge_total": _load(d / "edge_total.joblib", EdgeModel),
        "margin": _load(d / "margin.joblib", GameEnsemble), "total": _load(d / "total.joblib", GameEnsemble),
        "win": _load(d / "win.joblib", GameEnsemble),
    }
    if models["edge_spread"] is None and models["margin"] is None:
        raise SystemExit("no game models found; run train_models.py first")
    return models


# ---------------------------------------------------------------------------
# Lines at bet time
# ---------------------------------------------------------------------------
def attach_lines(g: pd.DataFrame, sched: pd.DataFrame, cfg, refresh: bool) -> pd.DataFrame:
    """Add opening / current lines for the games in ``g`` (ESPN feed refreshed for the current season)."""
    rows = sched[sched["game_id"].isin(g["game_id"])]
    lines = fetch_lines(rows, cfg, refresh=refresh, workers=8).set_index("game_id")
    g = g.copy()
    for c in ("spread_open", "total_open", "ml_h_open", "ml_a_open", "spread_close_espn", "total_close_espn", "ml_h_close", "ml_a_close"):
        g[c] = g["game_id"].map(lines[c]) if c in lines else np.nan
    s = sched.set_index("game_id")
    g["spread_now"] = g["spread_close_espn"].fillna(g["game_id"].map(s["spread_line"])).fillna(g["spread_open"])
    g["total_now"] = g["total_close_espn"].fillna(g["game_id"].map(s["total_line"])).fillna(g["total_open"])
    g["ml_home_now"] = g["ml_h_close"].fillna(g["game_id"].map(s["home_moneyline"]))
    g["ml_away_now"] = g["ml_a_close"].fillna(g["game_id"].map(s["away_moneyline"]))
    # keep the feature columns the models were trained on consistent with the freshest numbers
    g["spread_open"] = g["spread_open"].fillna(g["spread_now"])
    g["total_open"] = g["total_open"].fillna(g["total_now"])
    return g


# ---------------------------------------------------------------------------
# Games
# ---------------------------------------------------------------------------
def _side_pick(out: pd.DataFrame, p_home: np.ndarray, odds_home, odds_away, home_label, away_label, prefix: str, kf: float, cap: float, bankroll: float) -> None:
    p_home = np.clip(np.asarray(p_home, dtype=float), 1e-6, 1 - 1e-6)
    ev_h, ev_a = expected_value(p_home, odds_home), expected_value(1 - p_home, odds_away)
    home_side = ev_h >= ev_a
    out[f"{prefix}_pick"] = np.where(home_side, home_label, away_label)
    out[f"{prefix}_pick_p"] = np.where(home_side, p_home, 1 - p_home)
    out[f"{prefix}_ev"] = np.where(home_side, ev_h, ev_a)
    odds = np.where(home_side, odds_home, odds_away)
    out[f"{prefix}_pick_odds"] = odds
    out[f"{prefix}_edge_pct"] = market_edge_pct(out[f"{prefix}_pick_p"], odds)
    out[f"{prefix}_stake"] = kelly_fraction(out[f"{prefix}_pick_p"], odds, kf, cap) * bankroll


def predict_games(g: pd.DataFrame, models: dict, sched: pd.DataFrame, cfg, bankroll: float, bet_line: str = "current") -> pd.DataFrame:
    bcfg = cfg.get("betting", {})
    vig = bcfg.get("default_vig_odds", -110)
    kf, cap = bcfg.get("kelly_fraction", 0.25), bcfg.get("max_stake_frac", 0.03)
    s = sched.set_index("game_id")
    out = g[["game_id", "season", "week", "gameday", "home_team", "away_team", "home_qb_name", "away_qb_name",
             "spread_open", "spread_now", "total_open", "total_now", "ml_home_now", "ml_away_now", "ml_h_open", "ml_a_open",
             "temp_f", "wind_mph", "weather_imputed", "is_dome", "home_rest", "away_rest"]].copy()
    out["kickoff"] = out["game_id"].map(s["gametime"])
    out["stadium"] = out["game_id"].map(s["stadium"])
    for c in ("home_spread_odds", "away_spread_odds", "over_odds", "under_odds"):
        out[c] = out["game_id"].map(s[c]).fillna(vig) if c in s.columns else vig
    out["spread_open_to_now"] = out["spread_now"] - out["spread_open"]
    out["total_open_to_now"] = out["total_now"] - out["total_open"]
    line_sp = out["spread_now"] if bet_line == "current" else out["spread_open"]
    line_to = out["total_now"] if bet_line == "current" else out["total_open"]
    out["bet_spread_line"], out["bet_total_line"] = line_sp, line_to

    use_edge = models.get("edge_spread") is not None and models.get("pf_margin") is not None
    out["model_type"] = "edge" if use_edge else "legacy"
    if use_edge:
        pf_m = models["pf_margin"].predict(g)
        pf_t = models["pf_total"].predict(g)
        out["pf_margin"], out["pf_total"] = pf_m, pf_t
        es, et = models["edge_spread"], models["edge_total"]
        # The residual model is evaluated relative to the line being bet: the current line for live
        # bets (line moves carry information, so the open is not a valid anchor once it has moved)
        # and the opening line for the "at the open" view. Each anchor gets its own feature frame.
        frames = {"open": build_edge_frame(g, pf_m, pf_t, ref="open"),
                  "current": build_edge_frame(g, pf_m, pf_t, spread_col="spread_now", total_col="total_now")}
        res = {}
        for key, e in frames.items():
            res[key] = (es.predict(e[es.features_], e["spread_ref"].to_numpy()), et.predict(e[et.features_], e["total_ref"].to_numpy()))
        ps, pt = res[bet_line]
        ps_open, pt_open = res["open"]
        ps_now, pt_now = res["current"]
        out["model_margin"] = ps["mu"].to_numpy()
        out["model_margin_open"] = ps_open["mu"].to_numpy()
        out["model_margin_now"] = ps_now["mu"].to_numpy()
        out["spread_edge_pts"] = ps["edge_pts"].to_numpy()
        out["p_home_cover"] = ps["p_cover"].to_numpy()
        out["p_home_cover_open"] = ps_open["p_cover"].to_numpy()
        out["p_home_cover_now"] = ps_now["p_cover"].to_numpy()
        out["p_home_cover_normal"] = ps["p_cover_normal"].to_numpy()
        out["margin_sigma"] = es.sigma_
        out["model_total"] = pt["mu"].to_numpy()
        out["model_total_open"] = pt_open["mu"].to_numpy()
        out["model_total_now"] = pt_now["mu"].to_numpy()
        out["total_edge_pts"] = pt["edge_pts"].to_numpy()
        out["p_over"] = pt["p_cover"].to_numpy()
        out["p_over_open"] = pt_open["p_cover"].to_numpy()
        out["p_over_now"] = pt_now["p_cover"].to_numpy()
        out["total_sigma"] = et.sigma_
        out["p_home_win"] = ps["p_home_win"].to_numpy()
    else:  # legacy market-aware ensemble with Normal tails
        m = models["margin"]
        out["model_margin"] = m.predict(g)
        out["margin_sigma"] = m.sigma_ if np.isfinite(m.sigma_) else 13.5
        out["spread_edge_pts"] = out["model_margin"] - line_sp
        out["p_home_cover"] = prob_cover_home(out["model_margin"], out["margin_sigma"], line_sp)
        t = models["total"]
        out["model_total"] = t.predict(g)
        out["total_sigma"] = t.sigma_ if np.isfinite(t.sigma_) else 13.5
        out["total_edge_pts"] = out["model_total"] - line_to
        out["p_over"] = prob_over(out["model_total"], out["total_sigma"], line_to)
        p_margin = 1 - norm.cdf(-out["model_margin"] / out["margin_sigma"])
        if models.get("win") is not None:
            wm = models["win"]
            p_cls = np.clip(wm.predict(g), 1e-6, 1 - 1e-6)
            stack = getattr(wm, "stack_", None)
            if stack:
                z = stack["coef_cls"] * logit(p_cls) + stack["coef_margin"] * logit(np.clip(p_margin, 1e-6, 1 - 1e-6)) + stack["intercept"]
                out["p_home_win"] = expit(z)
            else:
                out["p_home_win"] = 0.5 * (p_cls + p_margin)
        else:
            out["p_home_win"] = p_margin
    out["p_away_cover"] = 1 - out["p_home_cover"]
    home_lbl = out["home_team"] + " " + (-line_sp).map(lambda v: f"{v:+.1f}" if pd.notna(v) else "n/a")
    away_lbl = out["away_team"] + " " + line_sp.map(lambda v: f"{v:+.1f}" if pd.notna(v) else "n/a")
    _side_pick(out, out["p_home_cover"], out["home_spread_odds"], out["away_spread_odds"], home_lbl, away_lbl, "spread", kf, cap, bankroll)
    _side_pick(out, out["p_over"], out["over_odds"], out["under_odds"], "OVER " + line_to.map(lambda v: f"{v:.1f}" if pd.notna(v) else "n/a"),
               "UNDER " + line_to.map(lambda v: f"{v:.1f}" if pd.notna(v) else "n/a"), "total", kf, cap, bankroll)
    ml_h = out["ml_home_now"].fillna(-110)
    ml_a = out["ml_away_now"].fillna(-110)
    out["market_p_home_win"] = american_to_prob(ml_h) / (american_to_prob(ml_h) + american_to_prob(ml_a))
    _side_pick(out, out["p_home_win"], ml_h, ml_a, out["home_team"], out["away_team"], "ml", kf, cap, bankroll)
    return out


# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------
def projected_starters(sched: pd.DataFrame, game_ids) -> pd.DataFrame:
    """Expected starting QB per (game_id, team) from the schedule.

    nflverse fills ``home_qb_id`` / ``away_qb_id`` for upcoming games with the expected
    starter. Measured against the actual pass-attempt leader on 2,238 completed team-games
    from 2022 on, it is right 95.1% of the time -- far better than the depth chart, which
    lags benchings and injury returns by days.
    """
    s = sched[sched["game_id"].isin(list(game_ids))]
    home = s[["game_id", "home_team", "home_qb_id", "home_qb_name"]].rename(
        columns={"home_team": "team", "home_qb_id": "starter_id", "home_qb_name": "starter_name"})
    away = s[["game_id", "away_team", "away_qb_id", "away_qb_name"]].rename(
        columns={"away_team": "team", "away_qb_id": "starter_id", "away_qb_name": "starter_name"})
    return pd.concat([home, away], ignore_index=True).dropna(subset=["starter_id"])


def predict_players(p: pd.DataFrame, cfg, players_per_team: int, sched: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Returns (projections, transforms) where transforms maps target -> 'none' | 'log1p' | 'sqrt'."""
    models = {}
    for t in cfg.get("models.player.targets", []):
        path = MODELS_DIR / "player" / f"{t}.joblib"
        if path.exists():
            models[t] = PlayerModel.load(path)
    transforms = {t: getattr(m, "transform", "none") for t, m in models.items()}
    if not models:
        log.warning("no player models found; skipping player projections")
        return pd.DataFrame(), transforms
    # Exclude only players confirmed out on a published report. Where the week's report is not
    # yet filed, statuses are carried forward as expected severity (see carry_forward_injuries)
    # and stay in the projection set, flagged, because a player listed Out last week plays more
    # often than not the following week.
    active = (p["inj_status"].fillna(0) < 3)
    # Quarterbacks: take the schedule's expected starter rather than the depth chart, which goes
    # stale on benchings and injury returns. Backups are dropped: only one QB per team carries a
    # meaningful passing projection, and the model's depth-chart feature cannot be trusted to
    # rank an unsettled room.
    starters = projected_starters(sched, p["game_id"].unique()) if sched is not None else pd.DataFrame()
    if len(starters):
        key = set(zip(starters["game_id"], starters["starter_id"]))
        is_starter = pd.Series([(g, i) in key for g, i in zip(p["game_id"], p["player_id"])], index=p.index)
        teams_with_starter = set(zip(starters["game_id"], starters["team"]))
        team_covered = pd.Series([(g, t) in teams_with_starter for g, t in zip(p["game_id"], p["team"])], index=p.index)
        qb_role = np.where(team_covered, is_starter, (p["depth_rank"] == 1) | (p["snap_pct_ewm"] >= 0.5))
    else:
        is_starter = pd.Series(False, index=p.index)
        qb_role = (p["depth_rank"] == 1) | (p["snap_pct_ewm"] >= 0.5)
    role = ((p["position"] == "QB") & pd.Series(qb_role, index=p.index)) | \
           ((p["position"] != "QB") & ((p["depth_rank"] <= 3) | (p["snap_pct_ewm"] >= 0.25) | (p["target_share_ewm"] >= 0.08)))
    u = p[active & role].copy()
    u["is_projected_starter"] = is_starter.reindex(u.index).fillna(False).astype(int)
    u = u.sort_values(["team", "position", "snap_pct_ewm"], ascending=[True, True, False])
    u = u.groupby(["team", "position"]).head(players_per_team).copy()
    keep_ids = ["player_id", "player_name", "position", "team", "opponent", "is_home", "season", "week", "game_id",
                "depth_rank", "inj_status", "snap_pct_ewm", "target_share_ewm", "carry_share_ewm", "team_implied_pts", "team_spread",
                "total_line"]
    keep_ids += [c for c in ("inj_report_available", "inj_carried", "inj_carried_from_week", "inj_prior_status", "is_projected_starter") if c in u.columns]
    out = u[keep_ids].copy()
    if "inj_prior_status" in out.columns:
        lab = {3.0: "Out", 2.0: "Doubtful", 1.0: "Questionable", 0.0: "Probable"}
        out["status_note"] = np.where(
            out["inj_carried"].fillna(0) == 1,
            "no report yet; wk " + out["inj_carried_from_week"].fillna(0).astype(int).astype(str) + " "
            + out["inj_prior_status"].map(lab).fillna("listed"),
            np.where(out["inj_report_available"].fillna(0) == 1,
                     np.where(out["inj_status"].fillna(0) > 0, "listed this week", "cleared this week"), "no report"))
    for t, model in models.items():
        mask = u["position"].isin(TARGET_POSITIONS[t])
        if not mask.any():
            continue
        rows = u.loc[mask]
        d = model.predict_dist(rows)
        out.loc[mask, f"{t}_mu"] = d["mu"].to_numpy()          # mean of the calibrated distribution
        out.loc[mask, f"{t}_median"] = d["median"].to_numpy()
        out.loc[mask, f"{t}_sigma"] = d["sigma"].to_numpy()    # ~ one standard deviation on the original scale
        out.loc[mask, f"{t}_mu_t"] = d["mu_t"].to_numpy()
        out.loc[mask, f"{t}_sigma_t"] = d["sigma_t"].to_numpy()
        if target_kind(t) == "poisson":
            out.loc[mask, f"{t}_p_any"] = poisson_prob_at_least(d["mu"].to_numpy(), 1)
            out.loc[mask, f"{t}_p_2plus"] = poisson_prob_at_least(d["mu"].to_numpy(), 2)
        else:
            out.loc[mask, f"{t}_q10"] = model.quantile(rows, 0.10)
            out.loc[mask, f"{t}_q50"] = d["median"].to_numpy()
            out.loc[mask, f"{t}_q90"] = model.quantile(rows, 0.90)
    return out.sort_values(["team", "position", "snap_pct_ewm"], ascending=[True, True, False]).reset_index(drop=True), transforms


def price_props(props: pd.DataFrame, players: pd.DataFrame, cfg, bankroll: float, transforms: dict | None = None) -> pd.DataFrame:
    """props columns: player_name, stat, line, over_odds, under_odds (odds optional -> -110)."""
    bcfg = cfg.get("betting", {})
    vig = bcfg.get("default_vig_odds", -110)
    transforms = transforms or {}
    rows = []
    idx = players.set_index(players["player_name"].str.lower())
    for _, r in props.iterrows():
        name, stat, line = str(r["player_name"]).lower(), str(r["stat"]), float(r["line"])
        if name not in idx.index or f"{stat}_mu" not in players.columns:
            rows.append({**r.to_dict(), "note": "no projection"})
            continue
        pr = idx.loc[name]
        pr = pr.iloc[0] if isinstance(pr, pd.DataFrame) else pr
        mu, sig = float(pr[f"{stat}_mu"]), float(pr[f"{stat}_sigma"])
        if pd.isna(mu):
            rows.append({**r.to_dict(), "note": "no projection for this stat"})
            continue
        p_over = float(dist_prob_over(float(pr[f"{stat}_mu_t"]), float(pr[f"{stat}_sigma_t"]), line, transforms.get(stat, "none"), target_kind(stat)))
        oo, uo = float(r.get("over_odds", vig) or vig), float(r.get("under_odds", vig) or vig)
        ev_o, ev_u = float(expected_value(p_over, oo)), float(expected_value(1 - p_over, uo))
        side = "OVER" if ev_o >= ev_u else "UNDER"
        p_side, odds = (p_over, oo) if side == "OVER" else (1 - p_over, uo)
        rows.append({**r.to_dict(), "model_mu": mu, "model_sigma": sig, "p_over": p_over, "pick": side, "pick_p": p_side,
                     "ev": max(ev_o, ev_u), "edge_pct": float(market_edge_pct(p_side, odds)),
                     "stake": float(kelly_fraction(p_side, odds, bcfg.get("kelly_fraction", 0.25), bcfg.get("max_stake_frac", 0.03)) * bankroll)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def _metrics_payload() -> dict:
    out = {}
    d = MODELS_DIR / "game"
    for k in ("edge_spread", "edge_total", "pf_margin", "pf_total"):
        p = d / f"{k}_metrics.json"
        if p.exists():
            m = json.load(open(p))
            m.pop("fold_features", None)
            m.pop("features", None)
            out[k] = m
    pl = {}
    for p in sorted((MODELS_DIR / "player").glob("*_metrics.json")):
        m = json.load(open(p))
        pl[m["target"]] = {"oof_overall": {k: v for k, v in m["oof_overall"].items() if k != "td_reliability"}, "n_features_selected": m["n_features_selected"],
                           "sigma_scale": m.get("sigma_scale"), "transform": m.get("transform", "none")}
    out["players"] = pl
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--week", type=int, default=None, help="defaults to the next unplayed week")
    ap.add_argument("--refresh", action="store_true", help="re-download current season data / lines / forecasts and rebuild features")
    ap.add_argument("--rebuild-features", action="store_true", help="rebuild features from cached raw data")
    ap.add_argument("--bet-line", default="current", choices=["current", "open"], help="line to price bets against")
    ap.add_argument("--no-forecast", action="store_true", help="skip kickoff weather forecasts (use climatology)")
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--min-edge", type=float, default=None, help="only print plays with >= this %% edge")
    ap.add_argument("--players-per-team", type=int, default=6, help="max players per team-position in the props sheet")
    ap.add_argument("--props", default=None, help="CSV of props to price: player_name, stat, line[, over_odds, under_odds]")
    ap.add_argument("--no-players", action="store_true")
    ap.add_argument("--no-dashboard", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    season = args.season or cfg.current_season

    if args.refresh or args.rebuild_features:
        from build_features import build
        gf, _, pf, _ = build(cfg, refresh=args.refresh)
    else:
        gf = pd.read_parquet(FEATURES_DIR / "game_features.parquet")
        pf = pd.read_parquet(FEATURES_DIR / "player_features.parquet")
    sched = load_schedules(cfg)
    if args.week is None:
        up = gf[(gf["season"] == season) & (gf["played"] == 0)]
        if up.empty:
            raise SystemExit(f"no unplayed games in season {season}")
        args.week = int(up["week"].min())
    g = gf[(gf["season"] == season) & (gf["week"] == args.week)].copy()
    if g.empty:
        raise SystemExit(f"no games for season {season} week {args.week}")
    if (g["played"] == 1).any():
        log.warning("week %s already has %s played games: these are in-sample numbers, not forecasts", args.week, int(g["played"].sum()))

    # freshest lines and kickoff weather for the slate
    g = attach_lines(g, sched, cfg, refresh=True)
    p = pf[(pf["season"] == season) & (pf["week"] == args.week)]
    fc = pd.DataFrame()
    if not args.no_forecast:
        fc = fetch_forecasts(sched[sched["game_id"].isin(g["game_id"])], sched, cfg)
        g = apply_forecasts(g, fc)
        p = apply_forecasts(p, fc)

    models = load_game_models()
    games = predict_games(g, models, sched, cfg, args.bankroll, bet_line=args.bet_line)
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    tag = f"{season}_week{args.week:02d}"
    games.to_csv(PREDICTIONS_DIR / f"{tag}_games.csv", index=False)

    print(f"\n=== GAMES: {season} week {args.week} ({len(games)} games, {games['model_type'].iloc[0]} models, bets priced at the {args.bet_line} line) ===")
    show = ["away_team", "home_team", "spread_open", "spread_now", "model_margin", "spread_pick", "spread_pick_p", "spread_edge_pct", "spread_stake",
            "total_open", "total_now", "model_total", "total_pick", "total_pick_p", "total_edge_pct", "p_home_win", "market_p_home_win", "ml_pick", "ml_edge_pct"]
    show = [c for c in show if c in games.columns]
    disp = games[show].copy()
    for c in disp.select_dtypes(include="number").columns:
        disp[c] = disp[c].round(2)
    print(disp.to_string(index=False))
    min_edge = args.min_edge if args.min_edge is not None else cfg.get("betting.min_edge_pct", 2.0)
    plays = []
    for kind, pcol, ecol, scol, ppcol in (("SPREAD", "spread_pick", "spread_edge_pct", "spread_stake", "spread_pick_p"),
                                          ("TOTAL", "total_pick", "total_edge_pct", "total_stake", "total_pick_p"),
                                          ("ML", "ml_pick", "ml_edge_pct", "ml_stake", "ml_pick_p")):
        for _, r in games[games[ecol] >= min_edge].iterrows():
            plays.append((kind, f"{r['away_team']}@{r['home_team']}", r[pcol], r[ppcol], r[ecol], r[scol]))
    if plays:
        print(f"\n--- plays with >= {min_edge:.1f}% edge (bankroll {args.bankroll:.0f}) ---")
        for k, gm, lbl, pp, e, st in sorted(plays, key=lambda x: -x[4]):
            print(f"  {k:<6} {gm:<9} {lbl:<18} p={pp:.3f}  edge={e:+.1f}%  stake={st:.0f}")
    else:
        print(f"\n(no plays clear the {min_edge:.1f}% edge threshold)")

    players = pd.DataFrame()
    priced = pd.DataFrame()
    transforms: dict = {}
    if not args.no_players:
        players, transforms = predict_players(p, cfg, args.players_per_team, sched)
        if len(players):
            players.to_csv(PREDICTIONS_DIR / f"{tag}_players.csv", index=False)
            print(f"\n=== PLAYERS: {len(players)} projections -> {PREDICTIONS_DIR / (tag + '_players.csv')} ===")
            for pos, cols, sort in (("QB", ["passing_yards_mu", "passing_yards_sigma", "passing_tds_mu", "passing_tds_p_any", "rushing_yards_mu"], "passing_yards_mu"),
                                    ("RB", ["rushing_yards_mu", "rushing_yards_sigma", "rushing_tds_p_any", "receiving_yards_mu"], "rushing_yards_mu"),
                                    ("WR", ["receiving_yards_mu", "receiving_yards_sigma", "receiving_yards_q10", "receiving_yards_q90", "receiving_tds_p_any"], "receiving_yards_mu"),
                                    ("TE", ["receiving_yards_mu", "receiving_yards_sigma", "receiving_tds_p_any"], "receiving_yards_mu")):
                cols = [c for c in cols if c in players.columns]
                if not cols or sort not in players.columns:
                    continue
                top = players[players["position"] == pos].sort_values(sort, ascending=False).head(8)
                print(f"\n  top {pos} by {sort}:")
                print(top[["player_name", "team", "opponent"] + cols].round(2).to_string(index=False))
            if args.props:
                priced = price_props(pd.read_csv(args.props), players, cfg, args.bankroll, transforms)
                priced.to_csv(PREDICTIONS_DIR / f"{tag}_props.csv", index=False)
                print("\n=== PRICED PROPS ===")
                print(priced.round(3).to_string(index=False))

    if not args.no_dashboard:
        from nfl_pipeline.dashboard import write_dashboard

        payload = {
            "season": season, "week": args.week, "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "bet_line": args.bet_line, "bankroll": args.bankroll,
            "betting": {"breakeven_110": float(american_to_prob(-110)), "kelly_fraction": cfg.get("betting.kelly_fraction", 0.25),
                        "max_stake_frac": cfg.get("betting.max_stake_frac", 0.03), "min_edge_pct": min_edge},
            "games": json.loads(games.to_json(orient="records", date_format="iso")),
            "players": json.loads(players.to_json(orient="records")) if len(players) else [],
            "props": json.loads(priced.to_json(orient="records")) if len(priced) else [],
            "transforms": transforms,
            "forecasts": json.loads(fc.to_json(orient="records")) if len(fc) else [],
            "metrics": _metrics_payload(),
        }
        out_html = PREDICTIONS_DIR / f"{tag}_dashboard.html"
        write_dashboard(payload, out_html)
        print(f"\ndashboard -> {out_html}")
    print(f"written -> {PREDICTIONS_DIR / (tag + '_games.csv')}")


if __name__ == "__main__":
    main()
