#!/usr/bin/env python
"""Generate predictions for one NFL week: games (spread / moneyline / total) and player props.

    python predict_week.py --season 2026 --week 2
    python predict_week.py --season 2026 --week 2 --refresh        # pull latest nflverse data & rebuild features
    python predict_week.py --season 2026 --week 2 --props my_props.csv

Outputs (predictions/):
    <season>_week<ww>_games.csv    one row per game with model numbers, market lines, edges, EV, Kelly stakes
    <season>_week<ww>_players.csv  one row per player with mu / sigma / quantiles / TD probabilities
    <season>_week<ww>_props.csv    (if --props) P(over) and EV for a user-supplied prop sheet
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
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from nfl_pipeline.config import FEATURES_DIR, MODELS_DIR, PREDICTIONS_DIR, load_config
from nfl_pipeline.ingest import load_schedules
from nfl_pipeline.models.betting import (expected_value, kelly_fraction, market_edge_pct, poisson_prob_at_least,
                                         prob_cover_home, prob_over)
from nfl_pipeline.models.game_ensemble import GameEnsemble
from nfl_pipeline.models.player_model import TARGET_POSITIONS, PlayerModel, target_kind
from nfl_pipeline.utils import log

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)


def _load_game_models() -> dict[str, GameEnsemble]:
    out = {}
    for t in ("margin", "total", "win"):
        p = MODELS_DIR / "game" / f"{t}.joblib"
        if p.exists():
            out[t] = GameEnsemble.load(p)
    if not out:
        raise SystemExit("no game models found; run train_models.py first")
    return out


def predict_games(g: pd.DataFrame, models: dict[str, GameEnsemble], sched: pd.DataFrame, cfg, bankroll: float) -> pd.DataFrame:
    bcfg = cfg.get("betting", {})
    vig = bcfg.get("default_vig_odds", -110)
    kf, cap = bcfg.get("kelly_fraction", 0.25), bcfg.get("max_stake_frac", 0.03)
    s = sched.set_index("game_id")
    out = g[["game_id", "season", "week", "gameday", "home_team", "away_team", "home_qb_name", "away_qb_name",
             "spread_line", "total_line"]].copy()
    out["kickoff"] = out["game_id"].map(s["gametime"])
    for c in ("home_moneyline", "away_moneyline", "home_spread_odds", "away_spread_odds", "over_odds", "under_odds"):
        out[c] = out["game_id"].map(s[c]) if c in s.columns else np.nan
    for c in ("home_spread_odds", "away_spread_odds", "over_odds", "under_odds"):
        out[c] = out[c].fillna(vig)

    if "margin" in models:
        m = models["margin"]
        out["model_margin"] = m.predict(g)
        sig = m.sigma_ if np.isfinite(m.sigma_) else 13.5
        out["margin_sigma"] = sig
        out["spread_edge_pts"] = out["model_margin"] - out["spread_line"]
        p_home_cover = prob_cover_home(out["model_margin"], sig, out["spread_line"])
        out["p_home_cover"] = p_home_cover
        out["p_away_cover"] = 1.0 - p_home_cover
        ev_home = expected_value(p_home_cover, out["home_spread_odds"])
        ev_away = expected_value(1 - p_home_cover, out["away_spread_odds"])
        home_side = ev_home >= ev_away
        out["spread_pick"] = np.where(home_side, out["home_team"] + " " + (-out["spread_line"]).map(lambda v: f"{v:+.1f}"),
                                      out["away_team"] + " " + out["spread_line"].map(lambda v: f"{v:+.1f}"))
        out["spread_pick_p"] = np.where(home_side, p_home_cover, 1 - p_home_cover)
        out["spread_ev"] = np.where(home_side, ev_home, ev_away)
        odds = np.where(home_side, out["home_spread_odds"], out["away_spread_odds"])
        out["spread_edge_pct"] = market_edge_pct(out["spread_pick_p"], odds)
        out["spread_stake"] = kelly_fraction(out["spread_pick_p"], odds, kf, cap) * bankroll
        out["p_home_win_from_margin"] = 1 - norm.cdf(-out["model_margin"] / sig)
    if "total" in models:
        m = models["total"]
        out["model_total"] = m.predict(g)
        sig_t = m.sigma_ if np.isfinite(m.sigma_) else 13.5
        out["total_sigma"] = sig_t
        out["total_edge_pts"] = out["model_total"] - out["total_line"]
        p_over = prob_over(out["model_total"], sig_t, out["total_line"])
        out["p_over"] = p_over
        ev_o, ev_u = expected_value(p_over, out["over_odds"]), expected_value(1 - p_over, out["under_odds"])
        over_side = ev_o >= ev_u
        out["total_pick"] = np.where(over_side, "OVER", "UNDER")
        out["total_pick_p"] = np.where(over_side, p_over, 1 - p_over)
        out["total_ev"] = np.where(over_side, ev_o, ev_u)
        odds = np.where(over_side, out["over_odds"], out["under_odds"])
        out["total_edge_pct"] = market_edge_pct(out["total_pick_p"], odds)
        out["total_stake"] = kelly_fraction(out["total_pick_p"], odds, kf, cap) * bankroll
    if "win" in models:
        p_cls = models["win"].predict(g)
        out["p_home_win_classifier"] = p_cls
        out["p_home_win"] = 0.5 * (p_cls + out["p_home_win_from_margin"]) if "p_home_win_from_margin" in out else p_cls
    elif "p_home_win_from_margin" in out:
        out["p_home_win"] = out["p_home_win_from_margin"]
    if "p_home_win" in out:
        ph = out["p_home_win"]
        ev_h = expected_value(ph, out["home_moneyline"])
        ev_a = expected_value(1 - ph, out["away_moneyline"])
        home_side = ev_h >= ev_a
        out["ml_pick"] = np.where(home_side, out["home_team"], out["away_team"])
        out["ml_pick_p"] = np.where(home_side, ph, 1 - ph)
        out["ml_ev"] = np.where(home_side, ev_h, ev_a)
        odds = np.where(home_side, out["home_moneyline"], out["away_moneyline"])
        out["ml_edge_pct"] = market_edge_pct(out["ml_pick_p"], odds)
        out["ml_stake"] = kelly_fraction(out["ml_pick_p"], odds, kf, cap) * bankroll
    return out


def predict_players(p: pd.DataFrame, cfg, players_per_team: int) -> pd.DataFrame:
    models = {}
    for t in cfg.get("models.player.targets", []):
        path = MODELS_DIR / "player" / f"{t}.joblib"
        if path.exists():
            models[t] = PlayerModel.load(path)
    if not models:
        log.warning("no player models found; skipping player projections")
        return pd.DataFrame()
    # projection universe: healthy, on the depth chart or with a meaningful snap share
    active = (p["inj_status"].fillna(0) < 3)
    role = ((p["position"] == "QB") & ((p["depth_rank"] == 1) | (p["snap_pct_ewm"] >= 0.5))) | \
           ((p["position"] != "QB") & ((p["depth_rank"] <= 3) | (p["snap_pct_ewm"] >= 0.25) | (p["target_share_ewm"] >= 0.08)))
    u = p[active & role].copy()
    u = u.sort_values(["team", "position", "snap_pct_ewm"], ascending=[True, True, False])
    u = u.groupby(["team", "position"]).head(players_per_team).copy()
    out = u[["player_id", "player_name", "position", "team", "opponent", "is_home", "season", "week", "game_id",
             "depth_rank", "inj_status", "snap_pct_ewm", "team_implied_pts", "team_spread", "total_line"]].copy()
    for t, model in models.items():
        mask = u["position"].isin(TARGET_POSITIONS[t])
        if not mask.any():
            continue
        d = model.predict_dist(u.loc[mask])
        out.loc[mask, f"{t}_mu"] = d["mu"].to_numpy()
        out.loc[mask, f"{t}_sigma"] = d["sigma"].to_numpy()
        if target_kind(t) == "poisson":
            out.loc[mask, f"{t}_p_any"] = poisson_prob_at_least(d["mu"].to_numpy(), 1)
            out.loc[mask, f"{t}_p_2plus"] = poisson_prob_at_least(d["mu"].to_numpy(), 2)
        else:
            mu, s = d["mu"].to_numpy(), d["sigma"].to_numpy()
            out.loc[mask, f"{t}_q10"] = np.maximum(mu - 1.2816 * s, 0)
            out.loc[mask, f"{t}_q50"] = mu
            out.loc[mask, f"{t}_q90"] = mu + 1.2816 * s
    return out.sort_values(["team", "position", "snap_pct_ewm"], ascending=[True, True, False]).reset_index(drop=True)


def price_props(props: pd.DataFrame, players: pd.DataFrame, cfg, bankroll: float) -> pd.DataFrame:
    """props columns: player_name, stat, line, over_odds, under_odds (odds optional -> -110)."""
    bcfg = cfg.get("betting", {})
    vig = bcfg.get("default_vig_odds", -110)
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
        if target_kind(stat) == "poisson":
            p_over = float(poisson_prob_at_least(mu, int(np.ceil(line + 1e-9))))
        else:
            p_over = float(prob_over(mu, sig, line))
        oo, uo = float(r.get("over_odds", vig) or vig), float(r.get("under_odds", vig) or vig)
        ev_o, ev_u = float(expected_value(p_over, oo)), float(expected_value(1 - p_over, uo))
        side = "OVER" if ev_o >= ev_u else "UNDER"
        p_side, odds = (p_over, oo) if side == "OVER" else (1 - p_over, uo)
        rows.append({**r.to_dict(), "model_mu": mu, "model_sigma": sig, "p_over": p_over, "pick": side, "pick_p": p_side,
                     "ev": max(ev_o, ev_u), "edge_pct": float(market_edge_pct(p_side, odds)),
                     "stake": float(kelly_fraction(p_side, odds, bcfg.get("kelly_fraction", 0.25), bcfg.get("max_stake_frac", 0.03)) * bankroll)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--week", type=int, default=None, help="defaults to the next unplayed week")
    ap.add_argument("--refresh", action="store_true", help="re-download current season data and rebuild features")
    ap.add_argument("--rebuild-features", action="store_true", help="rebuild features from cached raw data")
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--min-edge", type=float, default=None, help="only print plays with >= this %% edge")
    ap.add_argument("--players-per-team", type=int, default=6, help="max players per team-position in the props sheet")
    ap.add_argument("--props", default=None, help="CSV of props to price: player_name, stat, line[, over_odds, under_odds]")
    ap.add_argument("--no-players", action="store_true")
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
        log.warning("week %s already has %s played games: these are backtest predictions, not forecasts", args.week, int(g["played"].sum()))

    models = _load_game_models()
    games = predict_games(g, models, sched, cfg, args.bankroll)
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    tag = f"{season}_week{args.week:02d}"
    games.to_csv(PREDICTIONS_DIR / f"{tag}_games.csv", index=False)

    print(f"\n=== GAMES: {season} week {args.week} ({len(games)} games) ===")
    show = ["away_team", "home_team", "spread_line", "model_margin", "spread_pick", "spread_pick_p", "spread_edge_pct", "spread_stake",
            "total_line", "model_total", "total_pick", "total_pick_p", "total_edge_pct", "p_home_win", "ml_pick", "ml_edge_pct", "ml_stake"]
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
        if pcol in games:
            for _, r in games[games[ecol] >= min_edge].iterrows():
                lbl = r[pcol] if kind != "TOTAL" else f"{r[pcol]} {r['total_line']}"
                plays.append((kind, f"{r['away_team']}@{r['home_team']}", lbl, r[ppcol], r[ecol], r[scol]))
    if plays:
        print(f"\n--- plays with >= {min_edge:.1f}% edge (bankroll {args.bankroll:.0f}) ---")
        for k, gm, lbl, p, e, st in sorted(plays, key=lambda x: -x[4]):
            print(f"  {k:<6} {gm:<9} {lbl:<18} p={p:.3f}  edge={e:+.1f}%  stake={st:.0f}")
    else:
        print(f"\n(no plays clear the {min_edge:.1f}% edge threshold)")

    if not args.no_players:
        p = pf[(pf["season"] == season) & (pf["week"] == args.week)]
        players = predict_players(p, cfg, args.players_per_team)
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
                top = players[players["position"] == pos].sort_values(sort, ascending=False).head(12)
                print(f"\n  top {pos} by {sort}:")
                print(top[["player_name", "team", "opponent"] + cols].round(2).to_string(index=False))
            if args.props:
                props = pd.read_csv(args.props)
                priced = price_props(props, players, cfg, args.bankroll)
                priced.to_csv(PREDICTIONS_DIR / f"{tag}_props.csv", index=False)
                print("\n=== PRICED PROPS ===")
                print(priced.round(3).to_string(index=False))
    print(f"\nwritten -> {PREDICTIONS_DIR / (tag + '_games.csv')}")


if __name__ == "__main__":
    main()
