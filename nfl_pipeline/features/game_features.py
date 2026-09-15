"""Assemble the game-level feature matrix (one row per game, home perspective).

Feature families
----------------
team_ewm      : season-long EWMA of every team metric, all four sides
                (home_off, home_def, away_off, away_def)
team_recent   : 3- and 6-game trailing means for the core metric subset
net/matchup   : strength differentials and offense-vs-defense matchup sums
adjusted      : opponent-adjusted core metrics (raw minus opponent's prior EWM
                relative to the league mean), rolled the same way
qb            : starting-QB rolling EPA / CPOE and QB-change flag
context       : rest, travel, body clock, weather, venue, schedule slot
market        : closing spread / total / vig-free moneyline / implied totals
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import Config, FEATURES_DIR
from ..utils import log, timed
from .context import CONTEXT_FEATURES, MARKET_FEATURES, build_game_context
from .rolling import add_rolling_features
from .team_efficiency import metric_columns

CORE_METRICS = [
    "epa_pp", "epa_pp_neutral", "pass_epa_db", "rush_epa", "sr", "pass_sr", "rush_sr", "ed_epa", "cpoe",
    "explosive_rate", "sack_rate", "turnover_rate", "ppd", "drive_score_rate", "rz_td_rate", "third_conv",
    "sec_per_play", "plays", "pass_rate_neutral", "start_yl100",
]
ADJ_METRICS = ["epa_pp", "epa_pp_neutral", "pass_epa_db", "rush_epa", "sr", "pass_sr", "rush_sr", "ppd"]
LINE_METRICS = [  # OL vs DL matchup family (present depending on era)
    "sack_rate", "qb_hit_rate", "stuff_rate", "tfl_rate", "ed_ypc", "rush_sr", "pressure_rate_trk",
    "time_to_throw", "pass_rushers_trk", "box_count_trk", "blitz_rate_ftn", "pfr_pressured_rate",
    "pfr_pressures_rate", "pfr_blitzes_rate", "pfr_blitzed_rate", "def_missed_tackle_rate",
]


def _league_prior_mean(tg: pd.DataFrame, col: str) -> pd.Series:
    """Expanding league mean of ``col`` over games strictly before each game date."""
    daily = tg.groupby("gameday")[col].agg(["sum", "count"]).sort_index()
    cum = daily.cumsum().shift(1)
    mean = (cum["sum"] / cum["count"]).rename("m")
    return tg["gameday"].map(mean)


def add_opponent_adjusted(tg: pd.DataFrame, metrics: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Add ``off_<m>_adj`` / ``def_<m>_adj`` using the opponent's *prior* EWM."""
    opp = tg[["game_id", "team"] + [f"{s}_{m}_ewm" for s in ("off", "def") for m in metrics if f"{s}_{m}_ewm" in tg]]
    opp = opp.rename(columns={"team": "opponent"})
    opp.columns = ["game_id", "opponent"] + [f"opp_{c}" for c in opp.columns[2:]]
    tg = tg.merge(opp, on=["game_id", "opponent"], how="left")
    new_cols, new = [], {}
    for m in metrics:
        if f"off_{m}" not in tg or f"def_{m}" not in tg:
            continue
        lm_off = _league_prior_mean(tg, f"off_{m}")
        lm_def = _league_prior_mean(tg, f"def_{m}")
        # offense faced a defense that allows opp_def_ewm; subtract its strength relative to league
        new[f"off_{m}_adj"] = tg[f"off_{m}"] - (tg[f"opp_def_{m}_ewm"] - lm_def)
        new[f"def_{m}_adj"] = tg[f"def_{m}"] - (tg[f"opp_off_{m}_ewm"] - lm_off)
        new_cols += [f"off_{m}_adj", f"def_{m}_adj"]
    tg = tg.drop(columns=[c for c in tg.columns if c.startswith("opp_")])
    tg = pd.concat([tg, pd.DataFrame(new, index=tg.index)], axis=1)
    return tg, new_cols


def add_qb_features(tg: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, list[str]]:
    """Rolling QB form keyed on the starting QB (follows the player across teams)."""
    q = tg.loc[tg["qb_id"].notna(), ["game_id", "team", "qb_id", "gameday", "season", "qb_epa_db", "qb_cpoe", "qb_dropbacks", "played"]].copy()
    q = add_rolling_features(q, ["qb_epa_db", "qb_cpoe", "qb_dropbacks"], group_col="qb_id", date_col="gameday",
                             windows=(3, 6), ewm_halflife=cfg.get("features.ewma_halflife", 4.0),
                             ewm_shrink=cfg.get("features.ewma_prior_shrink", 0.6), cross_season=True)
    q = q.rename(columns={"n_prior_games": "qb_career_starts", "n_prior_games_season": "qb_season_starts"})
    feats = [c for c in q.columns if c.endswith(("_r3", "_r6", "_ewm"))] + ["qb_career_starts", "qb_season_starts"]
    tg = tg.merge(q[["game_id", "team"] + feats], on=["game_id", "team"], how="left")
    tg = tg.sort_values(["team", "gameday", "week"]).reset_index(drop=True)
    prev_qb = tg.groupby("team")["qb_id"].shift(1)
    tg = tg.assign(qb_changed=((tg["qb_id"] != prev_qb) & prev_qb.notna()).astype(float))
    feats.append("qb_changed")
    return tg, feats


def roll_team_games(tg: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Rolling + adjusted + QB features on the team-game table. Returns (table, families)."""
    fcfg = cfg.get("features", {})
    windows = tuple(fcfg.get("rolling_windows", [3, 6]))
    hl = fcfg.get("ewma_halflife", 4.0)
    shrink = fcfg.get("ewma_prior_shrink", 0.6)
    cross = fcfg.get("rolling_cross_season", True)
    base = metric_columns(tg)
    with timed("rolling: raw team metrics"):
        tg = add_rolling_features(tg, base, "team", "gameday", windows=windows, ewm_halflife=hl,
                                  ewm_shrink=shrink, cross_season=cross, order_cols=["gameday", "week"])
    with timed("opponent-adjusted metrics"):
        tg, adj_cols = add_opponent_adjusted(tg, ADJ_METRICS)
        tg = add_rolling_features(tg, adj_cols, "team", "gameday", windows=windows, ewm_halflife=hl,
                                  ewm_shrink=shrink, cross_season=cross, order_cols=["gameday", "week"])
        tg = tg.loc[:, ~tg.columns.duplicated()]
    with timed("QB features"):
        tg, qb_feats = add_qb_features(tg, cfg)
    families = {
        "base_metrics": base, "adjusted_metrics": adj_cols, "qb": qb_feats,
        "windows": [f"r{w}" for w in windows] + ["ewm"],
    }
    return tg, families


def _side_frame(tg: pd.DataFrame, cols: list[str], prefix: str, is_home: int) -> pd.DataFrame:
    side = tg[tg["is_home"] == is_home][["game_id"] + cols].copy()
    side.columns = ["game_id"] + [f"{prefix}{c}" for c in cols]
    return side


def build_game_features(tg_rolled: pd.DataFrame, schedules: pd.DataFrame, cfg: Config, families: dict) -> tuple[pd.DataFrame, dict]:
    """Game-level matrix from the rolled team-game table."""
    windows = families["windows"]
    base = families["base_metrics"]
    adj = families["adjusted_metrics"]
    # per-side feature lists
    ewm_all = [f"{c}_ewm" for c in base + adj if f"{c}_ewm" in tg_rolled]
    core_cols = [f"{s}_{m}" for s in ("off", "def") for m in CORE_METRICS + LINE_METRICS if f"{s}_{m}" in base]
    core_cols += adj
    recent = [f"{c}_{w}" for c in core_cols for w in windows if w != "ewm" and f"{c}_{w}" in tg_rolled]
    side_cols = ewm_all + recent + families["qb"] + ["n_prior_games", "n_prior_games_season"]
    side_cols = list(dict.fromkeys(side_cols))

    home = _side_frame(tg_rolled, side_cols, "home_", 1)
    away = _side_frame(tg_rolled, side_cols, "away_", 0)
    ids = tg_rolled[tg_rolled["is_home"] == 1][["game_id", "season", "week", "game_type", "gameday", "team", "opponent",
                                                  "points_for", "points_against", "played", "qb_id", "qb_name"]]
    ids = ids.rename(columns={"team": "home_team", "opponent": "away_team", "points_for": "home_score",
                              "points_against": "away_score", "qb_id": "home_qb_id", "qb_name": "home_qb_name"})
    away_qb = tg_rolled[tg_rolled["is_home"] == 0][["game_id", "qb_id", "qb_name"]].rename(columns={"qb_id": "away_qb_id", "qb_name": "away_qb_name"})
    g = ids.merge(away_qb, on="game_id").merge(home, on="game_id").merge(away, on="game_id")

    # net strength & matchup sums for core metrics across windows (built in one concat)
    new: dict[str, pd.Series] = {}
    net_cols, matchup_cols = [], []
    for m in CORE_METRICS + [f"{a}_adj" for a in ADJ_METRICS]:
        for w in windows:
            ho, hd, ao, ad = (f"home_off_{m}_{w}", f"home_def_{m}_{w}", f"away_off_{m}_{w}", f"away_def_{m}_{w}")
            if all(c in g for c in (ho, hd, ao, ad)):
                new[f"net_{m}_{w}"] = (g[ho] - g[hd]) - (g[ao] - g[ad])
                new[f"mu_home_{m}_{w}"] = g[ho] + g[ad]
                new[f"mu_away_{m}_{w}"] = g[ao] + g[hd]
                net_cols.append(f"net_{m}_{w}")
                matchup_cols += [f"mu_home_{m}_{w}", f"mu_away_{m}_{w}"]
    # OL/DL line matchup advantages: home offensive line vs away defensive front and vice versa
    line_cols = []
    for m in LINE_METRICS:
        for w in windows:
            ho, ad, ao, hd = f"home_off_{m}_{w}", f"away_def_{m}_{w}", f"away_off_{m}_{w}", f"home_def_{m}_{w}"
            if all(c in g for c in (ho, ad, ao, hd)):
                new[f"line_home_{m}_{w}"] = g[ho] - g[ad]
                new[f"line_away_{m}_{w}"] = g[ao] - g[hd]
                line_cols += [f"line_home_{m}_{w}", f"line_away_{m}_{w}"]
    # QB differentials
    qb_diff = []
    for c in ("qb_epa_db_ewm", "qb_epa_db_r3", "qb_cpoe_ewm", "qb_career_starts"):
        if f"home_{c}" in g and f"away_{c}" in g:
            new[f"d_{c}"] = g[f"home_{c}"] - g[f"away_{c}"]
            qb_diff.append(f"d_{c}")
    new["d_n_prior_games_season"] = g["home_n_prior_games_season"] - g["away_n_prior_games_season"]
    g = pd.concat([g, pd.DataFrame(new, index=g.index)], axis=1)

    # context + market
    ctx = build_game_context(schedules[schedules["game_id"].isin(g["game_id"])], cfg)
    g = g.merge(ctx.drop(columns=["season", "week"]), on="game_id", how="left")

    # targets
    tgt = {}
    tgt["home_margin"] = g["home_score"] - g["away_score"]
    tgt["total_points"] = g["home_score"] + g["away_score"]
    tgt["home_win"] = pd.Series(np.where(tgt["home_margin"].isna(), np.nan, (tgt["home_margin"] > 0).astype(float)), index=g.index)
    tgt["home_cover"] = pd.Series(np.where(tgt["home_margin"].isna() | g["spread_line"].isna(), np.nan,
                                           np.sign(tgt["home_margin"] - g["spread_line"])), index=g.index)
    tgt["over"] = pd.Series(np.where(tgt["total_points"].isna() | g["total_line"].isna(), np.nan,
                                     np.sign(tgt["total_points"] - g["total_line"])), index=g.index)
    g = pd.concat([g, pd.DataFrame(tgt, index=g.index)], axis=1)

    id_cols = ["game_id", "season", "week", "game_type", "gameday", "home_team", "away_team", "home_qb_id", "home_qb_name",
               "away_qb_id", "away_qb_name", "home_score", "away_score", "played"]
    target_cols = ["home_margin", "total_points", "home_win", "home_cover", "over"]
    fam = {
        "team_ewm": [f"{p}{c}" for p in ("home_", "away_") for c in ewm_all],
        "team_recent": [f"{p}{c}" for p in ("home_", "away_") for c in recent],
        "net_strength": net_cols,
        "matchup": matchup_cols,
        "line_matchup": line_cols,
        "qb": [f"{p}{c}" for p in ("home_", "away_") for c in families["qb"]] + qb_diff,
        "experience": ["home_n_prior_games", "away_n_prior_games", "home_n_prior_games_season", "away_n_prior_games_season", "d_n_prior_games_season"],
        "context": CONTEXT_FEATURES,
        "market": MARKET_FEATURES,
    }
    all_feats = [c for f in fam.values() for c in f if c in g.columns]
    all_feats = list(dict.fromkeys(all_feats))
    manifest = {"id_cols": id_cols, "target_cols": target_cols, "families": {k: [c for c in v if c in g.columns] for k, v in fam.items()},
                "features": all_feats, "n_rows": int(len(g)), "n_played": int(g["played"].sum())}
    keep = list(dict.fromkeys(id_cols + target_cols + all_feats))  # 'week' is both an id and a feature
    g = g[keep].sort_values(["gameday", "game_id"]).reset_index(drop=True)
    log.info("game features: %s games (%s played) x %s features", len(g), int(g["played"].sum()), len(all_feats))
    return g, manifest


def save_game_features(g: pd.DataFrame, manifest: dict) -> None:
    g.to_parquet(FEATURES_DIR / "game_features.parquet", index=False)
    with open(FEATURES_DIR / "game_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
