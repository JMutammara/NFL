"""Assemble the player-level feature matrix.

Feature families
----------------
usage_roll   : 3/6-game & EWM of volume, shares, efficiency, high-value touches
volatility   : 6-game rolling std of each target (heteroscedastic signal)
team_ctx     : own team's prior-only rolling offence profile (pass volume, pace, EPA)
opp_def      : opponent's prior-only rolling defence profile and defence-vs-position
expected_vol : share x team volume interactions (expected targets / carries / dropbacks)
status       : injury designation, practice status, depth-chart rank, snap share trend
static       : position one-hot, age, experience, draft capital, size
game         : home/away, spread & implied totals from the team's perspective, weather, rest
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import Config, FEATURES_DIR
from ..utils import log, timed
from .players import DVP_METRICS, PLAYER_ROLL_METRICS, TARGETS
from .rolling import add_rolling_features

TEAM_CTX = [
    "off_epa_pp_ewm", "off_pass_epa_db_ewm", "off_rush_epa_ewm", "off_plays_ewm", "off_dropbacks_ewm", "off_rushes_ewm",
    "off_pass_rate_neutral_ewm", "off_pass_oe_neutral_ewm", "off_sec_per_play_ewm", "off_ppd_ewm", "off_rz_trips_ewm",
    "off_adot_ewm", "off_explosive_pass_rate_ewm", "off_sack_rate_ewm", "off_drives_ewm", "off_plays_r3", "off_dropbacks_r3",
    "off_rushes_r3", "off_pass_rate_neutral_r3", "off_epa_pp_r3",
    "qb_epa_db_ewm", "qb_cpoe_ewm", "qb_changed", "qb_career_starts",
]
OPP_CTX = [
    "def_epa_pp_ewm", "def_pass_epa_db_ewm", "def_rush_epa_ewm", "def_sr_ewm", "def_pass_sr_ewm", "def_rush_sr_ewm",
    "def_plays_ewm", "def_dropbacks_ewm", "def_rushes_ewm", "def_ppd_ewm", "def_explosive_pass_rate_ewm",
    "def_explosive_rush_rate_ewm", "def_sack_rate_ewm", "def_pressure_rate_trk_ewm", "def_man_cov_rate_trk_ewm",
    "def_box_count_trk_ewm", "def_light_box_rate_trk_ewm", "def_blitz_rate_ftn_ewm", "def_cpoe_ewm", "def_adot_ewm",
    "def_sec_per_play_ewm", "def_pass_rate_neutral_ewm", "def_epa_pp_r3", "def_pass_epa_db_r3", "def_rush_epa_r3",
    "off_plays_ewm", "off_sec_per_play_ewm",  # opponent's offence drives game pace / script
]


def build_player_features(pg: pd.DataFrame, dvp: pd.DataFrame, tg_rolled: pd.DataFrame, game_feats: pd.DataFrame,
                          cfg: Config) -> tuple[pd.DataFrame, dict]:
    fcfg = cfg.get("features", {})
    windows = tuple(fcfg.get("rolling_windows", [3, 6]))
    hl, shrink = fcfg.get("ewma_halflife", 4.0), fcfg.get("ewma_prior_shrink", 0.6)
    roll_cols = [c for c in PLAYER_ROLL_METRICS if c in pg.columns]

    with timed("player rolling features"):
        pg = pg.copy()
        pg["played"] = (pg["is_projection"] == 0).astype(int)
        pf = add_rolling_features(pg, roll_cols, "player_id", "gameday", windows=windows, ewm_halflife=hl,
                                  ewm_shrink=shrink, cross_season=True, std_windows=(), prior_group_col="position")
        # volatility of the targets
        vol = add_rolling_features(pg[["player_id", "gameday", "season", "played"] + TARGETS], TARGETS, "player_id", "gameday",
                                   windows=(), std_windows=(6,), ewm_halflife=hl, ewm_shrink=shrink, cross_season=True,
                                   prior_group_col=None)
        vol_cols = [f"{t}_sd6" for t in TARGETS]
        pf = pd.concat([pf, vol[vol_cols]], axis=1)
        # season-to-date mean of targets, and games missed / gap
        extra = {f"{t}_std_mean": pf.groupby(["player_id", "season"])[t].transform(lambda s: s.shift(1).expanding().mean()) for t in TARGETS}
        extra["days_since_last"] = pf.groupby("player_id")["gameday"].diff().dt.days
        extra["games_played_season"] = pf["n_prior_games_season"]
        extra["career_games"] = pf["n_prior_games"]
        pf = pd.concat([pf, pd.DataFrame(extra, index=pf.index)], axis=1)

    with timed("team / opponent context"):
        tcols = [c for c in TEAM_CTX if c in tg_rolled.columns]
        ocols = [c for c in OPP_CTX if c in tg_rolled.columns]
        team = tg_rolled[["game_id", "team"] + tcols].copy()
        team.columns = ["game_id", "team"] + [f"tm_{c}" for c in tcols]
        opp = tg_rolled[["game_id", "team"] + ocols].copy()
        opp.columns = ["game_id", "opponent"] + [f"opp_{c}" for c in ocols]
        pf = pf.merge(team, on=["game_id", "team"], how="left").merge(opp, on=["game_id", "opponent"], how="left")

    with timed("defense vs position rolling"):
        # dvp is per (game_id, team=defense); attach schedule ordering for rolling
        sk = tg_rolled[["game_id", "team", "gameday", "season", "played", "week"]]
        d = sk.merge(dvp, on=["game_id", "team"], how="left")
        dcols = [c for c in DVP_METRICS if c in d.columns]
        d = add_rolling_features(d, dcols, "team", "gameday", windows=(6,), ewm_halflife=hl, ewm_shrink=shrink,
                                 cross_season=True, order_cols=["gameday", "week"])
        dfe = [f"{c}_{w}" for c in dcols for w in ("r6", "ewm")]
        d = d[["game_id", "team"] + dfe].rename(columns={"team": "opponent"})
        d.columns = ["game_id", "opponent"] + [f"opp_{c}" for c in dfe]
        pf = pf.merge(d, on=["game_id", "opponent"], how="left")

    with timed("game context & expected volume"):
        gcols = ["game_id", "spread_line", "total_line", "home_implied_pts", "away_implied_pts", "is_dome", "temp_f", "wind_mph",
                 "home_rest", "away_rest", "is_primetime", "is_playoff", "div_game", "away_travel_mi", "week"]
        g = game_feats[[c for c in gcols if c in game_feats.columns]]
        pf = pf.merge(g, on="game_id", how="left", suffixes=("", "_g"))
        if "week_g" in pf:
            pf = pf.drop(columns=["week_g"])
        home = pf["is_home"] == 1
        d = {}
        d["team_spread"] = pd.Series(np.where(home, -pf["spread_line"], pf["spread_line"]), index=pf.index)  # negative = favoured
        d["team_implied_pts"] = pd.Series(np.where(home, pf["home_implied_pts"], pf["away_implied_pts"]), index=pf.index)
        d["opp_implied_pts"] = pd.Series(np.where(home, pf["away_implied_pts"], pf["home_implied_pts"]), index=pf.index)
        d["team_rest"] = pd.Series(np.where(home, pf["home_rest"], pf["away_rest"]), index=pf.index)
        # expected volume = share x team volume
        d["exp_targets"] = pf["target_share_ewm"] * pf["tm_off_dropbacks_ewm"]
        d["exp_carries"] = pf["carry_share_ewm"] * pf["tm_off_rushes_ewm"]
        d["exp_rec_yards"] = d["exp_targets"] * pf["yards_per_target_ewm"]
        d["exp_rush_yards"] = d["exp_carries"] * pf["yards_per_carry_ewm"]
        d["exp_pass_yards"] = pf["tm_off_dropbacks_ewm"] * pf["yards_per_att_ewm"]
        d["snap_trend"] = pf["snap_pct_r3"] - pf["snap_pct_ewm"]
        d["target_share_trend"] = pf["target_share_r3"] - pf["target_share_ewm"]
        d["carry_share_trend"] = pf["carry_share_r3"] - pf["carry_share_ewm"]
        for p in ("QB", "RB", "WR", "TE"):
            d[f"pos_{p}"] = (pf["position"] == p).astype(int)
        pf = pf.drop(columns=["home_implied_pts", "away_implied_pts", "home_rest", "away_rest", "spread_line"])
        pf = pd.concat([pf, pd.DataFrame(d, index=pf.index)], axis=1)
        pf["inj_status"] = pf["inj_status"].fillna(0)
        pf["inj_practice"] = pf["inj_practice"].fillna(0)

    id_cols = ["player_id", "player_name", "position", "team", "opponent", "season", "week", "game_id", "gameday", "is_home",
               "played", "is_projection"]
    fam = {
        "usage_roll": [f"{c}_{w}" for c in roll_cols for w in [f"r{x}" for x in windows] + ["ewm"]],
        "volatility": vol_cols + [f"{t}_std_mean" for t in TARGETS],
        "experience": ["days_since_last", "games_played_season", "career_games"],
        "team_ctx": [f"tm_{c}" for c in tcols],
        "opp_def": [f"opp_{c}" for c in ocols] + [f"opp_{c}" for c in dfe],
        "expected_vol": ["exp_targets", "exp_carries", "exp_rec_yards", "exp_rush_yards", "exp_pass_yards", "snap_trend",
                         "target_share_trend", "carry_share_trend"],
        "status": ["inj_status", "inj_practice", "inj_listed", "depth_rank"],
        "static": ["pos_QB", "pos_RB", "pos_WR", "pos_TE", "age", "years_exp", "draft_round", "draft_number", "height", "weight"],
        "game": ["is_home", "team_spread", "total_line", "team_implied_pts", "opp_implied_pts", "is_dome", "temp_f", "wind_mph",
                 "team_rest", "is_primetime", "is_playoff", "div_game", "away_travel_mi", "week"],
    }
    feats = list(dict.fromkeys(c for v in fam.values() for c in v if c in pf.columns))
    for c in ("height", "weight", "years_exp", "draft_number"):
        if c in pf:
            pf[c] = pd.to_numeric(pf[c], errors="coerce")
    out = pf[id_cols + TARGETS + [c for c in ("targets", "carries", "attempts", "receptions") if c in pf] + feats]
    out = out.loc[:, ~out.columns.duplicated()].sort_values(["gameday", "game_id", "team", "player_id"]).reset_index(drop=True)
    manifest = {"id_cols": id_cols, "target_cols": TARGETS, "aux_targets": ["targets", "carries", "attempts", "receptions"],
                "families": {k: [c for c in v if c in out.columns] for k, v in fam.items()}, "features": feats,
                "n_rows": int(len(out)), "n_played": int((out["is_projection"] == 0).sum())}
    log.info("player features: %s rows x %s features", len(out), len(feats))
    return out, manifest


def save_player_features(pf: pd.DataFrame, manifest: dict) -> None:
    pf.to_parquet(FEATURES_DIR / "player_features.parquet", index=False)
    with open(FEATURES_DIR / "player_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
