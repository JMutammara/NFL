"""Team-game efficiency metrics from play-by-play.

One row per (game_id, team). Every metric is computed twice: from the
offense's perspective (``off_``) and from the defense's perspective
(``def_`` = what the team *allowed*). Line-of-scrimmage metrics use
tracking-derived participation data (2016+), FTN charting (2022+) and PFR
pressure counts (2018+) where available; play-by-play proxies (sack, QB hit,
stuff and TFL rates) cover every season.

Nothing here looks across games; rolling windows are applied later in
``rolling.py`` with a strict shift so a game never sees its own outcome.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..utils import log, safe_div, timed


def _prep_plays(pbp: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = pbp[pbp["posteam"].notna() & pbp["defteam"].notna()].copy()
    df = df[df["play_deleted"].fillna(0) != 1]
    df["scrim"] = (((df["pass"] == 1) | (df["rush"] == 1)) & df["epa"].notna()
                   & (df["qb_kneel"].fillna(0) != 1) & (df["qb_spike"].fillna(0) != 1))
    df["dropback"] = df["scrim"] & (df["qb_dropback"] == 1)
    df["rushplay"] = df["scrim"] & (df["rush"] == 1)
    df["attempt"] = df["scrim"] & (df["pass_attempt"] == 1) & (df["sack"].fillna(0) != 1)
    lo, hi = cfg.get("features.neutral_wp_band", [0.1, 0.9])
    g4 = cfg.get("features.garbage_time_qtr4_margin", 16)
    df["neutral"] = df["scrim"] & df["wp"].between(lo, hi) & ~((df["qtr"] >= 4) & (df["score_differential"].abs() > g4))
    df["early"] = df["scrim"] & df["down"].isin([1, 2])
    df["late"] = df["scrim"] & df["down"].isin([3, 4])
    df["explosive"] = df["scrim"] & (((df["pass"] == 1) & (df["yards_gained"] >= 15)) | ((df["rush"] == 1) & (df["yards_gained"] >= 10)))
    df["stuffed"] = df["rushplay"] & (df["yards_gained"] <= 0)
    df["rz"] = df["scrim"] & (df["yardline_100"] <= 20)
    df["turnover"] = df["scrim"] & ((df["interception"] == 1) | (df["fumble_lost"] == 1))
    df["is_pass_td"] = df["pass_touchdown"].fillna(0) == 1
    df["is_rush_td"] = df["rush_touchdown"].fillna(0) == 1
    df["third_att"] = (df["third_down_converted"].fillna(0) == 1) | (df["third_down_failed"].fillna(0) == 1)
    df["fourth_att"] = (df["fourth_down_converted"].fillna(0) == 1) | (df["fourth_down_failed"].fillna(0) == 1)
    return df


def _pace(df: pd.DataFrame, key: str) -> pd.Series:
    """Median seconds between consecutive neutral-script scrimmage plays within a drive."""
    d = df[df["neutral"]].sort_values(["game_id", "fixed_drive", "play_id"])
    same = (d["game_id"] == d["game_id"].shift(-1)) & (d["fixed_drive"] == d["fixed_drive"].shift(-1))
    gap = d["game_seconds_remaining"] - d["game_seconds_remaining"].shift(-1)
    gap = gap.where(same & (gap > 0) & (gap <= 50))
    d = d.assign(_gap=gap)
    return d.groupby(["game_id", key])["_gap"].median()


def _side_stats(df: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    """Aggregate play-level metrics to (game_id, key). key = posteam or defteam."""
    g = df.groupby(["game_id", key], sort=False)
    m = lambda mask, col: df[col].where(mask)  # noqa: E731

    agg = pd.DataFrame({
        "plays": g["scrim"].sum(),
        "dropbacks": g["dropback"].sum(),
        "rushes": g["rushplay"].sum(),
        "attempts": g["attempt"].sum(),
        "epa_pp": m(df["scrim"], "epa").groupby([df["game_id"], df[key]]).mean(),
        "epa_pp_neutral": m(df["neutral"], "epa").groupby([df["game_id"], df[key]]).mean(),
        "pass_epa_db": m(df["dropback"], "epa").groupby([df["game_id"], df[key]]).mean(),
        "pass_epa_neutral": m(df["dropback"] & df["neutral"], "epa").groupby([df["game_id"], df[key]]).mean(),
        "rush_epa": m(df["rushplay"], "epa").groupby([df["game_id"], df[key]]).mean(),
        "rush_epa_neutral": m(df["rushplay"] & df["neutral"], "epa").groupby([df["game_id"], df[key]]).mean(),
        "sr": m(df["scrim"], "success").groupby([df["game_id"], df[key]]).mean(),
        "sr_neutral": m(df["neutral"], "success").groupby([df["game_id"], df[key]]).mean(),
        "pass_sr": m(df["dropback"], "success").groupby([df["game_id"], df[key]]).mean(),
        "rush_sr": m(df["rushplay"], "success").groupby([df["game_id"], df[key]]).mean(),
        "ed_epa": m(df["early"], "epa").groupby([df["game_id"], df[key]]).mean(),
        "ed_sr": m(df["early"], "success").groupby([df["game_id"], df[key]]).mean(),
        "late_epa": m(df["late"], "epa").groupby([df["game_id"], df[key]]).mean(),
        "cpoe": m(df["attempt"], "cpoe").groupby([df["game_id"], df[key]]).mean(),
        "comp_pct": m(df["attempt"], "complete_pass").groupby([df["game_id"], df[key]]).mean(),
        "air_epa_db": m(df["dropback"], "air_epa").groupby([df["game_id"], df[key]]).mean(),
        "yac_epa_db": m(df["dropback"], "yac_epa").groupby([df["game_id"], df[key]]).mean(),
        "pass_oe_neutral": m(df["neutral"], "pass_oe").groupby([df["game_id"], df[key]]).mean(),
        "pass_rate_neutral": m(df["neutral"], "dropback").groupby([df["game_id"], df[key]]).mean(),
        "ed_pass_rate": m(df["early"] & df["neutral"], "dropback").groupby([df["game_id"], df[key]]).mean(),
        "adot": m(df["attempt"], "air_yards").groupby([df["game_id"], df[key]]).mean(),
        "yac_per_comp": m(df["attempt"] & (df["complete_pass"] == 1), "yards_after_catch").groupby([df["game_id"], df[key]]).mean(),
        "ypp": m(df["scrim"], "yards_gained").groupby([df["game_id"], df[key]]).mean(),
        "ypa": m(df["dropback"], "yards_gained").groupby([df["game_id"], df[key]]).mean(),
        "ypc": m(df["rushplay"], "yards_gained").groupby([df["game_id"], df[key]]).mean(),
        "ed_ypc": m(df["rushplay"] & df["early"], "yards_gained").groupby([df["game_id"], df[key]]).mean(),
        "explosive_rate": m(df["scrim"], "explosive").groupby([df["game_id"], df[key]]).mean(),
        "explosive_pass_rate": m(df["dropback"], "explosive").groupby([df["game_id"], df[key]]).mean(),
        "explosive_rush_rate": m(df["rushplay"], "explosive").groupby([df["game_id"], df[key]]).mean(),
        "sacks": (df["dropback"] & (df["sack"] == 1)).groupby([df["game_id"], df[key]]).sum(),
        "qb_hits": (df["dropback"] & (df["qb_hit"] == 1)).groupby([df["game_id"], df[key]]).sum(),
        "ints": (df["attempt"] & (df["interception"] == 1)).groupby([df["game_id"], df[key]]).sum(),
        "fumbles_lost": (df["scrim"] & (df["fumble_lost"] == 1)).groupby([df["game_id"], df[key]]).sum(),
        "turnovers": df["turnover"].groupby([df["game_id"], df[key]]).sum(),
        "stuffs": df["stuffed"].groupby([df["game_id"], df[key]]).sum(),
        "tfls": (df["rushplay"] & (df["tackled_for_loss"] == 1)).groupby([df["game_id"], df[key]]).sum(),
        "third_conv": m(df["third_att"], "third_down_converted").groupby([df["game_id"], df[key]]).mean(),
        "fourth_att": df["fourth_att"].groupby([df["game_id"], df[key]]).sum(),
        "pass_tds": df["is_pass_td"].groupby([df["game_id"], df[key]]).sum(),
        "rush_tds": df["is_rush_td"].groupby([df["game_id"], df[key]]).sum(),
        "rz_plays": df["rz"].groupby([df["game_id"], df[key]]).sum(),
        "rz_epa": m(df["rz"], "epa").groupby([df["game_id"], df[key]]).mean(),
        "no_huddle_rate": m(df["scrim"], "no_huddle").groupby([df["game_id"], df[key]]).mean(),
        "shotgun_rate": m(df["scrim"], "shotgun").groupby([df["game_id"], df[key]]).mean(),
        "scramble_rate": m(df["dropback"], "qb_scramble").groupby([df["game_id"], df[key]]).mean(),
    })
    # derived rates
    agg["sack_rate"] = safe_div(agg["sacks"], agg["dropbacks"])
    agg["qb_hit_rate"] = safe_div(agg["qb_hits"], agg["dropbacks"])
    agg["int_rate"] = safe_div(agg["ints"], agg["attempts"])
    agg["turnover_rate"] = safe_div(agg["turnovers"], agg["plays"])
    agg["stuff_rate"] = safe_div(agg["stuffs"], agg["rushes"])
    agg["tfl_rate"] = safe_div(agg["tfls"], agg["rushes"])
    agg["fourth_att_rate"] = safe_div(agg["fourth_att"], agg["plays"])
    agg["td_rate"] = safe_div(agg["pass_tds"] + agg["rush_tds"], agg["plays"])

    # drive-level: drives, points per drive, red-zone trips / TD rate, field position
    drives = df[df["scrim"] & df["fixed_drive"].notna()].sort_values("play_id")
    first = drives.groupby(["game_id", key, "fixed_drive"], sort=False).agg(
        start_yl=("yardline_100", "first"), inside20=("drive_inside20", "max"),
        result=("fixed_drive_result", "first"), scored=("drive_ended_with_score", "max"),
    ).reset_index()
    first["is_td"] = first["result"].astype("string").str.lower().eq("touchdown")
    first["is_to"] = first["result"].astype("string").str.lower().isin(["interception", "fumble", "turnover on downs", "opp touchdown"])
    dg = first.groupby(["game_id", key])
    dr = pd.DataFrame({
        "drives": dg.size(),
        "start_yl100": dg["start_yl"].mean(),
        "drive_score_rate": dg["scored"].mean(),
        "drive_td_rate": dg["is_td"].mean(),
        "drive_to_rate": dg["is_to"].mean(),
        "rz_trips": dg["inside20"].sum(),
        "rz_tds": dg.apply(lambda x: (x["inside20"].eq(1) & x["is_td"]).sum()),
    })
    dr["rz_td_rate"] = safe_div(dr["rz_tds"], dr["rz_trips"])
    agg = agg.join(dr, how="left")
    agg["sec_per_play"] = _pace(df, key)
    agg = agg.drop(columns=["sacks", "qb_hits", "ints", "fumbles_lost", "stuffs", "tfls", "fourth_att", "rz_tds"])
    agg.columns = [f"{prefix}{c}" for c in agg.columns]
    agg.index.names = ["game_id", "team"]
    return agg.reset_index()


def _qb_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Primary passer per (game, posteam) and his EPA/CPOE in that game."""
    d = df[df["dropback"] & df["passer_player_id"].notna()]
    g = d.groupby(["game_id", "posteam", "passer_player_id"]).agg(
        qb_dropbacks=("dropback", "sum"), qb_epa_db=("qb_epa", "mean"), qb_cpoe=("cpoe", "mean"),
        qb_name=("passer_player_name", "first"),
    ).reset_index()
    g = g.sort_values(["game_id", "posteam", "qb_dropbacks"], ascending=[True, True, False])
    g = g.drop_duplicates(["game_id", "posteam"], keep="first")
    return g.rename(columns={"posteam": "team", "passer_player_id": "qb_id"})


def _line_stats_participation(pbp: pd.DataFrame, part: pd.DataFrame) -> pd.DataFrame:
    """Tracking-derived line metrics (2016+): pressure, time to throw, box count, rushers."""
    if part is None or part.empty:
        return pd.DataFrame(columns=["game_id", "team"])
    p = part.rename(columns={"nflverse_game_id": "game_id"})
    keep = pbp.loc[pbp["scrim"], ["game_id", "play_id", "posteam", "defteam", "dropback", "rushplay"]]
    m = keep.merge(p, on=["game_id", "play_id"], how="inner")
    m["man"] = m["defense_man_zone_type"].astype("string").str.contains("MAN", na=False)
    m["heavy_box"] = m["defenders_in_box"] >= 8
    m["light_box"] = m["defenders_in_box"] <= 6
    out = []
    for key, prefix in (("posteam", "off_"), ("defteam", "def_")):
        g = m.groupby(["game_id", key])
        agg = pd.DataFrame({
            f"{prefix}pressure_rate_trk": m["was_pressure"].where(m["dropback"]).groupby([m["game_id"], m[key]]).mean(),
            f"{prefix}time_to_throw": m["time_to_throw"].where(m["dropback"]).groupby([m["game_id"], m[key]]).mean(),
            f"{prefix}pass_rushers_trk": m["number_of_pass_rushers"].where(m["dropback"]).groupby([m["game_id"], m[key]]).mean(),
            f"{prefix}box_count_trk": m["defenders_in_box"].where(m["rushplay"]).groupby([m["game_id"], m[key]]).mean(),
            f"{prefix}heavy_box_rate_trk": m["heavy_box"].where(m["rushplay"]).groupby([m["game_id"], m[key]]).mean(),
            f"{prefix}light_box_rate_trk": m["light_box"].where(m["rushplay"]).groupby([m["game_id"], m[key]]).mean(),
            f"{prefix}man_cov_rate_trk": m["man"].where(m["dropback"]).groupby([m["game_id"], m[key]]).mean(),
        })
        agg.index.names = ["game_id", "team"]
        out.append(agg)
    return out[0].join(out[1], how="outer").reset_index()


def _line_stats_ftn(pbp: pd.DataFrame, ftn: pd.DataFrame) -> pd.DataFrame:
    """FTN charting (2022+): blitz rate, rushers, play-action, motion, screens, drops."""
    if ftn is None or ftn.empty:
        return pd.DataFrame(columns=["game_id", "team"])
    f = ftn.rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
    keep = pbp.loc[pbp["scrim"], ["game_id", "play_id", "posteam", "defteam", "dropback", "rushplay", "attempt"]]
    m = keep.merge(f, on=["game_id", "play_id"], how="inner")
    for c in ("is_play_action", "is_motion", "is_screen_pass", "is_rpo", "is_qb_out_of_pocket", "is_drop",
              "is_contested_ball", "is_catchable_ball", "is_interception_worthy", "is_throw_away", "is_qb_fault_sack"):
        m[c] = m[c].astype("float")
    m["blitz"] = (m["n_blitzers"].fillna(0) > 0).astype(float)
    out = []
    for key, prefix in (("posteam", "off_"), ("defteam", "def_")):
        agg = pd.DataFrame({
            f"{prefix}blitz_rate_ftn": m["blitz"].where(m["dropback"]).groupby([m["game_id"], m[key]]).mean(),
            f"{prefix}pass_rushers_ftn": m["n_pass_rushers"].where(m["dropback"]).groupby([m["game_id"], m[key]]).mean(),
            f"{prefix}box_count_ftn": m["n_defense_box"].where(m["rushplay"]).groupby([m["game_id"], m[key]]).mean(),
            f"{prefix}play_action_rate": m["is_play_action"].where(m["dropback"]).groupby([m["game_id"], m[key]]).mean(),
            f"{prefix}motion_rate": m["is_motion"].where(m["scrim"] if "scrim" in m else m["dropback"] | m["rushplay"]).groupby([m["game_id"], m[key]]).mean(),
            f"{prefix}screen_rate": m["is_screen_pass"].where(m["dropback"]).groupby([m["game_id"], m[key]]).mean(),
            f"{prefix}rpo_rate": m["is_rpo"].where(m["dropback"] | m["rushplay"]).groupby([m["game_id"], m[key]]).mean(),
            f"{prefix}oop_rate": m["is_qb_out_of_pocket"].where(m["dropback"]).groupby([m["game_id"], m[key]]).mean(),
            f"{prefix}drop_rate": m["is_drop"].where(m["attempt"]).groupby([m["game_id"], m[key]]).mean(),
            f"{prefix}contested_rate": m["is_contested_ball"].where(m["attempt"]).groupby([m["game_id"], m[key]]).mean(),
            f"{prefix}int_worthy_rate": m["is_interception_worthy"].where(m["attempt"]).groupby([m["game_id"], m[key]]).mean(),
        })
        agg.index.names = ["game_id", "team"]
        out.append(agg)
    return out[0].join(out[1], how="outer").reset_index()


def _line_stats_pfr(pfr_pass: pd.DataFrame, pfr_def: pd.DataFrame) -> pd.DataFrame:
    """PFR (2018+): pressure / hurry / hit / blitz counts. Rates are per dropback (joined later)."""
    parts = []
    if pfr_pass is not None and not pfr_pass.empty:
        o = pfr_pass.groupby(["game_id", "team"]).agg(
            off_pfr_pressured=("times_pressured", "sum"), off_pfr_hurried=("times_hurried", "sum"),
            off_pfr_hit=("times_hit", "sum"), off_pfr_blitzed=("times_blitzed", "sum"),
            off_pfr_bad_throw_pct=("passing_bad_throw_pct", "mean"),
        )
        parts.append(o)
    if pfr_def is not None and not pfr_def.empty:
        d = pfr_def.groupby(["game_id", "team"]).agg(
            def_pfr_pressures=("def_pressures", "sum"), def_pfr_hurries=("def_times_hurried", "sum"),
            def_pfr_hits=("def_times_hitqb", "sum"), def_pfr_blitzes=("def_times_blitzed", "sum"),
            def_pfr_missed_tackles=("def_missed_tackles", "sum"), def_pfr_tackles=("def_tackles_combined", "sum"),
        )
        d["def_missed_tackle_rate"] = safe_div(d["def_pfr_missed_tackles"], d["def_pfr_tackles"] + d["def_pfr_missed_tackles"])
        parts.append(d.drop(columns=["def_pfr_missed_tackles", "def_pfr_tackles"]))
    if not parts:
        return pd.DataFrame(columns=["game_id", "team"])
    out = parts[0]
    for p in parts[1:]:
        out = out.join(p, how="outer")
    return out.reset_index()


METRIC_PREFIXES = ("off_", "def_")


def build_team_game_stats(
    pbp: pd.DataFrame,
    schedules: pd.DataFrame,
    cfg: Config,
    participation: pd.DataFrame | None = None,
    ftn: pd.DataFrame | None = None,
    pfr_pass: pd.DataFrame | None = None,
    pfr_def: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per (game_id, team) for every game in ``schedules`` (played or not).

    Unplayed games carry NaN metrics so that downstream rolling windows can
    produce prior-only features for upcoming fixtures.
    """
    with timed("prep plays"):
        df = _prep_plays(pbp, cfg)
    with timed("offense / defense aggregates"):
        off = _side_stats(df, "posteam", "off_")
        de = _side_stats(df, "defteam", "def_")
    with timed("QB, line, charting merges"):
        qb = _qb_stats(df)
        trk = _line_stats_participation(df, participation)
        chart = _line_stats_ftn(df, ftn)
        pfr = _line_stats_pfr(pfr_pass, pfr_def)

    # schedule skeleton: two rows per game
    sch = schedules[["game_id", "season", "week", "game_type", "gameday", "home_team", "away_team",
                     "home_score", "away_score", "home_qb_id", "away_qb_id", "home_qb_name", "away_qb_name",
                     "spread_line", "total_line"]].copy()
    home = sch.rename(columns={"home_team": "team", "away_team": "opponent", "home_score": "points_for",
                               "away_score": "points_against", "home_qb_id": "sched_qb_id", "home_qb_name": "sched_qb_name"})
    home = home.drop(columns=["away_qb_id", "away_qb_name"])
    home["is_home"] = 1
    away = sch.rename(columns={"away_team": "team", "home_team": "opponent", "away_score": "points_for",
                               "home_score": "points_against", "away_qb_id": "sched_qb_id", "away_qb_name": "sched_qb_name"})
    away = away.drop(columns=["home_qb_id", "home_qb_name"])
    away["is_home"] = 0
    tg = pd.concat([home, away], ignore_index=True)
    tg["margin"] = tg["points_for"] - tg["points_against"]
    # against-the-spread outcomes from the team's perspective (rolled later => prior-only ATS form)
    team_spread = np.where(tg["is_home"] == 1, tg["spread_line"], -tg["spread_line"])  # points the team was favoured by
    tg["ats_cover_margin"] = tg["margin"] - team_spread
    tg["ats_over_margin"] = (tg["points_for"] + tg["points_against"]) - tg["total_line"]
    tg = tg.drop(columns=["spread_line", "total_line"])
    tg["win"] = np.where(tg["margin"].isna(), np.nan, (tg["margin"] > 0).astype(float) + 0.5 * (tg["margin"] == 0))
    tg["played"] = tg["points_for"].notna().astype(int)
    tg["season_type"] = np.where(tg["game_type"] == "REG", "REG", "POST")

    tg = tg.merge(off, on=["game_id", "team"], how="left")
    tg = tg.merge(de, on=["game_id", "team"], how="left")
    tg = tg.merge(qb, on=["game_id", "team"], how="left")
    for extra in (trk, chart, pfr):
        if len(extra.columns) > 2:
            tg = tg.merge(extra, on=["game_id", "team"], how="left")
    # PFR rates per dropback
    if "off_pfr_pressured" in tg.columns:
        for c in ("pressured", "hurried", "hit", "blitzed"):
            tg[f"off_pfr_{c}_rate"] = safe_div(tg[f"off_pfr_{c}"], tg["off_dropbacks"])
        tg = tg.drop(columns=[f"off_pfr_{c}" for c in ("pressured", "hurried", "hit", "blitzed")])
    if "def_pfr_pressures" in tg.columns:
        for c in ("pressures", "hurries", "hits", "blitzes"):
            tg[f"def_pfr_{c}_rate"] = safe_div(tg[f"def_pfr_{c}"], tg["def_dropbacks"])
        tg = tg.drop(columns=[f"def_pfr_{c}" for c in ("pressures", "hurries", "hits", "blitzes")])

    # The QB who will start (schedule) vs who actually threw most (pbp). For
    # played games prefer pbp; for upcoming games use the schedule projection.
    tg["qb_id"] = tg["qb_id"].fillna(tg["sched_qb_id"])
    tg["qb_name"] = tg["qb_name"].fillna(tg["sched_qb_name"])
    tg = tg.drop(columns=["sched_qb_id", "sched_qb_name"])

    # points per drive uses actual points (includes defensive / ST scores; acceptable)
    tg["off_ppd"] = safe_div(tg["points_for"], tg["off_drives"])
    tg["def_ppd"] = safe_div(tg["points_against"], tg["def_drives"])

    mcols = [c for c in tg.columns if c.startswith(METRIC_PREFIXES)]
    tg[mcols] = tg[mcols].apply(pd.to_numeric, errors="coerce").astype("float64")
    tg = tg.sort_values(["team", "gameday", "week"]).reset_index(drop=True)
    played = tg["played"].sum()
    log.info("team-game table: %s rows (%s played), %s metric columns", len(tg), played,
             sum(c.startswith(("off_", "def_")) for c in tg.columns))
    return tg


def metric_columns(tg: pd.DataFrame) -> list[str]:
    return [c for c in tg.columns if c.startswith(METRIC_PREFIXES) and pd.api.types.is_numeric_dtype(tg[c])]
