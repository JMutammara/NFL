"""Player-game usage & production table (QB / RB / WR / TE).

One row per (player_id, game_id) for every played game *plus* projection rows
for upcoming games (active roster skill players), so rolling features can be
generated for next week's slate with the same prior-only logic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..utils import log, safe_div, timed

STAT_COLS = [
    "completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions", "sacks_suffered",
    "passing_air_yards", "passing_yards_after_catch", "passing_epa", "passing_cpoe", "pacr",
    "carries", "rushing_yards", "rushing_tds", "rushing_epa", "rushing_fumbles_lost",
    "receptions", "targets", "receiving_yards", "receiving_tds", "receiving_air_yards",
    "receiving_yards_after_catch", "receiving_epa", "racr", "target_share", "air_yards_share", "wopr",
    "fantasy_points_ppr",
]
TARGETS = ["passing_yards", "passing_tds", "rushing_yards", "rushing_tds", "receiving_yards", "receiving_tds"]


def _pbp_usage(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (game_id, player) high-value usage from play-by-play."""
    df = pbp[pbp["posteam"].notna() & (pbp["play_deleted"].fillna(0) != 1)]
    df = df[((df["pass"] == 1) | (df["rush"] == 1)) & (df["qb_kneel"].fillna(0) != 1) & (df["qb_spike"].fillna(0) != 1)]
    rz = df["yardline_100"] <= 20
    gl = df["yardline_100"] <= 5
    i10 = df["yardline_100"] <= 10

    # receiving
    t = df[df["receiver_player_id"].notna() & (df["pass_attempt"] == 1) & (df["sack"].fillna(0) != 1)].copy()
    t["rz"] = t["yardline_100"] <= 20
    t["ez"] = t["air_yards"] >= t["yardline_100"]
    t["deep"] = t["air_yards"] >= 20
    t["short"] = t["air_yards"] < 5
    rec = t.groupby(["game_id", "posteam", "receiver_player_id"]).agg(
        u_targets=("pass_attempt", "size"), rz_targets=("rz", "sum"), ez_targets=("ez", "sum"),
        deep_targets=("deep", "sum"), short_targets=("short", "sum"), rec_adot=("air_yards", "mean"),
        rec_epa_per_target=("epa", "mean"), rec_xyac=("xyac_epa", "mean"),
    ).reset_index().rename(columns={"receiver_player_id": "player_id", "posteam": "team"})
    team_t = t.groupby(["game_id", "posteam"]).agg(team_targets=("pass_attempt", "size"), team_rz_targets=("rz", "sum"),
                                                   team_deep_targets=("deep", "sum"), team_air_yards=("air_yards", "sum")).reset_index()
    rec = rec.merge(team_t.rename(columns={"posteam": "team"}), on=["game_id", "team"], how="left")
    rec["rz_target_share"] = safe_div(rec["rz_targets"], rec["team_rz_targets"], 0.0)
    rec["deep_target_share"] = safe_div(rec["deep_targets"], rec["team_deep_targets"], 0.0)

    # rushing
    r = df[df["rusher_player_id"].notna() & (df["rush"] == 1)].copy()
    r["rz"] = r["yardline_100"] <= 20
    r["gl"] = r["yardline_100"] <= 5
    r["i10"] = r["yardline_100"] <= 10
    r["explosive"] = r["yards_gained"] >= 10
    r["stuffed"] = r["yards_gained"] <= 0
    rush = r.groupby(["game_id", "posteam", "rusher_player_id"]).agg(
        u_carries=("rush", "size"), rz_carries=("rz", "sum"), gl_carries=("gl", "sum"), i10_carries=("i10", "sum"),
        explosive_carries=("explosive", "sum"), stuff_rate=("stuffed", "mean"), rush_epa_per_carry=("epa", "mean"),
        rush_sr=("success", "mean"),
    ).reset_index().rename(columns={"rusher_player_id": "player_id", "posteam": "team"})
    team_r = r.groupby(["game_id", "posteam"]).agg(team_carries=("rush", "size"), team_rz_carries=("rz", "sum"),
                                                   team_gl_carries=("gl", "sum")).reset_index()
    rush = rush.merge(team_r.rename(columns={"posteam": "team"}), on=["game_id", "team"], how="left")
    rush["carry_share"] = safe_div(rush["u_carries"], rush["team_carries"], 0.0)
    rush["rz_carry_share"] = safe_div(rush["rz_carries"], rush["team_rz_carries"], 0.0)
    rush["gl_carry_share"] = safe_div(rush["gl_carries"], rush["team_gl_carries"], 0.0)

    # passing
    p = df[df["passer_player_id"].notna() & (df["qb_dropback"] == 1)].copy()
    p["deep_att"] = (p["air_yards"] >= 20) & (p["pass_attempt"] == 1)
    p["scramble"] = p["qb_scramble"].fillna(0) == 1
    p["sacked"] = p["sack"].fillna(0) == 1
    p["rz"] = p["yardline_100"] <= 20
    pas = p.groupby(["game_id", "posteam", "passer_player_id"]).agg(
        dropbacks=("qb_dropback", "size"), pass_adot=("air_yards", "mean"), deep_att_rate=("deep_att", "mean"),
        scramble_rate=("scramble", "mean"), sack_rate_qb=("sacked", "mean"), qb_epa_per_db=("qb_epa", "mean"),
        qb_success=("success", "mean"), rz_dropbacks=("rz", "sum"), pass_oe_qb=("pass_oe", "mean"),
    ).reset_index().rename(columns={"passer_player_id": "player_id", "posteam": "team"})

    out = rec.merge(rush, on=["game_id", "team", "player_id"], how="outer").merge(pas, on=["game_id", "team", "player_id"], how="outer")
    for c in ("u_targets", "rz_targets", "ez_targets", "deep_targets", "short_targets", "u_carries", "rz_carries",
              "gl_carries", "i10_carries", "explosive_carries", "dropbacks", "rz_dropbacks"):
        out[c] = out[c].fillna(0)
    out["hv_touches"] = out["rz_carries"] + out["rz_targets"] + out["deep_targets"] + out["gl_carries"]
    return out


def _defense_vs_position(pbp: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Per (game_id, defteam): production allowed to WR / TE / RB / QB."""
    pos = players.set_index("gsis_id")["position"].to_dict()
    df = pbp[pbp["defteam"].notna() & (pbp["play_deleted"].fillna(0) != 1)]
    df = df[((df["pass"] == 1) | (df["rush"] == 1)) & (df["qb_kneel"].fillna(0) != 1)]
    t = df[df["receiver_player_id"].notna() & (df["pass_attempt"] == 1) & (df["sack"].fillna(0) != 1)].copy()
    t["rpos"] = t["receiver_player_id"].map(pos).fillna("OTH")
    t["rpos"] = t["rpos"].where(t["rpos"].isin(["WR", "TE", "RB"]), "OTH")
    t["rec_yards"] = t["receiving_yards"].fillna(0)
    t["rec_td"] = t["pass_touchdown"].fillna(0)
    parts = []
    for ps in ("WR", "TE", "RB"):
        s = t[t["rpos"] == ps].groupby(["game_id", "defteam"]).agg(**{
            f"dvp_{ps.lower()}_targets": ("pass_attempt", "size"), f"dvp_{ps.lower()}_rec_yards": ("rec_yards", "sum"),
            f"dvp_{ps.lower()}_rec_tds": ("rec_td", "sum"), f"dvp_{ps.lower()}_epa_per_target": ("epa", "mean"),
            f"dvp_{ps.lower()}_ypt": ("rec_yards", "mean"),
        })
        parts.append(s)
    r = df[df["rusher_player_id"].notna() & (df["rush"] == 1)].copy()
    r["rpos"] = r["rusher_player_id"].map(pos).fillna("OTH")
    r["ry"] = r["rushing_yards"].fillna(0)
    r["rtd"] = r["rush_touchdown"].fillna(0)
    rb = r[r["rpos"] == "RB"].groupby(["game_id", "defteam"]).agg(
        dvp_rb_carries=("rush", "size"), dvp_rb_rush_yards=("ry", "sum"), dvp_rb_rush_tds=("rtd", "sum"),
        dvp_rb_ypc=("ry", "mean"), dvp_rb_rush_epa=("epa", "mean"))
    qbr = r[r["rpos"] == "QB"].groupby(["game_id", "defteam"]).agg(dvp_qb_rush_yards=("ry", "sum"), dvp_qb_carries=("rush", "size"))
    p = df[(df["qb_dropback"] == 1)].copy()
    p["py"] = p["passing_yards"].fillna(0)
    p["ptd"] = p["pass_touchdown"].fillna(0)
    qb = p.groupby(["game_id", "defteam"]).agg(dvp_pass_yards=("py", "sum"), dvp_pass_tds=("ptd", "sum"),
                                                dvp_dropbacks=("qb_dropback", "size"), dvp_pass_epa_db=("epa", "mean"),
                                                dvp_cpoe=("cpoe", "mean"))
    out = parts[0]
    for x in parts[1:] + [rb, qbr, qb]:
        out = out.join(x, how="outer")
    out.index.names = ["game_id", "team"]
    return out.reset_index()


def _snap_counts(snaps: pd.DataFrame, players: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    if snaps is None or snaps.empty:
        return pd.DataFrame(columns=["game_id", "player_id", "snap_pct"])
    m1 = players.dropna(subset=["pfr_id"])[["pfr_id", "gsis_id"]]
    m2 = rosters.dropna(subset=["pfr_id", "gsis_id"])[["pfr_id", "gsis_id"]]
    mp = pd.concat([m1, m2]).drop_duplicates("pfr_id").set_index("pfr_id")["gsis_id"]
    s = snaps.copy()
    s["player_id"] = s["pfr_player_id"].map(mp)
    s = s.dropna(subset=["player_id"])
    out = s.groupby(["game_id", "player_id"]).agg(snap_pct=("offense_pct", "max"), offense_snaps=("offense_snaps", "max")).reset_index()
    return out


_STATUS_ORD = {"Out": 3, "Doubtful": 2, "Questionable": 1, "Probable": 0}
_PRACTICE_ORD = {"Did Not Participate In Practice": 2, "Limited Participation in Practice": 1, "Full Participation in Practice": 0}


def _injury_flags(inj: pd.DataFrame) -> pd.DataFrame:
    if inj is None or inj.empty:
        return pd.DataFrame(columns=["season", "week", "team", "player_id", "inj_status", "inj_practice"])
    d = inj.copy()
    d["inj_status"] = d["report_status"].map(_STATUS_ORD)
    d["inj_practice"] = d["practice_status"].map(_PRACTICE_ORD)
    d["inj_listed"] = 1.0
    d = d.rename(columns={"gsis_id": "player_id"})
    d["week"] = pd.to_numeric(d["week"], errors="coerce")
    d = d.dropna(subset=["week", "player_id"])
    d["week"] = d["week"].astype(int)
    d = d.sort_values("inj_status", ascending=False).drop_duplicates(["season", "week", "team", "player_id"])
    return d[["season", "week", "team", "player_id", "inj_status", "inj_practice", "inj_listed"]]


def _parse_height(h: pd.Series) -> pd.Series:
    """Height as inches; accepts numeric or 'ft-in' strings."""
    s = h.astype("string")
    ftin = s.str.extract(r"^(\d)-(\d{1,2})$")
    inches = pd.to_numeric(ftin[0], errors="coerce") * 12 + pd.to_numeric(ftin[1], errors="coerce")
    return inches.fillna(pd.to_numeric(s, errors="coerce")).astype(float)


def build_player_games(
    weekly: pd.DataFrame, pbp: pd.DataFrame, schedules: pd.DataFrame, rosters: pd.DataFrame, players: pd.DataFrame,
    snaps: pd.DataFrame, injuries: pd.DataFrame, depth: pd.DataFrame, cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (player_games, defense_vs_position)."""
    positions = cfg.get("features.player_positions", ["QB", "RB", "WR", "TE"])
    sch = schedules[["game_id", "season", "week", "game_type", "gameday", "home_team", "away_team", "home_score"]].copy()
    long = pd.concat([
        sch.rename(columns={"home_team": "team", "away_team": "opponent"}).assign(is_home=1),
        sch.rename(columns={"away_team": "team", "home_team": "opponent"}).assign(is_home=0),
    ], ignore_index=True)
    long["played"] = long["home_score"].notna().astype(int)
    long = long.drop(columns=["home_score"])

    with timed("weekly stats base"):
        w = weekly[weekly["position"].isin(positions)].copy()
        w = w.drop(columns=[c for c in ("player_name", "headshot_url") if c in w.columns])
        w = w.rename(columns={"player_display_name": "player_name"})
        keep = ["player_id", "player_name", "position", "season", "week", "team"] + [c for c in STAT_COLS if c in w.columns]
        w = w[keep]
        w = w.merge(long, on=["season", "week", "team"], how="inner")  # attaches game_id/opponent/gameday
        w = w[w["played"] == 1]

    with timed("play-by-play usage"):
        usage = _pbp_usage(pbp)
        w = w.merge(usage, on=["game_id", "team", "player_id"], how="left")

    with timed("snap counts / injuries / depth charts / roster attributes"):
        sc = _snap_counts(snaps, players, rosters)
        w = w.merge(sc, on=["game_id", "player_id"], how="left")

    # ---- projection rows for upcoming games -------------------------------
    upcoming = long[long["played"] == 0]
    proj = pd.DataFrame()
    if len(upcoming):
        ro = rosters[rosters["position"].isin(positions) & rosters["status"].isin(["ACT"])].copy()
        ro = ro.rename(columns={"gsis_id": "player_id", "full_name": "player_name"})
        rows = []
        for (season, week), grp in upcoming.groupby(["season", "week"]):
            rs = ro[ro["season"] == season]
            if rs.empty:
                continue
            wk = rs[rs["week"] == week]
            if wk.empty:  # fall back to the latest roster week available
                wk = rs[rs["week"] == rs["week"].max()]
            wk = wk.drop_duplicates(["team", "player_id"])
            r = grp.merge(wk[["team", "player_id", "player_name", "position"]], on="team", how="inner")
            rows.append(r)
        if rows:
            proj = pd.concat(rows, ignore_index=True)
            proj["is_projection"] = 1
    w["is_projection"] = 0
    pg = pd.concat([w, proj], ignore_index=True) if len(proj) else w

    # injuries & depth chart apply to both played and projection rows
    inj = _injury_flags(injuries)
    pg = pg.merge(inj, on=["season", "week", "team", "player_id"], how="left")
    pg["inj_listed"] = pg["inj_listed"].fillna(0)
    dc = depth.rename(columns={"gsis_id": "player_id"})[["season", "week", "team", "player_id", "depth_rank"]]
    pg = pg.merge(dc, on=["season", "week", "team", "player_id"], how="left")

    # roster attributes: age, experience, draft, size
    ra = rosters.rename(columns={"gsis_id": "player_id"}).dropna(subset=["player_id"])
    ra = ra.sort_values(["season", "week"]).drop_duplicates(["season", "player_id"], keep="last")
    ra = ra[["season", "player_id", "birth_date", "height", "weight", "years_exp", "draft_number"]]
    pg = pg.merge(ra, on=["season", "player_id"], how="left")
    pm = players.rename(columns={"gsis_id": "player_id"})[["player_id", "draft_round", "birth_date"]].rename(columns={"birth_date": "bd2"})
    pg = pg.merge(pm, on="player_id", how="left")
    bd = pd.to_datetime(pg["birth_date"], errors="coerce").fillna(pd.to_datetime(pg["bd2"], errors="coerce"))
    pg["age"] = (pg["gameday"] - bd).dt.days / 365.25
    pg["draft_round"] = pd.to_numeric(pg["draft_round"], errors="coerce").fillna(8)  # undrafted
    pg = pg.drop(columns=["birth_date", "bd2"])
    pg["height"] = _parse_height(pg["height"])
    for c in ("weight", "years_exp", "draft_number"):
        pg[c] = pd.to_numeric(pg[c], errors="coerce")
    pg["draft_number"] = pg["draft_number"].fillna(300)  # undrafted

    # derived per-game rates (NaN when no opportunity)
    pg["yards_per_target"] = safe_div(pg["receiving_yards"], pg["targets"])
    pg["yards_per_carry"] = safe_div(pg["rushing_yards"], pg["carries"])
    pg["yards_per_att"] = safe_div(pg["passing_yards"], pg["attempts"])
    pg["catch_rate"] = safe_div(pg["receptions"], pg["targets"])
    pg["touches"] = pg["carries"].fillna(0) + pg["receptions"].fillna(0)
    pg["opportunities"] = pg["carries"].fillna(0) + pg["targets"].fillna(0)

    with timed("defense vs position"):
        dvp = _defense_vs_position(pbp, players)

    pg = pg.sort_values(["player_id", "gameday"]).reset_index(drop=True)
    log.info("player-game table: %s rows (%s played, %s projection) for %s players",
             len(pg), int((pg["is_projection"] == 0).sum()), int((pg["is_projection"] == 1).sum()), pg["player_id"].nunique())
    return pg, dvp


PLAYER_ROLL_METRICS = [
    "attempts", "completions", "passing_yards", "passing_tds", "passing_interceptions", "passing_epa", "passing_cpoe",
    "passing_air_yards", "sacks_suffered", "dropbacks", "pass_adot", "deep_att_rate", "qb_epa_per_db", "yards_per_att",
    "carries", "rushing_yards", "rushing_tds", "rushing_epa", "rz_carries", "gl_carries", "i10_carries", "carry_share",
    "rz_carry_share", "gl_carry_share", "yards_per_carry", "explosive_carries", "rush_epa_per_carry",
    "targets", "receptions", "receiving_yards", "receiving_tds", "receiving_air_yards", "receiving_yards_after_catch",
    "receiving_epa", "target_share", "air_yards_share", "wopr", "racr", "rz_targets", "ez_targets", "deep_targets",
    "rz_target_share", "deep_target_share", "rec_adot", "yards_per_target", "catch_rate", "rec_epa_per_target",
    "snap_pct", "hv_touches", "touches", "opportunities", "fantasy_points_ppr",
]
DVP_METRICS = [
    "dvp_wr_targets", "dvp_wr_rec_yards", "dvp_wr_rec_tds", "dvp_wr_epa_per_target", "dvp_wr_ypt",
    "dvp_te_targets", "dvp_te_rec_yards", "dvp_te_rec_tds", "dvp_te_epa_per_target", "dvp_te_ypt",
    "dvp_rb_targets", "dvp_rb_rec_yards", "dvp_rb_rec_tds", "dvp_rb_epa_per_target", "dvp_rb_ypt",
    "dvp_rb_carries", "dvp_rb_rush_yards", "dvp_rb_rush_tds", "dvp_rb_ypc", "dvp_rb_rush_epa",
    "dvp_qb_rush_yards", "dvp_qb_carries", "dvp_pass_yards", "dvp_pass_tds", "dvp_dropbacks", "dvp_pass_epa_db", "dvp_cpoe",
]
