"""Data ingestion from nflverse (nflfastR) release artifacts.

Every loader follows the same contract:

* ``load_<asset>(seasons, cfg, refresh=False) -> pd.DataFrame``
* Each season is cached as parquet under ``data/raw/<asset>/``.
* Completed seasons are cached forever; the current season is re-downloaded
  when the cache is older than ``data.refresh_hours`` or ``refresh=True``.
* Missing assets for a season (e.g. FTN charting before 2022) are skipped
  with a warning, never fatal.

nfl_data_py wraps these same nflverse release files. We read them directly
with pyarrow column push-down (fast, schema-stable, works through corporate
proxies that block the package's secondary hosts) and route play-by-play
through ``nfl_data_py.import_pbp_data`` when ``data.pbp_source`` is set to
``nfl_data_py``.
"""
from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

from .config import RAW_DIR, Config, load_config
from .utils import log, norm_team, timed

NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download/"
NFLDATA_RAW = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/"

# ---------------------------------------------------------------------------
# Play-by-play columns we actually use (out of ~370). Keeping this list tight
# makes a 15-season load fit comfortably in memory.
# ---------------------------------------------------------------------------
PBP_COLUMNS: list[str] = [
    # identifiers / context
    "play_id", "game_id", "season", "week", "season_type", "game_date",
    "home_team", "away_team", "posteam", "defteam", "posteam_type",
    "qtr", "down", "ydstogo", "yardline_100", "goal_to_go",
    "game_seconds_remaining", "half_seconds_remaining", "score_differential",
    "wp", "vegas_wp", "drive", "fixed_drive", "fixed_drive_result", "series", "series_success",
    "drive_inside20", "drive_play_count", "drive_ended_with_score",
    # play classification
    "play_type", "play", "pass", "rush", "pass_attempt", "rush_attempt", "qb_dropback", "qb_kneel", "qb_spike", "qb_scramble",
    "special_teams_play", "aborted_play", "penalty", "no_huddle", "shotgun", "play_deleted",
    "pass_length", "pass_location", "run_location", "run_gap",
    # outcomes
    "yards_gained", "epa", "qb_epa", "air_epa", "yac_epa", "wpa", "success",
    "cp", "cpoe", "xpass", "pass_oe", "air_yards", "yards_after_catch", "xyac_epa",
    "complete_pass", "incomplete_pass", "interception", "sack", "qb_hit", "tackled_for_loss",
    "fumble", "fumble_lost", "touchdown", "pass_touchdown", "rush_touchdown",
    "first_down", "third_down_converted", "third_down_failed",
    "fourth_down_converted", "fourth_down_failed", "penalty_yards", "penalty_team",
    "field_goal_attempt", "field_goal_result", "kick_distance", "punt_attempt",
    "extra_point_result", "two_point_conv_result",
    # players
    "passer_player_id", "passer_player_name", "receiver_player_id", "receiver_player_name",
    "rusher_player_id", "rusher_player_name", "passing_yards", "receiving_yards", "rushing_yards",
    "td_player_id",
    # game meta
    "roof", "surface", "temp", "wind", "weather", "stadium", "stadium_id", "location",
    "total_home_score", "total_away_score", "home_score", "away_score", "result", "total",
    "spread_line", "total_line", "div_game",
]

PARTICIPATION_COLUMNS = [
    "nflverse_game_id", "play_id", "possession_team", "offense_formation", "offense_personnel",
    "defenders_in_box", "defense_personnel", "number_of_pass_rushers", "ngs_air_yards",
    "time_to_throw", "was_pressure", "route", "defense_man_zone_type", "defense_coverage_type",
]

FTN_COLUMNS = [
    "nflverse_game_id", "nflverse_play_id", "season", "week", "n_offense_backfield", "n_defense_box",
    "is_no_huddle", "is_motion", "is_play_action", "is_screen_pass", "is_rpo", "is_qb_out_of_pocket",
    "is_interception_worthy", "is_throw_away", "is_catchable_ball", "is_contested_ball", "is_drop",
    "n_blitzers", "n_pass_rushers", "is_qb_fault_sack",
]


# ---------------------------------------------------------------------------
# Download / cache plumbing
# ---------------------------------------------------------------------------
class AssetUnavailable(Exception):
    pass


def _http_get(url: str, retries: int = 4, timeout: int = 120) -> bytes:
    delay = 2.0
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 404:
                raise AssetUnavailable(url)
            r.raise_for_status()
            return r.content
        except AssetUnavailable:
            raise
        except Exception as e:  # noqa: BLE001 - retry any transport error
            last_err = e
            log.warning("download failed (%s/%s) %s: %s", attempt + 1, retries, url, e)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"giving up on {url}: {last_err}")


def _read_remote_parquet(url: str, columns: list[str] | None = None) -> pd.DataFrame:
    buf = io.BytesIO(_http_get(url))
    schema_names = set(pq.read_schema(buf).names)
    buf.seek(0)
    cols = [c for c in columns if c in schema_names] if columns else None
    if columns:
        missing = sorted(set(columns) - schema_names)
        if missing:
            log.debug("columns absent in %s: %s", url.rsplit("/", 1)[-1], missing)
    return pq.read_table(buf, columns=cols).to_pandas()


def _read_remote_csv(url: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(_http_get(url)), low_memory=False)


def _is_stale(path: Path, season: int, cfg: Config, refresh: bool) -> bool:
    if refresh or not path.exists():
        return True
    if season < cfg.current_season:
        return False
    age_h = (time.time() - path.stat().st_mtime) / 3600.0
    return age_h > float(cfg.get("data.refresh_hours", 6))


def _cached_season_asset(
    asset: str,
    season: int,
    fetch: Callable[[], pd.DataFrame],
    cfg: Config,
    refresh: bool = False,
) -> pd.DataFrame | None:
    d = RAW_DIR / asset
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{asset}_{season}.parquet"
    marker = d / f"{asset}_{season}.missing"
    if not _is_stale(path, season, cfg, refresh):
        return pd.read_parquet(path)
    if marker.exists() and not refresh and season < cfg.current_season:
        return None
    try:
        df = fetch()
    except AssetUnavailable:
        log.warning("%s not available for %s", asset, season)
        marker.touch()
        return None
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    if marker.exists():
        marker.unlink()
    return df


def _concat(parts: Iterable[pd.DataFrame | None]) -> pd.DataFrame:
    parts = [p for p in parts if p is not None and len(p)]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_schedules(cfg: Config | None = None, refresh: bool = False) -> pd.DataFrame:
    """All games (1999-present) with betting lines, rest days, stadium, weather."""
    cfg = cfg or load_config()
    path = RAW_DIR / "schedules" / "games.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if _is_stale(path, cfg.current_season, cfg, refresh):
        df = _read_remote_csv(NFLVERSE + "schedules/games.csv")
        df.to_parquet(path, index=False)
    else:
        df = pd.read_parquet(path)
    df = df[df["season"] >= cfg.get("data.first_season", 2012) - 1].copy()
    for c in ("home_team", "away_team"):
        df[c] = norm_team(df[c])
    df["gameday"] = pd.to_datetime(df["gameday"])
    if not cfg.get("data.include_postseason", True):
        df = df[df["game_type"] == "REG"]
    return df.reset_index(drop=True)


def load_pbp(seasons: Iterable[int], cfg: Config | None = None, refresh: bool = False) -> pd.DataFrame:
    cfg = cfg or load_config()
    source = cfg.get("data.pbp_source", "nflverse_direct")

    def _fetch(season: int) -> Callable[[], pd.DataFrame]:
        def go() -> pd.DataFrame:
            if source == "nfl_data_py":
                try:
                    import nfl_data_py as nfl  # noqa: WPS433

                    df = nfl.import_pbp_data([season], downcast=True, cache=False, include_participation=False)
                    keep = [c for c in PBP_COLUMNS if c in df.columns]
                    return df[keep]
                except Exception as e:  # noqa: BLE001
                    log.warning("nfl_data_py pbp failed for %s (%s); falling back to direct read", season, e)
            return _read_remote_parquet(NFLVERSE + f"pbp/play_by_play_{season}.parquet", PBP_COLUMNS)

        return go

    parts = []
    for s in seasons:
        with timed(f"pbp {s}"):
            parts.append(_cached_season_asset("pbp", s, _fetch(s), cfg, refresh))
    df = _concat(parts)
    if df.empty:
        return df
    for c in ("home_team", "away_team", "posteam", "defteam", "penalty_team"):
        if c in df.columns:
            df[c] = norm_team(df[c])
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def _season_loader(asset: str, url_fmt: str, columns: list[str] | None = None, team_cols: tuple[str, ...] = ()):
    def loader(seasons: Iterable[int], cfg: Config | None = None, refresh: bool = False) -> pd.DataFrame:
        cfg = cfg or load_config()
        parts = []
        for s in seasons:
            url = NFLVERSE + url_fmt.format(season=s)
            parts.append(_cached_season_asset(asset, s, lambda u=url: _read_remote_parquet(u, columns), cfg, refresh))
        df = _concat(parts)
        for c in team_cols:
            if c in df.columns:
                df[c] = norm_team(df[c])
        return df

    loader.__name__ = f"load_{asset}"
    return loader


load_weekly_player_stats = _season_loader(
    "stats_player_week", "stats_player/stats_player_week_{season}.parquet", team_cols=("team", "opponent_team")
)
load_weekly_rosters = _season_loader(
    "roster_weekly", "weekly_rosters/roster_weekly_{season}.parquet",
    columns=["season", "week", "game_type", "team", "position", "depth_chart_position", "status",
             "full_name", "birth_date", "height", "weight", "gsis_id", "pfr_id", "years_exp",
             "entry_year", "rookie_year", "draft_number"],
    team_cols=("team",),
)
load_snap_counts = _season_loader(
    "snap_counts", "snap_counts/snap_counts_{season}.parquet", team_cols=("team", "opponent")
)
load_injuries = _season_loader("injuries", "injuries/injuries_{season}.parquet", team_cols=("team",))
load_depth_charts = _season_loader("depth_charts", "depth_charts/depth_charts_{season}.parquet", team_cols=("team",))
load_participation = _season_loader(
    "pbp_participation", "pbp_participation/pbp_participation_{season}.parquet", columns=PARTICIPATION_COLUMNS
)
load_ftn = _season_loader("ftn_charting", "ftn_charting/ftn_charting_{season}.parquet", columns=FTN_COLUMNS)
load_pfr_pass = _season_loader("pfr_adv_pass", "pfr_advstats/advstats_week_pass_{season}.parquet", team_cols=("team", "opponent"))
load_pfr_def = _season_loader("pfr_adv_def", "pfr_advstats/advstats_week_def_{season}.parquet", team_cols=("team", "opponent"))


def load_players(cfg: Config | None = None, refresh: bool = False) -> pd.DataFrame:
    cfg = cfg or load_config()
    path = RAW_DIR / "players" / "players.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if _is_stale(path, cfg.current_season, cfg, refresh):
        df = _read_remote_parquet(
            NFLVERSE + "players/players.parquet",
            ["gsis_id", "display_name", "position", "position_group", "birth_date", "height", "weight",
             "rookie_season", "draft_year", "draft_round", "draft_pick", "pfr_id", "espn_id", "latest_team", "status"],
        )
        df.to_parquet(path, index=False)
        return df
    return pd.read_parquet(path)


def load_ngs(stat_type: str, cfg: Config | None = None, refresh: bool = False) -> pd.DataFrame:
    """Next Gen Stats weekly aggregates: 'passing' | 'receiving' | 'rushing'."""
    cfg = cfg or load_config()
    path = RAW_DIR / "ngs" / f"ngs_{stat_type}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if _is_stale(path, cfg.current_season, cfg, refresh):
        df = _read_remote_parquet(NFLVERSE + f"nextgen_stats/ngs_{stat_type}.parquet")
        df.to_parquet(path, index=False)
    else:
        df = pd.read_parquet(path)
    df = df[df["week"] > 0].copy()  # week 0 = season aggregate
    df["team_abbr"] = norm_team(df["team_abbr"])
    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def ingest_all(cfg: Config | None = None, refresh: bool = False, seasons: list[int] | None = None) -> dict[str, pd.DataFrame]:
    """Download / refresh every asset and return a dict of DataFrames."""
    cfg = cfg or load_config()
    seasons = seasons or cfg.seasons
    out: dict[str, pd.DataFrame] = {}
    with timed("schedules"):
        out["schedules"] = load_schedules(cfg, refresh)
    with timed(f"play-by-play {seasons[0]}-{seasons[-1]}"):
        out["pbp"] = load_pbp(seasons, cfg, refresh)
    with timed("weekly player stats"):
        out["weekly_stats"] = load_weekly_player_stats(seasons, cfg, refresh)
    with timed("weekly rosters"):
        out["rosters"] = load_weekly_rosters(seasons, cfg, refresh)
    with timed("snap counts"):
        out["snap_counts"] = load_snap_counts(seasons, cfg, refresh)
    with timed("injuries"):
        out["injuries"] = load_injuries(seasons, cfg, refresh)
    with timed("depth charts"):
        out["depth_charts"] = load_depth_charts(seasons, cfg, refresh)
    with timed("participation (tracking-derived, 2016+)"):
        out["participation"] = load_participation([s for s in seasons if s >= 2016], cfg, refresh)
    with timed("FTN charting (2022+)"):
        out["ftn"] = load_ftn([s for s in seasons if s >= 2022], cfg, refresh)
    with timed("PFR advanced passing / defense (2018+)"):
        out["pfr_pass"] = load_pfr_pass([s for s in seasons if s >= 2018], cfg, refresh)
        out["pfr_def"] = load_pfr_def([s for s in seasons if s >= 2018], cfg, refresh)
    with timed("players master"):
        out["players"] = load_players(cfg, refresh)
    with timed("Next Gen Stats"):
        for t in ("passing", "receiving", "rushing"):
            out[f"ngs_{t}"] = load_ngs(t, cfg, refresh)
    return out


def ingestion_report(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for k, df in data.items():
        if df is None or df.empty:
            rows.append({"asset": k, "rows": 0, "cols": 0, "seasons": "-", "mem_mb": 0.0})
            continue
        seasons = "-"
        if "season" in df.columns:
            seasons = f"{int(df['season'].min())}-{int(df['season'].max())}"
        rows.append({
            "asset": k, "rows": len(df), "cols": df.shape[1], "seasons": seasons,
            "mem_mb": round(df.memory_usage(deep=True).sum() / 1e6, 1),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Depth chart harmonisation. nflverse switched formats in 2025:
#   legacy (<=2024): season, week, club_code, gsis_id, position, depth_team (rank)
#   new    (>=2025): dt (snapshot timestamp), team, gsis_id, pos_abb, pos_rank
# We map every snapshot to the season-week whose first kickoff follows it, so
# the "week W depth chart" is the last chart published before week W games.
# ---------------------------------------------------------------------------
_SKILL_ABB = {"QB": "QB", "RB": "RB", "HB": "RB", "FB": "FB", "WR": "WR", "TE": "TE",
              "LWR": "WR", "RWR": "WR", "SWR": "WR", "SLWR": "WR"}


def normalize_depth_charts(dc: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Return columns: season, week, team, gsis_id, position, depth_rank."""
    if dc.empty:
        return pd.DataFrame(columns=["season", "week", "team", "gsis_id", "position", "depth_rank"])
    out = []
    legacy = dc[dc.get("club_code").notna()] if "club_code" in dc.columns else pd.DataFrame()
    if len(legacy):
        lg = legacy.copy()
        lg["team"] = norm_team(lg["club_code"])
        lg["position"] = lg["depth_position"].astype("string").str.upper().map(_SKILL_ABB)
        lg["position"] = lg["position"].fillna(lg["position"].astype("string").str.upper())
        lg["depth_rank"] = pd.to_numeric(lg["depth_team"], errors="coerce")
        lg["week"] = pd.to_numeric(lg["week"], errors="coerce")
        lg = lg[["season", "week", "team", "gsis_id", "position", "depth_rank"]].dropna(subset=["week", "gsis_id"])
        out.append(lg)
    new = dc[dc.get("dt").notna()] if "dt" in dc.columns else pd.DataFrame()
    if len(new):
        nw = new.copy()
        nw["dt"] = pd.to_datetime(nw["dt"], utc=True).dt.tz_convert(None)
        nw["team"] = norm_team(nw["team"])
        nw["position"] = nw["pos_abb"].astype("string").str.upper().map(_SKILL_ABB).fillna(nw["pos_abb"].astype("string").str.upper())
        nw["depth_rank"] = pd.to_numeric(nw["pos_rank"], errors="coerce")
        # season = the season whose window (Jul 1 .. next Jun 30) contains the snapshot
        nw["season"] = np.where(nw["dt"].dt.month >= 7, nw["dt"].dt.year, nw["dt"].dt.year - 1)
        week_start = (
            schedules.groupby(["season", "week"])["gameday"].min().reset_index()
            .rename(columns={"gameday": "first_kick"})
            .sort_values(["season", "first_kick"])
        )
        parts = []
        for season, grp in nw.groupby("season"):
            ws = week_start[week_start["season"] == season]
            if ws.empty:
                continue
            kicks = ws["first_kick"].to_numpy(dtype="datetime64[ns]")
            weeks = ws["week"].to_numpy()
            idx = np.searchsorted(kicks, grp["dt"].to_numpy(dtype="datetime64[ns]"), side="left")
            idx = np.clip(idx, 0, len(weeks) - 1)
            g = grp.copy()
            g["week"] = weeks[idx]
            parts.append(g)
        if parts:
            nw = pd.concat(parts)
            # keep only the latest snapshot per (season, week, team)
            latest = nw.groupby(["season", "week", "team"])["dt"].transform("max")
            nw = nw[nw["dt"] == latest]
            out.append(nw[["season", "week", "team", "gsis_id", "position", "depth_rank"]])
    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["season", "week", "team", "gsis_id", "position", "depth_rank"])
    res["season"] = res["season"].astype(int)
    res["week"] = res["week"].astype(int)
    res = res.dropna(subset=["depth_rank"])
    # one rank per player-week (lowest rank listed wins, e.g. WR1 listed twice)
    res = res.sort_values("depth_rank").drop_duplicates(["season", "week", "team", "gsis_id"], keep="first")
    return res.reset_index(drop=True)
