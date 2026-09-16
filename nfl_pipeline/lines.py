"""Opening and closing betting lines from ESPN's per-game odds feed.

nflverse schedules carry a single (closing) spread and total. Betting at the
open requires the opening numbers, which ESPN exposes per game and per
sportsbook. The record shape changed over the years:

* 2012-2014 : only model "providers" (accuscore, numberfire, ...) -> no book lines
* 2015-2018 : real books with ``initialSpread`` / ``initialOverUnder`` (opening)
              and ``spread`` / ``overUnder`` (closing), plus an "Opening" provider
* 2019-2022 : books with closing numbers only (no open in most records)
* 2023+     : books with ``open`` / ``close`` / ``current`` objects for spread,
              total and moneyline

Sign convention: everything is returned from the HOME perspective in the
nflverse style, i.e. ``spread_open > 0`` means the home team opened as the
favourite by that many points.

Every game's raw JSON is cached under ``data/raw/espn_odds/<season>/`` so the
scan is a one-off; the current season is refreshed with ``refresh=True``.
"""
from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import RAW_DIR, Config, load_config
from .utils import log

ODDS_URL = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{eid}/competitions/{eid}/odds"
MODEL_PROVIDERS = {"accuscore", "numberfire", "teamrankings"}  # predictions, not books
LINES_PATH = RAW_DIR / "espn_odds" / "lines.parquet"


def _num(x) -> float:
    """Parse ESPN's american-style strings ('-4.5', '+3', 'PK', 'EVEN', 'OFF') or numbers."""
    if x is None:
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().upper()
    if s in ("PK", "EVEN", "PICK", "0"):
        return 0.0
    m = re.match(r"^[+-]?\d+(\.\d+)?$", s)
    return float(s) if m else np.nan


def _get(d: dict, *path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def _fetch_one(eid: int, path: Path, session: requests.Session, retries: int = 3) -> dict | None:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    delay = 1.0
    for _ in range(retries):
        try:
            r = session.get(ODDS_URL.format(eid=eid), timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 404:
                d = {"items": []}
            else:
                r.raise_for_status()
                d = r.json()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(d), encoding="utf-8")
            return d
        except Exception as e:  # noqa: BLE001
            time.sleep(delay)
            delay *= 2
    log.warning("odds fetch failed for espn id %s", eid)
    return None


def parse_game(d: dict) -> dict:
    """Consensus open/close lines for one game (home perspective, nflverse sign)."""
    rows = []
    for it in d.get("items", []) or []:
        prov = str(_get(it, "provider", "name") or "").strip()
        pl = prov.lower()
        if pl in MODEL_PROVIDERS or "live odds" in pl or pl == "":
            continue
        home, away = it.get("homeTeamOdds", {}) or {}, it.get("awayTeamOdds", {}) or {}
        # ---- closing / current spread (home perspective; ESPN negative = home favoured)
        sp_close = _num(_get(home, "close", "pointSpread", "american"))
        if np.isnan(sp_close):
            sp_close = _num(_get(home, "current", "pointSpread", "american"))
        if np.isnan(sp_close):
            sp_close = _num(it.get("spread"))
        # ---- opening spread
        sp_open = _num(_get(home, "open", "pointSpread", "american"))
        if np.isnan(sp_open):
            sp_open = _num(it.get("initialSpread"))
        # ---- totals
        tot_close = _num(_get(it, "close", "total", "american"))
        if np.isnan(tot_close):
            tot_close = _num(_get(it, "current", "total", "american"))
        if np.isnan(tot_close):
            tot_close = _num(it.get("overUnder"))
        tot_open = _num(_get(it, "open", "total", "american"))
        if np.isnan(tot_open):
            tot_open = _num(it.get("initialOverUnder"))
        # ---- moneylines
        ml_h_close = _num(_get(home, "close", "moneyLine", "american"))
        if np.isnan(ml_h_close):
            ml_h_close = _num(home.get("moneyLine"))
        ml_a_close = _num(_get(away, "close", "moneyLine", "american"))
        if np.isnan(ml_a_close):
            ml_a_close = _num(away.get("moneyLine"))
        ml_h_open = _num(_get(home, "open", "moneyLine", "american"))
        ml_a_open = _num(_get(away, "open", "moneyLine", "american"))
        rows.append({"provider": prov, "sp_open": sp_open, "sp_close": sp_close, "tot_open": tot_open, "tot_close": tot_close,
                     "ml_h_open": ml_h_open, "ml_a_open": ml_a_open, "ml_h_close": ml_h_close, "ml_a_close": ml_a_close})
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    out: dict = {"n_books": int(len(df))}
    opening = df[df["provider"].str.lower() == "opening"]
    # opening spread: explicit "Opening" provider first, else median of per-book opens
    if len(opening) and not np.isnan(opening["sp_close"].iloc[0]):
        out["spread_open"] = -opening["sp_close"].iloc[0]
    elif df["sp_open"].notna().any():
        out["spread_open"] = -float(df["sp_open"].median())
    else:
        out["spread_open"] = np.nan
    if len(opening) and not np.isnan(opening["tot_close"].iloc[0]):
        out["total_open"] = opening["tot_close"].iloc[0]
    elif df["tot_open"].notna().any():
        out["total_open"] = float(df["tot_open"].median())
    else:
        out["total_open"] = np.nan
    books = df[df["provider"].str.lower() != "opening"]
    out["spread_close_espn"] = -float(books["sp_close"].median()) if books["sp_close"].notna().any() else np.nan
    out["total_close_espn"] = float(books["tot_close"].median()) if books["tot_close"].notna().any() else np.nan
    for k in ("ml_h_open", "ml_a_open", "ml_h_close", "ml_a_close"):
        out[k] = float(books[k].median()) if books[k].notna().any() else np.nan
    return out


def fetch_lines(schedules: pd.DataFrame, cfg: Config | None = None, refresh: bool = False, workers: int = 8) -> pd.DataFrame:
    """Fetch (cached) ESPN odds for every game in ``schedules`` and return the parsed line table."""
    cfg = cfg or load_config()
    s = schedules[schedules["espn"].notna()][["game_id", "season", "espn"]].copy()
    s["espn"] = s["espn"].astype(int)
    root = RAW_DIR / "espn_odds"
    todo = []
    for gid, season, eid in s.itertuples(index=False):
        p = root / str(int(season)) / f"{eid}.json"
        if refresh and int(season) >= cfg.current_season and p.exists():
            p.unlink()
        todo.append((gid, int(season), eid, p))
    results: dict[str, dict] = {}
    session = requests.Session()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_one, eid, p, session): gid for gid, _, eid, p in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            gid = futs[fut]
            d = fut.result()
            results[gid] = parse_game(d) if d else {}
            if i % 500 == 0:
                log.info("odds: %s / %s games (%.0fs)", i, len(todo), time.time() - t0)
    rows = []
    for gid, season, eid, _ in todo:
        r = {"game_id": gid, "season": season, "espn": eid}
        r.update(results.get(gid, {}))
        rows.append(r)
    lines = pd.DataFrame(rows)
    for c in ("n_books", "spread_open", "total_open", "spread_close_espn", "total_close_espn",
              "ml_h_open", "ml_a_open", "ml_h_close", "ml_a_close"):
        if c not in lines:
            lines[c] = np.nan
    LINES_PATH.parent.mkdir(parents=True, exist_ok=True)
    # merge into the master table: a weekly slate refresh must never shrink the historical lines
    if LINES_PATH.exists():
        master = pd.read_parquet(LINES_PATH)
        master = master[~master["game_id"].isin(lines["game_id"])]
        merged = pd.concat([master, lines], ignore_index=True).sort_values(["season", "game_id"]).reset_index(drop=True)
    else:
        merged = lines
    merged.to_parquet(LINES_PATH, index=False)
    cov = lines.groupby("season").agg(games=("game_id", "size"), open_spread=("spread_open", lambda x: x.notna().mean()),
                                      open_total=("total_open", lambda x: x.notna().mean()),
                                      close_spread=("spread_close_espn", lambda x: x.notna().mean()))
    log.info("opening-line coverage by season:\n%s", cov.round(2).to_string())
    return lines


def load_lines() -> pd.DataFrame:
    if LINES_PATH.exists():
        return pd.read_parquet(LINES_PATH)
    return pd.DataFrame(columns=["game_id", "spread_open", "total_open", "spread_close_espn", "total_close_espn",
                                 "ml_h_open", "ml_a_open", "ml_h_close", "ml_a_close", "n_books"])
