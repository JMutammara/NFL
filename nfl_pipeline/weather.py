"""Kickoff weather forecasts for upcoming outdoor games (Open-Meteo, no API key).

Training rows carry the observed game-time temperature and wind from the
nflverse schedule. For games that have not been played the schedule is empty,
so the feature build imputes climatology. This module replaces that with a
real forecast for games inside the forecast horizon, which matters most for
totals: wind is the best-documented weather effect on scoring.
"""
from __future__ import annotations

import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

from .config import RAW_DIR, Config, load_config
from .features.context import _resolve_venue, _stadium_team_lookup
from .utils import log

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_DIR = RAW_DIR / "forecasts"
ET_OFFSET = -5.0
# Home venues with a roof (fixed or retractable). When the schedule leaves ``roof`` blank for an
# upcoming game we assume these play indoors rather than fetching an outdoor forecast.
ROOFED_HOME_TEAMS = {"ARI", "ATL", "DAL", "DET", "HOU", "IND", "LA", "LAC", "LV", "MIN", "NO"}


def fetch_forecasts(schedule_rows: pd.DataFrame, all_schedules: pd.DataFrame, cfg: Config | None = None,
                    max_days_ahead: int = 15, refresh_hours: float = 6.0) -> pd.DataFrame:
    """Return game_id, temp_f, wind_mph, gust_mph, precip_prob, hours_ahead for outdoor games in the horizon."""
    cfg = cfg or load_config()
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    lookup = _stadium_team_lookup(all_schedules)
    today = pd.Timestamp.now().normalize()
    rows = []
    for _, r in schedule_rows.iterrows():
        roof = str(r.get("roof") or "").lower()
        if roof in ("dome", "closed"):
            continue
        if roof in ("", "nan", "none") and r.get("location") != "Neutral" and r.get("home_team") in ROOFED_HOME_TEAMS:
            continue
        gameday = pd.Timestamp(r["gameday"])
        days_ahead = (gameday - today).days
        if days_ahead < 0 or days_ahead > max_days_ahead:
            continue
        v = _resolve_venue(r, lookup)
        if v is None or not np.isfinite(v.lat):
            continue
        kick = pd.to_datetime(r.get("gametime"), format="%H:%M", errors="coerce")
        kick_h = 13.0 if pd.isna(kick) else kick.hour + kick.minute / 60.0
        local_h = int(round(kick_h + (v.tz - ET_OFFSET)))
        local_h = min(max(local_h, 0), 23)
        cache = FORECAST_DIR / f"{r['game_id']}.json"
        data = None
        if cache.exists() and (time.time() - cache.stat().st_mtime) / 3600.0 < refresh_hours:
            try:
                data = json.loads(cache.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                data = None
        if data is None:
            params = {"latitude": v.lat, "longitude": v.lon,
                      "hourly": "temperature_2m,wind_speed_10m,wind_gusts_10m,precipitation_probability",
                      "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "timezone": "auto", "forecast_days": 16}
            for attempt in range(3):
                try:
                    resp = requests.get(FORECAST_URL, params=params, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                    cache.write_text(json.dumps(data), encoding="utf-8")
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 2:
                        log.warning("forecast fetch failed for %s: %s", r["game_id"], e)
                    time.sleep(1.5 * (attempt + 1))
            if data is None:
                continue
        h = data.get("hourly", {})
        times = h.get("time", [])
        key = f"{gameday.strftime('%Y-%m-%d')}T{local_h:02d}:00"
        if key not in times:
            continue
        i = times.index(key)
        sl = slice(i, min(i + 4, len(times)))  # kickoff through roughly the end of the game
        temp = np.nanmean(np.array(h["temperature_2m"][sl], dtype=float))
        wind = np.nanmean(np.array(h["wind_speed_10m"][sl], dtype=float))
        gust = np.nanmax(np.array(h.get("wind_gusts_10m", [np.nan] * len(times))[sl], dtype=float))
        pp = np.nanmax(np.array(h.get("precipitation_probability", [np.nan] * len(times))[sl], dtype=float))
        rows.append({"game_id": r["game_id"], "temp_f": float(temp), "wind_mph": float(wind), "gust_mph": float(gust),
                     "precip_prob": float(pp), "hours_ahead": float((gameday + pd.Timedelta(hours=kick_h) - pd.Timestamp.now()).total_seconds() / 3600.0)})
    fc = pd.DataFrame(rows, columns=["game_id", "temp_f", "wind_mph", "gust_mph", "precip_prob", "hours_ahead"])
    log.info("forecasts: %s outdoor games within %s days", len(fc), max_days_ahead)
    return fc


def apply_forecasts(df: pd.DataFrame, fc: pd.DataFrame) -> pd.DataFrame:
    """Overwrite weather features in ``df`` (any frame with a game_id column) with forecast values."""
    if fc is None or fc.empty or "game_id" not in df:
        return df
    df = df.copy()
    f = fc.set_index("game_id")
    m = df["game_id"].isin(f.index)
    if not m.any():
        return df
    temp = df.loc[m, "game_id"].map(f["temp_f"])
    wind = df.loc[m, "game_id"].map(f["wind_mph"])
    for col, val in (("temp_f", temp), ("wind_mph", wind)):
        if col in df:
            df.loc[m, col] = val.to_numpy()
    if "weather_imputed" in df:
        df.loc[m, "weather_imputed"] = 0
    if "cold_game" in df:
        df.loc[m, "cold_game"] = (temp < 40).astype(int).to_numpy()
    if "windy_game" in df:
        df.loc[m, "windy_game"] = (wind >= 15).astype(int).to_numpy()
    return df
