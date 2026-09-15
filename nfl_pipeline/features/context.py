"""Game context features: rest, travel, time zones, weather, venue, scheduling.

Everything here is known before kickoff. Weather for upcoming games is
imputed from stadium-month climatology (flagged with ``weather_imputed``)
unless a forecast is supplied via ``forecast_override``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..static.venues import Venue, neutral_venue, team_venue
from ..utils import haversine_miles, log

ET_OFFSET = -5.0


def _stadium_team_lookup(schedules: pd.DataFrame) -> dict[str, str]:
    s = schedules[(schedules["location"] == "Home") & schedules["stadium"].notna()]
    return s.groupby("stadium")["home_team"].agg(lambda x: x.value_counts().index[0]).to_dict()


def _resolve_venue(row, lookup: dict[str, str]) -> Venue:
    season = int(row["season"])
    if row["location"] == "Neutral":
        v = neutral_venue(row["stadium"])
        if v is None:
            t = lookup.get(row["stadium"])
            if t:
                v = team_venue(t, season)
        if v is not None:
            return v
    return team_venue(row["home_team"], season) or Venue(np.nan, np.nan, ET_OFFSET)


def _climatology(schedules: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    played = schedules[schedules["home_score"].notna() & schedules["temp"].notna()].copy()
    played["month"] = played["gameday"].dt.month
    out = played[~played["roof"].isin(["dome", "closed"])]
    by_stadium = out.groupby(["stadium", "month"])[["temp", "wind"]].mean()
    by_team = out.groupby(["home_team", "month"])[["temp", "wind"]].mean()
    by_month = out.groupby("month")[["temp", "wind"]].mean()
    return by_stadium, by_team, by_month


def build_game_context(schedules: pd.DataFrame, cfg: Config, forecast_override: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per game_id with pre-game context features."""
    s = schedules.copy()
    lookup = _stadium_team_lookup(s)
    ctx = pd.DataFrame({"game_id": s["game_id"].values})
    ctx["season"] = s["season"].values
    ctx["week"] = s["week"].values
    ctx["is_playoff"] = (s["game_type"] != "REG").astype(int).values
    ctx["is_neutral"] = (s["location"] == "Neutral").astype(int).values
    ctx["div_game"] = s["div_game"].fillna(0).astype(int).values

    # kickoff & weekday
    hour = pd.to_datetime(s["gametime"], format="%H:%M", errors="coerce")
    ctx["kickoff_hour_et"] = (hour.dt.hour + hour.dt.minute / 60.0).fillna(13.0).values
    ctx["is_primetime"] = (ctx["kickoff_hour_et"] >= 19.5).astype(int)
    wd = s["gameday"].dt.dayofweek
    ctx["is_thu"] = (wd == 3).astype(int).values
    ctx["is_sat"] = (wd == 5).astype(int).values
    ctx["is_mon"] = (wd == 0).astype(int).values
    ctx["is_sun"] = (wd == 6).astype(int).values

    # rest
    ctx["home_rest"] = s["home_rest"].astype(float).values
    ctx["away_rest"] = s["away_rest"].astype(float).values
    ctx["rest_diff"] = ctx["home_rest"] - ctx["away_rest"]
    ctx["home_short_week"] = (ctx["home_rest"] <= 5).astype(int)
    ctx["away_short_week"] = (ctx["away_rest"] <= 5).astype(int)
    ctx["home_off_bye"] = (ctx["home_rest"] >= 13).astype(int)
    ctx["away_off_bye"] = (ctx["away_rest"] >= 13).astype(int)

    # venue / travel / body clock
    venues = [_resolve_venue(r, lookup) for _, r in s.iterrows()]
    v_lat = np.array([v.lat for v in venues]); v_lon = np.array([v.lon for v in venues])
    v_tz = np.array([v.tz for v in venues]); v_alt = np.array([v.alt_m for v in venues])
    hv = [team_venue(t, int(y)) for t, y in zip(s["home_team"], s["season"])]
    av = [team_venue(t, int(y)) for t, y in zip(s["away_team"], s["season"])]
    h_lat = np.array([v.lat if v else np.nan for v in hv]); h_lon = np.array([v.lon if v else np.nan for v in hv])
    a_lat = np.array([v.lat if v else np.nan for v in av]); a_lon = np.array([v.lon if v else np.nan for v in av])
    h_tz = np.array([v.tz if v else ET_OFFSET for v in hv]); a_tz = np.array([v.tz if v else ET_OFFSET for v in av])
    ctx["home_travel_mi"] = haversine_miles(h_lat, h_lon, v_lat, v_lon)
    ctx["away_travel_mi"] = haversine_miles(a_lat, a_lon, v_lat, v_lon)
    ctx["travel_diff"] = ctx["away_travel_mi"] - ctx["home_travel_mi"]
    ctx["home_tz_shift"] = v_tz - h_tz
    ctx["away_tz_shift"] = v_tz - a_tz
    ctx["home_body_clock"] = ctx["kickoff_hour_et"] + (h_tz - ET_OFFSET)
    ctx["away_body_clock"] = ctx["kickoff_hour_et"] + (a_tz - ET_OFFSET)
    ctx["venue_alt_m"] = v_alt
    ctx["intl_game"] = (np.abs(v_lon) < 30).astype(int) | (v_lat < 0).astype(int)  # Europe/Brazil/Aus

    # surface / roof / weather
    roof = s["roof"].astype("string").str.lower()
    ctx["is_dome"] = roof.isin(["dome", "closed"]).astype(int).values
    ctx["is_retractable"] = roof.isin(["closed", "open"]).astype(int).values
    ctx["is_grass"] = s["surface"].astype("string").str.lower().str.contains("grass", na=False).astype(int).values
    temp = s["temp"].astype(float).copy()
    wind = s["wind"].astype(float).copy()
    if forecast_override is not None and len(forecast_override):
        fo = forecast_override.set_index("game_id")
        m = s["game_id"].map(fo["temp"]) if "temp" in fo else pd.Series(np.nan, index=s.index)
        w = s["game_id"].map(fo["wind"]) if "wind" in fo else pd.Series(np.nan, index=s.index)
        temp = temp.fillna(m); wind = wind.fillna(w)
    dome = ctx["is_dome"].values == 1
    temp = temp.where(~dome, cfg.get("features.dome_temp_f", 70.0))
    wind = wind.where(~dome, cfg.get("features.dome_wind_mph", 0.0))
    missing = temp.isna() | wind.isna()
    by_stadium, by_team, by_month = _climatology(s, cfg)
    month = s["gameday"].dt.month
    imp_t = np.full(len(s), np.nan); imp_w = np.full(len(s), np.nan)
    for i, (st, ht, mo) in enumerate(zip(s["stadium"], s["home_team"], month)):
        for tbl, key in ((by_stadium, (st, mo)), (by_team, (ht, mo)), (by_month, mo)):
            if key in tbl.index:
                imp_t[i], imp_w[i] = tbl.loc[key, "temp"], tbl.loc[key, "wind"]
                break
    ctx["temp_f"] = temp.fillna(pd.Series(imp_t, index=s.index)).values
    ctx["wind_mph"] = wind.fillna(pd.Series(imp_w, index=s.index)).values
    ctx["weather_imputed"] = missing.astype(int).values
    ctx["cold_game"] = (ctx["temp_f"] < 40).astype(int)
    ctx["windy_game"] = (ctx["wind_mph"] >= 15).astype(int)

    # market (home perspective). nflverse spread_line > 0 => home favoured.
    ctx["spread_line"] = s["spread_line"].astype(float).values
    ctx["total_line"] = s["total_line"].astype(float).values
    from ..utils import american_to_prob
    ph = american_to_prob(s["home_moneyline"].astype(float).values)
    pa = american_to_prob(s["away_moneyline"].astype(float).values)
    with np.errstate(invalid="ignore"):
        ctx["home_ml_prob_novig"] = ph / (ph + pa)
    ctx["home_implied_pts"] = (ctx["total_line"] + ctx["spread_line"]) / 2.0
    ctx["away_implied_pts"] = (ctx["total_line"] - ctx["spread_line"]) / 2.0
    ctx["market_available"] = ctx["spread_line"].notna().astype(int)
    log.info("context: %s games, %s features; weather imputed for %s games", len(ctx), ctx.shape[1] - 3, int(ctx["weather_imputed"].sum()))
    return ctx


CONTEXT_FEATURES = [
    "week", "is_playoff", "is_neutral", "div_game", "kickoff_hour_et", "is_primetime", "is_thu", "is_sat", "is_mon",
    "home_rest", "away_rest", "rest_diff", "home_short_week", "away_short_week", "home_off_bye", "away_off_bye",
    "home_travel_mi", "away_travel_mi", "travel_diff", "home_tz_shift", "away_tz_shift", "home_body_clock",
    "away_body_clock", "venue_alt_m", "intl_game", "is_dome", "is_retractable", "is_grass", "temp_f", "wind_mph",
    "weather_imputed", "cold_game", "windy_game",
]
MARKET_FEATURES = ["spread_line", "total_line", "home_ml_prob_novig", "home_implied_pts", "away_implied_pts"]
