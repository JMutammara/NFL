"""Shared helpers: logging, timing, team-code normalisation, geo math."""
from __future__ import annotations

import logging
import math
import sys
import time
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import pandas as pd

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "nfl") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


log = get_logger()


@contextmanager
def timed(label: str) -> Iterator[None]:
    t0 = time.time()
    log.info("▶ %s", label)
    yield
    log.info("✔ %s (%.1fs)", label, time.time() - t0)


# ---------------------------------------------------------------------------
# Team code normalisation. nflverse uses historical codes for relocated
# franchises; we collapse everything onto the current code so a franchise is
# one continuous entity for rolling features. PFR / ESPN spellings are also
# mapped.
# ---------------------------------------------------------------------------
TEAM_ALIASES: dict[str, str] = {
    "OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "JAC": "JAX", "WSH": "WAS",
    "HST": "HOU", "BLT": "BAL", "CLV": "CLE", "ARZ": "ARI", "SL": "LA",
    # PFR abbreviations
    "GNB": "GB", "KAN": "KC", "NWE": "NE", "NOR": "NO", "SFO": "SF", "TAM": "TB",
    "LVR": "LV", "SDG": "LAC", "RAM": "LA", "RAI": "LV", "CRD": "ARI", "RAV": "BAL",
    "HTX": "HOU", "CLT": "IND", "OTI": "TEN", "NYG": "NYG", "NYJ": "NYJ",
}
CURRENT_TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB",
    "HOU", "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
]


def norm_team(x: pd.Series | str | None):
    """Map any historical/alternate code to the current franchise code."""
    if x is None:
        return None
    if isinstance(x, pd.Series):
        return x.astype("string").str.upper().replace(TEAM_ALIASES)
    return TEAM_ALIASES.get(str(x).upper(), str(x).upper())


# ---------------------------------------------------------------------------
# Geo
# ---------------------------------------------------------------------------
EARTH_RADIUS_MI = 3958.8


def haversine_miles(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance in miles."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_MI * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def safe_div(num, den, fill=np.nan):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den != 0, num / den, fill)
    return out


def american_to_prob(odds):
    """Implied probability (vig included) from American odds."""
    o = np.asarray(odds, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pos = 100.0 / (o + 100.0)
        neg = -o / (-o + 100.0)
    return np.where(o > 0, pos, neg)


def prob_to_american(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.where(p >= 0.5, -100 * p / (1 - p), 100 * (1 - p) / p)


def american_payout(odds):
    """Net profit per 1 unit staked for a winning bet at American odds."""
    o = np.asarray(odds, dtype=float)
    return np.where(o > 0, o / 100.0, 100.0 / np.abs(o))


def reduce_mem(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float64 -> float32 and int64 -> int32 where lossless."""
    for c in df.columns:
        dt = df[c].dtype
        if dt == "float64":
            df[c] = df[c].astype("float32")
        elif dt == "int64":
            if df[c].abs().max() < 2**31 - 1:
                df[c] = df[c].astype("int32")
    return df


def season_week_key(season, week):
    """Monotone integer key for ordering (season, week)."""
    return np.asarray(season, dtype=int) * 100 + np.asarray(week, dtype=int)
