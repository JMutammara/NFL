"""Leakage-safe rolling features.

For an entity (team or player) ordered by game date, each feature at game *g*
is a function of games strictly before *g*:

* ``_r{w}``  : simple mean of the previous ``w`` games (min 1)
* ``_ewm``   : exponentially weighted mean, halflife in games, season-long
               with carry-over — at week 1 the state is the previous
               season's final EWM shrunk toward the league (or position)
               mean, so early-season rows are informed without leaking.
* ``_sd{w}`` : rolling standard deviation over the previous ``w`` games
               (volatility features; players only by default)

The shift is applied *before* any window so the current game's outcome is
never included. Unplayed (NaN) games are skipped, not treated as zero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import log


def _shifted_rolling(df: pd.DataFrame, cols: list[str], group_col: str, window: int, stat: str,
                     season_col: str | None) -> pd.DataFrame:
    keys = [group_col] if season_col is None else [group_col, season_col]
    g = df.groupby(keys, sort=False)[cols]
    if stat == "mean":
        out = g.transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    elif stat == "std":
        out = g.transform(lambda s: s.shift(1).rolling(window, min_periods=2).std())
    else:
        raise ValueError(stat)
    return out


def _ewm_with_carryover(df: pd.DataFrame, cols: list[str], group_col: str, season_col: str,
                        halflife: float, shrink: float, prior_group_col: str | None) -> pd.DataFrame:
    """Shifted EWM per entity with season resets and shrunk carry-over priors."""
    alpha = 1.0 - np.exp(np.log(0.5) / halflife)
    X = df[cols].to_numpy(dtype=float)
    out = np.full_like(X, np.nan)

    # league / position mean of each metric in each season (used for the prior)
    if prior_group_col:
        lm = df.groupby([season_col, prior_group_col])[cols].mean().astype("float64")
    else:
        lm = df.groupby(season_col)[cols].mean().astype("float64")
    seasons = np.sort(df[season_col].unique())
    season_pos = {s: i for i, s in enumerate(seasons)}

    ent = df[group_col].to_numpy()
    sea = df[season_col].to_numpy()
    pg = df[prior_group_col].to_numpy() if prior_group_col else None
    # df must already be sorted by (group, date)
    order = np.arange(len(df))
    starts = np.flatnonzero(np.r_[True, ent[1:] != ent[:-1]])
    ends = np.r_[starts[1:], len(df)]
    for s0, s1 in zip(starts, ends):
        state = np.full(len(cols), np.nan)
        last_season = None
        for i in range(s0, s1):
            season = sea[i]
            if season != last_season:
                # season boundary: shrink last state toward previous season's mean
                if last_season is not None and not np.all(np.isnan(state)):
                    prev_s = seasons[season_pos[season] - 1] if season_pos[season] > 0 else None
                    try:
                        key = (prev_s, pg[i]) if prior_group_col else prev_s
                        mean_prev = lm.loc[key].to_numpy(dtype=float) if prev_s is not None else np.full(len(cols), np.nan)
                    except KeyError:
                        mean_prev = np.full(len(cols), np.nan)
                    mean_prev = np.where(np.isnan(mean_prev), state, mean_prev)
                    state = mean_prev + shrink * (state - mean_prev)
                last_season = season
            out[i] = state
            x = X[i]
            ok = ~np.isnan(x)
            if ok.any():
                st = state.copy()
                st[ok & np.isnan(state)] = x[ok & np.isnan(state)]
                upd = ok & ~np.isnan(state)
                st[upd] = state[upd] + alpha * (x[upd] - state[upd])
                state = st
    return pd.DataFrame(out, columns=cols, index=df.index)


def add_rolling_features(
    df: pd.DataFrame,
    cols: list[str],
    group_col: str,
    date_col: str,
    season_col: str = "season",
    windows: tuple[int, ...] = (3, 6),
    ewm_halflife: float = 4.0,
    ewm_shrink: float = 0.6,
    cross_season: bool = True,
    std_windows: tuple[int, ...] = (),
    prior_group_col: str | None = None,
    order_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Append rolling columns to ``df`` (returns a new sorted DataFrame)."""
    sort_cols = [group_col] + (order_cols or [date_col])
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    cols = [c for c in cols if c in df.columns]
    # nullable / object dtypes (pd.NA) break numpy math: force plain float64
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce").astype("float64")
    pieces = [df]
    season_key = None if cross_season else season_col
    for w in windows:
        r = _shifted_rolling(df, cols, group_col, w, "mean", season_key)
        r.columns = [f"{c}_r{w}" for c in cols]
        pieces.append(r)
    for w in std_windows:
        r = _shifted_rolling(df, cols, group_col, w, "std", season_key)
        r.columns = [f"{c}_sd{w}" for c in cols]
        pieces.append(r)
    e = _ewm_with_carryover(df, cols, group_col, season_col, ewm_halflife, ewm_shrink, prior_group_col)
    e.columns = [f"{c}_ewm" for c in cols]
    pieces.append(e)
    # count of prior games for the entity (all-time and this season)
    n_prior = df.groupby(group_col).cumcount()
    n_prior_season = df.groupby([group_col, season_col]).cumcount()
    # exclude unplayed games from the count when a 'played' flag exists
    if "played" in df.columns:
        n_prior = df.groupby(group_col)["played"].transform(lambda s: s.shift(1).fillna(0).cumsum())
        n_prior_season = df.groupby([group_col, season_col])["played"].transform(lambda s: s.shift(1).fillna(0).cumsum())
    pieces.append(pd.DataFrame({"n_prior_games": n_prior.astype(float), "n_prior_games_season": n_prior_season.astype(float)}, index=df.index))
    out = pd.concat(pieces, axis=1)
    log.info("rolling: %s base metrics -> %s features (windows=%s, ewm hl=%s)", len(cols),
             len(cols) * (len(windows) + len(std_windows) + 1), windows, ewm_halflife)
    return out


def rolling_feature_names(cols: list[str], windows=(3, 6), std_windows=()) -> list[str]:
    names = []
    for c in cols:
        names += [f"{c}_r{w}" for w in windows] + [f"{c}_sd{w}" for w in std_windows] + [f"{c}_ewm"]
    return names
