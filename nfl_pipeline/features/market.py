"""Market-implied team ratings from prior closing lines.

For every (season, week) block we fit a weighted ridge regression on the
closing spreads of games played strictly before the block's first kickoff:

    spread_line_g = r[home_g] - r[away_g] + hfa * (not neutral_g)

with exponential time decay (halflife in days) and a lookback window that
spans the previous season, so week 1 carries last year's market view. The
same is done for totals (``total_line_g = t[home] + t[away] + c``).

The interesting feature is the deviation of the actual closing line from
what the market's own ratings imply (``mkt_line_dev``): a line that has moved
away from the market's slow-moving team valuation often reflects a reaction
to recent noise rather than new information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import CURRENT_TEAMS, log

TEAM_IDX = {t: i for i, t in enumerate(CURRENT_TEAMS)}
N_TEAMS = len(CURRENT_TEAMS)


def _weighted_ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Solve (X'WX + diag(lam)) b = X'Wy."""
    Xw = X * w[:, None]
    A = X.T @ Xw + np.diag(lam)
    b = Xw.T @ y
    return np.linalg.solve(A, b)


def market_implied_ratings(schedules: pd.DataFrame, halflife_days: float = 60.0, lookback_days: float = 420.0,
                           ridge_team: float = 3.0, ridge_hfa: float = 0.5) -> pd.DataFrame:
    """Per-game market-implied ratings computed from games strictly before the game's week."""
    s = schedules[["game_id", "season", "week", "gameday", "home_team", "away_team", "location", "spread_line", "total_line"]].copy()
    s = s[s["home_team"].isin(TEAM_IDX) & s["away_team"].isin(TEAM_IDX)]
    s["gameday"] = pd.to_datetime(s["gameday"])
    s["neutral"] = (s["location"] == "Neutral").astype(float)
    s = s.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    hi = s["home_team"].map(TEAM_IDX).to_numpy()
    ai = s["away_team"].map(TEAM_IDX).to_numpy()
    days = s["gameday"].to_numpy(dtype="datetime64[D]").astype(np.int64)
    spread = s["spread_line"].to_numpy(dtype=float)
    total = s["total_line"].to_numpy(dtype=float)
    neutral = s["neutral"].to_numpy()

    out = np.full((len(s), 9), np.nan)
    blocks = s.groupby(["season", "week"], sort=False).indices
    keys = sorted(blocks, key=lambda k: days[blocks[k]].min())
    for key in keys:
        idx = blocks[key]
        ref = days[idx].min()
        prior = np.flatnonzero((days < ref) & (days >= ref - lookback_days) & ~np.isnan(spread))
        if len(prior) < 40:
            continue
        w = np.exp(-np.log(2.0) * (ref - days[prior]) / halflife_days)
        # spreads: columns = 32 team ratings + HFA
        X = np.zeros((len(prior), N_TEAMS + 1))
        X[np.arange(len(prior)), hi[prior]] += 1.0
        X[np.arange(len(prior)), ai[prior]] -= 1.0
        X[:, N_TEAMS] = 1.0 - neutral[prior]
        lam = np.r_[np.full(N_TEAMS, ridge_team), ridge_hfa]
        b = _weighted_ridge(X, spread[prior], w, lam)
        r, hfa = b[:N_TEAMS], b[N_TEAMS]
        # totals: columns = 32 team contributions + intercept
        pt = prior[~np.isnan(total[prior])]
        if len(pt) >= 40:
            wt = np.exp(-np.log(2.0) * (ref - days[pt]) / halflife_days)
            Xt = np.zeros((len(pt), N_TEAMS + 1))
            Xt[np.arange(len(pt)), hi[pt]] += 1.0
            Xt[np.arange(len(pt)), ai[pt]] += 1.0
            Xt[:, N_TEAMS] = 1.0
            lamt = np.r_[np.full(N_TEAMS, ridge_team), 1e-6]
            bt = _weighted_ridge(Xt, total[pt], wt, lamt)
            t, c = bt[:N_TEAMS], bt[N_TEAMS]
        else:
            t, c = np.full(N_TEAMS, np.nan), np.nan
        for i in idx:
            line_hat = r[hi[i]] - r[ai[i]] + hfa * (1.0 - neutral[i])
            total_hat = t[hi[i]] + t[ai[i]] + c
            out[i] = [r[hi[i]], r[ai[i]], hfa, line_hat, spread[i] - line_hat, t[hi[i]], t[ai[i]], total_hat, total[i] - total_hat]
    cols = ["mkt_rt_home", "mkt_rt_away", "mkt_hfa", "mkt_line_hat", "mkt_line_dev", "mkt_tot_home", "mkt_tot_away",
            "mkt_total_hat", "mkt_total_dev"]
    res = pd.DataFrame(out, columns=cols)
    res.insert(0, "game_id", s["game_id"].to_numpy())
    res["mkt_rt_diff"] = res["mkt_rt_home"] - res["mkt_rt_away"]
    ok = res["mkt_line_hat"].notna()
    log.info("market-implied ratings: %s of %s games (line-vs-rating deviation sd %.2f pts, total dev sd %.2f)",
             int(ok.sum()), len(res), res.loc[ok, "mkt_line_dev"].std(), res.loc[ok, "mkt_total_dev"].std())
    return res


MARKET_RATING_FEATURES = ["mkt_rt_home", "mkt_rt_away", "mkt_rt_diff", "mkt_hfa", "mkt_line_hat", "mkt_line_dev",
                          "mkt_tot_home", "mkt_tot_away", "mkt_total_hat", "mkt_total_dev"]
