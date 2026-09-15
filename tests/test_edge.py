import json

import numpy as np
import pandas as pd

from nfl_pipeline.features.market import market_implied_ratings
from nfl_pipeline.lines import parse_game
from nfl_pipeline.models.edge_model import breakeven, build_edge_frame, roi_table


def _schedule(n_weeks=12, seed=0):
    rng = np.random.default_rng(seed)
    teams = ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE"]
    true = {t: v for t, v in zip(teams, [-4, -2, 0, 1, 2, 3, 4, 6])}
    rows = []
    for season in (2023, 2024):
        for wk in range(1, n_weeks + 1):
            order = rng.permutation(teams)
            for i in range(0, len(teams), 2):
                h, a = order[i], order[i + 1]
                spread = true[h] - true[a] + 2.0 + rng.normal(0, 0.5)
                rows.append({"game_id": f"{season}_{wk:02d}_{a}_{h}", "season": season, "week": wk,
                             "gameday": pd.Timestamp(f"{season}-09-01") + pd.Timedelta(days=7 * wk), "home_team": h, "away_team": a,
                             "location": "Home", "spread_line": round(spread * 2) / 2, "total_line": 44 + rng.normal(0, 2)})
    return pd.DataFrame(rows)


def test_market_ratings_recover_team_strength_and_are_prior_only():
    sch = _schedule()
    r = market_implied_ratings(sch, halflife_days=400, lookback_days=800, ridge_team=0.1)
    m = sch.merge(r, on="game_id")
    late = m[(m.season == 2024) & (m.week >= 8)]
    # ratings fitted from prior lines should reproduce the current line closely
    assert late["mkt_line_dev"].abs().mean() < 1.0
    assert late["mkt_hfa"].mean() > 1.0
    # first block has no prior games -> NaN, never a peek at its own line
    first = m[(m.season == 2023) & (m.week == 1)]
    assert first["mkt_line_hat"].isna().all()


def test_parse_game_handles_both_espn_eras():
    old = {"items": [
        {"provider": {"name": "5Dimes.eu"}, "spread": -4.5, "overUnder": 47.0, "initialSpread": -3.0, "initialOverUnder": 46.5,
         "homeTeamOdds": {"moneyLine": -200}, "awayTeamOdds": {"moneyLine": 175}},
        {"provider": {"name": "Opening"}, "spread": -3.0, "overUnder": 46.5, "homeTeamOdds": {}, "awayTeamOdds": {}},
        {"provider": {"name": "numberfire"}, "spread": 1.0, "overUnder": 40.0, "homeTeamOdds": {}, "awayTeamOdds": {}},
    ]}
    p = parse_game(old)
    assert p["spread_open"] == 3.0 and p["spread_close_espn"] == 4.5   # nflverse sign: home favoured positive
    assert p["total_open"] == 46.5 and p["total_close_espn"] == 47.0
    assert p["ml_h_close"] == -200 and p["n_books"] == 2                 # model provider excluded
    new = {"items": [{"provider": {"name": "DraftKings"}, "spread": -4.5, "overUnder": 53.5,
                      "open": {"total": {"american": "52.5"}}, "current": {"total": {"american": "53.5"}},
                      "homeTeamOdds": {"moneyLine": -218, "open": {"pointSpread": {"american": "-3"}, "moneyLine": {"american": "-162"}},
                                       "current": {"pointSpread": {"american": "-4.5"}}},
                      "awayTeamOdds": {"moneyLine": 180, "open": {"moneyLine": {"american": "+136"}}}}]}
    p = parse_game(new)
    assert p["spread_open"] == 3.0 and p["spread_close_espn"] == 4.5
    assert p["total_open"] == 52.5 and p["total_close_espn"] == 53.5
    assert p["ml_h_open"] == -162 and p["ml_a_open"] == 136


def test_edge_frame_targets_and_open_fallback():
    gf = pd.DataFrame({"game_id": ["a", "b"], "spread_line": [3.0, -2.0], "total_line": [44.0, 50.0], "spread_open": [2.5, np.nan],
                       "total_open": [43.5, np.nan], "home_margin": [7.0, -10.0], "total_points": [51.0, 38.0],
                       "mkt_line_hat": [2.0, -1.0], "mkt_total_hat": [45.0, 49.0]})
    e = build_edge_frame(gf, np.array([4.0, -3.0]), np.array([46.0, 47.0]), ref="open")
    assert list(e["spread_ref"]) == [2.5, -2.0] and list(e["ref_is_open"]) == [1.0, 0.0]
    assert list(e["resid_margin"]) == [4.5, -8.0] and list(e["cover_home"]) == [1.0, -1.0]
    assert list(e["clv_spread"]) == [0.5, 0.0]
    assert np.isclose(e["pf_margin_dev"].iloc[0], 1.5)


def test_roi_table_flat_and_breakeven():
    p = np.array([0.6, 0.6, 0.4, 0.55, 0.5])
    cover = np.array([1, -1, -1, 0, 1])        # win, loss, win (took the under side), push, win
    r = roi_table(p, cover, clv=np.array([1.0, -1.0, -0.5, 0, 0]), thresholds=(0.5, 0.58), n_boot=50)
    flat = r["p>=0.50"]
    assert flat["n"] == 5 and np.isclose(flat["hit_rate"], 0.75)
    assert np.isclose(flat["units"], 3 * 100 / 110 - 1)
    assert r["p>=0.58"]["n"] == 3   # 0.40 is a 0.60-confidence pick on the other side
    assert np.isclose(breakeven(-110), 110 / 210)
