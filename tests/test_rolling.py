import numpy as np
import pandas as pd

from nfl_pipeline.features.rolling import add_rolling_features


def _toy():
    rows = []
    for team in ("A", "B"):
        for season in (2020, 2021):
            for wk in range(1, 6):
                rows.append({"team": team, "season": season, "week": wk, "gameday": pd.Timestamp(f"{season}-09-01") + pd.Timedelta(days=7 * wk),
                             "x": float(wk + (10 if season == 2021 else 0) + (100 if team == "B" else 0)), "played": 1})
    return pd.DataFrame(rows)


def test_rolling_never_sees_current_game():
    df = _toy()
    out = add_rolling_features(df, ["x"], "team", "gameday", windows=(3,), ewm_halflife=2.0, ewm_shrink=0.5, cross_season=True)
    a = out[out.team == "A"].sort_values("gameday").reset_index(drop=True)
    assert np.isnan(a.loc[0, "x_r3"]) and np.isnan(a.loc[0, "x_ewm"])
    assert a.loc[1, "x_r3"] == a.loc[0, "x"]
    assert a.loc[3, "x_r3"] == a.loc[0:2, "x"].mean()
    # feature at row i depends only on rows < i
    for i in range(1, len(a)):
        assert a.loc[i, "x_r3"] == a.loc[max(0, i - 3):i - 1, "x"].mean()


def test_ewm_season_carryover_is_shrunk_toward_league_mean():
    df = _toy()
    out = add_rolling_features(df, ["x"], "team", "gameday", windows=(), ewm_halflife=2.0, ewm_shrink=0.5, cross_season=True)
    a = out[out.team == "A"].sort_values("gameday").reset_index(drop=True)
    first_2021 = a[a.season == 2021].iloc[0]
    last_2020_state = a[a.season == 2021].iloc[0]["x_ewm"]  # equals shrunk carry-over
    league_mean_2020 = df[df.season == 2020]["x"].mean()
    # carried state must lie strictly between the team's 2020 EWM and the 2020 league mean
    # (team A is below league mean, so state > raw EWM)
    raw_2020_ewm_after_last = None
    alpha = 1 - np.exp(np.log(0.5) / 2.0)
    s = np.nan
    for v in df[(df.team == "A") & (df.season == 2020)].sort_values("week")["x"]:
        s = v if np.isnan(s) else s + alpha * (v - s)
    raw_2020_ewm_after_last = s
    assert raw_2020_ewm_after_last < last_2020_state < league_mean_2020
    assert np.isclose(last_2020_state, league_mean_2020 + 0.5 * (raw_2020_ewm_after_last - league_mean_2020))
    assert first_2021["n_prior_games_season"] == 0


def test_unplayed_rows_are_skipped_not_zeroed():
    df = _toy()
    df.loc[(df.team == "A") & (df.season == 2021) & (df.week == 5), ["x", "played"]] = [np.nan, 0]
    out = add_rolling_features(df, ["x"], "team", "gameday", windows=(3,), ewm_halflife=2.0, ewm_shrink=0.5, cross_season=True)
    a = out[(out.team == "A") & (out.season == 2021)].sort_values("week").reset_index(drop=True)
    assert a.loc[4, "x_r3"] == a.loc[1:3, "x"].mean()
    assert a.loc[4, "n_prior_games_season"] == 4
