```
==============================================================================
FEATURE ENGINEERING SUMMARY  (generated 2026-09-16 11:45)
==============================================================================

[GAME-LEVEL MATRIX]  data/features/game_features.parquet
  rows            : 4,101 games  (3,845 played for training, 256 upcoming)
  seasons         : 2012-2026   (postseason included: True)
  features        : 1,090
  targets         : home_margin, total_points, home_win, home_cover, over
  market coverage : spread 100.0%, total 100.0%, moneyline 100.0%
  target stats    : margin mean +2.05 sd 14.36 | total mean 45.7 sd 13.9 | home win 55.5%
  market baseline : spread MAE 9.98, total MAE 10.52
  families:
    team_ewm        364  (avg missing 13.3%)  e.g. home_off_plays_ewm, home_off_dropbacks_ewm, home_off_rushes_ewm, home_off_attempts_ewm ...
    team_recent     336  (avg missing  6.8%)  e.g. home_off_epa_pp_r3, home_off_epa_pp_r6, home_off_epa_pp_neutral_r3, home_off_epa_pp_neutral_r6 ...
    net_strength     84  (avg missing  0.5%)  e.g. net_epa_pp_r3, net_epa_pp_r6, net_epa_pp_ewm, net_epa_pp_neutral_r3 ...
    matchup         168  (avg missing  0.5%)  e.g. mu_home_epa_pp_r3, mu_away_epa_pp_r3, mu_home_epa_pp_r6, mu_away_epa_pp_r6 ...
    line_matchup     66  (avg missing 16.9%)  e.g. line_home_sack_rate_r3, line_away_sack_rate_r3, line_home_sack_rate_r6, line_away_sack_rate_r6 ...
    qb               28  (avg missing  2.2%)  e.g. home_qb_epa_db_r3, home_qb_cpoe_r3, home_qb_dropbacks_r3, home_qb_epa_db_r6 ...
    experience        5  (avg missing  0.0%)  e.g. home_n_prior_games, away_n_prior_games, home_n_prior_games_season, away_n_prior_games_season ...
    context          33  (avg missing  0.0%)  e.g. week, is_playoff, is_neutral, div_game ...
    market            5  (avg missing  0.0%)  e.g. spread_line, total_line, home_ml_prob_novig, home_implied_pts ...
    market_ratings   10  (avg missing  1.2%)  e.g. mkt_rt_home, mkt_rt_away, mkt_rt_diff, mkt_hfa ...
    lines             7  (avg missing 41.6%)  e.g. spread_open, total_open, ml_h_open, ml_a_open ...
  next slate      : season 2026 week 2 -> 16 games

[PLAYER-LEVEL MATRIX]  data/features/player_features.parquet
  rows            : 89,731 player-games  (81,747 played, 7,984 projection rows)
  players         : 2,301   by position: WR 34,057, RB 21,688, TE 16,933, QB 9,069
  features        : 312
  targets         : passing_yards, passing_tds, rushing_yards, rushing_tds, receiving_yards, receiving_tds  (aux: targets, carries, attempts, receptions)
    passing_yards    mean  23.10  sd  74.11  zero-share 89.3%
    passing_tds      mean   0.14  sd   0.57  zero-share 92.2%
    rushing_yards    mean  10.68  sd  24.56  zero-share 64.0%
    rushing_tds      mean   0.08  sd   0.32  zero-share 93.3%
    receiving_yards  mean  22.92  sd  31.08  zero-share 34.6%
    receiving_tds    mean   0.14  sd   0.40  zero-share 87.4%
  families:
    usage_roll      156  (avg missing 19.7%)  e.g. attempts_r3, attempts_r6, attempts_ewm, completions_r3 ...
    volatility       12  (avg missing  7.8%)  e.g. passing_yards_sd6, passing_tds_sd6, rushing_yards_sd6, rushing_tds_sd6 ...
    experience        3  (avg missing  0.9%)  e.g. days_since_last, games_played_season, career_games
    team_ctx         24  (avg missing  0.6%)  e.g. tm_off_epa_pp_ewm, tm_off_pass_epa_db_ewm, tm_off_rush_epa_ewm, tm_off_plays_ewm ...
    opp_def          81  (avg missing  2.6%)  e.g. opp_def_epa_pp_ewm, opp_def_pass_epa_db_ewm, opp_def_rush_epa_ewm, opp_def_sr_ewm ...
    expected_vol      8  (avg missing 28.5%)  e.g. exp_targets, exp_carries, exp_rec_yards, exp_rush_yards ...
    status            4  (avg missing  1.0%)  e.g. inj_status, inj_practice, inj_listed, depth_rank
    static           10  (avg missing  0.0%)  e.g. pos_QB, pos_RB, pos_WR, pos_TE ...
    game             14  (avg missing  0.0%)  e.g. is_home, team_spread, total_line, team_implied_pts ...

[LEAKAGE GUARDS]
  * every rolling / EWM feature is shifted one game before windowing (game never sees its own stats)
  * opponent adjustment uses the opponent's prior EWM and an expanding league mean
  * season-start EWM state = last season's final EWM shrunk toward last season's league/position mean
  * rolling-origin CV in train_models.py trains only on (season, week) strictly before each validation block

NEXT STEP: review the families above. To approve and train:
    python build_features.py --approve        # writes data/features/APPROVED
    python train_models.py                    # or: python train_models.py --approve-features
==============================================================================
```
