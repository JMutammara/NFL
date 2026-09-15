# NFL Quantitative Prediction Pipeline

Production-style Python pipeline that predicts two families of targets from
nflverse / nflfastR data and prices them against the betting market:

1. **Game outcomes** – spread, moneyline and total. A market-free
   gradient-boosting power model (stage 1) feeds a market-relative *edge
   model* (stage 2) that predicts the residual of the outcome against the
   **opening line** and calibrates cover probabilities on out-of-fold
   predictions. Evaluation is flat-stake ROI against the opening line with
   bootstrap confidence intervals and closing-line value.
2. **Player performance** – weekly passing / rushing / receiving yards and
   touchdowns via a heteroscedastic neural network (mean + volatility head,
   Gaussian for yards, Poisson for TDs) blended with a LightGBM
   mean/variance model.

Every feature is strictly pre-game. Validation is rolling-origin: models
never see a week that has not happened yet relative to the validation block.
A self-contained interactive dashboard is written with every weekly run.

---

## Quick start

```bash
pip install -r requirements.txt

python ingest_data.py                     # Stage 1: download + cache + verify nflverse assets
python build_features.py                  # Stage 2: engineer features (fetches ESPN opening lines on first run)
python build_features.py --approve        #          approve the feature set for training
python train_models.py                    # Stage 3: stage-1 power models, edge models, player models
python predict_week.py --season 2026 --week 2 --refresh   # weekly predictions + dashboard
```

`train_models.py --smoke --targets games` runs the game pipeline with tiny
models in a few minutes to verify the code path. Open
`predictions/<season>_week<ww>_dashboard.html` in a browser after a run.

## Repository layout

```
config/pipeline.yaml            all knobs: seasons, windows, selection thresholds, model params, staking
config/best_params/             Optuna outputs (auto-loaded by train_models.py when present)
nfl_pipeline/
  ingest.py                     nflverse loaders with per-season parquet cache and freshness control
  lines.py                      opening / closing lines from ESPN's per-game odds feed (cached JSON per game)
  weather.py                    kickoff forecasts for upcoming outdoor games (Open-Meteo, no key)
  features/team_efficiency.py   EPA / success rate / CPOE / explosive / pressure / pace per team-game, ATS results
  features/rolling.py           shifted 3g, 6g, season-long EWM (with shrunk carry-over) + rolling std
  features/context.py           rest, travel miles, time-zone shift, body clock, weather (climatology
                                imputation), venue, schedule slot, closing market lines
  features/market.py            market-implied team ratings from prior closing lines (weighted ridge, per week)
  features/game_features.py     opponent adjustment, QB form, net / matchup / OL-DL line features, lines merge
  features/players.py           usage & high-value touches from play-by-play, snaps, injuries, depth charts,
                                defence-vs-position
  features/player_features.py   player rolling + volatility + team/opponent context + expected volume
  validation.py                 rolling-origin CV (season / half / week blocks) + causal assertions
  feature_selection.py          missingness, variance, Spearman collinearity clusters, null-importance filter
  models/game_ensemble.py       stage 1: LGBM + XGB + CatBoost + ridge, early stopping, NNLS blend
  models/edge_model.py          stage 2: residual-vs-line ridge + bagged CatBoost + cover classifier,
                                OOF-calibrated cover probability, ROI / CLV evaluation
  models/player_model.py        PyTorch heteroscedastic net + LightGBM mean/variance fallback, sigma calibration
  models/betting.py             cover / over / moneyline probabilities, EV, fractional Kelly
  dashboard.py                  self-contained HTML dashboard (slate, players, prop pricer, model report)
ingest_data.py                  Stage 1 CLI
build_features.py               Stage 2 CLI (prints the feature summary; --approve gates training)
train_models.py                 Stage 3 CLI
predict_week.py                 weekly predictions, prop pricing (--props sheet.csv), dashboard
optuna_studies/                 hyper-parameter studies to run locally (never inside the training loop)
tests/                          leakage, CV causality, line parsing, market ratings, betting-math tests (pytest)
```

## Data sources

| asset | seasons | used for |
|---|---|---|
| play-by-play (nflfastR) | 2012+ | EPA, success, CPOE, xpass, drives, pace, usage |
| schedules (games.csv) | 2012+ | results, closing lines, rest, stadium, weather, projected QBs |
| ESPN odds feed | 2014-16, 2023+ opens; closes throughout | opening spreads / totals / moneylines, closing-line value |
| Open-Meteo forecasts | live | kickoff temperature and wind for upcoming outdoor games |
| stats_player_week | 2012+ | player targets, shares, WOPR, RACR, PACR |
| weekly rosters / players | 2012+ | projection universe, age, experience, draft capital |
| snap counts | 2013+ | snap share |
| injuries | 2012+ | designation and practice status |
| depth charts (both formats) | 2012+ | depth rank |
| participation (tracking) | 2016-2025 | pressure rate, time to throw, box counts, coverage |
| FTN charting | 2022-2025 | blitz, play-action, motion, drops, contested targets |
| PFR advanced | 2018+ | pressures, hurries, hits, blitzes, missed tackles |

Opening lines are only published for some seasons. Where none exists the
edge models train against the closing line (flagged by `ref_is_open`), and
the headline "beat the open" evaluation is restricted to games with a real
opening number.

## Leakage guards

* Every rolling feature is computed on a series shifted by one game **before**
  windowing. Unit tests assert a game's feature depends only on earlier games.
* Season-start EWM state is last season's final value shrunk toward last
  season's league (or position) mean – informed but never peeking.
* Opponent adjustment subtracts the opponent's *prior* EWM relative to an
  expanding league mean computed over earlier dates only.
* Market-implied ratings for a week are fitted on lines of games played
  before that week's first kickoff.
* Rolling-origin CV trains on `(season, week)` strictly before each block;
  early stopping uses a chronological tail of the training rows only.
* Feature selection runs inside every CV fold on that fold's training rows
  only; the final model re-selects on all played rows.
* Stage-2 edge models consume stage-1 predictions that are out-of-fold, so a
  power-model prediction for a game never saw that game.
* Weather for future games comes from a forecast where available, otherwise
  climatology, and is flagged either way.

## Outputs

`predict_week.py` writes, under `predictions/`:

* `<season>_week<ww>_games.csv` – opening and current lines, stage-1 power
  numbers, edge-model margin / total, calibrated cover, over and win
  probabilities (at the current line and at the open), the EV-best side for
  spread, total and moneyline, edge versus the price's implied probability
  and fractional-Kelly stakes.
* `<season>_week<ww>_players.csv` – `mu`, `sigma`, 10/50/90 quantiles for
  yards and P(≥1), P(≥2) for touchdowns.
* `<season>_week<ww>_props.csv` – when `--props sheet.csv` is supplied
  (`player_name, stat, line[, over_odds, under_odds]`).
* `<season>_week<ww>_dashboard.html` – interactive board: plays above a
  threshold, the full slate with expandable detail, player projections with
  filters, a live prop pricer, and the out-of-fold validation report.

`--bet-line open` prices bets at the opening line instead of the current one.

## Tuning

```bash
python optuna_studies/tune_game_models.py --target margin --member lgbm --n-trials 100
python optuna_studies/tune_player_model.py --target receiving_yards --engine nn --n-trials 40
```

Results land in `config/best_params/` and are picked up automatically by the
next `train_models.py` run.

## Notes on modelling choices

* Stage 1 is deliberately market-free so that its disagreement with the line
  is informative. Stage 2 sees that disagreement, the deviation of the line
  from the market's own implied ratings, against-the-spread form, rest,
  travel, weather, quarterback changes and a handful of efficiency
  differentials, and predicts the residual against the reference line with
  heavy regularisation.
* Cover probabilities are a logistic calibration over (predicted edge in
  points, cover-classifier logit) fitted on out-of-fold rows, not a Normal
  tail. Kelly stakes therefore only appear where validated accuracy supports
  them.
* Win probability comes from the edge-model margin through a calibrated
  Normal, evaluated against the vig-free moneyline.
* The legacy market-aware ensemble (`--targets margin total win`) remains
  available for comparison and as a fallback when no edge models exist.
