# NFL Quantitative Prediction Pipeline

Production-style Python pipeline that predicts two families of targets from
nflverse / nflfastR data:

1. **Game outcomes** – spread (home margin), moneyline (home win probability)
   and total points, via a gradient-boosting ensemble (LightGBM + XGBoost +
   CatBoost + a ridge stabiliser) blended with non-negative least squares on
   out-of-fold predictions.
2. **Player performance** – weekly passing / rushing / receiving yards and
   touchdowns via a heteroscedastic neural network (mean + volatility head,
   Gaussian NLL for yards, Poisson for TDs) blended with a LightGBM
   mean/variance model.

Every feature is strictly pre-game. Validation is rolling-origin: models
never see a week that has not happened yet relative to the validation block.

---

## Quick start

```bash
pip install -r requirements.txt

python ingest_data.py                     # Stage 1: download + cache + verify nflverse assets
python build_features.py                  # Stage 2: engineer features, print summary for approval
python build_features.py --approve        #          (or train_models.py --approve-features)
python train_models.py                    # Stage 3: rolling-origin CV + final fit (game + player)
python predict_week.py --season 2026 --week 2 --refresh   # weekly predictions
```

`train_models.py --smoke` runs the whole modelling stage with tiny models and
two folds in a few minutes to verify the code path before a full run.

## Repository layout

```
config/pipeline.yaml            all knobs: seasons, windows, selection thresholds, model params, staking
config/best_params/             Optuna outputs (auto-loaded by train_models.py when present)
nfl_pipeline/
  ingest.py                     nflverse loaders with per-season parquet cache and freshness control
  features/team_efficiency.py   EPA / success rate / CPOE / explosive / pressure / pace per team-game
  features/rolling.py           shifted 3g, 6g, season-long EWM (with shrunk carry-over) + rolling std
  features/context.py           rest, travel miles, time-zone shift, body clock, weather (with climatology
                                imputation), venue, schedule slot, market lines
  features/game_features.py     opponent adjustment, QB form, net / matchup / OL-DL line features
  features/players.py           usage & high-value touches from play-by-play, snaps, injuries, depth charts,
                                defence-vs-position
  features/player_features.py   player rolling + volatility + team/opponent context + expected volume
  validation.py                 rolling-origin CV (season / half / week blocks) + causal assertions
  feature_selection.py          missingness, variance, Spearman collinearity clusters, null-importance filter
  models/game_ensemble.py       LGBM + XGB + CatBoost + ridge, early stopping on a chronological tail, NNLS blend
  models/player_model.py        PyTorch heteroscedastic net + LightGBM mean/variance fallback, sigma calibration
  models/betting.py             cover / over / moneyline probabilities, EV, fractional Kelly
ingest_data.py                  Stage 1 CLI
build_features.py               Stage 2 CLI (prints the feature summary; --approve gates training)
train_models.py                 Stage 3 CLI
predict_week.py                 weekly predictions + optional prop pricing (--props sheet.csv)
optuna_studies/                 hyper-parameter studies to run locally (never inside the training loop)
tests/                          leakage, CV causality and betting-math unit tests (pytest)
```

## Data sources

All assets are nflverse release artifacts (the same files `nfl_data_py`
reads). Play-by-play is read with column push-down straight from the parquet
releases; set `data.pbp_source: nfl_data_py` in the config to route it
through the package instead. Coverage used by default:

| asset | seasons | used for |
|---|---|---|
| play-by-play (nflfastR) | 2012+ | EPA, success, CPOE, xpass, drives, pace, usage |
| schedules (games.csv) | 2012+ | results, lines, rest, stadium, weather, projected QBs |
| stats_player_week | 2012+ | player targets, shares, WOPR, RACR, PACR |
| weekly rosters / players | 2012+ | projection universe, age, experience, draft capital |
| snap counts | 2013+ | snap share |
| injuries | 2012+ | designation and practice status |
| depth charts (both formats) | 2012+ | depth rank |
| participation (tracking) | 2016-2025 | pressure rate, time to throw, box counts, coverage |
| FTN charting | 2022-2025 | blitz, play-action, motion, drops, contested targets |
| PFR advanced | 2018+ | pressures, hurries, hits, blitzes, missed tackles |

## Leakage guards

* Every rolling feature is computed on a series shifted by one game **before**
  windowing. Unit tests assert a game's feature depends only on earlier games.
* Season-start EWM state is last season's final value shrunk toward last
  season's league (or position) mean – informed but never peeking.
* Opponent adjustment subtracts the opponent's *prior* EWM relative to an
  expanding league mean computed over earlier dates only.
* Rolling-origin CV trains on `(season, week)` strictly before each block;
  early stopping uses a chronological tail of the training rows only.
* Feature selection runs inside every CV fold on that fold's training rows
  only; the final model re-selects on all played rows. Per-fold feature lists
  are stored in each `<target>_metrics.json`.
* Weather for future games is imputed from climatology and flagged.

## Outputs

`predict_week.py` writes `predictions/<season>_week<ww>_games.csv` with model
margin / total / win probability, cover and over probabilities, the EV-best
side for spread, total and moneyline, edge in percentage points versus the
market's implied probability, and fractional-Kelly stakes. The players file
carries `mu`, `sigma`, 10/50/90 quantiles for yards and P(≥1), P(≥2) for TDs.
Supply a CSV (`player_name, stat, line[, over_odds, under_odds]`) via
`--props` to price a prop sheet.

## Tuning

```bash
python optuna_studies/tune_game_models.py --target margin --member lgbm --n-trials 100
python optuna_studies/tune_player_model.py --target receiving_yards --engine nn --n-trials 40
```

Results land in `config/best_params/` and are picked up automatically by the
next `train_models.py` run.

## Notes on modelling choices

* Closing lines are included as features by default (`models.game.use_market_features`).
  The model's value is in disagreement with the market, reported as
  `spread_edge_pts` / `total_edge_pts`; the evaluation prints accuracy
  against the line bucketed by edge size. `--no-market` trains a pure
  power-rating variant.
* Margin and total residual sigma come from out-of-fold residuals; player
  sigma is learned per row and rescaled so standardised OOF residuals have
  unit variance.
* Win probability is a logistic stack of the calibrated classifier and the
  probability implied by the margin model, with the two weights fitted on
  out-of-fold predictions (`win_stack` in `win_metrics.json`). Train `margin`
  before `win`; without a margin OOF file the stack falls back to a 50/50 blend.
