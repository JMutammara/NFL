"""NFL quantitative prediction pipeline.

Modules
-------
config          : YAML-backed configuration and project paths
ingest          : nflverse / nflfastR data ingestion with local parquet cache
features.*      : team efficiency, rolling windows, context, line matchups, players
validation      : rolling-origin (time-series) cross-validation
feature_selection: variance / collinearity / null-importance filters
models.*        : game ensemble (XGB+LGBM+CatBoost), player volatility model, betting math
"""
__version__ = "0.1.0"
