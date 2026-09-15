"""Rolling-origin (time-series) cross-validation.

Each fold trains on every row whose (season, week) is strictly earlier than
the validation block and validates on the block. Blocks are whole seasons by
default (``fold_unit='season'``), season halves (``'half'``), or single weeks
(``'week'``). A chronological tail of the training rows is reserved as the
early-stopping set so no validation row ever influences model fitting.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Fold:
    name: str
    train_idx: np.ndarray
    val_idx: np.ndarray
    es_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    fit_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))


def _order_key(df: pd.DataFrame, season_col: str, week_col: str) -> np.ndarray:
    return df[season_col].to_numpy(dtype=int) * 100 + df[week_col].to_numpy(dtype=int)


def rolling_origin_folds(
    df: pd.DataFrame,
    season_col: str = "season",
    week_col: str = "week",
    date_col: str | None = "gameday",
    min_train_seasons: int = 5,
    fold_unit: str = "season",
    es_tail_frac: float = 0.12,
    val_seasons: list[int] | None = None,
    max_folds: int | None = None,
) -> list[Fold]:
    """Return chronological folds over ``df`` (index positions, not labels)."""
    key = _order_key(df, season_col, week_col)
    seasons = np.sort(df[season_col].unique())
    if len(seasons) <= min_train_seasons:
        raise ValueError("not enough seasons for the requested min_train_seasons")
    cand = seasons[min_train_seasons:]
    if val_seasons:
        cand = [s for s in cand if s in set(val_seasons)]
    order = np.argsort(df[date_col].to_numpy() if date_col else key, kind="mergesort")

    blocks: list[tuple[str, np.ndarray]] = []
    for s in cand:
        in_s = df[season_col].to_numpy() == s
        if fold_unit == "season":
            blocks.append((f"{s}", np.flatnonzero(in_s)))
        elif fold_unit == "half":
            wk = df[week_col].to_numpy()
            blocks.append((f"{s}H1", np.flatnonzero(in_s & (wk <= 9))))
            blocks.append((f"{s}H2", np.flatnonzero(in_s & (wk > 9))))
        elif fold_unit == "week":
            for w in np.sort(df.loc[in_s, week_col].unique()):
                blocks.append((f"{s}W{int(w)}", np.flatnonzero(in_s & (df[week_col].to_numpy() == w))))
        else:
            raise ValueError(fold_unit)
    folds: list[Fold] = []
    for name, val_idx in blocks:
        if len(val_idx) == 0:
            continue
        cutoff = key[val_idx].min()
        train_idx = np.flatnonzero(key < cutoff)
        if len(train_idx) == 0:
            continue
        # chronological tail of train for early stopping
        tr_sorted = train_idx[np.argsort(key[train_idx], kind="mergesort")]
        n_es = int(round(len(tr_sorted) * es_tail_frac))
        es_idx = tr_sorted[-n_es:] if n_es > 0 else np.array([], dtype=int)
        fit_idx = tr_sorted[:-n_es] if n_es > 0 else tr_sorted
        folds.append(Fold(name, train_idx, val_idx, es_idx, fit_idx))
    if max_folds:
        folds = folds[-max_folds:]
    return folds


def assert_folds_are_causal(df: pd.DataFrame, folds: list[Fold], season_col="season", week_col="week") -> None:
    key = _order_key(df, season_col, week_col)
    for f in folds:
        assert key[f.train_idx].max() < key[f.val_idx].min(), f"fold {f.name}: training data overlaps validation period"
        if len(f.es_idx):
            assert key[f.es_idx].max() < key[f.val_idx].min()
            assert not set(f.es_idx) & set(f.fit_idx)
