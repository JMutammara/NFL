import numpy as np
import pandas as pd

from nfl_pipeline.validation import assert_folds_are_causal, rolling_origin_folds


def _games():
    rows = []
    for s in range(2015, 2021):
        for w in range(1, 19):
            for g in range(8):
                rows.append({"season": s, "week": w, "gameday": pd.Timestamp(f"{s}-09-01") + pd.Timedelta(days=7 * w), "y": np.random.randn()})
    return pd.DataFrame(rows)


def test_folds_are_causal_and_expanding():
    df = _games()
    folds = rolling_origin_folds(df, min_train_seasons=3, fold_unit="season", es_tail_frac=0.1)
    assert [f.name for f in folds] == ["2018", "2019", "2020"]
    assert_folds_are_causal(df, folds)
    sizes = [len(f.train_idx) for f in folds]
    assert sizes == sorted(sizes)
    for f in folds:
        assert set(f.es_idx).isdisjoint(f.fit_idx)
        assert len(f.es_idx) + len(f.fit_idx) == len(f.train_idx)
        assert df.iloc[f.es_idx]["season"].max() >= df.iloc[f.fit_idx]["season"].max()  # ES tail is the most recent slice


def test_week_folds_train_only_on_earlier_weeks():
    df = _games()
    folds = rolling_origin_folds(df, min_train_seasons=5, fold_unit="week", es_tail_frac=0.05)
    assert_folds_are_causal(df, folds)
    f = folds[3]  # 2020 week 4
    assert f.name == "2020W4"
    assert (df.iloc[f.train_idx][["season", "week"]].apply(tuple, axis=1) < (2020, 4)).all()
