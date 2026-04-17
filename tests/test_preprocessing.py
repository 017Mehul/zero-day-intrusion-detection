"""Unit tests for data preprocessing module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sa_zd_nids.data.preprocessing import DataPreprocessor, PreparedData


def _make_df(n: int = 200, n_features: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features)).astype(float)
    labels = np.array(["BENIGN"] * n)
    labels[rng.choice(n, size=max(1, n // 5), replace=False)] = "ATTACK"
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["timestamp"] = np.arange(n)
    df["label"] = labels
    return df


def _base_cfg(scale: str = "standard", use_mi: bool = False) -> dict:
    return {
        "project": {"random_state": 42},
        "data": {
            "label_col": "label",
            "timestamp_col": "timestamp",
            "scale": scale,
            "test_size": 0.2,
            "val_size": 0.1,
            "drop_columns": [],
            "benign_label": "BENIGN",
        },
        "features": {"use_mutual_info": use_mi, "mutual_info_k": 5, "impute_strategy": "median"},
    }


def test_fit_transform_returns_prepared_data():
    df = _make_df()
    pre = DataPreprocessor(_base_cfg())
    result = pre.fit_transform(df)
    assert isinstance(result, PreparedData)


def test_split_sizes_are_correct():
    n = 200
    df = _make_df(n)
    pre = DataPreprocessor(_base_cfg())
    result = pre.fit_transform(df)
    n_test = int(n * 0.2)
    n_val = int(n * 0.1)
    n_train = n - n_test - n_val
    assert len(result.X_train) == n_train
    assert len(result.X_val) == n_val
    assert len(result.X_test) == n_test


def test_feature_names_consistent():
    df = _make_df()
    pre = DataPreprocessor(_base_cfg())
    result = pre.fit_transform(df)
    assert len(result.feature_names) == result.X_train.shape[1]


def test_minmax_scale():
    df = _make_df()
    pre = DataPreprocessor(_base_cfg(scale="minmax"))
    result = pre.fit_transform(df)
    assert result.X_train.max() <= 1.01
    assert result.X_train.min() >= -0.01


def test_no_scale():
    df = _make_df()
    pre = DataPreprocessor(_base_cfg(scale="none"))
    result = pre.fit_transform(df)
    assert result.X_train is not None


def test_mutual_info_feature_selection():
    df = _make_df(n=300, n_features=15)
    cfg = _base_cfg(use_mi=True)
    cfg["features"]["mutual_info_k"] = 5
    pre = DataPreprocessor(cfg)
    result = pre.fit_transform(df)
    assert result.X_train.shape[1] == 5


def test_transform_for_inference():
    df = _make_df(n=300)
    pre = DataPreprocessor(_base_cfg())
    result = pre.fit_transform(df)
    df_infer = _make_df(n=50)
    X_inf = pre.transform_for_inference(df_infer, result.feature_names)
    assert X_inf.shape[1] == len(result.feature_names)


def test_missing_label_column_raises():
    df = _make_df()
    df = df.drop(columns=["label"])
    pre = DataPreprocessor(_base_cfg())
    with pytest.raises(ValueError, match="Missing label column"):
        pre.fit_transform(df)


def test_empty_train_raises_on_tiny_data():
    # test_size + val_size >= 1.0 means n_train <= 0
    df = _make_df(n=100)
    cfg = _base_cfg()
    cfg["data"]["test_size"] = 0.6
    cfg["data"]["val_size"] = 0.5
    pre = DataPreprocessor(cfg)
    with pytest.raises(ValueError, match="empty training set"):
        pre.fit_transform(df)


def test_drop_columns():
    df = _make_df()
    df["extra"] = 99.0
    cfg = _base_cfg()
    cfg["data"]["drop_columns"] = ["extra"]
    pre = DataPreprocessor(cfg)
    result = pre.fit_transform(df)
    assert "extra" not in result.feature_names


def test_nan_imputation():
    df = _make_df(n=300)
    df.loc[0:10, "f0"] = np.nan
    pre = DataPreprocessor(_base_cfg())
    result = pre.fit_transform(df)
    assert not np.isnan(result.X_train).any()
