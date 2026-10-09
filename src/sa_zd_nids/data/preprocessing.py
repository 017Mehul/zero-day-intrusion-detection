"""Leakage-safe preprocessing for tabular network-flow data."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass
class PreparedData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    timestamps_train: pd.Series
    timestamps_val: pd.Series
    timestamps_test: pd.Series
    feature_names: list[str]


class DataPreprocessor:
    """Fit preprocessing only on training data and reuse it for inference."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.data_cfg = config.get("data", {})
        self.feature_cfg = config.get("features", {})
        self.imputer: SimpleImputer | None = None
        self.scaler: StandardScaler | MinMaxScaler | None = None
        self.feature_names_: list[str] = []
        self.selected_feature_names_: list[str] = []

    def load_raw(self, path: str | Path) -> pd.DataFrame:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {p}")
        return pd.read_csv(p, low_memory=False)

    def _feature_frame(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        label_col = self.data_cfg.get("label_col", "label")
        timestamp_col = self.data_cfg.get("timestamp_col", "timestamp")
        if label_col not in df.columns:
            raise ValueError(f"Missing label column: {label_col}")
        y = df[label_col].astype(str).to_numpy()
        timestamps = df[timestamp_col] if timestamp_col in df.columns else pd.Series(np.arange(len(df)))
        drop = set(self.data_cfg.get("drop_columns", [])) | {label_col, timestamp_col}
        candidates = [c for c in df.columns if c not in drop]
        if not candidates:
            raise ValueError("No feature columns remain after dropping label/timestamp columns")
        numeric = df[candidates].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        usable = [c for c in numeric.columns if not numeric[c].isna().all()]
        if not usable:
            raise ValueError("No numeric feature columns found in dataset")
        return numeric[usable], pd.Series(y, index=df.index), timestamps.reset_index(drop=True)

    def fit_transform(self, df: pd.DataFrame) -> PreparedData:
        X_df, y, timestamps = self._feature_frame(df.reset_index(drop=True))
        n = len(X_df)
        test_n = int(n * float(self.data_cfg.get("test_size", 0.2)))
        val_n = int(n * float(self.data_cfg.get("val_size", 0.1)))
        train_n = n - test_n - val_n
        if train_n <= 0:
            raise ValueError("empty training set: test_size + val_size leaves no training samples")

        X_train_df = X_df.iloc[:train_n].copy()
        X_val_df = X_df.iloc[train_n:train_n + val_n].copy()
        X_test_df = X_df.iloc[train_n + val_n:].copy()
        y_train = y.iloc[:train_n].to_numpy()
        y_val = y.iloc[train_n:train_n + val_n].to_numpy()
        y_test = y.iloc[train_n + val_n:].to_numpy()

        self.feature_names_ = list(X_train_df.columns)
        strategy = self.feature_cfg.get("impute_strategy", "median")
        self.imputer = SimpleImputer(strategy=strategy)
        Xt = self.imputer.fit_transform(X_train_df)
        Xv = self.imputer.transform(X_val_df)
        Xte = self.imputer.transform(X_test_df)

        if self.feature_cfg.get("use_mutual_info", False):
            k = min(int(self.feature_cfg.get("mutual_info_k", 40)), Xt.shape[1])
            if k < Xt.shape[1]:
                scores = mutual_info_classif(Xt, y_train, random_state=self.config.get("project", {}).get("random_state", 42))
                keep = np.argsort(scores)[::-1][:k]
                keep.sort()
                self.feature_names_ = [self.feature_names_[i] for i in keep]
                Xt, Xv, Xte = Xt[:, keep], Xv[:, keep], Xte[:, keep]

        self.selected_feature_names_ = list(self.feature_names_)
        scale = self.data_cfg.get("scale", "standard")
        self.scaler = None
        if scale == "standard":
            self.scaler = StandardScaler()
        elif scale == "minmax":
            self.scaler = MinMaxScaler()
        elif scale != "none":
            raise ValueError(f"Unsupported scale: {scale}")
        if self.scaler is not None:
            Xt = self.scaler.fit_transform(Xt)
            Xv = self.scaler.transform(Xv)
            Xte = self.scaler.transform(Xte)

        return PreparedData(
            Xt.astype(np.float32), Xv.astype(np.float32), Xte.astype(np.float32),
            y_train, y_val, y_test,
            timestamps.iloc[:train_n].reset_index(drop=True),
            timestamps.iloc[train_n:train_n + val_n].reset_index(drop=True),
            timestamps.iloc[train_n + val_n:].reset_index(drop=True),
            list(self.selected_feature_names_),
        )

    def transform_for_inference(self, df: pd.DataFrame, feature_names: list[str] | None = None) -> np.ndarray:
        if self.imputer is None:
            raise RuntimeError("Preprocessor has not been fitted")
        names = feature_names or self.selected_feature_names_ or self.feature_names_
        numeric = df.copy()
        drop = set(self.data_cfg.get("drop_columns", [])) | {
            self.data_cfg.get("label_col", "label"), self.data_cfg.get("timestamp_col", "timestamp")
        }
        numeric = numeric.drop(columns=[c for c in drop if c in numeric.columns], errors="ignore")
        numeric = numeric.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        # Restore the exact training feature order; missing columns become NaN and are imputed.
        numeric = numeric.reindex(columns=names)
        X = self.imputer.transform(numeric)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return np.asarray(X, dtype=np.float32)
