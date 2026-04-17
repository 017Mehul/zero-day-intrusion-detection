"""Unit tests for the streaming engine."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.requires_river

try:
    from sa_zd_nids.models.autoencoder import ZeroDayAutoencoder
    from sa_zd_nids.models.classifier import KnownAttackClassifier
    from sa_zd_nids.streaming.engine import StreamArtifacts, StreamingEngine
except ModuleNotFoundError:
    pytest.skip("river not installed", allow_module_level=True)


def _make_data(n: int = 300, n_features: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    y = np.array(["BENIGN"] * n)
    y[rng.choice(n, size=n // 5, replace=False)] = "ATTACK"
    ts = pd.Series(np.arange(n))
    return X, y, ts


def _build_engine(cfg: dict, X_train, y_train, X_val, y_val):
    clf = KnownAttackClassifier(cfg)
    clf.fit(X_train, y_train, X_val, y_val)
    ae = ZeroDayAutoencoder(cfg)
    normal_mask = y_train == "BENIGN"
    X_normal = X_train[normal_mask] if normal_mask.any() else X_train[:10]
    val_normal_mask = y_val == "BENIGN"
    X_val_normal = X_val[val_normal_mask] if val_normal_mask.any() else X_val[:5]
    ae.fit(X_normal, X_val_normal)
    return StreamingEngine(cfg, clf, ae)


def _cfg(tmp_path) -> dict:
    return {
        "project": {"random_state": 42},
        "data": {"benign_label": "BENIGN"},
        "classifier": {
            "model_type": "random_forest",
            "confidence_threshold": 0.5,
            "random_forest": {"n_estimators": 5, "max_depth": 3},
        },
        "anomaly": {"hidden_dims": [16, 8, 4], "epochs": 1, "batch_size": 32, "threshold_percentile": 95},
        "drift": {
            "adwin_delta": 0.002,
            "window_size": 100,
            "min_adapt_samples": 50,
            "past_replay_fraction": 0.3,
            "confidence_low_threshold": 0.5,
            "accuracy_weight": 0.6,
            "confidence_weight": 0.3,
            "feature_shift_weight": 0.1,
            "adapt_validation_fraction": 0.2,
            "adapt_accept_tolerance": 0.0,
        },
        "streaming": {"batch_size": 50, "max_batches": None},
        "logging": {
            "output_csv": str(tmp_path / "preds.csv"),
            "events_jsonl": str(tmp_path / "events.jsonl"),
            "metrics_json": str(tmp_path / "metrics.json"),
        },
    }


def test_streaming_returns_artifacts(tmp_path):
    X, y, ts = _make_data()
    cfg = _cfg(tmp_path)
    engine = _build_engine(cfg, X[:200], y[:200], X[200:240], y[200:240])
    arts = engine.run(X[240:], y[240:], ts[240:], benign_label="BENIGN")
    assert isinstance(arts, StreamArtifacts)
    assert len(arts.records) > 0


def test_streaming_metrics_keys(tmp_path):
    X, y, ts = _make_data()
    cfg = _cfg(tmp_path)
    engine = _build_engine(cfg, X[:200], y[:200], X[200:240], y[200:240])
    arts = engine.run(X[240:], y[240:], ts[240:], benign_label="BENIGN")
    for key in ("accuracy", "f1_macro", "samples", "num_drifts"):
        assert key in arts.metrics


def test_streaming_output_csv_created(tmp_path):
    X, y, ts = _make_data()
    cfg = _cfg(tmp_path)
    engine = _build_engine(cfg, X[:200], y[:200], X[200:240], y[200:240])
    engine.run(X[240:], y[240:], ts[240:], benign_label="BENIGN")
    assert (tmp_path / "preds.csv").exists()


def test_streaming_events_jsonl_created(tmp_path):
    X, y, ts = _make_data()
    cfg = _cfg(tmp_path)
    engine = _build_engine(cfg, X[:200], y[:200], X[200:240], y[200:240])
    engine.run(X[240:], y[240:], ts[240:], benign_label="BENIGN")
    assert (tmp_path / "events.jsonl").exists()


def test_streaming_max_batches(tmp_path):
    X, y, ts = _make_data(n=500)
    cfg = _cfg(tmp_path)
    cfg["streaming"]["max_batches"] = 2
    engine = _build_engine(cfg, X[:300], y[:300], X[300:360], y[300:360])
    arts = engine.run(X[360:], y[360:], ts[360:], benign_label="BENIGN")
    # with batch_size=50 and max_batches=2, at most 100 samples processed
    assert arts.metrics["samples"] <= 100


def test_streaming_with_known_labels(tmp_path):
    X, y, ts = _make_data()
    cfg = _cfg(tmp_path)
    engine = _build_engine(cfg, X[:200], y[:200], X[200:240], y[200:240])
    arts = engine.run(
        X[240:], y[240:], ts[240:],
        benign_label="BENIGN",
        known_labels=["BENIGN", "ATTACK"],
    )
    assert "zero_day_detection_rate" in arts.metrics
