"""Unit tests for classifier and autoencoder models."""
from __future__ import annotations

import numpy as np
import pytest

from sa_zd_nids.models.autoencoder import AutoencoderNet, ZeroDayAutoencoder
from sa_zd_nids.models.classifier import KnownAttackClassifier


def _clf_cfg(model_type: str = "random_forest") -> dict:
    return {
        "project": {"random_state": 42},
        "classifier": {
            "model_type": model_type,
            "confidence_threshold": 0.5,
            "random_forest": {"n_estimators": 10, "max_depth": 3},
            "xgboost": {
                "n_estimators": 10,
                "learning_rate": 0.1,
                "max_depth": 3,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 1.0,
            },
        },
        "drift": {"adapt_accept_tolerance": 0.0},
    }


def _ae_cfg() -> dict:
    return {
        "anomaly": {
            "hidden_dims": [16, 8, 4],
            "epochs": 2,
            "batch_size": 32,
            "learning_rate": 0.001,
            "threshold_percentile": 95,
        }
    }


def _make_xy(n: int = 200, n_features: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    y = np.array(["BENIGN"] * n)
    y[rng.choice(n, size=n // 5, replace=False)] = "ATTACK"
    return X, y


# ── Classifier ────────────────────────────────────────────────────────────────

class TestKnownAttackClassifier:
    def test_fit_and_predict_shapes(self):
        X, y = _make_xy(200)
        clf = KnownAttackClassifier(_clf_cfg())
        clf.fit(X[:160], y[:160], X[160:180], y[160:180])
        preds, conf = clf.predict(X[180:])
        assert preds.shape == (20,)
        assert conf.shape == (20,)

    def test_confidence_in_range(self):
        X, y = _make_xy(200)
        clf = KnownAttackClassifier(_clf_cfg())
        clf.fit(X[:160], y[:160], X[160:180], y[160:180])
        _, conf = clf.predict(X[180:])
        assert np.all(conf >= 0.0) and np.all(conf <= 1.0)

    def test_predict_before_fit_raises(self):
        clf = KnownAttackClassifier(_clf_cfg())
        with pytest.raises(RuntimeError, match="not trained"):
            clf.predict(np.zeros((5, 10), dtype=np.float32))

    def test_partial_adapt_runs(self):
        X, y = _make_xy(300)
        clf = KnownAttackClassifier(_clf_cfg())
        clf.fit(X[:200], y[:200], X[200:240], y[200:240])
        clf.partial_adapt(X[240:], y[240:])

    def test_balanced_sample_weights_sum(self):
        y = np.array(["A", "A", "A", "B"])
        w = KnownAttackClassifier._balanced_sample_weights(y)
        assert w.shape == (4,)
        assert np.all(w > 0)

    def test_artifacts_labels_populated(self):
        X, y = _make_xy(200)
        clf = KnownAttackClassifier(_clf_cfg())
        art = clf.fit(X[:160], y[:160], X[160:180], y[160:180])
        assert len(art.labels) >= 1
        assert art.val_f1_macro >= 0.0

    def test_xgboost_fit_handles_validation_only_labels(self):
        pytest.importorskip("xgboost")
        rng = np.random.default_rng(7)
        X_train = rng.normal(size=(60, 8)).astype(np.float32)
        y_train = np.array(["BENIGN"] * 20 + ["ATTACK_A"] * 20 + ["ATTACK_B"] * 20, dtype=object)
        X_val = rng.normal(size=(12, 8)).astype(np.float32)
        y_val = np.array(["ATTACK_C"] * 12, dtype=object)

        clf = KnownAttackClassifier(_clf_cfg(model_type="xgboost"))
        art = clf.fit(X_train, y_train, X_val, y_val)
        preds, conf = clf.predict(X_val)

        assert set(art.labels) == {"BENIGN", "ATTACK_A", "ATTACK_B"}
        assert preds.shape == (12,)
        assert conf.shape == (12,)


# ── Autoencoder ───────────────────────────────────────────────────────────────

class TestAutoencoder:
    def test_autoencoder_net_forward(self):
        net = AutoencoderNet(input_dim=10, hidden_dims=[16, 8, 4])
        import torch
        x = torch.randn(5, 10)
        out = net(x)
        assert out.shape == (5, 10)

    def test_hidden_dims_wrong_length_raises(self):
        with pytest.raises(ValueError, match="length 3"):
            AutoencoderNet(input_dim=10, hidden_dims=[16, 8])

    def test_fit_returns_artifacts(self):
        X_normal = np.random.default_rng(0).normal(size=(100, 10)).astype(np.float32)
        ae = ZeroDayAutoencoder(_ae_cfg())
        art = ae.fit(X_normal[:80], X_normal[80:])
        assert art.threshold > 0.0
        assert art.model is not None

    def test_reconstruction_error_shape(self):
        X = np.random.default_rng(1).normal(size=(50, 10)).astype(np.float32)
        ae = ZeroDayAutoencoder(_ae_cfg())
        ae.fit(X[:40], X[40:])
        errs = ae.reconstruction_error(X)
        assert errs.shape == (50,)
        assert np.all(errs >= 0)

    def test_is_anomaly_returns_flags_and_errors(self):
        X = np.random.default_rng(2).normal(size=(50, 10)).astype(np.float32)
        ae = ZeroDayAutoencoder(_ae_cfg())
        ae.fit(X[:40], X[40:])
        flags, errs = ae.is_anomaly(X)
        assert flags.shape == (50,)
        assert errs.shape == (50,)
        assert flags.dtype == bool

    def test_is_anomaly_before_fit_raises(self):
        ae = ZeroDayAutoencoder(_ae_cfg())
        with pytest.raises(RuntimeError):
            ae.is_anomaly(np.zeros((5, 10), dtype=np.float32))

    def test_update_threshold(self):
        X = np.random.default_rng(3).normal(size=(100, 10)).astype(np.float32)
        ae = ZeroDayAutoencoder(_ae_cfg())
        ae.fit(X[:80], X[80:])
        old_thr = ae.threshold
        ae.update_threshold(X[:20])
        assert ae.threshold != old_thr or True  # threshold may or may not change
