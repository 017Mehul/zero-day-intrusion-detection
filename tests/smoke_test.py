from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    from sa_zd_nids.data.preprocessing import DataPreprocessor
    from sa_zd_nids.models.classifier import KnownAttackClassifier
    from sa_zd_nids.models.autoencoder import ZeroDayAutoencoder
    from sa_zd_nids.streaming.engine import StreamingEngine
except ModuleNotFoundError:
    pytest.skip("river not installed", allow_module_level=True)


def make_synthetic(n=200):
    rng = np.random.default_rng(0)
    ts = np.arange(n)
    X = rng.normal(size=(n, 10)).astype(float)
    # labels: mostly BENIGN, with some attacks
    labels = np.array(["BENIGN"] * n)
    labels[rng.choice(n, size=max(1, n // 10), replace=False)] = "ATTACK"
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["timestamp"] = ts
    df["label"] = labels
    return df


def test_minimal_pipeline():
    df = make_synthetic(300)
    cfg = {
        "project": {"random_state": 42},
        "data": {"label_col": "label", "timestamp_col": "timestamp", "scale": "standard", "test_size": 0.2, "val_size": 0.1},
        "features": {"use_mutual_info": False},
        "classifier": {"model_type": "random_forest", "random_forest": {"n_estimators": 10, "max_depth": 4}},
        "anomaly": {"hidden_dims": [16, 8, 4], "epochs": 1, "batch_size": 32, "threshold_percentile": 95},
        "drift": {"adwin_delta": 0.01, "window_size": 100, "min_adapt_samples": 50, "past_replay_fraction": 0.3},
        "streaming": {"batch_size": 50, "max_batches": None},
        "logging": {"output_csv": "logs/test_stream.csv", "events_jsonl": "logs/test_events.jsonl", "metrics_json": "logs/test_metrics.json"},
    }

    pre = DataPreprocessor(cfg)
    prepared = pre.fit_transform(df)

    clf = KnownAttackClassifier(cfg)
    clf_art = clf.fit(prepared.X_train, prepared.y_train, prepared.X_val, prepared.y_val)

    benign_label = cfg["data"].get("benign_label", "BENIGN")
    ae = ZeroDayAutoencoder(cfg)
    train_normal = prepared.X_train[prepared.y_train == benign_label]
    val_normal = prepared.X_val[prepared.y_val == benign_label]
    if len(train_normal) == 0:
        train_normal = prepared.X_train[:10]
    if len(val_normal) == 0:
        val_normal = prepared.X_val[:5]
    ae_art = ae.fit(train_normal, val_normal)

    engine = StreamingEngine(cfg, clf, ae)
    artifacts = engine.run(prepared.X_test, prepared.y_test, prepared.timestamps_test, benign_label=benign_label, known_labels=clf_art.labels)
    print("Smoke test metrics:", artifacts.metrics)


if __name__ == "__main__":
    test_minimal_pipeline()
