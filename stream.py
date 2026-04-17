from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sa_zd_nids.config import load_config, validate_config
from sa_zd_nids.data.preprocessing import DataPreprocessor
from sa_zd_nids.models.autoencoder import AutoencoderNet, ZeroDayAutoencoder
from sa_zd_nids.models.classifier import KnownAttackClassifier
from sa_zd_nids.streaming.engine import StreamingEngine
from sa_zd_nids.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SA-ZD-NIDS streaming simulation")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", default=None, help="Optional override for stream input CSV")
    args = parser.parse_args()

    cfg = load_config(args.config)
    validate_config(cfg)
    input_path = args.input or cfg["data"]["input_path"]

    pre: DataPreprocessor = joblib.load("models/preprocessor.joblib")
    clf: KnownAttackClassifier = joblib.load("models/classifier.joblib")
    metadata = json.loads(Path("models/metadata.json").read_text(encoding="utf-8"))

    ae = ZeroDayAutoencoder(cfg)
    input_dim = len(metadata["feature_names"])
    ae.model = AutoencoderNet(input_dim, cfg["anomaly"].get("hidden_dims", [128, 64, 32])).to(ae.device)
    ae.model.load_state_dict(torch.load("models/autoencoder.pt", map_location=ae.device))
    ae.threshold = float(metadata["autoencoder_threshold"])

    df = pre.load_raw(input_path)
    X = pre.transform_for_inference(df, metadata["feature_names"])
    y = df[cfg["data"]["label_col"]].to_numpy()
    ts_col = cfg["data"]["timestamp_col"]
    timestamps = df[ts_col] if ts_col in df.columns else np.arange(len(df))

    split = int(len(X) * (1 - cfg["data"].get("test_size", 0.2)))
    X_stream, y_stream = X[split:], y[split:]
    ts_stream = timestamps.iloc[split:] if hasattr(timestamps, "iloc") else timestamps[split:]

    recent_n = min(cfg["drift"].get("window_size", 1000), split)
    recent_cache = (X[split - recent_n : split], y[split - recent_n : split]) if recent_n > 0 else None

    engine = StreamingEngine(cfg, clf, ae)
    artifacts = engine.run(
        X=X_stream,
        y=y_stream,
        timestamps=ts_stream,
        benign_label=metadata["benign_label"],
        known_labels=metadata.get("known_labels"),
        baseline_feature_mean=np.asarray(metadata.get("train_feature_mean")) if metadata.get("train_feature_mean") is not None else None,
        baseline_feature_std=np.asarray(metadata.get("train_feature_std")) if metadata.get("train_feature_std") is not None else None,
        recent_train_cache=recent_cache,
    )

    write_json(cfg["logging"]["metrics_json"], artifacts.metrics)
    print("Streaming complete")
    print(json.dumps(artifacts.metrics, indent=2))


if __name__ == "__main__":
    main()
