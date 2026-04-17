from __future__ import annotations

"""Manual adaptation entrypoint for periodic offline refresh.

Usage:
  python adapt.py --config config.yaml --recent data/recent_window.csv
"""

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
from sa_zd_nids.utils.io import save_model_atomic, save_metadata_atomic


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt SA-ZD-NIDS models on recent data")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--recent", required=True, help="CSV path with recent labeled traffic")
    args = parser.parse_args()

    cfg = load_config(args.config)
    validate_config(cfg)
    pre: DataPreprocessor = joblib.load("models/preprocessor.joblib")
    clf: KnownAttackClassifier = joblib.load("models/classifier.joblib")
    metadata = json.loads(Path("models/metadata.json").read_text(encoding="utf-8"))

    df_recent = pre.load_raw(args.recent)
    X_recent = pre.transform_for_inference(df_recent, metadata["feature_names"])
    y_recent = df_recent[cfg["data"]["label_col"]].to_numpy()

    # perform adaptation with no validation in this manual flow
    clf.partial_adapt(X_recent, y_recent, X_past=None, y_past=None)
    save_model_atomic("models/classifier.joblib", clf, method="joblib")

    ae = ZeroDayAutoencoder(cfg)
    ae.model = AutoencoderNet(len(metadata["feature_names"]), cfg["anomaly"].get("hidden_dims", [128, 64, 32])).to(ae.device)
    ae.model.load_state_dict(torch.load("models/autoencoder.pt", map_location=ae.device))
    ae.threshold = float(metadata["autoencoder_threshold"])

    benign = metadata["benign_label"]
    normal_mask = y_recent == benign
    if np.any(normal_mask):
        new_thr = ae.update_threshold(X_recent[normal_mask])
        metadata["autoencoder_threshold"] = float(new_thr)
        # bump metadata version and timestamp
        metadata["version"] = int(metadata.get("version", 0)) + 1
        from datetime import datetime

        metadata["last_updated"] = datetime.utcnow().isoformat()
        metadata["last_adapted_at"] = datetime.utcnow().isoformat()
        save_metadata_atomic("models/metadata.json", metadata)

    print("Adaptation complete")


if __name__ == "__main__":
    main()
