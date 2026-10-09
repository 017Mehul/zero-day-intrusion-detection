"""Cross-dataset validation against a trained SA-ZD-NIDS artifact set."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sa_zd_nids.config import load_config
from sa_zd_nids.evaluation.metrics import compute_classification_metrics, compute_zero_day_metrics
from sa_zd_nids.models.autoencoder import AutoencoderNet, ZeroDayAutoencoder
from sa_zd_nids.utils.io import write_json

def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-dataset validation")
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="logs/external_validation.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    pre = joblib.load("models/preprocessor.joblib")
    clf = joblib.load("models/classifier.joblib")
    metadata = json.loads(Path("models/metadata.json").read_text(encoding="utf-8"))
    df = pre.load_raw(args.data)
    label_col = cfg["data"].get("label_col", "label")
    if label_col not in df.columns:
        raise ValueError(f"External dataset must contain label column '{label_col}'")
    required = list(getattr(pre, "raw_feature_names_", []))
    missing = [name for name in required if name not in df.columns]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"External dataset is missing {len(missing)} trained feature columns: {preview}")
    X = pre.transform_for_inference(df, metadata["feature_names"])
    y = df[label_col].astype(str).to_numpy()
    pred, conf = clf.predict(X)
    pred = pred.astype(object)
    flags = conf < float(cfg["classifier"].get("confidence_threshold", 0.7))
    if np.any(flags):
        import torch
        ae = ZeroDayAutoencoder(cfg)
        ae.model = AutoencoderNet(len(metadata["feature_names"]), cfg["anomaly"].get("hidden_dims", [128, 64, 32])).to(ae.device)
        ae.model.load_state_dict(torch.load("models/autoencoder.pt", map_location=ae.device))
        ae.model.eval()
        ae.threshold = float(metadata["autoencoder_threshold"])
        anomaly, _ = ae.is_anomaly(X[flags])
        for idx, is_anomaly in zip(np.where(flags)[0], anomaly):
            pred[idx] = "ZERO_DAY" if is_anomaly else metadata["benign_label"]
    metrics = compute_classification_metrics(y, pred)
    metrics.update(compute_zero_day_metrics(
        y_true=y, y_pred=pred,
        known_labels=set(metadata.get("known_labels", [])),
        benign_label=metadata["benign_label"],
    ))
    metrics.update({
        "dataset": str(args.data),
        "samples": int(len(y)),
        "mean_confidence": float(np.mean(conf)),
        "low_confidence_rate": float(np.mean(flags)),
        "feature_count": int(X.shape[1]),
    })
    write_json(args.output, metrics)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
