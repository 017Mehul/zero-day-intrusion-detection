from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import torch
from datetime import datetime

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sa_zd_nids.config import load_config, validate_config
from sa_zd_nids.data.preprocessing import DataPreprocessor
from sa_zd_nids.models.autoencoder import ZeroDayAutoencoder
from sa_zd_nids.models.classifier import KnownAttackClassifier
from sa_zd_nids.utils.io import write_json, save_model_atomic, save_metadata_atomic


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SA-ZD-NIDS models")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    validate_config(cfg)
    pre = DataPreprocessor(cfg)
    df = pre.load_raw(cfg["data"]["input_path"])
    prepared = pre.fit_transform(df)

    clf = KnownAttackClassifier(cfg)
    clf_art = clf.fit(prepared.X_train, prepared.y_train, prepared.X_val, prepared.y_val)

    benign_label = cfg["data"]["benign_label"]
    train_normal = prepared.X_train[prepared.y_train == benign_label]
    val_normal = prepared.X_val[prepared.y_val == benign_label]

    if len(train_normal) == 0 or len(val_normal) == 0:
        raise ValueError(
            f"No normal samples found for benign label '{benign_label}'. "
            "Check config.data.benign_label and dataset labels."
        )

    ae = ZeroDayAutoencoder(cfg)
    ae_art = ae.fit(train_normal, val_normal)

    Path("models").mkdir(parents=True, exist_ok=True)
    # atomic saves
    save_model_atomic("models/classifier.joblib", clf, method="joblib")
    # save autoencoder state dict via torch
    save_model_atomic("models/autoencoder.pt", ae.model.state_dict(), method="torch")

    metadata = {
        "feature_names": prepared.feature_names,
        "known_labels": clf_art.labels,
        "autoencoder_threshold": ae_art.threshold,
        "classifier_val_f1_macro": clf_art.val_f1_macro,
        "classifier_report_val": clf_art.report,
        "benign_label": benign_label,
        "classifier_training_device": getattr(clf, "training_device", "cpu"),
        "autoencoder_training_device": str(ae.device),
        "train_feature_mean": prepared.X_train.mean(axis=0).astype(float).tolist(),
        "train_feature_std": prepared.X_train.std(axis=0).astype(float).tolist(),
        "version": 1,
        "last_updated": datetime.utcnow().isoformat(),
    }
    save_metadata_atomic("models/metadata.json", metadata)
    save_model_atomic("models/preprocessor.joblib", pre, method="joblib")

    print("Training complete")
    print(f"Classifier macro-F1 (val): {clf_art.val_f1_macro:.4f}")
    print(f"Autoencoder threshold: {ae_art.threshold:.6f}")
    print(f"Classifier device: {metadata['classifier_training_device']}")
    print(f"Autoencoder device: {metadata['autoencoder_training_device']}")

    # Optional: export SHAP feature importance if shap is installed
    try:
        from sa_zd_nids.utils.explain import compute_shap_importance
        shap_k = min(20, len(prepared.feature_names))
        importance = compute_shap_importance(clf.model, prepared.X_val[:200], top_k=shap_k)
        shap_out = {prepared.feature_names[i]: score for i, score in importance}
        write_json("models/shap_importance.json", shap_out)
        print("SHAP feature importance saved to models/shap_importance.json")
    except Exception:
        pass  # shap not installed or model not supported — skip silently


if __name__ == "__main__":
    main()
