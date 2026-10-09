"""Model loading, inference, drift detection and safe serving-time adaptation."""
from __future__ import annotations

import json
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from sa_zd_nids.drift.detector import DriftDetector
from sa_zd_nids.models.autoencoder import AutoencoderNet, ZeroDayAutoencoder
from sa_zd_nids.models.classifier import KnownAttackClassifier
from sa_zd_nids.config import load_config
from sa_zd_nids.utils.io import save_metadata_atomic, save_model_atomic


class ModelManager:
    """Own trained artifacts and conservative serving-time adaptation."""

    def __init__(self, config_path: str | None = None):
        root = Path(__file__).resolve().parents[2]
        self.root = Path(os.getenv("SA_ZD_NIDS_ROOT", root))
        config = config_path or os.getenv("SA_ZD_NIDS_CONFIG", str(self.root / "config.yaml"))
        self.config = load_config(config)
        self.model_dir = Path(os.getenv("SA_ZD_NIDS_MODEL_DIR", str(self.root / "models")))
        self.start_time = time.time()

        self.metadata = self._load_metadata()
        self.expected_features = len(self.metadata.get("feature_names", []))
        if self.expected_features <= 0:
            raise RuntimeError("Model metadata contains no feature_names")

        self.classifier = self._load_classifier()
        self.autoencoder = self._load_autoencoder()
        self.drift_detector = DriftDetector(delta=self.config["drift"].get("adwin_delta", 0.002))

        self.prediction_stats = {
            "total_predictions": 0,
            "zero_day_detections": 0,
            "total_confidence": 0.0,
            "total_latency_ms": 0.0,
        }
        self.recent_predictions = deque(maxlen=1000)
        self.recent_confidences = deque(maxlen=1000)
        self.recent_samples = deque(maxlen=int(self.config["drift"].get("window_size", 1000)))
        self.last_drift_time: str | None = None
        self.last_adaptation: dict[str, Any] | None = None

    def _load_metadata(self) -> dict[str, Any]:
        path = self.model_dir / "metadata.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Failed to load model metadata from {path}: {exc}") from exc

    def _load_classifier(self) -> KnownAttackClassifier:
        path = self.model_dir / "classifier.joblib"
        try:
            return joblib.load(path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load classifier from {path}: {exc}") from exc

    def _load_autoencoder(self) -> ZeroDayAutoencoder:
        try:
            ae = ZeroDayAutoencoder(self.config)
            ae.model = AutoencoderNet(
                self.expected_features,
                self.config["anomaly"].get("hidden_dims", [128, 64, 32]),
            ).to(ae.device)
            state_dict = torch.load(self.model_dir / "autoencoder.pt", map_location=ae.device)
            ae.model.load_state_dict(state_dict)
            ae.model.eval()
            ae.threshold = float(self.metadata.get("autoencoder_threshold", 0.0))
            return ae
        except Exception as exc:
            raise RuntimeError(f"Failed to load autoencoder: {exc}") from exc

    def predict(self, features: np.ndarray):
        start_time = time.perf_counter()
        vector = np.asarray(features, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.expected_features:
            raise ValueError(f"Expected {self.expected_features} features, got {vector.shape[0]}")

        pred_cls, confidence = self.classifier.predict(vector.reshape(1, -1))
        pred_cls = str(pred_cls[0])
        confidence = float(confidence[0])
        is_zero_day = False
        reconstruction_error = None

        if confidence < self.config["classifier"].get("confidence_threshold", 0.7):
            flags, errors = self.autoencoder.is_anomaly(vector.reshape(1, -1))
            is_zero_day = bool(flags[0])
            reconstruction_error = float(errors[0])
            pred_cls = "ZERO_DAY" if is_zero_day else self.metadata.get("benign_label", "BENIGN")

        self.prediction_stats["total_predictions"] += 1
        self.prediction_stats["total_confidence"] += confidence
        self.prediction_stats["total_latency_ms"] += (time.perf_counter() - start_time) * 1000
        if is_zero_day:
            self.prediction_stats["zero_day_detections"] += 1
        self.recent_predictions.append(pred_cls)
        self.recent_confidences.append(confidence)
        self.recent_samples.append((vector.copy(), pred_cls, confidence, is_zero_day))
        return pred_cls, confidence, is_zero_day, reconstruction_error

    def predict_batch(self, features: np.ndarray) -> dict[str, list]:
        batch = np.asarray(features, dtype=np.float32)
        if batch.ndim != 2 or batch.shape[0] == 0:
            raise ValueError("batch_features must be a non-empty 2D array")
        predictions, confidences, flags, errors = [], [], [], []
        for row in batch:
            pred, conf, flag, err = self.predict(row)
            predictions.append(pred)
            confidences.append(conf)
            flags.append(flag)
            errors.append(err)
        return {
            "predictions": predictions,
            "confidences": confidences,
            "zero_day_flags": flags,
            "reconstruction_errors": errors,
        }

    def check_drift(self, confidence: float, prediction: str) -> dict[str, Any]:
        drift_signal = 1.0 - float(confidence)
        event = self.drift_detector.update(drift_signal, self.prediction_stats["total_predictions"])
        if event.detected:
            self.last_drift_time = datetime.utcnow().isoformat()
        return {
            "drift_detected": bool(event.detected),
            "drift_score": drift_signal,
            "drift_threshold": self.drift_detector.adwin.delta,
            "method": "ADWIN",
            "num_drifts_total": self.drift_detector.num_drifts,
            "last_drift_time": self.last_drift_time,
        }

    def trigger_adaptation(self) -> dict[str, Any]:
        """Perform conservative unlabeled adaptation using high-confidence pseudo-labels."""
        samples = list(self.recent_samples)
        threshold = max(0.9, float(self.config["classifier"].get("confidence_threshold", 0.7)))
        high_conf = [s for s in samples if s[2] >= threshold and not s[3]]
        minimum = min(20, max(5, int(self.config["drift"].get("min_adapt_samples", 500)) // 10))
        if len(high_conf) < minimum:
            result = {"status": "deferred", "reason": "insufficient_high_confidence_samples", "samples_available": len(high_conf)}
            self.last_adaptation = result
            return result

        X = np.vstack([s[0] for s in high_conf])
        y = np.asarray([s[1] for s in high_conf], dtype=str)
        accepted = bool(self.classifier.partial_adapt(X, y))

        benign = self.metadata.get("benign_label", "BENIGN")
        normal_mask = y == benign
        threshold_updated = bool(np.any(normal_mask))
        if threshold_updated:
            self.autoencoder.update_threshold(X[normal_mask])
            self.metadata["autoencoder_threshold"] = float(self.autoencoder.threshold)

        if accepted or threshold_updated:
            self.metadata["version"] = int(self.metadata.get("version", 1)) + 1
            self.metadata["last_updated"] = datetime.utcnow().isoformat()
            self.metadata["last_adapted_at"] = datetime.utcnow().isoformat()
            save_model_atomic(self.model_dir / "classifier.joblib", self.classifier, method="joblib")
            save_model_atomic(self.model_dir / "autoencoder.pt", self.autoencoder.model.state_dict(), method="torch")
            save_metadata_atomic(self.model_dir / "metadata.json", self.metadata)

        result = {
            "status": "accepted" if (accepted or threshold_updated) else "rejected",
            "pseudo_labeled_samples": len(y),
            "classifier_updated": accepted,
            "autoencoder_threshold_updated": threshold_updated,
            "model_version": self.metadata.get("version", 1),
        }
        self.last_adaptation = result
        return result

    def get_prediction_stats(self) -> dict[str, Any]:
        total = self.prediction_stats["total_predictions"]
        return {
            "total_predictions": total,
            "zero_day_detections": self.prediction_stats["zero_day_detections"],
            "avg_confidence": self.prediction_stats["total_confidence"] / max(1, total),
            "avg_latency_ms": self.prediction_stats["total_latency_ms"] / max(1, total),
            "model_version": self.metadata.get("version", 1),
            "uptime_hours": (time.time() - self.start_time) / 3600,
            "last_drift_time": self.last_drift_time,
            "last_adaptation": self.last_adaptation,
        }


_model_manager: ModelManager | None = None


def get_model_manager() -> ModelManager:
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
