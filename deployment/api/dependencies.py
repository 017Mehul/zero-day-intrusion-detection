"""Dependencies and model management for FastAPI app."""
import joblib
import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
from collections import deque
import time

from sa_zd_nids.models.classifier import KnownAttackClassifier
from sa_zd_nids.models.autoencoder import ZeroDayAutoencoder, AutoencoderNet
from sa_zd_nids.drift.detector import DriftDetector
from sa_zd_nids.utils.io import load_config

class ModelManager:
    """Manages model loading, inference, and drift detection."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.start_time = time.time()
        
        # Load models
        self.classifier = self._load_classifier()
        self.autoencoder = self._load_autoencoder()
        self.metadata = self._load_metadata()
        
        # Initialize drift detection
        self.drift_detector = DriftDetector(
            delta=self.config["drift"].get("adwin_delta", 0.002)
        )
        
        # Prediction statistics
        self.prediction_stats = {
            "total_predictions": 0,
            "zero_day_detections": 0,
            "total_confidence": 0.0,
            "total_latency_ms": 0.0
        }
        
        # Recent predictions for drift detection
        self.recent_predictions = deque(maxlen=1000)
        self.recent_confidences = deque(maxlen=1000)
        
        # Feature information
        self.expected_features = len(self.metadata.get("feature_names", []))
        
        print(f"ModelManager initialized with {self.expected_features} features")
    
    def _load_classifier(self) -> KnownAttackClassifier:
        """Load the trained classifier."""
        try:
            classifier = joblib.load("models/classifier.joblib")
            return classifier
        except Exception as e:
            raise RuntimeError(f"Failed to load classifier: {e}")
    
    def _load_autoencoder(self) -> ZeroDayAutoencoder:
        """Load the trained autoencoder."""
        try:
            ae = ZeroDayAutoencoder(self.config)
            input_dim = len(self.metadata.get("feature_names", []))
            ae.model = AutoencoderNet(
                input_dim, 
                self.config["anomaly"].get("hidden_dims", [128, 64, 32])
            ).to(ae.device)
            
            # Load state dict
            state_dict = torch.load("models/autoencoder.pt", map_location=ae.device)
            ae.model.load_state_dict(state_dict)
            ae.threshold = float(self.metadata.get("autoencoder_threshold", 0.0))
            
            return ae
        except Exception as e:
            raise RuntimeError(f"Failed to load autoencoder: {e}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load model metadata."""
        try:
            with open("models/metadata.json", "r") as f:
                return json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load metadata: {e}")
    
    def predict(self, features: np.ndarray) -> Tuple[str, float, bool, float]:
        """Make a single prediction."""
        start_time = time.time()
        
        # Classifier prediction
        pred_cls, confidence = self.classifier.predict(features.reshape(1, -1))
        pred_cls = pred_cls[0]
        confidence = confidence[0]
        
        # Zero-day detection
        is_zero_day = False
        reconstruction_error = None
        
        if confidence < self.config["classifier"].get("confidence_threshold", 0.7):
            # Use autoencoder for zero-day detection
            flags, errors = self.autoencoder.is_anomaly(features.reshape(1, -1))
            is_zero_day = flags[0]
            reconstruction_error = errors[0]
            
            if is_zero_day:
                pred_cls = "ZERO_DAY"
            else:
                pred_cls = self.metadata.get("benign_label", "BENIGN")
        
        # Update statistics
        self.prediction_stats["total_predictions"] += 1
        self.prediction_stats["total_confidence"] += confidence
        self.prediction_stats["total_latency_ms"] += (time.time() - start_time) * 1000
        
        if is_zero_day:
            self.prediction_stats["zero_day_detections"] += 1
        
        # Store for drift detection
        self.recent_predictions.append(pred_cls)
        self.recent_confidences.append(confidence)
        
        return pred_cls, confidence, is_zero_day, reconstruction_error
    
    def predict_batch(self, features: np.ndarray) -> Dict[str, List]:
        """Make batch predictions."""
        batch_size = features.shape[0]
        predictions = []
        confidences = []
        zero_day_flags = []
        reconstruction_errors = []
        
        for i in range(batch_size):
            pred, conf, is_zd, recon_err = self.predict(features[i])
            predictions.append(pred)
            confidences.append(conf)
            zero_day_flags.append(is_zd)
            reconstruction_errors.append(recon_err)
        
        return {
            "predictions": predictions,
            "confidences": confidences,
            "zero_day_flags": zero_day_flags,
            "reconstruction_errors": reconstruction_errors
        }
    
    def check_drift(self, confidence: float, prediction: str) -> Dict[str, Any]:
        """Check for concept drift."""
        # Simple drift detection based on confidence
        drift_signal = 1.0 - confidence  # Lower confidence = higher drift signal
        
        drift_event = self.drift_detector.update(drift_signal, self.prediction_stats["total_predictions"])
        
        return {
            "drift_detected": drift_event.detected,
            "drift_score": drift_signal,
            "drift_threshold": self.drift_detector.adwin.delta,
            "method": "ADWIN",
            "num_drifts_total": self.drift_detector.num_drifts
        }
    
    def trigger_adaptation(self):
        """Trigger model adaptation (placeholder)."""
        print("Adaptation triggered - would retrain models with recent data")
        # In production, this would collect recent data and retrain
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get prediction statistics."""
        total_preds = self.prediction_stats["total_predictions"]
        
        return {
            "total_predictions": total_preds,
            "zero_day_detections": self.prediction_stats["zero_day_detections"],
            "avg_confidence": self.prediction_stats["total_confidence"] / max(1, total_preds),
            "avg_latency_ms": self.prediction_stats["total_latency_ms"] / max(1, total_preds),
            "model_version": self.metadata.get("version", 1),
            "uptime_hours": (time.time() - self.start_time) / 3600
        }

# Global model manager instance
_model_manager = None

def get_model_manager() -> ModelManager:
    """Get or create model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
