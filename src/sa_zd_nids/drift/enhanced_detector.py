"""Enhanced drift detection with multiple methods (ADWIN, DDM, EDDM)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np
from enum import Enum

# Import River drift detection methods
try:
    from river.drift import ADWIN, DDM, EDDM
except ImportError:
    print("Warning: River library not found. Install with: pip install river")
    ADWIN = DDM = EDDM = None


class DriftMethod(Enum):
    """Available drift detection methods."""
    ADWIN = "adwin"
    DDM = "ddm"
    EDDM = "eddm"


@dataclass
class DriftEvent:
    """Drift detection event."""
    index: int
    value: float
    detected: bool
    method: str
    confidence: float
    warning: bool = False


class EnhancedDriftDetector:
    """Enhanced drift detector supporting multiple algorithms."""
    
    def __init__(self, 
                 method: DriftMethod = DriftMethod.ADWIN,
                 **kwargs):
        self.method = method
        self.num_drifts = 0
        self.num_warnings = 0
        self.detector = self._create_detector(method, **kwargs)
        
        # Track statistics
        self.values_seen = 0
        self.drift_history: List[DriftEvent] = []
        self.warning_history: List[DriftEvent] = []
        
    def _create_detector(self, method: DriftMethod, **kwargs):
        """Create drift detector based on method."""
        if ADWIN is None:
            raise ImportError("River library is required. Install with: pip install river")
        
        if method == DriftMethod.ADWIN:
            delta = kwargs.get("delta", 0.002)
            return ADWIN(delta=delta)
        
        elif method == DriftMethod.DDM:
            # DDM parameters
            warn_threshold = kwargs.get("warn_threshold", 2.0)
            drift_threshold = kwargs.get("drift_threshold", 3.0)
            return DDM(warn_threshold=warn_threshold, drift_threshold=drift_threshold)
        
        elif method == DriftMethod.EDDM:
            # EDDM parameters
            alpha = kwargs.get("alpha", 0.95)
            beta = kwargs.get("beta", 0.9)
            return EDDM(alpha=alpha, beta=beta)
        
        else:
            raise ValueError(f"Unknown drift method: {method}")
    
    def update(self, metric_value: float, index: int) -> DriftEvent:
        """Update detector with new metric value."""
        self.values_seen += 1
        
        # Convert metric to appropriate format for River
        # For classification accuracy/error, we need to convert to success/failure
        if self.method in [DriftMethod.DDM, DriftMethod.EDDM]:
            # DDM and EDDM work with binary predictions (success/failure)
            # Convert metric to binary (1 for success, 0 for failure)
            # Assuming metric_value is accuracy, so > 0.5 is success
            binary_value = 1 if metric_value > 0.5 else 0
            self.detector.update(binary_value)
        else:
            # ADWIN works with continuous values
            self.detector.update(metric_value)
        
        # Check for drift and warnings
        drift_detected = False
        warning_detected = False
        confidence = 0.0
        
        if hasattr(self.detector, 'drift_detected') and self.detector.drift_detected:
            drift_detected = True
            self.num_drifts += 1
            confidence = 1.0
        
        if hasattr(self.detector, 'warning_detected') and self.detector.warning_detected:
            warning_detected = True
            self.num_warnings += 1
            confidence = 0.7
        
        # Create event
        event = DriftEvent(
            index=index,
            value=metric_value,
            detected=drift_detected,
            method=self.method.value,
            confidence=confidence,
            warning=warning_detected
        )
        
        # Store in history
        if drift_detected:
            self.drift_history.append(event)
        if warning_detected:
            self.warning_history.append(event)
        
        return event
    
    def get_drift_rate(self, window_size: Optional[int] = None) -> float:
        """Calculate drift rate over a window."""
        if window_size is None:
            # Overall drift rate
            return self.num_drifts / max(1, self.values_seen)
        else:
            # Recent drift rate
            recent_drifts = sum(1 for event in self.drift_history 
                             if event.index >= self.values_seen - window_size)
            return recent_drifts / max(1, window_size)
    
    def get_warning_rate(self, window_size: Optional[int] = None) -> float:
        """Calculate warning rate over a window."""
        if window_size is None:
            return self.num_warnings / max(1, self.values_seen)
        else:
            recent_warnings = sum(1 for event in self.warning_history 
                                if event.index >= self.values_seen - window_size)
            return recent_warnings / max(1, window_size)
    
    def reset(self):
        """Reset detector state."""
        self.detector = self._create_detector(self.method)
        self.num_drifts = 0
        self.num_warnings = 0
        self.values_seen = 0
        self.drift_history.clear()
        self.warning_history.clear()


class MultiMethodDriftDetector:
    """Drift detector that combines multiple methods for robust detection."""
    
    def __init__(self, methods: List[DriftMethod] = None, voting_strategy: str = "majority"):
        if methods is None:
            methods = [DriftMethod.ADWIN, DriftMethod.DDM, DriftMethod.EDDM]
        
        self.methods = methods
        self.voting_strategy = voting_strategy
        self.detectors = {method: EnhancedDriftDetector(method) for method in methods}
        
        # Combined statistics
        self.combined_drifts = 0
        self.events_history: List[Dict[str, Any]] = []
        
    def update(self, metric_value: float, index: int) -> Dict[str, Any]:
        """Update all detectors and return combined result."""
        # Update each detector
        events = {}
        for method, detector in self.detectors.items():
            events[method.value] = detector.update(metric_value, index)
        
        # Combine results based on voting strategy
        combined_result = self._combine_results(events)
        
        # Store combined event
        combined_event = {
            "index": index,
            "value": metric_value,
            "combined_drift": combined_result["drift_detected"],
            "combined_warning": combined_result["warning_detected"],
            "confidence": combined_result["confidence"],
            "individual_events": events,
            "voting_strategy": self.voting_strategy
        }
        
        self.events_history.append(combined_event)
        
        if combined_result["drift_detected"]:
            self.combined_drifts += 1
        
        return combined_event
    
    def _combine_results(self, events: Dict[str, DriftEvent]) -> Dict[str, Any]:
        """Combine results from multiple detectors."""
        drift_votes = sum(1 for event in events.values() if event.detected)
        warning_votes = sum(1 for event in events.values() if event.warning)
        total_votes = len(events)
        
        if self.voting_strategy == "majority":
            drift_detected = drift_votes > total_votes / 2
            warning_detected = warning_votes > total_votes / 2
            confidence = max(drift_votes, warning_votes) / total_votes
        
        elif self.voting_strategy == "unanimous":
            drift_detected = drift_votes == total_votes
            warning_detected = warning_votes == total_votes
            confidence = 1.0 if drift_detected else (drift_votes / total_votes)
        
        elif self.voting_strategy == "any":
            drift_detected = drift_votes > 0
            warning_detected = warning_votes > 0
            confidence = max(drift_votes, warning_votes) / total_votes
        
        else:
            # Default to majority
            drift_detected = drift_votes > total_votes / 2
            warning_detected = warning_votes > total_votes / 2
            confidence = max(drift_votes, warning_votes) / total_votes
        
        return {
            "drift_detected": drift_detected,
            "warning_detected": warning_detected,
            "confidence": confidence,
            "drift_votes": drift_votes,
            "warning_votes": warning_votes,
            "total_votes": total_votes
        }
    
    def get_method_comparison(self) -> Dict[str, Any]:
        """Get comparison of different drift detection methods."""
        comparison = {}
        
        for method, detector in self.detectors.items():
            comparison[method.value] = {
                "num_drifts": detector.num_drifts,
                "num_warnings": detector.num_warnings,
                "drift_rate": detector.get_drift_rate(),
                "warning_rate": detector.get_warning_rate(),
                "values_seen": detector.values_seen
            }
        
        # Add combined statistics
        comparison["combined"] = {
            "num_drifts": self.combined_drifts,
            "drift_rate": self.combined_drifts / max(1, len(self.events_history)),
            "voting_strategy": self.voting_strategy
        }
        
        return comparison
    
    def reset_all(self):
        """Reset all detectors."""
        for detector in self.detectors.values():
            detector.reset()
        self.combined_drifts = 0
        self.events_history.clear()


def create_drift_detector(method: str = "adwin", **kwargs) -> EnhancedDriftDetector:
    """Factory function to create drift detector."""
    try:
        method_enum = DriftMethod(method.lower())
        return EnhancedDriftDetector(method_enum, **kwargs)
    except ValueError:
        raise ValueError(f"Unknown drift method: {method}. Available: {[m.value for m in DriftMethod]}")


def create_multi_method_detector(methods: List[str] = None, 
                                 voting_strategy: str = "majority") -> MultiMethodDriftDetector:
    """Factory function to create multi-method drift detector."""
    if methods is None:
        methods = ["adwin", "ddm", "eddm"]
    
    method_enums = []
    for method in methods:
        try:
            method_enums.append(DriftMethod(method.lower()))
        except ValueError:
            print(f"Warning: Unknown drift method '{method}', skipping...")
    
    if not method_enums:
        raise ValueError("No valid drift methods provided")
    
    return MultiMethodDriftDetector(method_enums, voting_strategy)


# Usage examples and recommendations
DRIFT_METHOD_RECOMMENDATIONS = {
    "adwin": {
        "best_for": "Gradual drift, streaming data",
        "parameters": {"delta": 0.002},
        "pros": ["Adaptive window size", "No parameter tuning needed"],
        "cons": ["May be slower", "Less sensitive to sudden changes"]
    },
    "ddm": {
        "best_for": "Sudden/abrupt drift",
        "parameters": {"warn_threshold": 2.0, "drift_threshold": 3.0},
        "pros": ["Fast detection of sudden changes", "Simple to understand"],
        "cons": ["Requires binary predictions", "Fixed thresholds"]
    },
    "eddm": {
        "best_for": "Gradual drift with delayed detection",
        "parameters": {"alpha": 0.95, "beta": 0.9},
        "pros": ["Good for gradual drift", "Statistically motivated"],
        "cons": ["Slower to detect", "More complex"]
    }
}
