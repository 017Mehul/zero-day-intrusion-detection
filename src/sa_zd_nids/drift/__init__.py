"""Drift detection via ADWIN, DDM, EDDM, and ensemble methods."""

__all__ = ["DriftDetector", "DriftEvent", "EnhancedDriftDetector", "MultiMethodDriftDetector", 
           "DriftMethod", "create_drift_detector", "create_multi_method_detector"]


def __getattr__(name):
    if name in ["DriftDetector", "DriftEvent"]:
        from sa_zd_nids.drift.detector import DriftDetector, DriftEvent  # noqa: F401
        return {"DriftDetector": DriftDetector, "DriftEvent": DriftEvent}[name]
    elif name in ["EnhancedDriftDetector", "MultiMethodDriftDetector", "DriftMethod", 
                  "create_drift_detector", "create_multi_method_detector"]:
        from sa_zd_nids.drift.enhanced_detector import (  # noqa: F401
            EnhancedDriftDetector, MultiMethodDriftDetector, DriftMethod, 
            create_drift_detector, create_multi_method_detector
        )
        return {
            "EnhancedDriftDetector": EnhancedDriftDetector,
            "MultiMethodDriftDetector": MultiMethodDriftDetector,
            "DriftMethod": DriftMethod,
            "create_drift_detector": create_drift_detector,
            "create_multi_method_detector": create_multi_method_detector
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
