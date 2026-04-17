from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

_REQUIRED_SECTIONS = ("project", "data", "features", "classifier", "anomaly", "drift", "streaming", "logging")
_VALID_SCALES = ("standard", "minmax", "none")
_VALID_MODEL_TYPES = ("xgboost", "random_forest", "lightgbm")


def load_config(path: str | Path = "config.yaml") -> Dict[str, Any]:
    """Load YAML config from *path*. Raises FileNotFoundError if missing."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_config(cfg: Dict[str, Any]) -> None:
    """Validate config structure and value ranges.

    Raises KeyError for missing required sections/keys.
    Raises ValueError for invalid values.
    """
    for section in _REQUIRED_SECTIONS:
        if section not in cfg:
            raise KeyError(f"Missing required config section: '{section}'")

    data = cfg["data"]
    scale = data.get("scale", "standard")
    if scale not in _VALID_SCALES:
        raise ValueError(f"Invalid scale '{scale}'. Must be one of {_VALID_SCALES}")

    for size_key in ("test_size", "val_size"):
        val = data.get(size_key, 0.2)
        if not (0.0 < val < 1.0):
            raise ValueError(f"data.{size_key}={val} must be in (0, 1)")

    if data.get("test_size", 0.2) + data.get("val_size", 0.2) >= 1.0:
        raise ValueError("test_size + val_size must be < 1.0")

    model_type = cfg["classifier"].get("model_type", "xgboost")
    if model_type not in _VALID_MODEL_TYPES:
        raise ValueError(f"Invalid classifier.model_type '{model_type}'. Must be one of {_VALID_MODEL_TYPES}")

    conf_thr = cfg["classifier"].get("confidence_threshold", 0.7)
    if not (0.0 <= conf_thr <= 1.0):
        raise ValueError(f"classifier.confidence_threshold={conf_thr} must be in [0, 1]")

    hidden = cfg["anomaly"].get("hidden_dims", [128, 64, 32])
    if len(hidden) != 3:
        raise ValueError(f"anomaly.hidden_dims must have exactly 3 elements, got {len(hidden)}")

    pct = cfg["anomaly"].get("threshold_percentile", 95)
    if not (0 < pct <= 100):
        raise ValueError(f"anomaly.threshold_percentile={pct} must be in (0, 100]")

    delta = cfg["drift"].get("adwin_delta", 0.002)
    if delta <= 0:
        raise ValueError(f"drift.adwin_delta={delta} must be > 0")

    for log_key in ("output_csv", "events_jsonl", "metrics_json"):
        if log_key not in cfg["logging"]:
            raise KeyError(f"Missing logging.{log_key} in config")
