"""Unit tests for config loading and validation."""
from __future__ import annotations

import pytest
import yaml

from sa_zd_nids.config import load_config, validate_config


def test_load_config_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nonexistent.yaml")


def test_load_config_valid(tmp_path):
    cfg_data = {
        "project": {"random_state": 42, "log_level": "INFO"},
        "data": {
            "input_path": "data/processed.csv",
            "timestamp_col": "timestamp",
            "label_col": "label",
            "benign_label": "BENIGN",
            "test_size": 0.2,
            "val_size": 0.1,
            "scale": "standard",
        },
        "features": {"use_mutual_info": False, "mutual_info_k": 40, "impute_strategy": "median"},
        "classifier": {"model_type": "random_forest", "confidence_threshold": 0.7, "random_forest": {"n_estimators": 10}},
        "anomaly": {"hidden_dims": [128, 64, 32], "epochs": 5, "threshold_percentile": 95},
        "drift": {"adwin_delta": 0.002, "window_size": 1000, "min_adapt_samples": 500},
        "streaming": {"batch_size": 500},
        "logging": {"output_csv": "logs/out.csv", "events_jsonl": "logs/ev.jsonl", "metrics_json": "logs/m.json"},
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg_data))
    cfg = load_config(p)
    assert cfg["project"]["random_state"] == 42


def test_validate_config_missing_section(tmp_path):
    cfg_data = {"project": {"random_state": 42}}
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg_data))
    cfg = load_config(p)
    with pytest.raises(KeyError):
        validate_config(cfg)


def test_validate_config_invalid_scale(tmp_path):
    cfg_data = {
        "project": {"random_state": 42},
        "data": {"scale": "invalid", "test_size": 0.2, "val_size": 0.1, "label_col": "label", "timestamp_col": "ts", "benign_label": "BENIGN"},
        "features": {"use_mutual_info": False, "impute_strategy": "median"},
        "classifier": {"model_type": "random_forest", "confidence_threshold": 0.7},
        "anomaly": {"hidden_dims": [128, 64, 32], "epochs": 5, "threshold_percentile": 95},
        "drift": {"adwin_delta": 0.002, "window_size": 1000, "min_adapt_samples": 500},
        "streaming": {"batch_size": 500},
        "logging": {"output_csv": "a", "events_jsonl": "b", "metrics_json": "c"},
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg_data))
    cfg = load_config(p)
    with pytest.raises(ValueError, match="scale"):
        validate_config(cfg)


def test_validate_config_invalid_test_size(tmp_path):
    cfg_data = {
        "project": {"random_state": 42},
        "data": {"scale": "standard", "test_size": 1.5, "val_size": 0.1, "label_col": "label", "timestamp_col": "ts", "benign_label": "BENIGN"},
        "features": {"use_mutual_info": False, "impute_strategy": "median"},
        "classifier": {"model_type": "random_forest", "confidence_threshold": 0.7},
        "anomaly": {"hidden_dims": [128, 64, 32], "epochs": 5, "threshold_percentile": 95},
        "drift": {"adwin_delta": 0.002, "window_size": 1000, "min_adapt_samples": 500},
        "streaming": {"batch_size": 500},
        "logging": {"output_csv": "a", "events_jsonl": "b", "metrics_json": "c"},
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg_data))
    cfg = load_config(p)
    with pytest.raises(ValueError, match="test_size"):
        validate_config(cfg)
