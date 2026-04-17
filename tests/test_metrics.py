"""Unit tests for evaluation metrics."""
from __future__ import annotations

import numpy as np
import pytest

from sa_zd_nids.evaluation.metrics import compute_classification_metrics, compute_zero_day_metrics


def test_perfect_classification():
    y = np.array(["A", "B", "A", "B"])
    m = compute_classification_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["f1_macro"] == 1.0


def test_all_wrong_classification():
    y_true = np.array(["A", "A", "A"])
    y_pred = np.array(["B", "B", "B"])
    m = compute_classification_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.0


def test_classification_metrics_keys():
    y = np.array(["A", "B"])
    m = compute_classification_metrics(y, y)
    for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        assert key in m


def test_zero_day_detection_rate_perfect():
    y_true = np.array(["BENIGN", "NEW_ATTACK", "NEW_ATTACK"])
    y_pred = np.array(["BENIGN", "ZERO_DAY", "ZERO_DAY"])
    m = compute_zero_day_metrics(y_true, y_pred, known_labels={"BENIGN"}, benign_label="BENIGN")
    assert m["zero_day_detection_rate"] == 1.0
    assert m["zero_day_false_positive_rate_benign"] == 0.0


def test_zero_day_false_positive():
    y_true = np.array(["BENIGN", "BENIGN", "NEW_ATTACK"])
    y_pred = np.array(["ZERO_DAY", "BENIGN", "ZERO_DAY"])
    m = compute_zero_day_metrics(y_true, y_pred, known_labels={"BENIGN"}, benign_label="BENIGN")
    assert m["zero_day_false_positive_rate_benign"] == pytest.approx(0.5)
    assert m["zero_day_detection_rate"] == 1.0


def test_zero_day_no_unseen_samples():
    y_true = np.array(["BENIGN", "BENIGN"])
    y_pred = np.array(["BENIGN", "BENIGN"])
    m = compute_zero_day_metrics(y_true, y_pred, known_labels={"BENIGN"}, benign_label="BENIGN")
    assert m["zero_day_detection_rate"] == 0.0
    assert m["zero_day_unseen_samples"] == 0


def test_zero_day_no_benign_samples():
    y_true = np.array(["ATTACK", "ATTACK"])
    y_pred = np.array(["ZERO_DAY", "ZERO_DAY"])
    m = compute_zero_day_metrics(y_true, y_pred, known_labels={"BENIGN"}, benign_label="BENIGN")
    assert m["zero_day_false_positive_rate_benign"] == 0.0
