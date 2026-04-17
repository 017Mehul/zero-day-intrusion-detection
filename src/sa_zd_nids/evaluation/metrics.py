"""Evaluation metrics for classification and zero-day detection."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def try_multiclass_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    try:
        return float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
    except Exception:
        return None


def compute_zero_day_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    known_labels: set[str],
    benign_label: str,
    zero_day_label: str = "ZERO_DAY",
) -> dict[str, Any]:
    y_true_s = np.asarray(y_true, dtype=object).astype(str)
    y_pred_s = np.asarray(y_pred, dtype=object).astype(str)
    unseen_mask = ~np.isin(y_true_s, list(known_labels))
    benign_mask = y_true_s == benign_label
    zd_pred = y_pred_s == zero_day_label

    unseen_total = int(unseen_mask.sum())
    benign_total = int(benign_mask.sum())
    zd_detected = int(np.logical_and(unseen_mask, zd_pred).sum())
    zd_false_positive = int(np.logical_and(benign_mask, zd_pred).sum())

    detection_rate = float(zd_detected / unseen_total) if unseen_total > 0 else 0.0
    benign_fpr = float(zd_false_positive / benign_total) if benign_total > 0 else 0.0

    return {
        "zero_day_unseen_samples": unseen_total,
        "zero_day_detected": zd_detected,
        "zero_day_detection_rate": detection_rate,
        "benign_samples": benign_total,
        "zero_day_false_positive_on_benign": zd_false_positive,
        "zero_day_false_positive_rate_benign": benign_fpr,
    }
