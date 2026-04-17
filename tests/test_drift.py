"""Unit tests for drift detector."""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.requires_river

try:
    from sa_zd_nids.drift.detector import DriftDetector, DriftEvent
except ModuleNotFoundError:
    pytest.skip("river not installed", allow_module_level=True)


def test_drift_event_fields():
    det = DriftDetector(delta=0.002)
    event = det.update(metric_value=0.5, index=10)
    assert isinstance(event, DriftEvent)
    assert event.index == 10
    assert event.value == 0.5
    assert isinstance(event.detected, bool)


def test_no_drift_on_stable_signal():
    det = DriftDetector(delta=0.002)
    detected_any = False
    for i in range(200):
        e = det.update(metric_value=0.1, index=i)
        if e.detected:
            detected_any = True
    # stable signal should not trigger drift
    assert not detected_any


def test_drift_detected_on_sudden_shift():
    det = DriftDetector(delta=0.002)
    detected = False
    # feed stable low values then sudden high values
    for i in range(300):
        val = 0.05 if i < 200 else 0.95
        e = det.update(metric_value=val, index=i)
        if e.detected:
            detected = True
    assert detected


def test_num_drifts_increments():
    det = DriftDetector(delta=0.002)
    for i in range(300):
        val = 0.05 if i < 200 else 0.95
        det.update(metric_value=val, index=i)
    assert det.num_drifts >= 1
