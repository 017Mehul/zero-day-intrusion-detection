"""Drift detection using the ADWIN algorithm from the River library."""
from __future__ import annotations

from dataclasses import dataclass

from river.drift import ADWIN


@dataclass
class DriftEvent:
    index: int
    value: float
    detected: bool


class DriftDetector:
    def __init__(self, delta: float = 0.002):
        self.adwin = ADWIN(delta=delta)
        self.num_drifts = 0

    def update(self, metric_value: float, index: int) -> DriftEvent:
        self.adwin.update(metric_value)
        detected = bool(self.adwin.drift_detected)
        if detected:
            self.num_drifts += 1
        return DriftEvent(index=index, value=metric_value, detected=detected)
