"""Monitoring Agent: deterministic, no LLM.

Reduces each window to features and scores them against the robust per-load-bucket baseline.
Named as an agent because it is the stage that raises candidates for the Diagnosis Agent; it
contains no model and no learned parameters.

`analyze` is CPU-bound and stateless, so callers can run it in parallel across assets;
`score` mutates the shared detector and must be serialised by the caller.
"""

from __future__ import annotations

from datetime import datetime

from backend.detection.baseline import (
    DetectorConfig,
    ObservationResult,
    SustainedExceedanceDetector,
)
from backend.detection.features import ThreePhaseCapture, WindowFeatures, analyze_window
from backend.models.motor import MotorSpec


class MonitoringAgent:
    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.detector = SustainedExceedanceDetector(config)

    @staticmethod
    def analyze(spec: MotorSpec, capture: ThreePhaseCapture) -> WindowFeatures:
        return analyze_window(spec, capture)

    def score(
        self, spec: MotorSpec, features: WindowFeatures, timestamp: datetime
    ) -> ObservationResult | None:
        """Score a window; None when it is not stationary (load transient) and was skipped."""
        if not features.stationary:
            return None
        return self.detector.observe(
            spec.asset_id, features.load.load_factor, features.measurements, timestamp
        )
