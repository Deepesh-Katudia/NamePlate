"""Detection outputs: per-window bin measurements and fault candidates.

A `Candidate` is not yet an alert: it is the Monitoring stage's statistical claim that a fault
class's bins have exceeded their baseline for several consecutive windows. It carries enough
evidence (equation, measured level, baseline median/MAD, z-score) to be checked by hand.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from backend.models.fault_map import FaultClass


class BinMeasurement(BaseModel):
    """One monitored quantity in one window: a fault bin's level, or a sequence feature."""

    model_config = ConfigDict(frozen=True)

    key: str = Field(description="Stable identifier across windows: class:position:label")
    label: str
    fault_class: FaultClass
    bearing_position: str | None
    frequency_hz: float | None = Field(description="None for non-spectral features")
    equation: str
    inputs: dict[str, float]
    is_primary: bool
    level_db: float = Field(description="Line level relative to the fundamental, dB")
    ambiguous_with: list[str] = Field(default_factory=list)


class BinEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    label: str
    frequency_hz: float | None
    equation: str
    inputs: dict[str, float]
    measured_db: float = Field(description="Line level relative to the fundamental, dB")
    baseline_median_db: float
    baseline_mad_db: float
    z_score: float
    consecutive_windows: int
    ambiguous_with: list[str] = Field(default_factory=list)


class Candidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    asset_id: str
    fault_class: FaultClass
    bearing_position: str | None
    load_bucket: str
    load_factor: float
    window_index: int
    raised_at: datetime
    evidence: list[BinEvidence] = Field(min_length=1)
