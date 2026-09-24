"""API request/response schemas and the stored records behind them."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from backend.models.alert import Candidate
from backend.models.fault_map import ConfirmationItem, FaultClass, FaultMap, UnavailableFault
from backend.models.motor import MotorSpec
from backend.simulator.faults import Confounder, FaultInjection

T = TypeVar("T")


class ApiError(BaseModel):
    code: str
    message: str
    details: list[dict] | None = None


class Envelope(BaseModel, Generic[T]):
    """Every response: success flag, data (null on error), error (null on success)."""

    success: bool
    data: T | None = None
    error: ApiError | None = None
    meta: dict | None = None


def ok(data: T, meta: dict | None = None) -> Envelope[T]:
    return Envelope[T](success=True, data=data, meta=meta)


class HealthState(StrEnum):
    ALERT = "alert"
    WATCH = "watch"
    LEARNING = "learning"
    HEALTHY = "healthy"


HEALTH_SEVERITY_ORDER = [
    HealthState.ALERT,
    HealthState.WATCH,
    HealthState.LEARNING,
    HealthState.HEALTHY,
]


class AlertStatus(StrEnum):
    ACTIVE = "active"
    CONFIRMED = "confirmed"
    DISCARDED = "discarded"
    INCONCLUSIVE = "inconclusive"


# --- stored records -------------------------------------------------------------------


class SimulationProfile(BaseModel):
    """Where this asset's data comes from in the prototype: the simulator."""

    model_config = ConfigDict(frozen=True)

    load_factor: float = Field(default=0.8, ge=0.05, le=1.3)
    load_jitter: float = Field(default=0.03, ge=0, le=0.2)
    faults: list[FaultInjection] = Field(default_factory=list)
    confounders: list[Confounder] = Field(default_factory=list)


class AssetRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    spec: MotorSpec
    commissioned_at: datetime
    commissioning_fault_map: FaultMap
    simulation: SimulationProfile
    health: HealthState = HealthState.LEARNING
    windows_processed: int = 0


class BinScoreView(BaseModel):
    key: str
    label: str
    fault_class: FaultClass
    bearing_position: str | None
    frequency_hz: float | None
    equation: str
    level_db: float
    baseline_count: int
    baseline_median_db: float | None
    baseline_mad_db: float | None
    z_score: float | None
    consecutive_windows: int
    ambiguous_with: list[str]


class SlipView(BaseModel):
    slip: float
    rotor_speed_rpm: float
    confidence: float
    source: str
    notes: list[str]


class LoadView(BaseModel):
    load_factor: float
    load_factor_low: float
    load_factor_high: float
    input_power_kw: float
    shaft_power_kw: float
    efficiency_indicative: float
    assumptions: list[str]


class UnbalanceView(BaseModel):
    raw_current_unbalance: float
    voltage_unbalance: float
    supply_attributable_unbalance: float | None
    net_current_unbalance: float | None
    method: str
    note: str


class WindowSnapshot(BaseModel):
    asset_id: str
    window_index: int
    captured_at: datetime
    stationary: bool
    stationarity_cv: float
    scored: bool
    load_bucket: str | None
    learning: bool
    load: LoadView
    slip: SlipView
    unbalance: UnbalanceView | None
    resolution_hz: float
    spectrum_hz: list[float]
    spectrum_db: list[float] = Field(description="Phase-averaged power, dB re fundamental")
    scores: list[BinScoreView]
    excluded_bins: dict[str, str]
    operating_fault_map: FaultMap


class Alert(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    asset_id: str
    fault_class: FaultClass
    bearing_position: str | None
    status: AlertStatus
    first_raised_at: datetime
    last_seen_at: datetime
    windows_seen: int
    candidate: Candidate


# --- requests -------------------------------------------------------------------------


class AssetCreate(BaseModel):
    spec: MotorSpec
    simulation: SimulationProfile = Field(default_factory=SimulationProfile)


class InjectRequest(BaseModel):
    asset_id: str
    faults: list[FaultInjection] = Field(default_factory=list)
    confounders: list[Confounder] = Field(default_factory=list)
    advance_windows: int = Field(default=0, ge=0, le=50)


class AdvanceRequest(BaseModel):
    asset_id: str
    windows: int = Field(default=1, ge=1, le=50)


# --- responses ------------------------------------------------------------------------


class AssetSummary(BaseModel):
    asset_id: str
    rated_power_kw: float
    poles: int
    supply_frequency_hz: float
    health: HealthState
    load_factor: float | None
    slip: float | None
    active_alerts: int
    unavailable_fault_classes: list[FaultClass]
    needs_confirmation: int
    windows_processed: int


class AssetDetail(BaseModel):
    summary: AssetSummary
    spec: MotorSpec
    commissioning_fault_map: FaultMap
    needs_confirmation: list[ConfirmationItem]
    unavailable: list[UnavailableFault]
    simulation: SimulationProfile
    baseline_assumption: str
    latest_window: WindowSnapshot | None


class SpectrumView(BaseModel):
    asset_id: str
    window_index: int
    resolution_hz: float
    supply_frequency_hz: float
    frequencies_hz: list[float]
    level_db: list[float]
    bins: list[BinScoreView]
    excluded_bins: dict[str, str]
    slip: SlipView


class WindowResultView(BaseModel):
    asset_id: str
    window_index: int
    scored: bool
    health: HealthState
    load_factor: float
    slip: float
    new_alerts: list[Alert]


class EnergyAsset(BaseModel):
    asset_id: str
    rated_power_kw: float
    load_factor: float
    load_factor_low: float
    load_factor_high: float
    shaft_power_kw: float
    input_power_kw: float
    oversizing_candidate: bool
    estimated_waste_kw: float
    estimated_waste_mwh_per_year: float


class FleetEnergy(BaseModel):
    assets: list[EnergyAsset]
    oversizing_candidates: list[str]
    assumptions: list[str]
