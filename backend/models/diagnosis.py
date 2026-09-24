"""Diagnosis, work order, and receipt models.

The receipt is the product: everything a technician needs to check an alert by hand
(predicted bins with their equations, measured levels against baseline, operating point,
the ordered evidence chain, and the reading) as one typed object, not a formatted string.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from backend.models.alert import BinEvidence
from backend.models.fault_map import FaultClass


class VerdictStatus(StrEnum):
    CONFIRMED = "confirmed"
    DISCARDED = "discarded"
    INCONCLUSIVE = "inconclusive"


class EvidenceStep(BaseModel):
    """One diagnostic in the chain: why it ran, what it returned, what that implies."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=1)
    tool: str
    arguments: dict[str, Any]
    rationale: str = Field(description="The agent's stated hypothesis before running the tool")
    result: dict[str, Any]
    implication: str = Field(description="Deterministic reading of the result by the tool")
    is_error: bool = False


class Verdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: VerdictStatus
    fault_class: FaultClass
    bearing_position: str | None
    confidence: float = Field(ge=0, le=1)
    summary: str
    reading: str = Field(description="Plain-language explanation for a technician")
    discard_reason: str | None = None
    evidence: list[EvidenceStep]
    steps_used: int
    step_budget: int
    model: str
    guardrail_notes: list[str] = Field(default_factory=list)


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class WorkOrderDraft(BaseModel):
    """The only part of a work order the Action Agent writes."""

    confirming_offline_test: str = Field(
        description="Offline test that would confirm the fault before intervention"
    )
    parts: list[str] = Field(description="Parts likely needed; empty if none")
    recommended_window: str = Field(description="When to act, e.g. 'next planned outage'")
    actions: list[str] = Field(description="Ordered maintenance actions")
    safety_notes: list[str] = Field(description="Safety considerations for the work")


class WorkOrder(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    asset_id: str
    alert_id: str
    fault_class: FaultClass
    bearing_position: str | None
    severity: Severity
    severity_basis: str
    confirming_offline_test: str
    parts: list[str]
    recommended_window: str
    actions: list[str]
    safety_notes: list[str]
    created_at: datetime
    model: str


class ReceiptOperatingPoint(BaseModel):
    load_factor: float
    load_factor_low: float
    load_factor_high: float
    load_bucket: str
    slip: float
    rotor_speed_rpm: float
    slip_source: str
    slip_confidence: float


class Receipt(BaseModel):
    """Everything needed to verify an alert independently."""

    model_config = ConfigDict(frozen=True)

    alert_id: str
    asset_id: str
    fault_class: FaultClass
    bearing_position: str | None
    generated_at: datetime
    predicted_bins: list[BinEvidence]
    operating_point: ReceiptOperatingPoint
    verdict: Verdict
    reading: str
    work_order: WorkOrder | None = None
    limitations: list[str] = Field(default_factory=list)
