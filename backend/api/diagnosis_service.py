"""Diagnosis orchestration: candidate -> Diagnosis Agent -> Action Agent -> receipt.

Separate from MonitoringService so window processing never depends on the LLM being
available. LLM calls run without holding the service's locks; only the final alert update is
serialised.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from backend.agents.action import ActionAgent
from backend.agents.commissioning import CommissioningAgent, CommissioningResult
from backend.agents.diagnosis import DiagnosisAgent
from backend.agents.llm import AgentUnavailableError, LLMClient
from backend.agents.tools import DiagnosticContext
from backend.api.schemas import Alert, AlertStatus, CommissioningPreviewRequest
from backend.api.service import BASELINE_ASSUMPTION, MonitoringService, max_z
from backend.detection.baseline import LoadBucket
from backend.models.diagnosis import Receipt, ReceiptOperatingPoint, VerdictStatus
from backend.models.errors import DomainError

logger = logging.getLogger(__name__)

_STATUS_FOR_VERDICT = {
    VerdictStatus.CONFIRMED: AlertStatus.CONFIRMED,
    VerdictStatus.DISCARDED: AlertStatus.DISCARDED,
    VerdictStatus.INCONCLUSIVE: AlertStatus.INCONCLUSIVE,
}


class DiagnosisService:
    def __init__(self, monitoring: MonitoringService, llm: LLMClient | None) -> None:
        self.monitoring = monitoring
        self.llm = llm

    def _require_llm(self) -> LLMClient:
        if self.llm is None:
            raise AgentUnavailableError(
                "Diagnosis needs the Anthropic API: set ANTHROPIC_API_KEY in the environment"
            )
        return self.llm

    def commissioning_preview(self, request: CommissioningPreviewRequest) -> CommissioningResult:
        agent = CommissioningAgent(self.llm)
        if request.spec is not None:
            data = dict(request.spec)
            if request.asset_id:
                data["asset_id"] = request.asset_id
            return agent.from_structured(data)
        assert request.nameplate_text is not None  # enforced by the request validator
        return agent.from_text(request.nameplate_text, request.asset_id)

    def _select_alert(self, asset_id: str, alert_id: str | None) -> Alert:
        alerts = self.monitoring.repo.list_alerts(asset_id)
        if alert_id is not None:
            alert = next((a for a in alerts if a.id == alert_id), None)
            if alert is None:
                raise DomainError(f"alert '{alert_id}' not found on asset '{asset_id}'")
            return alert
        open_alerts = [
            a for a in alerts if a.status in (AlertStatus.ACTIVE, AlertStatus.INCONCLUSIVE)
        ]
        if not open_alerts:
            raise DomainError(f"asset '{asset_id}' has no open alert to diagnose")
        return max(open_alerts, key=lambda a: max_z(a.candidate))

    def diagnose(self, asset_id: str, alert_id: str | None = None) -> Receipt:
        llm = self._require_llm()
        record = self.monitoring.get(asset_id)
        alert = self._select_alert(asset_id, alert_id)
        latest = self.monitoring.latest(asset_id)
        others = tuple(
            a.candidate
            for a in self.monitoring.repo.list_alerts(asset_id)
            if a.id != alert.id and a.status != AlertStatus.DISCARDED
        )
        ctx = DiagnosticContext(
            asset_id=asset_id,
            spec=record.spec,
            candidate=alert.candidate,
            features=latest.features,
            capture=latest.capture,
            load_bucket=LoadBucket(alert.candidate.load_bucket),
            baselines=self.monitoring.baseline_snapshot(asset_id),
            detector_config=self.monitoring.detector.config,
            recapture=lambda duration: self.monitoring.recapture(asset_id, duration),
            other_candidates=others,
        )
        verdict = DiagnosisAgent(llm).run(ctx)
        work_order = (
            ActionAgent(llm).run(verdict, alert.candidate, record.spec, alert.id)
            if verdict.status == VerdictStatus.CONFIRMED
            else None
        )
        receipt = _receipt(alert, ctx, verdict, work_order)
        self.monitoring.record_diagnosis(
            alert.model_copy(
                update={
                    "status": _STATUS_FOR_VERDICT[verdict.status],
                    "receipt": receipt,
                    "discarded_at_z": max_z(alert.candidate)
                    if verdict.status == VerdictStatus.DISCARDED
                    else None,
                }
            )
        )
        logger.info(
            "diagnosed %s: %s (%.2f) in %d steps",
            alert.id,
            verdict.status.value,
            verdict.confidence,
            verdict.steps_used,
        )
        return receipt


def _receipt(alert: Alert, ctx: DiagnosticContext, verdict, work_order) -> Receipt:
    f = ctx.features
    limitations = [
        BASELINE_ASSUMPTION.format(n=ctx.detector_config.min_baseline_windows),
        *(
            f"{u.fault_class.value}"
            + (f" ({u.bearing_position})" if u.bearing_position else "")
            + f" not monitored: {u.reason}"
            for u in f.fault_map.unavailable
        ),
    ]
    if f.slip.source != "principal_slot_harmonic":
        limitations.append(
            "Slip was scaled from the nameplate by load, not measured from the slot harmonic; "
            "slip-dependent bins carry wider positional uncertainty"
        )
    ambiguous = sorted({k for e in alert.candidate.evidence for k in e.ambiguous_with})
    if ambiguous:
        limitations.append(
            "Some triggering bins cannot be separated at this resolution from: "
            + ", ".join(ambiguous)
        )
    return Receipt(
        alert_id=alert.id,
        asset_id=alert.asset_id,
        fault_class=alert.fault_class,
        bearing_position=alert.bearing_position,
        generated_at=datetime.now(UTC),
        predicted_bins=alert.candidate.evidence,
        operating_point=ReceiptOperatingPoint(
            load_factor=f.load.load_factor,
            load_factor_low=f.load.load_factor_low,
            load_factor_high=f.load.load_factor_high,
            load_bucket=alert.candidate.load_bucket,
            slip=f.slip.slip,
            rotor_speed_rpm=f.slip.rotor_speed_rpm,
            slip_source=f.slip.source,
            slip_confidence=f.slip.confidence,
        ),
        verdict=verdict,
        reading=verdict.reading,
        work_order=work_order,
        limitations=limitations,
    )
