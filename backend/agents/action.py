"""Action Agent: confirmed verdict -> work order.

Bounded generation. Identity fields (asset, alert, fault, bearing position) are copied from the
verdict, and severity is computed deterministically from the evidence, so the model cannot
change what was diagnosed or how urgent it is. The model writes only the practical content a
technician needs: confirming offline test, parts, window, actions, safety notes.
"""

from __future__ import annotations

from datetime import UTC, datetime

from backend.agents.llm import LLMClient
from backend.models.alert import Candidate
from backend.models.diagnosis import Severity, Verdict, VerdictStatus, WorkOrder, WorkOrderDraft
from backend.models.errors import DomainError
from backend.models.fault_map import FaultClass
from backend.models.motor import MotorSpec
from backend.physics.severity import RotorBarSeverity, classify_rotor_bar

MAX_TOKENS = 4_000
MAX_LIST_ITEMS = 10

# Illustrative, site-tunable tiers for classes without a published amplitude standard.
# Rise of the strongest bin above its load-matched baseline median, in dB.
LINE_RISE_TIERS_DB: tuple[tuple[float, Severity], ...] = (
    (20.0, Severity.CRITICAL),
    (12.0, Severity.HIGH),
    (6.0, Severity.MEDIUM),
    (0.0, Severity.LOW),
)
# Net negative-sequence current unbalance (level in dB re I1) for stator winding faults.
STATOR_TIERS_DB: tuple[tuple[float, Severity], ...] = (
    (-26.0, Severity.CRITICAL),  # >= 5 %
    (-34.0, Severity.HIGH),  # >= 2 %
    (-80.0, Severity.MEDIUM),
)

_ROTOR_BAR_TO_SEVERITY: dict[RotorBarSeverity, Severity] = {
    RotorBarSeverity.EXCELLENT: Severity.LOW,
    RotorBarSeverity.GOOD: Severity.LOW,
    RotorBarSeverity.MODERATE: Severity.MEDIUM,
    RotorBarSeverity.CRACKED_BAR_OR_HIGH_RESISTANCE_JOINT: Severity.HIGH,
    RotorBarSeverity.BROKEN_BARS: Severity.CRITICAL,
    RotorBarSeverity.MULTIPLE_BROKEN_BARS: Severity.CRITICAL,
    RotorBarSeverity.INSUFFICIENT_LOAD: Severity.MEDIUM,
}

SYSTEM_PROMPT = """\
You write maintenance work orders for industrial induction motors from a confirmed diagnosis. \
The fault, its location and its severity are already decided; do not restate or change them. \
Write only: the offline test that would confirm the fault before intervention (for example \
rotor-bar: single-phase rotation test or growler; bearing: vibration envelope measurement at \
the named position; stator: surge test or insulation resistance and winding resistance \
balance), the parts likely needed (use the bearing designation given, if any), a recommended \
window proportionate to the severity, ordered actions, and safety notes including isolation \
and lock-out. Be concise and specific; no markdown."""


def _tiered(value: float, tiers: tuple[tuple[float, Severity], ...]) -> Severity:
    return next(sev for threshold, sev in tiers if value >= threshold)


def assess_severity(candidate: Candidate) -> tuple[Severity, str]:
    """Deterministic severity from the candidate's evidence; the model never sets it."""
    fc = candidate.fault_class
    if fc == FaultClass.BROKEN_ROTOR_BAR:
        primary = next((e for e in candidate.evidence if e.label == "BRB lower k=1"), None)
        if primary is not None:
            band = classify_rotor_bar(min(primary.measured_db, 0.0), candidate.load_factor)
            return _ROTOR_BAR_TO_SEVERITY[band.state], (
                f"Rotor-bar band '{band.state.value}' (lower sideband {primary.measured_db:.1f} dB "
                f"re fundamental at {candidate.load_factor:.0%} load; Thomson & Fenger bands)"
            )
    if fc == FaultClass.STATOR_WINDING:
        level = candidate.evidence[0].measured_db
        return _tiered(level, STATOR_TIERS_DB), (
            f"Net negative-sequence current {10 ** (level / 20):.1%} of positive sequence"
        )
    rise = max(e.measured_db - e.baseline_median_db for e in candidate.evidence)
    return _tiered(rise, LINE_RISE_TIERS_DB), (
        f"Strongest line {rise:.1f} dB above its load-matched baseline"
    )


class ActionAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(
        self, verdict: Verdict, candidate: Candidate, spec: MotorSpec, alert_id: str
    ) -> WorkOrder:
        if verdict.status != VerdictStatus.CONFIRMED:
            raise DomainError(
                f"work orders are issued only for confirmed verdicts (got {verdict.status.value})"
            )
        severity, basis = assess_severity(candidate)
        bearing = next(
            (b.designation for b in spec.bearings if b.position.value == verdict.bearing_position),
            None,
        )
        prompt = (
            f"Asset {spec.asset_id}: {spec.rated_power_kw} kW, {spec.poles}-pole, "
            f"{spec.rated_voltage_v} V, {spec.supply_frequency_hz} Hz induction motor.\n"
            f"Confirmed fault: {verdict.fault_class.value}"
            + (f" at the {verdict.bearing_position} bearing ({bearing})" if bearing else "")
            + f".\nSeverity: {severity.value} ({basis}).\n"
            f"Diagnosis summary: {verdict.summary}\nReading: {verdict.reading}"
        )
        response = self.llm.parse(
            output_format=WorkOrderDraft,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        draft: WorkOrderDraft = response.parsed_output
        if draft is None:
            raise DomainError("Action Agent returned no work order draft")
        return WorkOrder(
            id=f"WO-{alert_id}",
            asset_id=spec.asset_id,
            alert_id=alert_id,
            fault_class=verdict.fault_class,
            bearing_position=verdict.bearing_position,
            severity=severity,
            severity_basis=basis,
            confirming_offline_test=draft.confirming_offline_test,
            parts=draft.parts[:MAX_LIST_ITEMS],
            recommended_window=draft.recommended_window,
            actions=draft.actions[:MAX_LIST_ITEMS],
            safety_notes=draft.safety_notes[:MAX_LIST_ITEMS],
            created_at=datetime.now(UTC),
            model=self.llm.model,
        )
