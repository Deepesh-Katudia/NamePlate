"""Agent-layer tests. The LLM is always a scripted fake; tools run on simulated data."""

from __future__ import annotations

import pytest

from backend.agents.action import ActionAgent, assess_severity
from backend.agents.commissioning import (
    CommissioningAgent,
    ExtractedBearing,
    NameplateExtraction,
    UncertainField,
)
from backend.agents.diagnosis import STEP_BUDGET, DiagnosisAgent
from backend.agents.llm import AgentUnavailableError
from backend.agents.tools import DiagnosticContext, run_tool
from backend.api.repository import InMemoryRepository
from backend.api.schemas import AssetCreate, SimulationProfile
from backend.api.service import MonitoringService
from backend.api.settings import Settings
from backend.detection.baseline import DetectorConfig, LoadBucket
from backend.models.diagnosis import Severity, VerdictStatus, WorkOrderDraft
from backend.models.errors import DomainError
from backend.models.fault_map import FaultClass
from backend.tests.fakes import ScriptedLLM, calls, final_text, tool_call, verdict
from backend.tests.test_physics import make_spec

FAST = Settings(window_duration_s=8.0, sample_rate_hz=4000.0)
FAST_DETECTOR = DetectorConfig(min_baseline_windows=4, consecutive_windows=3)
BASELINE_WINDOWS = 5
FAULT_WINDOWS = 3


def build_context(
    faults=(), confounders=(), fault_class: FaultClass | None = None, **spec_overrides
) -> tuple[MonitoringService, DiagnosticContext]:
    """Commission, learn a baseline, inject, and return a context for the raised alert."""
    service = MonitoringService(InMemoryRepository(), FAST, FAST_DETECTOR)
    spec = make_spec(locked_rotor_current_ratio=6.5, **spec_overrides)
    service.commission(AssetCreate(spec=spec, simulation=SimulationProfile(load_jitter=0.0)))
    service.advance(spec.asset_id, BASELINE_WINDOWS)
    service.set_simulation(
        spec.asset_id,
        SimulationProfile(load_jitter=0.0, faults=list(faults), confounders=list(confounders)),
    )
    service.advance(spec.asset_id, FAULT_WINDOWS)
    alerts = service.repo.list_alerts(spec.asset_id)
    alert = next((a for a in alerts if fault_class is None or a.fault_class == fault_class), None)
    latest = service.latest(spec.asset_id)
    candidate = alert.candidate if alert else None
    ctx = DiagnosticContext(
        asset_id=spec.asset_id,
        spec=spec,
        candidate=candidate,  # type: ignore[arg-type]
        features=latest.features,
        capture=latest.capture,
        load_bucket=LoadBucket(candidate.load_bucket) if candidate else LoadBucket.ABOVE_75,
        baselines=service.baseline_snapshot(spec.asset_id),
        detector_config=service.detector.config,
        recapture=lambda d: service.recapture(spec.asset_id, d),
    )
    return service, ctx


@pytest.fixture(scope="module")
def outer_race_ctx():
    return build_context(
        faults=[{"fault": "bearing_outer", "severity": 0.6, "bearing_position": "DE"}],
        fault_class=FaultClass.BEARING_OUTER,
    )[1]


@pytest.fixture(scope="module")
def rotor_bar_ctx():
    return build_context(
        faults=[{"fault": "broken_rotor_bar", "severity": 0.6}],
        fault_class=FaultClass.BROKEN_ROTOR_BAR,
    )[1]


# --- tools ------------------------------------------------------------------------------


class TestTools:
    def test_envelope_finds_outer_race_modulation(self, outer_race_ctx):
        out = run_tool(outer_race_ctx, "envelope_analysis", {"bearing_position": "DE"})
        chars = out.result["characteristics"]
        assert chars["BPFO"]["present"] is True
        assert not chars["BPFI"]["present"]
        assert "BPFO" in out.implication

    def test_envelope_quiet_on_healthy_bearing(self, rotor_bar_ctx):
        out = run_tool(rotor_bar_ctx, "envelope_analysis", {"bearing_position": "DE"})
        assert not any(c["present"] for c in out.result["characteristics"].values())
        assert "No DE bearing" in out.implication

    def test_envelope_without_bearing_geometry_is_domain_error(self):
        _, ctx = build_context(
            faults=[{"fault": "broken_rotor_bar", "severity": 0.6}],
            bearings=[],
            fault_class=FaultClass.BROKEN_ROTOR_BAR,
        )
        with pytest.raises(DomainError, match="No bearing geometry"):
            run_tool(ctx, "envelope_analysis", {"bearing_position": "DE"})

    def test_refine_slip_tracks_rotor_bar_lines(self, rotor_bar_ctx):
        out = run_tool(rotor_bar_ctx, "refine_slip", {"duration_s": 32})
        assert out.result["refined_slip_source"] == "principal_slot_harmonic"
        assert out.result["refined_slip"] == pytest.approx(0.8 * make_spec().rated_slip, abs=2e-3)
        assert out.result["candidate_lines"]
        assert all(line["tracks_prediction"] for line in out.result["candidate_lines"])

    def test_harmonic_order_consistent_for_rotor_bar_series(self, rotor_bar_ctx):
        out = run_tool(
            rotor_bar_ctx,
            "check_harmonic_order",
            {"fault_class": "broken_rotor_bar", "bearing_position": "none"},
        )
        assert out.result["series_consistent"] is True
        assert out.result["elevated_orders"][0] == 1

    def test_sideband_symmetry_reports_lower_minus_upper(self, rotor_bar_ctx):
        out = run_tool(
            rotor_bar_ctx, "check_sideband_symmetry", {"fault_class": "broken_rotor_bar"}
        )
        # the simulator places the upper sideband 2 dB below the lower
        assert out.result["asymmetry_db"] == pytest.approx(2.0, abs=1.0)

    def test_phase_balance_nets_out_supply_unbalance(self):
        _, ctx = build_context(
            faults=[{"fault": "broken_rotor_bar", "severity": 0.6}],
            confounders=[{"kind": "supply_voltage_unbalance", "voltage_unbalance": 0.02}],
            fault_class=FaultClass.BROKEN_ROTOR_BAR,
        )
        out = run_tool(ctx, "check_phase_balance", {})
        assert out.result["raw_current_unbalance_pct"] > 8.0
        assert out.result["net_current_unbalance_pct"] < 1.0

    def test_load_matched_compare_uses_same_bucket_history(self, rotor_bar_ctx):
        out = run_tool(rotor_bar_ctx, "load_matched_compare", {})
        assert out.result["load_bucket"] == LoadBucket.ABOVE_75.value
        primary = next(b for b in out.result["bins"] if b["label"] == "BRB lower k=1")
        assert primary["baseline_windows"] >= FAST_DETECTOR.min_baseline_windows
        assert primary["level_db"] - primary["median_db"] > 20


# --- diagnosis agent -----------------------------------------------------------------------


class TestDiagnosisAgent:
    def test_selects_envelope_and_confirms_with_evidence_chain(self, outer_race_ctx):
        llm = ScriptedLLM(
            [
                tool_call(
                    "envelope_analysis",
                    {"bearing_position": "DE"},
                    "Testing for outer-race modulation at the DE bearing.",
                ),
                verdict("confirmed", 0.9),
            ]
        )
        v = DiagnosisAgent(llm).run(outer_race_ctx)
        assert v.status == VerdictStatus.CONFIRMED
        assert [s.tool for s in v.evidence] == ["envelope_analysis"]
        step = v.evidence[0]
        assert step.rationale.startswith("Testing for outer-race")
        assert step.result["characteristics"]["BPFO"]["present"]
        assert "BPFO" in step.implication
        # the tool result went back to the model with the implication attached
        sent = llm.tool_results(1)[0]
        assert "implication" in sent["content"] and not sent.get("is_error")
        # every request carries strict tools and adaptive thinking
        req = llm.create_calls[0]
        assert req["thinking"] == {"type": "adaptive"}
        assert all(t.get("strict") for t in req["tools"])

    def test_step_budget_enforced(self, rotor_bar_ctx):
        script = [tool_call("load_matched_compare", {}) for _ in range(STEP_BUDGET + 2)]
        script += [final_text("done"), final_text("still done")]
        llm = ScriptedLLM(script)
        v = DiagnosisAgent(llm).run(rotor_bar_ctx)
        assert v.steps_used == STEP_BUDGET
        assert v.status == VerdictStatus.INCONCLUSIVE
        assert "No verdict submitted" in " ".join(v.guardrail_notes)
        over_budget = llm.tool_results(STEP_BUDGET + 1)[0]
        assert over_budget["is_error"] and "budget" in over_budget["content"]

    def test_confirmation_without_diagnostics_is_downgraded(self, rotor_bar_ctx):
        v = DiagnosisAgent(ScriptedLLM([verdict("confirmed", 0.95)])).run(rotor_bar_ctx)
        assert v.status == VerdictStatus.INCONCLUSIVE
        assert v.confidence <= 0.3
        assert any("downgraded" in n for n in v.guardrail_notes)

    def test_confirmation_backed_only_by_irrelevant_diagnostic_is_downgraded(self, outer_race_ctx):
        # a successful but unrelated diagnostic must not unlock a confirmation
        llm = ScriptedLLM([tool_call("check_phase_balance", {}), verdict("confirmed", 0.95)])
        v = DiagnosisAgent(llm).run(outer_race_ctx)
        assert v.status == VerdictStatus.INCONCLUSIVE
        assert any("independent" in n for n in v.guardrail_notes)

    def test_load_matched_compare_alone_cannot_confirm(self, rotor_bar_ctx):
        llm = ScriptedLLM([tool_call("load_matched_compare", {}), verdict("confirmed", 0.9)])
        assert DiagnosisAgent(llm).run(rotor_bar_ctx).status == VerdictStatus.INCONCLUSIVE

    def test_envelope_at_wrong_position_cannot_confirm(self, outer_race_ctx):
        llm = ScriptedLLM(
            [tool_call("envelope_analysis", {"bearing_position": "NDE"}), verdict("confirmed")]
        )
        assert DiagnosisAgent(llm).run(outer_race_ctx).status == VerdictStatus.INCONCLUSIVE

    def test_pause_turn_is_resumed_without_a_nudge(self, rotor_bar_ctx):
        paused = final_text("thinking...")
        paused.stop_reason = "pause_turn"
        llm = ScriptedLLM(
            [
                paused,
                tool_call(
                    "check_harmonic_order",
                    {"fault_class": "broken_rotor_bar", "bearing_position": "none"},
                ),
                verdict("confirmed"),
            ]
        )
        v = DiagnosisAgent(llm).run(rotor_bar_ctx)
        assert v.status == VerdictStatus.CONFIRMED
        resumed = llm.create_calls[1]["messages"]
        assert resumed[-1]["role"] == "assistant"  # sent back as-is, no user text inserted

    def test_baseline_snapshot_is_isolated_from_later_windows(self):
        service, ctx = build_context(
            faults=[{"fault": "broken_rotor_bar", "severity": 0.6}],
            fault_class=FaultClass.BROKEN_ROTOR_BAR,
        )
        before = run_tool(ctx, "load_matched_compare", {}).result
        service.set_simulation(ctx.asset_id, SimulationProfile(load_jitter=0.0))
        service.advance(ctx.asset_id, 3)  # healthy windows extend the live baseline
        assert run_tool(ctx, "load_matched_compare", {}).result == before

    def test_invalid_verdict_is_rejected_then_retried(self, rotor_bar_ctx):
        bad = verdict("confirmed", 1.7)
        llm = ScriptedLLM(
            [
                tool_call(
                    "check_harmonic_order",
                    {"fault_class": "broken_rotor_bar", "bearing_position": "none"},
                ),
                bad,
                verdict("confirmed", 0.8),
            ]
        )
        v = DiagnosisAgent(llm).run(rotor_bar_ctx)
        assert v.status == VerdictStatus.CONFIRMED and v.confidence == 0.8
        assert llm.tool_results(2)[0]["is_error"]

    def test_unknown_tool_and_misuse_become_error_steps(self, rotor_bar_ctx):
        llm = ScriptedLLM(
            [
                calls(
                    tool_call("read_the_manual", {}),
                    tool_call("check_sideband_symmetry", {"fault_class": "bearing_outer"}),
                ),
                verdict("inconclusive", 0.2),
            ]
        )
        v = DiagnosisAgent(llm).run(rotor_bar_ctx)
        assert [s.is_error for s in v.evidence] == [True, True]
        assert all(r["is_error"] for r in llm.tool_results(1))

    def test_discard_without_reason_uses_summary(self, rotor_bar_ctx):
        llm = ScriptedLLM([tool_call("load_matched_compare", {}), verdict("discarded", 0.7)])
        v = DiagnosisAgent(llm).run(rotor_bar_ctx)
        assert v.status == VerdictStatus.DISCARDED
        assert v.discard_reason == "discarded after diagnostics"
        assert any("reason" in n for n in v.guardrail_notes)


# --- action agent --------------------------------------------------------------------------

DRAFT = WorkOrderDraft(
    confirming_offline_test="Single-phase rotation test",
    parts=["Rotor or rewind"],
    recommended_window="Next planned outage",
    actions=["Isolate", "Test", "Repair"],
    safety_notes=["Lock out"],
)


class TestActionAgent:
    def _confirmed(self, ctx):
        llm = ScriptedLLM(
            [
                tool_call(
                    "check_harmonic_order",
                    {"fault_class": "broken_rotor_bar", "bearing_position": "none"},
                ),
                verdict("confirmed"),
            ]
        )
        return DiagnosisAgent(llm).run(ctx)

    def test_work_order_copies_identity_and_computes_severity(self, rotor_bar_ctx):
        v = self._confirmed(rotor_bar_ctx)
        llm = ScriptedLLM(parsed=DRAFT)
        wo = ActionAgent(llm).run(v, rotor_bar_ctx.candidate, rotor_bar_ctx.spec, "A-1")
        assert (wo.asset_id, wo.alert_id, wo.fault_class) == (
            rotor_bar_ctx.spec.asset_id,
            "A-1",
            FaultClass.BROKEN_ROTOR_BAR,
        )
        assert wo.severity == assess_severity(rotor_bar_ctx.candidate)[0]
        assert "Thomson" in wo.severity_basis
        assert llm.parse_calls[0]["output_format"] is WorkOrderDraft
        assert wo.confirming_offline_test == DRAFT.confirming_offline_test

    def test_refuses_unconfirmed_verdict(self, rotor_bar_ctx):
        v = DiagnosisAgent(ScriptedLLM([verdict("inconclusive", 0.3)])).run(rotor_bar_ctx)
        with pytest.raises(DomainError, match="confirmed"):
            ActionAgent(ScriptedLLM(parsed=DRAFT)).run(
                v, rotor_bar_ctx.candidate, rotor_bar_ctx.spec, "A-1"
            )

    def test_bearing_severity_from_line_rise(self, outer_race_ctx):
        severity, basis = assess_severity(outer_race_ctx.candidate)
        assert severity in (Severity.HIGH, Severity.CRITICAL)
        assert "above its load-matched baseline" in basis


# --- commissioning agent -------------------------------------------------------------------


def _extraction(**overrides) -> NameplateExtraction:
    base = dict(
        asset_id="P-9",
        rated_power_kw=15.0,
        rated_voltage_v=460.0,
        rated_current_a=22.0,
        supply_frequency_hz=60.0,
        poles=4,
        rated_speed_rpm=1750.0,
        rated_efficiency=0.91,
        locked_rotor_current_ratio=None,
        rotor_slots=None,
        bearings=[ExtractedBearing(position="DE", designation="6205-2Z")],
        uncertain_fields=[UncertainField(field="poles", note="inferred from 1750 rpm at 60 Hz")],
    )
    base.update(overrides)
    return NameplateExtraction(**base)


class TestCommissioningAgent:
    def test_structured_input_needs_no_llm(self):
        result = CommissioningAgent(None).from_structured(
            make_spec(bearings=[], rotor_slots=None).model_dump(mode="json")
        )
        assert result.status == "ready" and not result.used_llm
        params = {c.parameter for c in result.needs_confirmation}
        assert "rotor_slots" in params and "bearing_designation[DE]" in params
        assert FaultClass.BEARING_OUTER in {u.fault_class for u in result.unavailable}

    def test_structured_invalid_reports_physical_reason(self):
        data = make_spec().model_dump(mode="json")
        data["poles"] = 5
        result = CommissioningAgent(None).from_structured(data)
        assert result.status == "invalid"
        assert any("even" in e for e in result.validation_errors)

    def test_free_text_complete_nameplate_with_inferred_field(self):
        llm = ScriptedLLM(parsed=_extraction())
        result = CommissioningAgent(llm).from_text("15 kW 460 V 22 A 1750 rpm ...")
        assert result.status == "ready" and result.used_llm
        assert result.spec is not None and result.spec.poles == 4
        poles_item = next(c for c in result.needs_confirmation if c.parameter == "poles")
        assert "inferred" in poles_item.reason
        # the missing optional fields surface too, never defaulted
        assert "locked_rotor_current_ratio" in {c.parameter for c in result.needs_confirmation}

    def test_free_text_missing_required_field_produces_no_spec(self):
        llm = ScriptedLLM(parsed=_extraction(rated_speed_rpm=None))
        result = CommissioningAgent(llm).from_text("15 kW motor, speed illegible")
        assert result.status == "needs_input"
        assert result.spec is None
        assert result.missing_required == ["rated_speed_rpm"]

    def test_free_text_without_llm_is_unavailable(self):
        with pytest.raises(AgentUnavailableError):
            CommissioningAgent(None).from_text("15 kW motor")
