"""End-to-end agent demonstration against the live Anthropic API.

Commission a motor, learn its baseline, inject a fault, let the Monitoring Agent raise an
alert, then run the Diagnosis Agent (and the Action Agent if confirmed) and print the receipt.

Usage (from the repository root, ANTHROPIC_API_KEY set in the environment or .env):
    python -m backend.scripts.demo_diagnosis [fault] [bearing_position]

Default: bearing_outer at DE. Runs in-process; no server needed.
"""

from __future__ import annotations

import json
import logging
import sys
import textwrap

from dotenv import load_dotenv

from backend.agents.llm import llm_from_env
from backend.api.diagnosis_service import DiagnosisService
from backend.api.repository import InMemoryRepository
from backend.api.schemas import AssetCreate, SimulationProfile
from backend.api.service import MonitoringService
from backend.api.settings import Settings
from backend.models.diagnosis import Receipt
from backend.scripts.verify_injection import DEMO_SPEC

BASELINE_WINDOWS = 10
FAULT_WINDOWS = 3
FAULT_SEVERITY = 0.6
WRAP = 96


def _print_receipt(receipt: Receipt) -> None:
    v = receipt.verdict
    print(f"\n=== RECEIPT {receipt.alert_id} ===")
    op = receipt.operating_point
    print(
        f"Operating point: load {op.load_factor:.0%} ({op.load_bucket}), slip {op.slip:.5f} "
        f"({op.slip_source}, confidence {op.slip_confidence:.2f})"
    )
    print("\nPredicted bins that triggered the alert:")
    for b in receipt.predicted_bins:
        print(
            f"  {b.label:18} {b.frequency_hz or 0:8.2f} Hz  {b.equation:34} measured "
            f"{b.measured_db:6.1f} dB  baseline {b.baseline_median_db:6.1f} dB  z={b.z_score:5.1f}"
        )
    print(f"\nEvidence chain ({v.steps_used}/{v.step_budget} diagnostics, model {v.model}):")
    for s in v.evidence:
        print(
            f"  [{s.index}] {s.tool}({json.dumps(s.arguments)})" + ("  ERROR" if s.is_error else "")
        )
        if s.rationale:
            print(textwrap.indent(textwrap.fill("why: " + s.rationale, WRAP), "      "))
        print(textwrap.indent(textwrap.fill("-> " + s.implication, WRAP), "      "))
    print(f"\nVerdict: {v.status.value.upper()} (confidence {v.confidence:.2f})")
    print(textwrap.fill("Summary: " + v.summary, WRAP))
    if v.discard_reason:
        print(textwrap.fill("Discarded because: " + v.discard_reason, WRAP))
    for note in v.guardrail_notes:
        print("Guardrail: " + note)
    print("\nReading:\n" + textwrap.indent(textwrap.fill(receipt.reading, WRAP), "  "))
    wo = receipt.work_order
    if wo is not None:
        print(f"\nWork order {wo.id}: severity {wo.severity.value} ({wo.severity_basis})")
        print(textwrap.fill("  Confirming test: " + wo.confirming_offline_test, WRAP))
        print("  Parts: " + (", ".join(wo.parts) or "none"))
        print("  Window: " + wo.recommended_window)
        for i, action in enumerate(wo.actions, 1):
            print(f"  {i}. {action}")
    print("\nLimitations:")
    for lim in receipt.limitations:
        print(textwrap.indent(textwrap.fill("- " + lim, WRAP - 2), "  "))


def main(fault: str, position: str) -> int:
    load_dotenv()
    logging.basicConfig(level=logging.WARNING)
    llm = llm_from_env()
    if llm is None:
        print("ANTHROPIC_API_KEY is not set (environment or .env).", file=sys.stderr)
        return 2
    monitoring = MonitoringService(InMemoryRepository(), Settings())
    diagnosis = DiagnosisService(monitoring, llm)
    asset_id = DEMO_SPEC.asset_id
    monitoring.commission(AssetCreate(spec=DEMO_SPEC, simulation=SimulationProfile()))
    print(f"Commissioned {asset_id}; learning baseline over {BASELINE_WINDOWS} windows...")
    monitoring.advance(asset_id, BASELINE_WINDOWS)
    injection = {"fault": fault, "severity": FAULT_SEVERITY}
    if fault.startswith("bearing"):
        injection["bearing_position"] = position
    monitoring.set_simulation(asset_id, SimulationProfile(faults=[injection]))
    print(f"Injected {injection}; monitoring {FAULT_WINDOWS} windows...")
    for result in monitoring.advance(asset_id, FAULT_WINDOWS):
        for alert in result.new_alerts:
            print(f"  window {result.window_index}: alert raised {alert.id}")
    alerts = monitoring.repo.list_alerts(asset_id)
    if not alerts:
        print("No alert raised; nothing to diagnose.", file=sys.stderr)
        return 1
    for alert in alerts:
        print(f"\nDiagnosing {alert.id} with {llm.model}...")
        _print_receipt(diagnosis.diagnose(asset_id, alert.id))
    return 0


if __name__ == "__main__":
    sys.exit(
        main(
            sys.argv[1] if len(sys.argv) > 1 else "bearing_outer",
            sys.argv[2] if len(sys.argv) > 2 else "DE",
        )
    )
