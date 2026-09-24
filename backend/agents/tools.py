"""Diagnostic tools available to the Diagnosis Agent.

Each tool is deterministic: it computes facts from the capture, the fault map and the
baseline, and returns them with an `implication` string that states what the facts mean
without deciding the verdict. The agent chooses which tool to run next and weighs the
results; it never computes a number itself.

Tool schemas are strict (additionalProperties false, every property required) so the model's
arguments are guaranteed to validate.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from backend.detection.baseline import BinBaseline, DetectorConfig, LoadBucket
from backend.detection.features import ThreePhaseCapture, WindowFeatures, analyze_window
from backend.models.alert import BinMeasurement, Candidate
from backend.models.errors import DomainError
from backend.models.fault_map import FaultClass
from backend.models.motor import MotorSpec
from backend.signal.envelope import envelope_spectrum
from backend.signal.spectrum import Spectrum, average_spectra
from backend.signal.symmetrical import (
    negative_sequence_admittance_from_nameplate,
    unbalance_net_of_supply,
)

REFINED_RESOLUTION_HZ: dict[int, float] = {32: 0.1, 64: 0.05}
ENVELOPE_RESOLUTION_HZ = 0.25
ENVELOPE_PRESENT_DB = 15.0  # line prominence above local envelope floor
ENVELOPE_FLOOR_HALFWIDTH_HZ = 5.0
ENVELOPE_FLOOR_GAP_HZ = 1.0
ELEVATED_Z = 3.0
BEARING_CHARACTERISTICS = {"BPFO": "outer race", "BPFI": "inner race", "BSF": "ball", "FTF": "cage"}
_FLOOR = 1e-30


@dataclass(frozen=True)
class DiagnosticContext:
    asset_id: str
    spec: MotorSpec
    candidate: Candidate
    features: WindowFeatures
    capture: ThreePhaseCapture
    load_bucket: LoadBucket
    baselines: Mapping[tuple[LoadBucket, str], BinBaseline]  # snapshot at diagnosis start
    detector_config: DetectorConfig
    recapture: Callable[[float], ThreePhaseCapture]
    other_candidates: tuple[Candidate, ...] = ()


@dataclass(frozen=True)
class ToolOutput:
    result: dict[str, Any]
    implication: str


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    run: Callable[..., ToolOutput]

    def definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "strict": True,
            "input_schema": self.input_schema,
        }


def _schema(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _baseline(ctx: DiagnosticContext, bucket: LoadBucket, key: str) -> BinBaseline:
    return ctx.baselines.get((bucket, key), BinBaseline())


def _baseline_stats(ctx: DiagnosticContext, m: BinMeasurement) -> dict[str, Any]:
    base = _baseline(ctx, ctx.load_bucket, m.key)
    if len(base.values) < ctx.detector_config.min_baseline_windows:
        return {"baseline_windows": len(base.values), "median_db": None, "mad_db": None, "z": None}
    mad = max(base.mad, ctx.detector_config.min_mad_db)
    return {
        "baseline_windows": len(base.values),
        "median_db": round(base.median, 2),
        "mad_db": round(base.mad, 3),
        "z": round((m.level_db - base.median) / (1.4826 * mad), 2),
    }


def _measurements(
    features: WindowFeatures, fault_class: FaultClass, position: str | None
) -> list[BinMeasurement]:
    return [
        m
        for m in features.measurements
        if m.fault_class == fault_class and (position is None or m.bearing_position == position)
    ]


# --- refine_slip --------------------------------------------------------------------------


def refine_slip(ctx: DiagnosticContext, duration_s: int) -> ToolOutput:
    resolution = REFINED_RESOLUTION_HZ[duration_s]
    refined = analyze_window(ctx.spec, ctx.recapture(float(duration_s)), resolution_hz=resolution)
    keys = {e.key for e in ctx.candidate.evidence}
    lines = []
    for m in refined.measurements:
        if m.key not in keys or m.frequency_hz is None:
            continue
        peak = refined.spectrum.tone(m.frequency_hz, 2 * refined.spectrum.resolution_hz)
        offset = peak.frequency_hz - m.frequency_hz
        lines.append(
            {
                "label": m.label,
                "predicted_hz": round(m.frequency_hz, 3),
                "peak_hz": round(peak.frequency_hz, 3),
                "offset_hz": round(offset, 3),
                "level_db": round(m.level_db, 1),
                "tracks_prediction": abs(offset) <= refined.spectrum.resolution_hz,
            }
        )
    window = ctx.features.slip
    tracking = sum(line["tracks_prediction"] for line in lines)
    result = {
        "window_slip": round(window.slip, 5),
        "window_slip_source": window.source,
        "refined_slip": round(refined.slip.slip, 5),
        "refined_slip_source": refined.slip.source,
        "refined_confidence": round(refined.slip.confidence, 2),
        "resolution_hz": refined.spectrum.resolution_hz,
        "candidate_lines": lines,
    }
    verb = "tracks" if lines and tracking == len(lines) else "does not fully track"
    implication = (
        f"Refined slip {refined.slip.slip:.5f} ({refined.slip.source}, confidence "
        f"{refined.slip.confidence:.2f}) vs window slip {window.slip:.5f}. "
        f"{tracking}/{len(lines)} candidate lines peak within one bin of the refined prediction "
        f"at {refined.spectrum.resolution_hz} Hz resolution: the energy {verb} the "
        "slip-dependent fault frequencies."
    )
    return ToolOutput(result, implication)


# --- envelope_analysis -----------------------------------------------------------------


def _envelope_prominence_db(spec: Spectrum, f: float) -> float:
    band = np.abs(spec.frequencies_hz - f)
    floor_mask = (band <= ENVELOPE_FLOOR_HALFWIDTH_HZ) & (band >= ENVELOPE_FLOOR_GAP_HZ)
    floor = float(np.median(spec.power[floor_mask]))
    line = spec.tone(f, 2 * spec.resolution_hz).power
    return 10 * math.log10(max(line, _FLOOR) / max(floor, _FLOOR))


def envelope_analysis(ctx: DiagnosticContext, bearing_position: str) -> ToolOutput:
    chars = {
        b.source: b.inputs["f_char"]
        for b in ctx.features.fault_map.bins
        if b.bearing_position == bearing_position
        and b.source in BEARING_CHARACTERISTICS
        and b.harmonic == 1
    }
    if not chars:
        raise DomainError(
            f"No bearing geometry at {bearing_position}: the bearing is unknown or not "
            "commissioned, so its characteristic frequencies cannot be computed"
        )
    env = average_spectra(
        [
            envelope_spectrum(phase, ctx.capture.sample_rate_hz, ENVELOPE_RESOLUTION_HZ)
            for phase in ctx.capture.currents
        ]
    )
    lines: dict[str, Any] = {}
    for name, f_char in chars.items():
        k1 = _envelope_prominence_db(env, f_char)
        k2 = _envelope_prominence_db(env, 2 * f_char)
        lines[name] = {
            "component": BEARING_CHARACTERISTICS[name],
            "f_char_hz": round(f_char, 3),
            "k1_prominence_db": round(k1, 1),
            "k2_prominence_db": round(k2, 1),
            "present": k1 >= ENVELOPE_PRESENT_DB,
        }
    present = [n for n, v in lines.items() if v["present"]]
    if present:
        detail = ", ".join(
            f"{n} ({lines[n]['component']}, {lines[n]['f_char_hz']} Hz, "
            f"+{lines[n]['k1_prominence_db']} dB, 2x: +{lines[n]['k2_prominence_db']} dB)"
            for n in present
        )
        implication = (
            f"Envelope of the {bearing_position} current shows modulation at {detail}: the "
            "current is amplitude-modulated at that bearing characteristic frequency."
        )
    else:
        strongest = max(lines, key=lambda n: lines[n]["k1_prominence_db"])
        implication = (
            f"No {bearing_position} bearing characteristic frequency stands out of the envelope "
            f"(threshold +{ENVELOPE_PRESENT_DB:g} dB; strongest {strongest} at "
            f"+{lines[strongest]['k1_prominence_db']} dB): no bearing modulation detected."
        )
    return ToolOutput(
        {
            "bearing_position": bearing_position,
            "threshold_db": ENVELOPE_PRESENT_DB,
            "characteristics": lines,
        },
        implication,
    )


# --- check_sideband_symmetry -----------------------------------------------------------


def check_sideband_symmetry(ctx: DiagnosticContext, fault_class: str) -> ToolOutput:
    fc = FaultClass(fault_class)
    k1 = {m.label.split()[1]: m for m in _measurements(ctx.features, fc, None) if "k=1" in m.label}
    if not {"lower", "upper"} <= set(k1):
        raise DomainError(
            f"k=1 sidebands for {fc.value} are not both measurable in this window "
            "(excluded as unresolvable from the fundamental)"
        )
    lower, upper = k1["lower"], k1["upper"]
    asym = lower.level_db - upper.level_db
    result = {
        "fault_class": fc.value,
        "lower": {
            "hz": round(lower.frequency_hz or 0, 3),
            "level_db": round(lower.level_db, 1),
            **_baseline_stats(ctx, lower),
        },
        "upper": {
            "hz": round(upper.frequency_hz or 0, 3),
            "level_db": round(upper.level_db, 1),
            **_baseline_stats(ctx, upper),
        },
        "asymmetry_db": round(asym, 1),
    }
    implication = (
        f"{fc.value} k=1 sidebands: lower {lower.level_db:.1f} dB, upper {upper.level_db:.1f} dB "
        f"re fundamental; lower minus upper = {asym:+.1f} dB."
    )
    return ToolOutput(result, implication)


# --- check_phase_balance ---------------------------------------------------------------


def check_phase_balance(ctx: DiagnosticContext) -> ToolOutput:
    admittance = (
        negative_sequence_admittance_from_nameplate(ctx.spec)
        if ctx.spec.locked_rotor_current_ratio is not None
        else None
    )
    u = unbalance_net_of_supply(
        ctx.capture.currents,
        ctx.capture.voltages,
        ctx.capture.sample_rate_hz,
        ctx.spec.supply_frequency_hz,
        admittance,
    )
    result = {
        "raw_current_unbalance_pct": round(100 * u.raw_current_unbalance, 2),
        "voltage_unbalance_pct": round(100 * u.voltage_unbalance, 2),
        "supply_attributable_pct": None
        if u.supply_attributable_unbalance is None
        else round(100 * u.supply_attributable_unbalance, 2),
        "net_current_unbalance_pct": None
        if u.net_current_unbalance is None
        else round(100 * u.net_current_unbalance, 2),
        "method": u.method,
    }
    if u.net_current_unbalance is None:
        implication = (
            f"Raw current unbalance {result['raw_current_unbalance_pct']}% with voltage unbalance "
            f"{result['voltage_unbalance_pct']}%; supply share cannot be removed ({u.note})."
        )
    else:
        implication = (
            f"Current unbalance {result['raw_current_unbalance_pct']}% raw, of which "
            f"{result['supply_attributable_pct']}% is explained by "
            f"{result['voltage_unbalance_pct']}% supply voltage unbalance; net "
            f"{result['net_current_unbalance_pct']}% is attributable to the machine ({u.method})."
        )
    return ToolOutput(result, implication)


# --- check_harmonic_order --------------------------------------------------------------


def check_harmonic_order(
    ctx: DiagnosticContext, fault_class: str, bearing_position: str
) -> ToolOutput:
    fc = FaultClass(fault_class)
    position = None if bearing_position == "none" else bearing_position
    ms = _measurements(ctx.features, fc, position)
    if not ms:
        raise DomainError(f"no measurable {fc.value} bins at position {bearing_position}")
    rows = []
    for m in sorted(ms, key=lambda m: m.label):
        stats = _baseline_stats(ctx, m)
        rows.append(
            {
                "label": m.label,
                "hz": None if m.frequency_hz is None else round(m.frequency_hz, 3),
                "level_db": round(m.level_db, 1),
                **stats,
                "elevated": stats["z"] is not None and stats["z"] > ELEVATED_Z,
            }
        )
    harmonic_of = {r["label"]: int(r["label"].split("k=")[1]) for r in rows if "k=" in r["label"]}
    by_k: dict[int, list[dict]] = {}
    for r in rows:
        if r["label"] in harmonic_of:
            by_k.setdefault(harmonic_of[r["label"]], []).append(r)
    elevated_orders = sorted(k for k, rs in by_k.items() if any(r["elevated"] for r in rs))
    mean_level = {k: float(np.mean([r["level_db"] for r in rs])) for k, rs in by_k.items()}
    decreasing = all(mean_level[a] >= mean_level[b] for a, b in pairwise(sorted(mean_level)))
    consistent = 1 in elevated_orders and len(elevated_orders) >= 2 and decreasing
    implication = (
        f"Orders elevated above baseline (z>{ELEVATED_Z:g}): {elevated_orders or 'none'}; mean "
        f"level by order {', '.join(f'k={k}: {v:.1f} dB' for k, v in sorted(mean_level.items()))}. "
        + (
            "Higher orders are present and weaker than k=1, as a genuine fault series would be."
            if consistent
            else "The harmonic series is not the pattern a genuine fault series would produce."
        )
    )
    return ToolOutput(
        {
            "fault_class": fc.value,
            "bins": rows,
            "elevated_orders": elevated_orders,
            "levels_decrease_with_order": decreasing,
            "series_consistent": consistent,
        },
        implication,
    )


# --- load_matched_compare ---------------------------------------------------------------


def load_matched_compare(ctx: DiagnosticContext) -> ToolOutput:
    keys = {e.key for e in ctx.candidate.evidence}
    rows = []
    for m in ctx.features.measurements:
        if m.key not in keys:
            continue
        other_buckets = {}
        for bucket in LoadBucket:
            if bucket == ctx.load_bucket:
                continue
            base = _baseline(ctx, bucket, m.key)
            if base.values:
                other_buckets[bucket.value] = {
                    "windows": len(base.values),
                    "median_db": round(base.median, 2),
                }
        rows.append(
            {
                "label": m.label,
                "level_db": round(m.level_db, 1),
                **_baseline_stats(ctx, m),
                "other_load_buckets": other_buckets,
            }
        )
    counts = [r["baseline_windows"] for r in rows]
    deltas = [r["level_db"] - r["median_db"] for r in rows if r["median_db"] is not None]
    implication = (
        f"Against {min(counts, default=0)}-{max(counts, default=0)} historical windows in the "
        f"same load bucket ({ctx.load_bucket.value}), candidate lines sit "
        f"{min(deltas, default=0):+.1f} to {max(deltas, default=0):+.1f} dB from their "
        "load-matched median."
    )
    return ToolOutput(
        {
            "load_bucket": ctx.load_bucket.value,
            "load_factor": round(ctx.candidate.load_factor, 3),
            "bins": rows,
        },
        implication,
    )


_SPECTRAL_CLASSES = [c.value for c in FaultClass if c != FaultClass.STATOR_WINDING]

DIAGNOSTIC_TOOLS: dict[str, ToolSpec] = {
    t.name: t
    for t in (
        ToolSpec(
            "refine_slip",
            "Capture a longer record and re-estimate slip from the principal slot harmonic at "
            "finer resolution, then re-locate the candidate's lines. Tests whether the energy "
            "sits on the slip-dependent fault frequencies.",
            _schema({"duration_s": {"type": "integer", "enum": [32, 64]}}),
            refine_slip,
        ),
        ToolSpec(
            "envelope_analysis",
            "Hilbert envelope spectrum of the phase currents, checked at BPFO, BPFI, BSF and FTF "
            "of the bearing at the given position. Use for bearing hypotheses.",
            _schema({"bearing_position": {"type": "string", "enum": ["DE", "NDE"]}}),
            envelope_analysis,
        ),
        ToolSpec(
            "check_sideband_symmetry",
            "Compare the k=1 lower and upper sideband levels of a rotor-bar or eccentricity "
            "series, with their load-matched baselines.",
            _schema(
                {"fault_class": {"type": "string", "enum": ["broken_rotor_bar", "eccentricity"]}}
            ),
            check_sideband_symmetry,
        ),
        ToolSpec(
            "check_phase_balance",
            "Negative-sequence current unbalance, net of the share explained by supply voltage "
            "unbalance. Use for stator-winding hypotheses and to rule out supply effects.",
            _schema({}),
            check_phase_balance,
        ),
        ToolSpec(
            "check_harmonic_order",
            "Check whether the higher-order (k=2,3) bins of a fault series are elevated and "
            "weaker than k=1, as a genuine fault series would be.",
            _schema(
                {
                    "fault_class": {"type": "string", "enum": _SPECTRAL_CLASSES},
                    "bearing_position": {"type": "string", "enum": ["DE", "NDE", "none"]},
                }
            ),
            check_harmonic_order,
        ),
        ToolSpec(
            "load_matched_compare",
            "Compare the candidate's lines with historical windows in the same load bucket, and "
            "show their medians at other loads.",
            _schema({}),
            load_matched_compare,
        ),
    )
}


# --- which results independently support a fault class ----------------------------------

NET_UNBALANCE_SUPPORT_PCT = 1.0
_BEARING_CHAR_FOR_CLASS = {
    FaultClass.BEARING_OUTER: "BPFO",
    FaultClass.BEARING_INNER: "BPFI",
    FaultClass.BEARING_BALL: "BSF",
    FaultClass.BEARING_CAGE: "FTF",
}


def supports_fault(
    fault_class: FaultClass,
    bearing_position: str | None,
    tool: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
) -> bool:
    """True when a diagnostic's deterministic result independently supports the fault.

    load_matched_compare never counts: it re-reads the bins that raised the candidate.
    """
    if fault_class in _BEARING_CHAR_FOR_CLASS:
        char = _BEARING_CHAR_FOR_CLASS[fault_class]
        return (
            tool == "envelope_analysis"
            and arguments.get("bearing_position") == bearing_position
            and bool(result.get("characteristics", {}).get(char, {}).get("present"))
        )
    if fault_class in (FaultClass.BROKEN_ROTOR_BAR, FaultClass.ECCENTRICITY):
        if tool == "check_harmonic_order":
            return result.get("fault_class") == fault_class.value and bool(
                result.get("series_consistent")
            )
        if tool == "refine_slip":
            lines = result.get("candidate_lines", [])
            return bool(lines) and all(line["tracks_prediction"] for line in lines)
        return False
    if fault_class == FaultClass.STATOR_WINDING:
        net = result.get("net_current_unbalance_pct")
        return (
            tool == "check_phase_balance" and net is not None and net >= NET_UNBALANCE_SUPPORT_PCT
        )
    return False


def _validate_arguments(spec: ToolSpec, arguments: dict[str, Any]) -> None:
    """Enforce the tool's own schema. Strict mode guarantees this at the API, but the tools
    must not rely on the caller: an out-of-enum class would silently measure the wrong bins."""
    properties = spec.input_schema["properties"]
    unexpected = set(arguments) - set(properties)
    missing = set(properties) - set(arguments)
    if unexpected or missing:
        raise DomainError(
            f"{spec.name} arguments invalid (unexpected {sorted(unexpected)}, "
            f"missing {sorted(missing)})"
        )
    for key, value in arguments.items():
        allowed = properties[key].get("enum")
        if allowed is not None and value not in allowed:
            raise DomainError(f"{spec.name}: {key}={value!r} not one of {allowed}")


def run_tool(ctx: DiagnosticContext, name: str, arguments: dict[str, Any]) -> ToolOutput:
    """Execute a diagnostic by name. Raises KeyError for unknown tools, DomainError on misuse."""
    spec = DIAGNOSTIC_TOOLS[name]
    _validate_arguments(spec, arguments)
    return spec.run(ctx, **arguments)
