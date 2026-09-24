"""Per-window feature extraction: from a three-phase capture to per-bin levels.

This is where signal meets physics. For each window:
1. Estimate load (energy) and slip (PSH if credible, else load-scaled nameplate, labelled).
2. Rebuild the fault map at that operating speed, so bins follow slip as load changes.
3. Measure each bin's line power relative to the fundamental, in dB.
4. Add the stator-winding feature: net-of-supply current unbalance, if it can be computed.

Bins that the achievable resolution cannot separate from the fundamental are excluded; bins
from different fault classes that cannot be separated from each other are marked ambiguous so
downstream reasoning knows the attribution is open.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from backend.energy.load import LoadEstimate, estimate_load
from backend.models.alert import BinMeasurement
from backend.models.fault_map import FaultBin, FaultClass, FaultMap
from backend.models.motor import MotorSpec
from backend.physics.fault_map import build_fault_map
from backend.signal.fundamental import cancel_supply_tones
from backend.signal.slip import SlipEstimationUnavailableError, estimate_slip_psh
from backend.signal.spectrum import (
    Spectrum,
    average_spectra,
    required_resolution_hz,
    welch_spectrum,
)
from backend.signal.symmetrical import (
    UnbalanceResult,
    negative_sequence_admittance_from_nameplate,
    unbalance_net_of_supply,
)

DEFAULT_RESOLUTION_HZ = 0.25
MIN_SEPARATION_BINS = 4.0
PSH_MIN_CONFIDENCE = 0.5
LOAD_SCALED_SLIP_CONFIDENCE = 0.3
SEARCH_HALFWIDTH_BINS = 2.0
NET_UNBALANCE_FLOOR = 0.005  # below 0.5 % net unbalance is treated as measurement noise
STATIONARITY_BLOCK_S = 1.0
STATIONARITY_MAX_CV = 0.02  # >2 % RMS variation across 1 s blocks: load not steady
MIN_LOAD_FOR_SLIP_SCALING = 0.05
STATOR_KEY = "stator_winding:-:net negative sequence"
_FLOOR = 1e-30


class ThreePhaseCapture(Protocol):
    currents: NDArray[np.float64]
    voltages: NDArray[np.float64]
    sample_rate_hz: float


@dataclass(frozen=True)
class SlipResult:
    slip: float
    rotor_speed_rpm: float
    confidence: float
    source: Literal["principal_slot_harmonic", "load_scaled_nameplate"]
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BinConflicts:
    excluded: dict[str, str]  # key -> reason
    ambiguous: dict[str, list[str]]  # key -> keys it cannot be separated from


@dataclass(frozen=True)
class WindowFeatures:
    spectrum: Spectrum  # supply tones cancelled; what the bins are measured on
    display_spectrum: Spectrum  # raw phase-averaged spectrum, for presentation
    fundamental_hz: float
    fundamental_power: float  # A_rms^2, from the least-squares fit
    fault_map: FaultMap
    slip: SlipResult
    load: LoadEstimate
    unbalance: UnbalanceResult | None
    measurements: list[BinMeasurement]
    conflicts: BinConflicts
    stationary: bool
    stationarity_cv: float


def bin_key(b: FaultBin) -> str:
    return f"{b.fault_class.value}:{b.bearing_position or '-'}:{b.label}"


def find_bin_conflicts(
    fmap: FaultMap, resolution_hz: float, min_separation_bins: float = MIN_SEPARATION_BINS
) -> BinConflicts:
    """Bins too close to the fundamental (excluded) or to another class's bin (ambiguous)."""
    min_sep = min_separation_bins * resolution_hz
    f_s = fmap.operating_point.supply_frequency_hz
    excluded: dict[str, str] = {}
    for b in fmap.bins:
        if abs(b.frequency_hz - f_s) < min_sep:
            excluded[bin_key(b)] = (
                f"{b.frequency_hz:.2f} Hz is within {min_separation_bins:g} bins of the "
                f"fundamental at {resolution_hz:.3f} Hz resolution"
            )
        elif b.frequency_hz < min_sep:
            excluded[bin_key(b)] = f"{b.frequency_hz:.2f} Hz is too close to DC to resolve"
    ambiguous: dict[str, list[str]] = {}
    for i, a in enumerate(fmap.bins):
        for b in fmap.bins[i + 1 :]:
            if a.fault_class == b.fault_class or abs(a.frequency_hz - b.frequency_hz) >= min_sep:
                continue
            ambiguous.setdefault(bin_key(a), []).append(bin_key(b))
            ambiguous.setdefault(bin_key(b), []).append(bin_key(a))
    return BinConflicts(excluded=excluded, ambiguous=ambiguous)


def _slip(spec: MotorSpec, spectrum: Spectrum, load: LoadEstimate) -> SlipResult:
    notes: list[str] = []
    try:
        est = estimate_slip_psh(spectrum, spec)
        if est.confidence >= PSH_MIN_CONFIDENCE:
            return SlipResult(
                est.slip, est.rotor_speed_rpm, est.confidence, "principal_slot_harmonic", est.notes
            )
        notes.append(f"PSH estimate rejected (confidence {est.confidence:.2f})")
        notes.extend(est.notes)
    except SlipEstimationUnavailableError as err:
        notes.append(str(err))
    slip = spec.rated_slip * max(load.load_factor, MIN_LOAD_FOR_SLIP_SCALING)
    notes.append("Slip scaled from nameplate rated slip by estimated load factor")
    return SlipResult(
        slip,
        spec.synchronous_speed_rpm * (1 - slip),
        LOAD_SCALED_SLIP_CONFIDENCE,
        "load_scaled_nameplate",
        notes,
    )


def _stationarity_cv(currents: NDArray[np.float64], sample_rate_hz: float) -> float:
    block = int(STATIONARITY_BLOCK_S * sample_rate_hz)
    n_blocks = currents.shape[1] // block
    if n_blocks < 2:
        return 0.0
    trimmed = currents[:, : n_blocks * block].reshape(3, n_blocks, block)
    rms = np.sqrt(np.mean(trimmed**2, axis=2)).mean(axis=0)
    return float(np.std(rms) / np.mean(rms))


def _stator_measurement(
    spec: MotorSpec, capture: ThreePhaseCapture
) -> tuple[UnbalanceResult | None, BinMeasurement | None]:
    if spec.locked_rotor_current_ratio is None:
        return None, None
    result = unbalance_net_of_supply(
        capture.currents,
        capture.voltages,
        capture.sample_rate_hz,
        spec.supply_frequency_hz,
        negative_sequence_admittance_from_nameplate(spec),
    )
    assert result.net_current_unbalance is not None  # admittance was supplied
    level = 20 * math.log10(max(result.net_current_unbalance, NET_UNBALANCE_FLOOR))
    return result, BinMeasurement(
        key=STATOR_KEY,
        label="net negative-sequence current",
        fault_class=FaultClass.STATOR_WINDING,
        bearing_position=None,
        frequency_hz=None,
        equation="I2_net = |I2| - |Y2| * |V2|,  |Y2| = LRC * I_rated / V_phase",
        inputs={
            "raw_current_unbalance": result.raw_current_unbalance,
            "voltage_unbalance": result.voltage_unbalance,
            "supply_attributable_unbalance": result.supply_attributable_unbalance or 0.0,
        },
        is_primary=True,
        level_db=level,
    )


def _bin_measurement(
    b: FaultBin,
    spectrum: Spectrum,
    fundamental_power: float,
    halfwidth_hz: float,
    conflicts: BinConflicts,
) -> BinMeasurement:
    line = spectrum.tone(b.frequency_hz, halfwidth_hz).power
    return BinMeasurement(
        key=bin_key(b),
        label=b.label,
        fault_class=b.fault_class,
        bearing_position=b.bearing_position,
        frequency_hz=b.frequency_hz,
        equation=b.equation,
        inputs=b.inputs,
        is_primary=b.is_primary,
        level_db=10 * math.log10(max(line, _FLOOR) / fundamental_power),
        ambiguous_with=conflicts.ambiguous.get(bin_key(b), []),
    )


def analyze_window(
    spec: MotorSpec, capture: ThreePhaseCapture, resolution_hz: float | None = None
) -> WindowFeatures:
    """Reduce one three-phase capture to the measurements the detector scores."""
    res = resolution_hz or min(
        DEFAULT_RESOLUTION_HZ,
        required_resolution_hz(spec.rated_slip * 0.5, spec.supply_frequency_hz),
    )
    cancelled = cancel_supply_tones(
        capture.currents, capture.sample_rate_hz, spec.supply_frequency_hz
    )
    spectrum = average_spectra(
        [welch_spectrum(phase, capture.sample_rate_hz, res) for phase in cancelled.residual]
    )
    load = estimate_load(capture.voltages, capture.currents, spec)
    slip = _slip(spec, spectrum, load)
    fmap = build_fault_map(spec, rotor_speed_rpm=slip.rotor_speed_rpm)
    conflicts = find_bin_conflicts(fmap, spectrum.resolution_hz)
    fundamental = cancelled.fundamental_power
    halfwidth = SEARCH_HALFWIDTH_BINS * spectrum.resolution_hz
    nyquist_margin = (
        spectrum.frequencies_hz[-1] - (SEARCH_HALFWIDTH_BINS + 3) * spectrum.resolution_hz
    )

    conflicts = BinConflicts(
        excluded={
            **conflicts.excluded,
            **{
                bin_key(b): f"{b.frequency_hz:.1f} Hz is too close to the Nyquist limit"
                for b in fmap.bins
                if b.frequency_hz >= nyquist_margin
            },
        },
        ambiguous=conflicts.ambiguous,
    )
    measurements = [
        _bin_measurement(b, spectrum, fundamental, halfwidth, conflicts)
        for b in fmap.bins
        if bin_key(b) not in conflicts.excluded
    ]
    unbalance, stator = _stator_measurement(spec, capture)
    if stator is not None:
        measurements.append(stator)
    cv = _stationarity_cv(capture.currents, capture.sample_rate_hz)
    display = average_spectra(
        [welch_spectrum(phase, capture.sample_rate_hz, res) for phase in capture.currents]
    )
    return WindowFeatures(
        spectrum=spectrum,
        display_spectrum=display,
        fundamental_hz=cancelled.fundamental_hz,
        fundamental_power=cancelled.fundamental_power,
        fault_map=fmap,
        slip=slip,
        load=load,
        unbalance=unbalance,
        measurements=measurements,
        conflicts=conflicts,
        stationary=cv <= STATIONARITY_MAX_CV,
        stationarity_cv=cv,
    )
