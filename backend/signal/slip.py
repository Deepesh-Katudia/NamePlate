"""Sensorless slip estimation from the principal slot harmonic (PSH).

Forward model (physics engine): f_psh = f_s * [(R/p)(1 - s) +/- 1]
Inverse (here):                 s = 1 - (f_psh / f_s -/+ 1) * p / R

The search neighbourhood is the PSH band swept over plausible slip, (0, 1.5 * rated slip].
Both PSH lines are searched; the stronger one gives the estimate and the other, if visible,
cross-checks it. Confidence is reduced, never overridden, when:
- the peak barely rises above the band's noise (a wrong slot count finds only noise),
- the two PSH lines imply different slips,
- the estimate exceeds the nameplate rated slip by more than the overload tolerance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from backend.models.fault_map import Sideband
from backend.models.motor import MotorSpec
from backend.physics.fault_frequencies import principal_slot_harmonic_frequencies
from backend.signal.spectrum import Spectrum

MIN_SEARCH_SLIP = 1e-4
MAX_SLIP_OVER_RATED = 1.5  # search ceiling as a multiple of rated slip
OVERLOAD_TOLERANCE = 1.25  # slip beyond this multiple of rated is implausible in service
SNR_FLOOR_DB = 6.0  # below this a "peak" is indistinguishable from noise
SNR_FULL_DB = 20.0  # at or above this the peak is unambiguous
VISIBLE_SNR_DB = 10.0
AGREEMENT_BINS = 2.0  # PSH pair agreement tolerance, in frequency bins
PENALTY_DISAGREE = 0.5
PENALTY_SINGLE_LINE = 0.8
PENALTY_NAMEPLATE = 0.4

_SIGN: dict[Sideband, int] = {"lower": -1, "upper": +1}


class SlipEstimationUnavailableError(ValueError):
    """PSH slip estimation cannot run for this asset or spectrum."""


@dataclass(frozen=True)
class SlipEstimate:
    slip: float
    rotor_speed_rpm: float
    confidence: float
    psh_frequency_hz: float
    sideband: Sideband
    snr_db: float
    nameplate_rated_slip: float
    method: str = "principal_slot_harmonic"
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _PshCandidate:
    sideband: Sideband
    frequency_hz: float
    slip: float
    snr_db: float


def _invert(f_psh: float, f_s: float, rotor_slots: int, pole_pairs: int, side: Sideband) -> float:
    return 1.0 - (f_psh / f_s - _SIGN[side]) * pole_pairs / rotor_slots


def _search(spectrum: Spectrum, spec: MotorSpec, side: Sideband) -> _PshCandidate:
    f_s, r, p = spec.supply_frequency_hz, spec.rotor_slots, spec.pole_pairs
    assert r is not None  # checked by caller
    s_max = min(spec.rated_slip * MAX_SLIP_OVER_RATED, 0.99)
    edges = [
        next(
            b.frequency_hz
            for b in principal_slot_harmonic_frequencies(f_s, r, p, s)
            if b.sideband == side
        )
        for s in (MIN_SEARCH_SLIP, s_max)
    ]
    lo, hi = min(edges), max(edges)
    if hi >= spectrum.frequencies_hz[-1]:
        raise SlipEstimationUnavailableError(
            f"PSH band {lo:.0f}-{hi:.0f} Hz exceeds the spectrum's "
            f"{spectrum.frequencies_hz[-1]:.0f} Hz upper limit"
        )
    band = (spectrum.frequencies_hz >= lo) & (spectrum.frequencies_hz <= hi)
    noise = float(np.median(spectrum.power[band]))
    centre = (lo + hi) / 2
    tone = spectrum.tone(centre, (hi - lo) / 2)
    peak_bin = spectrum.power[tone.bin_index]
    snr = 10 * math.log10(max(peak_bin, 1e-30) / max(noise, 1e-30))
    return _PshCandidate(side, tone.frequency_hz, _invert(tone.frequency_hz, f_s, r, p, side), snr)


def _snr_score(snr_db: float) -> float:
    return float(np.clip((snr_db - SNR_FLOOR_DB) / (SNR_FULL_DB - SNR_FLOOR_DB), 0.0, 1.0))


def estimate_slip_psh(spectrum: Spectrum, spec: MotorSpec) -> SlipEstimate:
    """Estimate slip from a phase-current spectrum. Raises if rotor slots are unknown."""
    if spec.rotor_slots is None:
        raise SlipEstimationUnavailableError(
            "rotor_slots unknown: PSH cannot be located; use the nameplate rated point"
        )
    lower, upper = _search(spectrum, spec, "lower"), _search(spectrum, spec, "upper")
    best, other = (lower, upper) if lower.snr_db >= upper.snr_db else (upper, lower)
    notes: list[str] = []
    confidence = _snr_score(best.snr_db)
    if confidence < 1.0:
        notes.append(f"PSH peak only {best.snr_db:.1f} dB above band noise")

    # df_psh/ds = f_s * R / p, so a two-bin frequency tolerance maps to this slip tolerance
    slip_tol = (
        AGREEMENT_BINS
        * spectrum.resolution_hz
        * spec.pole_pairs
        / (spec.supply_frequency_hz * spec.rotor_slots)
    )
    if other.snr_db < VISIBLE_SNR_DB:
        confidence *= PENALTY_SINGLE_LINE
        notes.append(f"Only the {best.sideband} PSH line is visible; no cross-check")
    elif abs(other.slip - best.slip) > slip_tol:
        confidence *= PENALTY_DISAGREE
        notes.append(f"PSH lines disagree: s={best.slip:.4f} vs s={other.slip:.4f}")

    if not 0 < best.slip <= spec.rated_slip * OVERLOAD_TOLERANCE:
        confidence *= PENALTY_NAMEPLATE
        notes.append(
            f"Estimated slip {best.slip:.4f} inconsistent with rated slip {spec.rated_slip:.4f}"
        )

    return SlipEstimate(
        slip=best.slip,
        rotor_speed_rpm=spec.synchronous_speed_rpm * (1.0 - best.slip),
        confidence=confidence,
        psh_frequency_hz=best.frequency_hz,
        sideband=best.sideband,
        snr_db=best.snr_db,
        nameplate_rated_slip=spec.rated_slip,
        notes=notes,
    )
