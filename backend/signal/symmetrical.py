"""Symmetrical components and supply-compensated current unbalance.

I1 = (Ia + a*Ib + a^2*Ic) / 3,   I2 = (Ia + a^2*Ib + a*Ic) / 3,   a = exp(2*pi*j/3)

Raw current unbalance is a classic false positive for winding faults: a healthy induction
motor draws negative-sequence current I2 = Y2 * V2 whenever the supply is unbalanced, and
because Y2 is close to the locked-rotor admittance, 1 % voltage unbalance produces roughly
6-7 % current unbalance. Only the part of I2 *not* explained by V2 points at the winding.
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from backend.models.motor import MotorSpec

A = cmath.exp(2j * math.pi / 3)


@dataclass(frozen=True)
class SequenceComponents:
    zero: complex
    positive: complex
    negative: complex

    @property
    def unbalance_factor(self) -> float:
        """|I2| / |I1|."""
        return abs(self.negative) / abs(self.positive) if self.positive else math.inf


@dataclass(frozen=True)
class NegativeSequenceAdmittance:
    magnitude_siemens: float
    angle_rad: float | None  # None: magnitude known only (nameplate-derived)
    source: Literal["nameplate_lrc_ratio", "commissioning_calibration"]


@dataclass(frozen=True)
class UnbalanceResult:
    raw_current_unbalance: float
    voltage_unbalance: float
    supply_attributable_unbalance: float | None
    net_current_unbalance: float | None
    method: Literal["vector", "magnitude_only", "unavailable"]
    note: str


def fundamental_phasor(
    x: NDArray[np.float64], sample_rate_hz: float, frequency_hz: float
) -> complex:
    """Peak-amplitude phasor P such that x ~= Re(P * exp(j*2*pi*f*t)), by least squares."""
    x = np.asarray(x, dtype=float)
    t = np.arange(x.size) / sample_rate_hz
    w = 2 * np.pi * frequency_hz * t
    basis = np.column_stack([np.cos(w), np.sin(w), np.ones_like(t)])
    (a, b, _), *_ = np.linalg.lstsq(basis, x, rcond=None)
    return complex(a, -b)


def sequence_components(phasors: Sequence[complex]) -> SequenceComponents:
    if len(phasors) != 3:
        raise ValueError(f"expected three phase phasors (got {len(phasors)})")
    pa, pb, pc = phasors
    return SequenceComponents(
        zero=(pa + pb + pc) / 3,
        positive=(pa + A * pb + A**2 * pc) / 3,
        negative=(pa + A**2 * pb + A * pc) / 3,
    )


def negative_sequence_admittance_from_nameplate(spec: MotorSpec) -> NegativeSequenceAdmittance:
    """|Y2| ~= I_locked_rotor / V_phase. Angle unknown from the nameplate alone."""
    if spec.locked_rotor_current_ratio is None:
        raise ValueError(
            "locked_rotor_current_ratio is required to estimate the negative-sequence admittance"
        )
    v_ph = spec.rated_voltage_v / math.sqrt(3.0)
    return NegativeSequenceAdmittance(
        magnitude_siemens=spec.locked_rotor_current_ratio * spec.rated_current_a / v_ph,
        angle_rad=None,
        source="nameplate_lrc_ratio",
    )


def _phase_sequences(
    x: NDArray[np.float64], sample_rate_hz: float, frequency_hz: float
) -> SequenceComponents:
    if x.ndim != 2 or x.shape[0] != 3:
        raise ValueError(f"expected shape (3, N) (got {x.shape})")
    return sequence_components([fundamental_phasor(ch, sample_rate_hz, frequency_hz) for ch in x])


def unbalance_net_of_supply(
    currents: NDArray[np.float64],
    voltages: NDArray[np.float64],
    sample_rate_hz: float,
    supply_frequency_hz: float,
    admittance: NegativeSequenceAdmittance | None,
) -> UnbalanceResult:
    """Current unbalance with the supply-driven negative sequence removed."""
    i = _phase_sequences(currents, sample_rate_hz, supply_frequency_hz)
    v = _phase_sequences(voltages, sample_rate_hz, supply_frequency_hz)
    i1 = abs(i.positive)
    raw, v_unbalance = i.unbalance_factor, v.unbalance_factor

    if admittance is None:
        return UnbalanceResult(
            raw_current_unbalance=raw,
            voltage_unbalance=v_unbalance,
            supply_attributable_unbalance=None,
            net_current_unbalance=None,
            method="unavailable",
            note=(
                "Cannot separate supply unbalance from winding asymmetry: supply "
                "locked_rotor_current_ratio or a commissioning calibration"
            ),
        )
    expected_mag = admittance.magnitude_siemens * abs(v.negative)
    if admittance.angle_rad is not None:
        y2 = cmath.rect(admittance.magnitude_siemens, admittance.angle_rad)
        net = abs(i.negative - y2 * v.negative) / i1
        method: Literal["vector", "magnitude_only"] = "vector"
        note = f"Vector subtraction using {admittance.source} admittance"
    else:
        # Without the angle, subtract magnitudes. This is a lower bound on the residual,
        # which errs toward fewer false winding alarms.
        net = max(abs(i.negative) - expected_mag, 0.0) / i1
        method = "magnitude_only"
        note = f"Magnitude subtraction (lower bound) using {admittance.source} admittance"
    return UnbalanceResult(
        raw_current_unbalance=raw,
        voltage_unbalance=v_unbalance,
        supply_attributable_unbalance=expected_mag / i1,
        net_current_unbalance=net,
        method=method,
        note=note,
    )
