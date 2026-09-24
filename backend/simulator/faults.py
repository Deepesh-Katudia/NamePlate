"""Fault and confounder definitions for the motor simulator.

Fault frequencies are never computed here: every line sits on a bin from the physics engine's
`FaultMap`, so the simulator and the detector cannot drift apart. This module only decides
*how strong* each line is for a given severity.

Severity-to-amplitude mapping is illustrative, chosen to span the Thomson & Fenger rotor-bar
bands (about -60 dB healthy to -25 dB severe) and the weaker, typical current-signature levels
of bearing defects. It is a test fixture, not a claim about any specific machine.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from backend.models.errors import DomainError
from backend.models.fault_map import FaultBin, FaultClass, FaultMap

STATOR_NEG_SEQ_AT_FULL_SEVERITY = 0.10  # negative-sequence current as fraction of I1
HARMONIC_OFFSET_DB = 6.0  # each additional sideband order k sits this much lower
UPPER_SIDEBAND_OFFSET_DB = 2.0  # speed-ripple effect makes the BRB upper sideband weaker


class SimFault(StrEnum):
    BROKEN_ROTOR_BAR = "broken_rotor_bar"
    BEARING_OUTER = "bearing_outer"
    BEARING_INNER = "bearing_inner"
    ECCENTRICITY = "eccentricity"
    STATOR_TURN_FAULT = "stator_turn_fault"
    MISALIGNMENT = "misalignment"


class FaultInjection(BaseModel):
    model_config = ConfigDict(frozen=True)

    fault: SimFault
    severity: float = Field(gt=0, le=1, description="0 = incipient, 1 = severe")
    bearing_position: Literal["DE", "NDE"] = "DE"


class LoadTransient(BaseModel):
    """Step change in load, settling exponentially. Modulates the fundamental envelope."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["load_transient"] = "load_transient"
    step_fraction: float = Field(ge=-0.9, le=1.0, description="Relative current step")
    at_s: float = Field(ge=0)
    time_constant_s: float = Field(default=0.2, gt=0)


class SupplyUnbalance(BaseModel):
    """Negative-sequence supply voltage, as a fraction of positive sequence."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["supply_voltage_unbalance"] = "supply_voltage_unbalance"
    voltage_unbalance: float = Field(gt=0, le=0.1)


Confounder = LoadTransient | SupplyUnbalance


class InjectedComponent(BaseModel):
    """One sinusoid added to the three phases; recorded as ground truth."""

    model_config = ConfigDict(frozen=True)

    label: str
    frequency_hz: float = Field(ge=0)
    amplitude_rel: float = Field(ge=0, description="Peak amplitude relative to fundamental")
    phase_order: int = Field(description="Phase m is shifted by -phase_order * m * 120 deg")
    phase_rad: float
    channel: Literal["current", "voltage"] = "current"
    fault: SimFault | None = None


# (dB at severity -> 0, dB at severity 1) for the primary bin of each bin-based fault
_LEVELS_DB: dict[SimFault, tuple[float, float]] = {
    SimFault.BROKEN_ROTOR_BAR: (-60.0, -25.0),
    SimFault.BEARING_OUTER: (-70.0, -40.0),
    SimFault.BEARING_INNER: (-70.0, -40.0),
    SimFault.ECCENTRICITY: (-60.0, -30.0),
    SimFault.MISALIGNMENT: (-60.0, -35.0),
}

_BEARING_SOURCE: dict[SimFault, str] = {
    SimFault.BEARING_OUTER: "BPFO",
    SimFault.BEARING_INNER: "BPFI",
}


def _db_to_ratio(db: float) -> float:
    return float(10 ** (db / 20.0))


def _primary_level_db(fault: SimFault, severity: float) -> float:
    lo, hi = _LEVELS_DB[fault]
    return lo + (hi - lo) * severity


def _select_bins(fmap: FaultMap, injection: FaultInjection) -> list[FaultBin]:
    fault = injection.fault
    if fault == SimFault.BROKEN_ROTOR_BAR:
        return [b for b in fmap.bins if b.fault_class == FaultClass.BROKEN_ROTOR_BAR]
    if fault == SimFault.ECCENTRICITY:
        return [b for b in fmap.bins if b.fault_class == FaultClass.ECCENTRICITY]
    if fault == SimFault.MISALIGNMENT:
        # Misalignment appears in current mainly as the k=1 rotational sidebands.
        return [
            b for b in fmap.bins if b.fault_class == FaultClass.ECCENTRICITY and b.harmonic == 1
        ]
    source = _BEARING_SOURCE[fault]
    bins = [
        b
        for b in fmap.bins
        if b.source == source and b.bearing_position == injection.bearing_position
    ]
    if not bins:
        raise DomainError(
            f"Cannot inject {fault.value} at {injection.bearing_position}: the fault map has no "
            "bearing bins there (bearing unknown or not supplied)"
        )
    return bins


def _bin_level_db(fault: SimFault, primary_db: float, b: FaultBin) -> float:
    level = primary_db - HARMONIC_OFFSET_DB * (b.harmonic - 1)
    if fault == SimFault.BROKEN_ROTOR_BAR and b.sideband == "upper":
        level -= UPPER_SIDEBAND_OFFSET_DB
    return level


def fault_components(
    fmap: FaultMap, injection: FaultInjection, rng: np.random.Generator
) -> list[InjectedComponent]:
    """Current components for one injected fault, placed on physics-engine bins."""
    fault = injection.fault
    f_s = fmap.operating_point.supply_frequency_hz
    if fault == SimFault.STATOR_TURN_FAULT:
        return [
            InjectedComponent(
                label="stator turn fault negative sequence",
                frequency_hz=f_s,
                amplitude_rel=STATOR_NEG_SEQ_AT_FULL_SEVERITY * injection.severity,
                phase_order=-1,
                phase_rad=float(rng.uniform(0, 2 * np.pi)),
                fault=fault,
            )
        ]
    primary_db = _primary_level_db(fault, injection.severity)
    return [
        InjectedComponent(
            label=f"{fault.value}: {b.label}",
            frequency_hz=b.frequency_hz,
            amplitude_rel=_db_to_ratio(_bin_level_db(fault, primary_db, b)),
            phase_order=1,
            phase_rad=float(rng.uniform(0, 2 * np.pi)),
            fault=fault,
        )
        for b in _select_bins(fmap, injection)
    ]
