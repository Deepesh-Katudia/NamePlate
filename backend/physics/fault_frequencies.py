"""Closed-form fault frequencies for induction motors (Motor Current Signature Analysis).

Every function here is pure and signal-agnostic: numbers in, predicted frequencies out.
Each returned bin carries its governing equation and the inputs substituted into it, so a
technician can recompute any prediction by hand. Derivations and references: docs/PHYSICS.md.
"""

from __future__ import annotations

from backend.models.fault_map import FrequencyBin, Sideband
from backend.physics.bearings import BearingCharacteristicFrequencies

BRB_HARMONICS = (1, 2, 3)
ECCENTRICITY_HARMONICS = (1, 2, 3)
BEARING_HARMONICS = (1, 2)

EQ_SYNCHRONOUS_SPEED = "n_s = 120 * f_s / P"
EQ_SLIP = "s = (n_s - n_r) / n_s"
EQ_ROTOR_FREQUENCY = "f_r = n_r / 60"
EQ_BROKEN_ROTOR_BAR = "f_brb = f_s * (1 +/- 2*k*s)"
EQ_ECCENTRICITY = "f_ecc = f_s +/- k * f_r"
EQ_BEARING_CURRENT = "f_bearing = |f_s +/- k * f_char|"
EQ_PRINCIPAL_SLOT_HARMONIC = "f_psh = f_s * [(R/p) * (1 - s) +/- 1]"

_SIDES: tuple[tuple[Sideband, int], ...] = (("lower", -1), ("upper", +1))


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive (got {value})")


def synchronous_speed_rpm(supply_frequency_hz: float, poles: int) -> float:
    """n_s = 120 * f_s / P, with P the pole count."""
    _require_positive("supply_frequency_hz", supply_frequency_hz)
    if poles < 2 or poles % 2 != 0:
        raise ValueError(f"pole count must be an even integer >= 2 (got {poles})")
    return 120.0 * supply_frequency_hz / poles


def per_unit_slip(synchronous_rpm: float, rotor_rpm: float) -> float:
    """s = (n_s - n_r) / n_s. Restricted to motoring operation, 0 < s < 1."""
    _require_positive("synchronous_rpm", synchronous_rpm)
    _require_positive("rotor_rpm", rotor_rpm)
    slip = (synchronous_rpm - rotor_rpm) / synchronous_rpm
    if not 0 < slip < 1:
        raise ValueError(
            f"slip {slip:.5f} outside motoring range (0, 1): rotor speed {rotor_rpm} rpm vs "
            f"synchronous {synchronous_rpm} rpm"
        )
    return slip


def rotor_frequency_hz(rotor_rpm: float) -> float:
    """f_r = n_r / 60, the rotor mechanical rotation frequency."""
    _require_positive("rotor_rpm", rotor_rpm)
    return rotor_rpm / 60.0


def broken_rotor_bar_frequencies(
    supply_frequency_hz: float, slip: float, harmonics: tuple[int, ...] = BRB_HARMONICS
) -> list[FrequencyBin]:
    """f_brb = f_s(1 +/- 2ks). The k=1 lower sideband is the primary indicator."""
    _require_positive("supply_frequency_hz", supply_frequency_hz)
    if not 0 < slip < 1:
        raise ValueError(f"slip must be in (0, 1) (got {slip})")
    return [
        FrequencyBin(
            label=f"BRB {side} k={k}",
            frequency_hz=abs(supply_frequency_hz * (1 + sign * 2 * k * slip)),
            harmonic=k,
            sideband=side,
            equation=EQ_BROKEN_ROTOR_BAR,
            inputs={"f_s": supply_frequency_hz, "s": slip, "k": float(k)},
            source="slip",
            is_primary=(k == 1 and side == "lower"),
        )
        for k in harmonics
        for side, sign in _SIDES
    ]


def eccentricity_frequencies(
    supply_frequency_hz: float,
    rotor_freq_hz: float,
    harmonics: tuple[int, ...] = ECCENTRICITY_HARMONICS,
) -> list[FrequencyBin]:
    """f_ecc = f_s +/- k*f_r (mixed static/dynamic air-gap eccentricity)."""
    _require_positive("supply_frequency_hz", supply_frequency_hz)
    _require_positive("rotor_freq_hz", rotor_freq_hz)
    return [
        FrequencyBin(
            label=f"ECC {side} k={k}",
            frequency_hz=abs(supply_frequency_hz + sign * k * rotor_freq_hz),
            harmonic=k,
            sideband=side,
            equation=EQ_ECCENTRICITY,
            inputs={"f_s": supply_frequency_hz, "f_r": rotor_freq_hz, "k": float(k)},
            source="f_r",
            is_primary=(k == 1),
        )
        for k in harmonics
        for side, sign in _SIDES
    ]


def bearing_current_frequencies(
    supply_frequency_hz: float,
    characteristics: BearingCharacteristicFrequencies,
    harmonics: tuple[int, ...] = BEARING_HARMONICS,
) -> list[FrequencyBin]:
    """f_bearing = |f_s +/- k*f_char| for f_char in {BPFO, BPFI, BSF, FTF}."""
    _require_positive("supply_frequency_hz", supply_frequency_hz)
    bins: list[FrequencyBin] = []
    for name, f_char in characteristics.as_dict().items():
        for k in harmonics:
            for side, sign in _SIDES:
                bins.append(
                    FrequencyBin(
                        label=f"{name} {side} k={k}",
                        frequency_hz=abs(supply_frequency_hz + sign * k * f_char),
                        harmonic=k,
                        sideband=side,
                        equation=EQ_BEARING_CURRENT,
                        inputs={"f_s": supply_frequency_hz, "f_char": f_char, "k": float(k)},
                        source=name,
                        is_primary=(k == 1),
                    )
                )
    return bins


def principal_slot_harmonic_frequencies(
    supply_frequency_hz: float, rotor_slots: int, pole_pairs: int, slip: float
) -> list[FrequencyBin]:
    """f_psh = f_s[(R/p)(1 - s) +/- 1]. Used by the signal engine for sensorless slip."""
    _require_positive("supply_frequency_hz", supply_frequency_hz)
    if rotor_slots <= 0 or pole_pairs <= 0:
        raise ValueError("rotor_slots and pole_pairs must be positive integers")
    if not 0 < slip < 1:
        raise ValueError(f"slip must be in (0, 1) (got {slip})")
    base = (rotor_slots / pole_pairs) * (1 - slip)
    return [
        FrequencyBin(
            label=f"PSH {side}",
            frequency_hz=abs(supply_frequency_hz * (base + sign)),
            harmonic=1,
            sideband=side,
            equation=EQ_PRINCIPAL_SLOT_HARMONIC,
            inputs={
                "f_s": supply_frequency_hz,
                "R": float(rotor_slots),
                "p": float(pole_pairs),
                "s": slip,
            },
            source="rotor_slots",
        )
        for side, sign in _SIDES
    ]
