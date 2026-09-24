"""Compose a `MotorSpec` into a `FaultMap` at a given operating point.

This is where missing information is made explicit. Rotor-bar and eccentricity bins need only
the nameplate and are always produced; bearing and slot-harmonic outputs are produced only when
their inputs are known, and otherwise appear as `unavailable` / `needs_confirmation` entries.
"""

from __future__ import annotations

from backend.models.fault_map import (
    BEARING_FAULT_CLASSES,
    ConfirmationItem,
    FaultBin,
    FaultClass,
    FaultMap,
    FrequencyBin,
    OperatingPoint,
    OperatingPointSource,
    UnavailableFault,
)
from backend.models.motor import BearingGeometry, BearingPosition, BearingSpec, MotorSpec
from backend.physics.bearings import (
    GeometryProvenance,
    UnknownBearingError,
    bearing_characteristic_frequencies,
    lookup_bearing,
)
from backend.physics.fault_frequencies import (
    bearing_current_frequencies,
    broken_rotor_bar_frequencies,
    eccentricity_frequencies,
    per_unit_slip,
    principal_slot_harmonic_frequencies,
    rotor_frequency_hz,
)

SLIP_ESTIMATION = "sensorless_slip_estimation"
STATOR_UNAVAILABLE_REASON = (
    "locked_rotor_current_ratio unknown: supply-voltage unbalance cannot be separated "
    "from winding asymmetry, and raw current unbalance would false-alarm"
)

_BEARING_SOURCE_TO_CLASS: dict[str, FaultClass] = {
    "BPFO": FaultClass.BEARING_OUTER,
    "BPFI": FaultClass.BEARING_INNER,
    "BSF": FaultClass.BEARING_BALL,
    "FTF": FaultClass.BEARING_CAGE,
}


def operating_point(spec: MotorSpec, rotor_speed_rpm: float | None = None) -> OperatingPoint:
    """Operating point from a measured speed, or the nameplate rated point if none is given."""
    source = (
        OperatingPointSource.NAMEPLATE_RATED
        if rotor_speed_rpm is None
        else OperatingPointSource.MEASURED
    )
    speed = spec.rated_speed_rpm if rotor_speed_rpm is None else rotor_speed_rpm
    n_s = spec.synchronous_speed_rpm
    return OperatingPoint(
        supply_frequency_hz=spec.supply_frequency_hz,
        rotor_speed_rpm=speed,
        synchronous_speed_rpm=n_s,
        slip=per_unit_slip(n_s, speed),
        rotor_frequency_hz=rotor_frequency_hz(speed),
        source=source,
    )


def _tag(bins: list[FrequencyBin], fault_class: FaultClass) -> list[FaultBin]:
    return [FaultBin(**b.model_dump(), fault_class=fault_class) for b in bins]


def _bearing_unavailable(position: BearingPosition, reason: str) -> list[UnavailableFault]:
    return [
        UnavailableFault(fault_class=fc, reason=reason, bearing_position=position.value)
        for fc in BEARING_FAULT_CLASSES
    ]


def _resolve_geometry(
    bearing: BearingSpec,
) -> tuple[BearingGeometry, GeometryProvenance, ConfirmationItem | None]:
    """User geometry wins; if it contradicts a database entry, ask a human which is right."""
    if bearing.geometry is None:
        record = lookup_bearing(bearing.designation)
        return record.geometry, record.provenance, None
    try:
        record = lookup_bearing(bearing.designation)
    except UnknownBearingError:
        return bearing.geometry, GeometryProvenance.USER_SUPPLIED, None
    conflict = None
    if record.geometry != bearing.geometry:
        conflict = ConfirmationItem(
            parameter=f"bearing_geometry[{bearing.position.value}]",
            reason=(
                f"Supplied geometry for {bearing.designation} differs from the database entry "
                f"({record.provenance.value}); using the supplied values, confirm they are correct"
            ),
        )
    return bearing.geometry, GeometryProvenance.USER_SUPPLIED, conflict


def _bearing_section(
    spec: MotorSpec, op: OperatingPoint
) -> tuple[list[FaultBin], list[UnavailableFault], list[ConfirmationItem]]:
    bins: list[FaultBin] = []
    unavailable: list[UnavailableFault] = []
    confirmations: list[ConfirmationItem] = []
    fitted = {b.position: b for b in spec.bearings}

    for position in BearingPosition:
        param = f"bearing_designation[{position.value}]"
        bearing = fitted.get(position)
        if bearing is None:
            reason = f"No bearing designation supplied for {position.value} position"
            unavailable.extend(_bearing_unavailable(position, reason))
            confirmations.append(
                ConfirmationItem(parameter=param, reason=reason, blocks=list(BEARING_FAULT_CLASSES))
            )
            continue
        try:
            geometry, provenance, conflict = _resolve_geometry(bearing)
        except UnknownBearingError as err:
            unavailable.extend(_bearing_unavailable(position, str(err)))
            confirmations.append(
                ConfirmationItem(
                    parameter=f"bearing_geometry[{position.value}]",
                    reason=str(err),
                    blocks=list(BEARING_FAULT_CLASSES),
                )
            )
            continue
        if conflict is not None:
            confirmations.append(conflict)
        if provenance == GeometryProvenance.NOMINAL:
            confirmations.append(
                ConfirmationItem(
                    parameter=f"bearing_geometry[{position.value}]",
                    reason=(
                        f"Bearing {bearing.designation} uses nominal geometry; confirm ball count "
                        "and diameters against the manufacturer's defect-frequency sheet"
                    ),
                )
            )
        chars = bearing_characteristic_frequencies(geometry, op.rotor_frequency_hz)
        for b in bearing_current_frequencies(op.supply_frequency_hz, chars):
            bins.append(
                FaultBin(
                    **b.model_dump(),
                    fault_class=_BEARING_SOURCE_TO_CLASS[b.source],
                    bearing_position=position.value,
                )
            )
    return bins, unavailable, confirmations


def _slot_harmonics(
    spec: MotorSpec, op: OperatingPoint
) -> tuple[list[FrequencyBin], list[ConfirmationItem]]:
    if spec.rotor_slots is None:
        return [], [
            ConfirmationItem(
                parameter="rotor_slots",
                reason=(
                    "Rotor slot count unknown: principal slot harmonic cannot be located, so slip "
                    "must come from the nameplate rated point rather than sensorless estimation"
                ),
                blocks=[SLIP_ESTIMATION],
            )
        ]
    psh = principal_slot_harmonic_frequencies(
        op.supply_frequency_hz, spec.rotor_slots, spec.pole_pairs, op.slip
    )
    return psh, []


def _stator_coverage(
    spec: MotorSpec,
) -> tuple[list[UnavailableFault], list[ConfirmationItem]]:
    if spec.locked_rotor_current_ratio is not None:
        return [], []
    return (
        [UnavailableFault(fault_class=FaultClass.STATOR_WINDING, reason=STATOR_UNAVAILABLE_REASON)],
        [
            ConfirmationItem(
                parameter="locked_rotor_current_ratio",
                reason=STATOR_UNAVAILABLE_REASON + " (read it from the NEMA code letter)",
                blocks=[FaultClass.STATOR_WINDING],
            )
        ],
    )


def build_fault_map(spec: MotorSpec, rotor_speed_rpm: float | None = None) -> FaultMap:
    """Derive every monitored frequency bin for `spec` at the given (or rated) speed."""
    op = operating_point(spec, rotor_speed_rpm)
    bins = [
        *_tag(
            broken_rotor_bar_frequencies(op.supply_frequency_hz, op.slip),
            FaultClass.BROKEN_ROTOR_BAR,
        ),
        *_tag(
            eccentricity_frequencies(op.supply_frequency_hz, op.rotor_frequency_hz),
            FaultClass.ECCENTRICITY,
        ),
    ]
    bearing_bins, unavailable, bearing_confirmations = _bearing_section(spec, op)
    psh, psh_confirmations = _slot_harmonics(spec, op)
    stator_unavailable, stator_confirmations = _stator_coverage(spec)
    return FaultMap(
        asset_id=spec.asset_id,
        operating_point=op,
        bins=[*bins, *bearing_bins],
        slot_harmonics=psh,
        unavailable=[*unavailable, *stator_unavailable],
        needs_confirmation=[
            *bearing_confirmations,
            *psh_confirmations,
            *stator_confirmations,
        ],
    )
