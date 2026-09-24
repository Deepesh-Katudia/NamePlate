"""Fault map: the physics engine's output.

A `FaultMap` lists every frequency bin to watch on one machine, each carrying the equation
and inputs that produced it, plus an explicit account of what could not be computed.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Sideband = Literal["lower", "upper"]


class FaultClass(StrEnum):
    BROKEN_ROTOR_BAR = "broken_rotor_bar"
    ECCENTRICITY = "eccentricity"
    BEARING_OUTER = "bearing_outer"
    BEARING_INNER = "bearing_inner"
    BEARING_BALL = "bearing_ball"
    BEARING_CAGE = "bearing_cage"
    STATOR_WINDING = "stator_winding"  # sequence-based, not a frequency bin


BEARING_FAULT_CLASSES: tuple[FaultClass, ...] = (
    FaultClass.BEARING_OUTER,
    FaultClass.BEARING_INNER,
    FaultClass.BEARING_BALL,
    FaultClass.BEARING_CAGE,
)


class OperatingPointSource(StrEnum):
    NAMEPLATE_RATED = "nameplate_rated"
    MEASURED = "measured"


class FrequencyBin(BaseModel):
    """One predicted spectral line, traceable to its governing equation."""

    model_config = ConfigDict(frozen=True)

    label: str
    frequency_hz: float = Field(ge=0)
    harmonic: int = Field(ge=1, description="Harmonic index k")
    sideband: Sideband
    equation: str
    inputs: dict[str, float]
    source: str = Field(description="Characteristic the bin derives from, e.g. BPFO, slip")
    is_primary: bool = False


class FaultBin(FrequencyBin):
    fault_class: FaultClass
    bearing_position: str | None = None


class OperatingPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    supply_frequency_hz: float = Field(gt=0)
    rotor_speed_rpm: float = Field(gt=0)
    synchronous_speed_rpm: float = Field(gt=0)
    slip: float = Field(gt=0, lt=1)
    rotor_frequency_hz: float = Field(gt=0)
    source: OperatingPointSource


class ConfirmationItem(BaseModel):
    """A parameter a human must supply or confirm; never silently defaulted."""

    model_config = ConfigDict(frozen=True)

    parameter: str
    reason: str
    blocks: list[FaultClass | str] = Field(default_factory=list)


class UnavailableFault(BaseModel):
    model_config = ConfigDict(frozen=True)

    fault_class: FaultClass
    reason: str
    bearing_position: str | None = None


class FaultMap(BaseModel):
    model_config = ConfigDict(frozen=True)

    asset_id: str
    operating_point: OperatingPoint
    bins: list[FaultBin]
    slot_harmonics: list[FrequencyBin] = Field(
        default_factory=list, description="Principal slot harmonics, used for slip estimation"
    )
    unavailable: list[UnavailableFault] = Field(default_factory=list)
    needs_confirmation: list[ConfirmationItem] = Field(default_factory=list)
