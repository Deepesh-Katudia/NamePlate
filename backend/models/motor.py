"""Motor nameplate model.

`MotorSpec` is the single input to the physics engine. Validation here is physical, not
cosmetic: a spec that passes is one the fault-frequency equations are defined for.
"""

from __future__ import annotations

import math
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MIN_SUPPLY_FREQUENCY_HZ = 1.0
MAX_SUPPLY_FREQUENCY_HZ = 400.0  # upper bound of typical VFD output


class BearingPosition(StrEnum):
    DRIVE_END = "DE"
    NON_DRIVE_END = "NDE"


class BearingGeometry(BaseModel):
    """Rolling-element bearing geometry needed for characteristic defect frequencies."""

    model_config = ConfigDict(frozen=True)

    ball_count: int = Field(gt=2, description="Number of rolling elements, N_b")
    ball_diameter_mm: float = Field(gt=0, description="Rolling element diameter, B_d")
    pitch_diameter_mm: float = Field(gt=0, description="Pitch (cage) diameter, P_d")
    contact_angle_rad: float = Field(
        ge=0, lt=math.pi / 2, description="Contact angle phi in radians (0 for deep-groove radial)"
    )

    @model_validator(mode="after")
    def _ball_fits_pitch_circle(self) -> BearingGeometry:
        if self.ball_diameter_mm >= self.pitch_diameter_mm:
            raise ValueError(
                "ball_diameter_mm must be smaller than pitch_diameter_mm "
                f"(got B_d={self.ball_diameter_mm}, P_d={self.pitch_diameter_mm})"
            )
        if self.ball_count * self.ball_diameter_mm >= math.pi * self.pitch_diameter_mm:
            raise ValueError(
                f"{self.ball_count} balls of {self.ball_diameter_mm} mm cannot fit on a pitch "
                f"circle of {self.pitch_diameter_mm} mm diameter (N_b * B_d >= pi * P_d)"
            )
        return self


class BearingSpec(BaseModel):
    """A bearing fitted to the motor.

    `geometry` is optional: when omitted the designation is looked up in the bearing database.
    When supplied it takes precedence, which is how an operator resolves an unknown designation.
    """

    model_config = ConfigDict(frozen=True)

    position: BearingPosition
    designation: str = Field(min_length=1, max_length=64)
    geometry: BearingGeometry | None = None


class MotorSpec(BaseModel):
    """Induction motor nameplate plus the construction details MCSA needs."""

    model_config = ConfigDict(frozen=True)

    asset_id: str = Field(min_length=1, max_length=64)
    rated_power_kw: float = Field(gt=0)
    rated_voltage_v: float = Field(gt=0)
    rated_current_a: float = Field(gt=0)
    supply_frequency_hz: float = Field(ge=MIN_SUPPLY_FREQUENCY_HZ, le=MAX_SUPPLY_FREQUENCY_HZ)
    poles: int = Field(ge=2, description="Pole count P (not pole pairs)")
    rated_speed_rpm: float = Field(gt=0)
    rated_efficiency: float | None = Field(default=None, gt=0, le=1)
    stator_resistance_ohm: float | None = Field(
        default=None, gt=0, description="Per-phase stator resistance, if known"
    )
    rotor_slots: int | None = Field(
        default=None, gt=0, description="Rotor slot count R; None means unknown, not zero"
    )
    bearings: list[BearingSpec] = Field(default_factory=list)

    @field_validator("poles")
    @classmethod
    def _poles_even(cls, v: int) -> int:
        if v % 2 != 0:
            raise ValueError(f"pole count must be even (got {v}); poles come in N-S pairs")
        return v

    @field_validator("bearings")
    @classmethod
    def _unique_positions(cls, v: list[BearingSpec]) -> list[BearingSpec]:
        positions = [b.position for b in v]
        if len(positions) != len(set(positions)):
            raise ValueError("each bearing position (DE/NDE) may appear at most once")
        return v

    @model_validator(mode="after")
    def _motoring_below_synchronous(self) -> MotorSpec:
        n_s = self.synchronous_speed_rpm
        if self.rated_speed_rpm >= n_s:
            raise ValueError(
                f"rated_speed_rpm {self.rated_speed_rpm} must be below synchronous speed "
                f"{n_s:.1f} rpm for a motoring induction machine (slip must be positive)"
            )
        return self

    @property
    def pole_pairs(self) -> int:
        return self.poles // 2

    @property
    def synchronous_speed_rpm(self) -> float:
        return 120.0 * self.supply_frequency_hz / self.poles

    @property
    def rated_slip(self) -> float:
        n_s = self.synchronous_speed_rpm
        return (n_s - self.rated_speed_rpm) / n_s
