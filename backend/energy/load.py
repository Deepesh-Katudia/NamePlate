"""Shaft-load estimation from three-phase voltage and current.

P_in    = mean(v_a*i_a + v_b*i_b + v_c*i_c)
P_shaft = P_in - stator copper - core/friction/windage - rotor copper/stray

Losses are scaled from the nameplate rated efficiency using a documented loss split (typical
for TEFC induction motors; see docs/PHYSICS.md). Stator copper uses the measured per-phase
stator resistance when the nameplate supplies it, otherwise it is scaled like the other
load-dependent losses and the uncertainty band widens.

The load factor is the reliable output: a few percent of loss-model error moves it by a few
percent. Absolute efficiency is not: it is a ratio of two nearly equal numbers, so it is
returned only as an indicative figure and labelled as such in the type.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.models.errors import MissingParameterError
from backend.models.motor import MotorSpec

__all__ = ["MissingParameterError"]  # re-exported for callers of this module

LOSS_SPLIT_TOLERANCE = 1e-6
MEASUREMENT_UNCERTAINTY = 0.01  # fraction of P_in: VT/CT and sampling error
LOSS_MODEL_UNCERTAINTY = 0.30  # fraction of any loss term derived from the split
NEAR_NO_LOAD = 0.05


class LossSplit(BaseModel):
    """Fraction of rated total losses in each loss mechanism. Must sum to 1."""

    model_config = ConfigDict(frozen=True)

    stator_copper: float = Field(ge=0, le=1)
    rotor_copper: float = Field(ge=0, le=1)
    core: float = Field(ge=0, le=1)
    friction_windage: float = Field(ge=0, le=1)
    stray: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _sums_to_one(self) -> LossSplit:
        total = (
            self.stator_copper + self.rotor_copper + self.core + self.friction_windage + self.stray
        )
        if abs(total - 1.0) > LOSS_SPLIT_TOLERANCE:
            raise ValueError(f"loss split fractions must sum to 1 (got {total:.4f})")
        return self


DEFAULT_LOSS_SPLIT = LossSplit(
    stator_copper=0.35, rotor_copper=0.20, core=0.20, friction_windage=0.10, stray=0.15
)


class LossBreakdown(BaseModel):
    model_config = ConfigDict(frozen=True)

    stator_copper_w: float
    fixed_w: float = Field(description="Core + friction + windage, load independent")
    rotor_copper_and_stray_w: float
    stator_copper_from_resistance: bool

    @property
    def total_w(self) -> float:
        return self.stator_copper_w + self.fixed_w + self.rotor_copper_and_stray_w

    @property
    def split_derived_w(self) -> float:
        """Loss terms that come from the assumed split rather than a measurement."""
        stator = 0.0 if self.stator_copper_from_resistance else self.stator_copper_w
        return stator + self.fixed_w + self.rotor_copper_and_stray_w


class LoadEstimate(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_power_w: float
    shaft_power_w: float
    load_factor: float
    load_factor_low: float
    load_factor_high: float
    losses: LossBreakdown
    efficiency_indicative: float
    efficiency_is_indicative_only: Literal[True] = True
    assumptions: list[str]


def rated_losses_w(spec: MotorSpec) -> float:
    if spec.rated_efficiency is None:
        raise MissingParameterError("rated_efficiency", "loss estimation")
    return spec.rated_power_kw * 1000.0 * (1.0 / spec.rated_efficiency - 1.0)


def loss_breakdown(
    spec: MotorSpec, current_rms_a: float, split: LossSplit = DEFAULT_LOSS_SPLIT
) -> LossBreakdown:
    """Losses at a given per-phase RMS current. Load-dependent terms scale with (I/I_rated)^2."""
    rated = rated_losses_w(spec)
    current_ratio_sq = (current_rms_a / spec.rated_current_a) ** 2
    if spec.stator_resistance_ohm is not None:
        stator = 3.0 * current_rms_a**2 * spec.stator_resistance_ohm
    else:
        stator = split.stator_copper * rated * current_ratio_sq
    return LossBreakdown(
        stator_copper_w=stator,
        fixed_w=(split.core + split.friction_windage) * rated,
        rotor_copper_and_stray_w=(split.rotor_copper + split.stray) * rated * current_ratio_sq,
        stator_copper_from_resistance=spec.stator_resistance_ohm is not None,
    )


def _validate_three_phase(name: str, x: NDArray[np.float64]) -> None:
    if x.ndim != 2 or x.shape[0] != 3 or x.shape[1] < 2:
        raise ValueError(f"{name} must have shape (3, N) with N >= 2 (got {x.shape})")


def input_power_w(voltages: NDArray[np.float64], currents: NDArray[np.float64]) -> float:
    """P_in = mean(v_a i_a + v_b i_b + v_c i_c), phase-to-neutral voltages."""
    _validate_three_phase("voltages", voltages)
    _validate_three_phase("currents", currents)
    if voltages.shape != currents.shape:
        raise ValueError(f"shape mismatch: voltages {voltages.shape} vs currents {currents.shape}")
    return float(np.mean(np.sum(voltages * currents, axis=0)))


def estimate_load(
    voltages: NDArray[np.float64],
    currents: NDArray[np.float64],
    spec: MotorSpec,
    split: LossSplit = DEFAULT_LOSS_SPLIT,
) -> LoadEstimate:
    """Estimate shaft power and load factor with an explicit uncertainty band."""
    p_in = input_power_w(voltages, currents)
    i_rms = float(np.mean(np.sqrt(np.mean(currents**2, axis=1))))
    losses = loss_breakdown(spec, i_rms, split)
    p_rated = spec.rated_power_kw * 1000.0
    p_shaft = p_in - losses.total_w
    uncertainty_w = LOSS_MODEL_UNCERTAINTY * losses.split_derived_w + MEASUREMENT_UNCERTAINTY * abs(
        p_in
    )
    load = p_shaft / p_rated

    assumptions = [
        f"Losses scaled from rated efficiency {spec.rated_efficiency} using split {split}",
        f"Loss-model terms carry +/-{LOSS_MODEL_UNCERTAINTY:.0%}, measurement "
        f"+/-{MEASUREMENT_UNCERTAINTY:.0%} of input power",
    ]
    if spec.stator_resistance_ohm is None:
        assumptions.append(
            "stator_resistance_ohm not supplied: stator copper loss scaled from rated losses"
        )
    if load < NEAR_NO_LOAD:
        assumptions.append("Near no-load: estimate dominated by the loss model")

    return LoadEstimate(
        input_power_w=p_in,
        shaft_power_w=p_shaft,
        load_factor=load,
        load_factor_low=(p_shaft - uncertainty_w) / p_rated,
        load_factor_high=(p_shaft + uncertainty_w) / p_rated,
        losses=losses,
        efficiency_indicative=p_shaft / p_in if p_in > 0 else 0.0,
        assumptions=assumptions,
    )
