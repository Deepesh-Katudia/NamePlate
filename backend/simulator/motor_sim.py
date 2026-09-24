"""Three-phase voltage and current synthesis for a commissioned motor.

The fundamental current magnitude and power-factor angle come from an energy balance: shaft
power at the requested load plus losses from `backend.energy.load`, with a constant magnetising
current calibrated so the nameplate rated point reproduces rated current. Slip scales linearly
with load (valid in the normal operating region). All fault lines come from the physics engine.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from backend.energy.load import MissingParameterError, loss_breakdown
from backend.models.fault_map import FaultMap
from backend.models.motor import MotorSpec
from backend.physics.fault_map import build_fault_map
from backend.simulator.faults import (
    Confounder,
    FaultInjection,
    InjectedComponent,
    LoadTransient,
    SupplyUnbalance,
    fault_components,
)

_OPERATING_POINT_ITERATIONS = 30
_TWO_PI = 2.0 * math.pi


class SimulationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    sample_rate_hz: float = Field(default=10_000.0, ge=1_000.0)
    duration_s: float = Field(default=16.0, gt=0, le=120.0)
    load_factor: float = Field(default=0.8, ge=0.05, le=1.3)
    harmonics: dict[int, float] = Field(
        default_factory=lambda: {5: 0.03, 7: 0.02},
        description="Supply harmonic order -> amplitude relative to the fundamental",
    )
    noise_fraction: float = Field(
        default=1e-3, ge=0, description="White noise std relative to fundamental peak"
    )
    psh_level_db: float = Field(default=-40.0, le=0, description="Slot harmonic level, dB rel")
    negative_sequence_angle_deg: float = Field(
        default=-75.0, description="Angle of the motor's negative-sequence admittance"
    )
    seed: int | None = 0


class GroundTruth(BaseModel):
    model_config = ConfigDict(frozen=True)

    load_factor: float
    slip: float
    rotor_speed_rpm: float
    current_rms_a: float
    faults: list[FaultInjection]
    confounders: list[Confounder]
    components: list[InjectedComponent]


@dataclass(frozen=True)
class SimulationResult:
    currents: NDArray[np.float64]  # shape (3, N), amperes
    voltages: NDArray[np.float64]  # shape (3, N), volts phase-to-neutral
    sample_rate_hz: float
    ground_truth: GroundTruth
    fault_map: FaultMap


def _phase_voltage_rms(spec: MotorSpec) -> float:
    return spec.rated_voltage_v / math.sqrt(3.0)


def operating_current(spec: MotorSpec, load_factor: float) -> tuple[float, float]:
    """Per-phase RMS current and power-factor angle at a load, from the energy balance."""
    if spec.rated_efficiency is None:
        raise MissingParameterError("rated_efficiency", "simulating the operating point")
    v_ph = _phase_voltage_rms(spec)
    p_rated = spec.rated_power_kw * 1000.0
    i_active_rated = p_rated / spec.rated_efficiency / (3.0 * v_ph)
    if i_active_rated >= spec.rated_current_a:
        raise ValueError(
            "Nameplate inconsistent: rated input power implies power factor >= 1 "
            f"(active current {i_active_rated:.1f} A vs rated {spec.rated_current_a} A)"
        )
    i_magnetising = math.sqrt(spec.rated_current_a**2 - i_active_rated**2)
    current = spec.rated_current_a
    for _ in range(_OPERATING_POINT_ITERATIONS):
        p_in = load_factor * p_rated + loss_breakdown(spec, current).total_w
        current = math.hypot(p_in / (3.0 * v_ph), i_magnetising)
    p_in = load_factor * p_rated + loss_breakdown(spec, current).total_w
    return current, math.atan2(i_magnetising, p_in / (3.0 * v_ph))


def _three_phase(
    t: NDArray[np.float64], peak: float, components: Sequence[InjectedComponent]
) -> NDArray[np.float64]:
    out = np.zeros((3, t.size))
    for c in components:
        omega_t = _TWO_PI * c.frequency_hz * t
        for m in range(3):
            shift = c.phase_rad - c.phase_order * m * _TWO_PI / 3.0
            out[m] += c.amplitude_rel * peak * np.cos(omega_t + shift)
    return out


def _background_components(
    spec: MotorSpec, fmap: FaultMap, config: SimulationConfig, rng: np.random.Generator
) -> list[InjectedComponent]:
    f_s = spec.supply_frequency_hz
    comps = [
        InjectedComponent(
            label=f"supply harmonic {h}",
            frequency_hz=h * f_s,
            amplitude_rel=a,
            phase_order=h,
            phase_rad=float(rng.uniform(0, _TWO_PI)),
        )
        for h, a in sorted(config.harmonics.items())
    ]
    comps += [
        InjectedComponent(
            label=b.label,
            frequency_hz=b.frequency_hz,
            amplitude_rel=10 ** (config.psh_level_db / 20.0),
            phase_order=1,
            phase_rad=float(rng.uniform(0, _TWO_PI)),
        )
        for b in fmap.slot_harmonics
    ]
    return comps


def _supply_unbalance(
    spec: MotorSpec, u: SupplyUnbalance, config: SimulationConfig, i_rms: float, rng
) -> tuple[InjectedComponent, InjectedComponent]:
    """Voltage negative sequence and the motor's current response I2 = Y2 * V2."""
    if spec.locked_rotor_current_ratio is None:
        raise ValueError(
            "Simulating supply unbalance needs locked_rotor_current_ratio to set the motor's "
            "negative-sequence admittance"
        )
    v_ph = _phase_voltage_rms(spec)
    y2 = spec.locked_rotor_current_ratio * spec.rated_current_a / v_ph
    theta = float(rng.uniform(0, _TWO_PI))
    voltage = InjectedComponent(
        label="supply voltage negative sequence",
        frequency_hz=spec.supply_frequency_hz,
        amplitude_rel=u.voltage_unbalance,
        phase_order=-1,
        phase_rad=theta,
        channel="voltage",
    )
    current = InjectedComponent(
        label="current response to supply negative sequence",
        frequency_hz=spec.supply_frequency_hz,
        amplitude_rel=y2 * u.voltage_unbalance * v_ph / i_rms,
        phase_order=-1,
        phase_rad=theta + math.radians(config.negative_sequence_angle_deg),
    )
    return voltage, current


def _load_envelope(t: NDArray[np.float64], transients: Sequence[LoadTransient]):
    env = np.ones_like(t)
    for tr in transients:
        active = t >= tr.at_s
        env[active] += tr.step_fraction * (1 - np.exp(-(t[active] - tr.at_s) / tr.time_constant_s))
    return env


def _read_only(x: NDArray[np.float64]) -> NDArray[np.float64]:
    x.setflags(write=False)
    return x


def simulate(
    spec: MotorSpec,
    config: SimulationConfig | None = None,
    faults: Sequence[FaultInjection] = (),
    confounders: Sequence[Confounder] = (),
) -> SimulationResult:
    """Synthesize three-phase voltage and current with ground-truth labels."""
    cfg = config or SimulationConfig()
    rng = np.random.default_rng(cfg.seed)
    slip = spec.rated_slip * cfg.load_factor
    rotor_rpm = spec.synchronous_speed_rpm * (1.0 - slip)
    fmap = build_fault_map(spec, rotor_speed_rpm=rotor_rpm)
    i_rms, pf_angle = operating_current(spec, cfg.load_factor)

    t = np.arange(int(round(cfg.duration_s * cfg.sample_rate_hz))) / cfg.sample_rate_hz
    i_peak = math.sqrt(2.0) * i_rms
    v_peak = math.sqrt(2.0) * _phase_voltage_rms(spec)
    f_s = spec.supply_frequency_hz

    fundamental_i = InjectedComponent(
        label="fundamental", frequency_hz=f_s, amplitude_rel=1.0, phase_order=1, phase_rad=-pf_angle
    )
    fundamental_v = fundamental_i.model_copy(update={"phase_rad": 0.0, "channel": "voltage"})
    enveloped = [fundamental_i, *_background_components(spec, fmap, cfg, rng)]
    fault_comps = [c for f in faults for c in fault_components(fmap, f, rng)]
    voltage_comps: list[InjectedComponent] = [fundamental_v]
    for u in (c for c in confounders if isinstance(c, SupplyUnbalance)):
        v_comp, i_comp = _supply_unbalance(spec, u, cfg, i_rms, rng)
        voltage_comps.append(v_comp)
        fault_comps.append(i_comp)

    envelope = _load_envelope(t, [c for c in confounders if isinstance(c, LoadTransient)])
    currents = _three_phase(t, i_peak, enveloped) * envelope + _three_phase(t, i_peak, fault_comps)
    voltages = _three_phase(t, v_peak, voltage_comps)
    currents += rng.normal(scale=cfg.noise_fraction * i_peak, size=currents.shape)
    voltages += rng.normal(scale=cfg.noise_fraction * v_peak, size=voltages.shape)

    return SimulationResult(
        currents=_read_only(currents),
        voltages=_read_only(voltages),
        sample_rate_hz=cfg.sample_rate_hz,
        ground_truth=GroundTruth(
            load_factor=cfg.load_factor,
            slip=slip,
            rotor_speed_rpm=rotor_rpm,
            current_rms_a=i_rms,
            faults=list(faults),
            confounders=list(confounders),
            components=[*enveloped, *fault_comps, *voltage_comps],
        ),
        fault_map=fmap,
    )
