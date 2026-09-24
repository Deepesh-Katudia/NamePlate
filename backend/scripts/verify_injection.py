"""Inject a fault with the simulator and compare predicted vs measured spectral lines.

Usage (from the repository root):
    python -m backend.scripts.verify_injection [fault] [severity]

The fault map is built from the *true* operating speed, then the same lines are read from the
Welch spectrum of phase A current, alongside the same lines on an otherwise identical healthy
machine. Exits non-zero if any predicted line is missed by more than one frequency bin.
"""

from __future__ import annotations

import sys

from backend.models.motor import BearingPosition, BearingSpec, MotorSpec
from backend.signal.spectrum import welch_spectrum
from backend.simulator.faults import FaultInjection, SimFault
from backend.simulator.motor_sim import SimulationConfig, simulate

RESOLUTION_HZ = 0.25
DEMO_SPEC = MotorSpec(
    asset_id="DEMO-15KW",
    rated_power_kw=15.0,
    rated_voltage_v=460.0,
    rated_current_a=22.0,
    supply_frequency_hz=60.0,
    poles=4,
    rated_speed_rpm=1750.0,
    rated_efficiency=0.91,
    rotor_slots=28,
    bearings=[
        BearingSpec(position=BearingPosition.DRIVE_END, designation="6205"),
        BearingSpec(position=BearingPosition.NON_DRIVE_END, designation="6203"),
    ],
)


def main(fault: SimFault, severity: float) -> int:
    config = SimulationConfig(load_factor=0.8, duration_s=16.0, seed=2)
    faulty = simulate(DEMO_SPEC, config, faults=[FaultInjection(fault=fault, severity=severity)])
    healthy = simulate(DEMO_SPEC, config)
    f_s = DEMO_SPEC.supply_frequency_hz
    spec_f = welch_spectrum(faulty.currents[0], faulty.sample_rate_hz, RESOLUTION_HZ)
    spec_h = welch_spectrum(healthy.currents[0], healthy.sample_rate_hz, RESOLUTION_HZ)
    gt = faulty.ground_truth
    injected = {round(c.frequency_hz, 9) for c in gt.components if c.fault == fault}
    lines = [b for b in faulty.fault_map.bins if round(b.frequency_hz, 9) in injected]

    print(f"Fault: {fault.value}, severity {severity}, load {gt.load_factor:.0%}")
    print(f"True slip {gt.slip:.5f}, speed {gt.rotor_speed_rpm:.1f} rpm")
    print(f"Resolution {spec_f.resolution_hz:.3f} Hz, {spec_f.n_segments} Welch segments\n")
    print(
        f"{'line':22} {'equation':32} {'predicted Hz':>12} {'measured Hz':>12} "
        f"{'error Hz':>9} {'faulty dB':>10} {'healthy dB':>11}"
    )
    worst = 0.0
    for b in sorted(lines, key=lambda b: b.frequency_hz):
        tone = spec_f.tone(b.frequency_hz, search_halfwidth_hz=2 * spec_f.resolution_hz)
        err = tone.frequency_hz - b.frequency_hz
        worst = max(worst, abs(err))
        print(
            f"{b.label:22} {b.equation:32} {b.frequency_hz:12.3f} {tone.frequency_hz:12.3f} "
            f"{err:+9.3f} {spec_f.db_relative(b.frequency_hz, f_s, 0.25):10.1f} "
            f"{spec_h.db_relative(b.frequency_hz, f_s, 0.25):11.1f}"
        )
    ok = worst <= spec_f.resolution_hz
    print(f"\nWorst frequency error {worst:.3f} Hz ({'PASS' if ok else 'FAIL'}: <= one bin)")
    return 0 if ok else 1


if __name__ == "__main__":
    fault_arg = SimFault(sys.argv[1]) if len(sys.argv) > 1 else SimFault.BROKEN_ROTOR_BAR
    severity_arg = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
    sys.exit(main(fault_arg, severity_arg))
