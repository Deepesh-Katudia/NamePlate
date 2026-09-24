"""Simulator tests: consistency with the physics engine and ground-truth labelling."""

import numpy as np
import pytest

from backend.models.fault_map import FaultClass
from backend.physics.fault_map import build_fault_map
from backend.signal.spectrum import welch_spectrum
from backend.signal.symmetrical import fundamental_phasor, sequence_components
from backend.simulator.faults import FaultInjection, LoadTransient, SimFault, SupplyUnbalance
from backend.simulator.motor_sim import SimulationConfig, simulate
from backend.tests.test_physics import make_spec


def spectrum_of(sim):
    return welch_spectrum(sim.currents[0], sim.sample_rate_hz, resolution_hz=0.25)


class TestSimulator:
    def test_deterministic_with_seed(self):
        a = simulate(make_spec(), SimulationConfig(duration_s=1.0, seed=7))
        b = simulate(make_spec(), SimulationConfig(duration_s=1.0, seed=7))
        np.testing.assert_array_equal(a.currents, b.currents)

    def test_outputs_are_read_only(self):
        sim = simulate(make_spec(), SimulationConfig(duration_s=1.0, seed=7))
        with pytest.raises(ValueError):
            sim.currents[0, 0] = 1.0

    def test_balanced_fundamental(self):
        spec = make_spec()
        sim = simulate(spec, SimulationConfig(duration_s=2.0, seed=1))
        phasors = [fundamental_phasor(c, sim.sample_rate_hz, 60.0) for c in sim.currents]
        comps = sequence_components(phasors)
        assert comps.unbalance_factor < 0.002

    def test_rated_load_draws_about_rated_current(self):
        spec = make_spec(rated_efficiency=0.91)
        sim = simulate(spec, SimulationConfig(load_factor=1.0, duration_s=2.0, seed=1))
        rms = float(np.sqrt(np.mean(sim.currents[0] ** 2)))
        assert rms == pytest.approx(spec.rated_current_a, rel=0.03)

    def test_slip_scales_with_load(self):
        spec = make_spec()
        sim = simulate(spec, SimulationConfig(load_factor=0.5, duration_s=1.0, seed=1))
        assert sim.ground_truth.slip == pytest.approx(0.5 * spec.rated_slip)

    def test_ground_truth_labels_faults_and_confounders(self):
        sim = simulate(
            make_spec(),
            SimulationConfig(duration_s=1.0, seed=1),
            faults=[FaultInjection(fault=SimFault.BROKEN_ROTOR_BAR, severity=0.5)],
            confounders=[LoadTransient(step_fraction=0.2, at_s=0.5)],
        )
        gt = sim.ground_truth
        assert [f.fault for f in gt.faults] == [SimFault.BROKEN_ROTOR_BAR]
        assert len(gt.confounders) == 1
        assert any(c.fault == SimFault.BROKEN_ROTOR_BAR for c in gt.components)

    @pytest.mark.parametrize(
        "fault,fault_class",
        [
            (SimFault.BROKEN_ROTOR_BAR, FaultClass.BROKEN_ROTOR_BAR),
            (SimFault.BEARING_OUTER, FaultClass.BEARING_OUTER),
            (SimFault.BEARING_INNER, FaultClass.BEARING_INNER),
            (SimFault.ECCENTRICITY, FaultClass.ECCENTRICITY),
        ],
    )
    def test_fault_components_sit_on_physics_engine_bins(self, fault, fault_class):
        spec = make_spec()
        sim = simulate(
            spec,
            SimulationConfig(duration_s=1.0, seed=1),
            faults=[FaultInjection(fault=fault, severity=0.8)],
        )
        fmap = build_fault_map(spec, rotor_speed_rpm=sim.ground_truth.rotor_speed_rpm)
        predicted = {
            round(b.frequency_hz, 9)
            for b in fmap.bins
            if b.fault_class == fault_class and b.bearing_position in (None, "DE")
        }
        injected = {round(c.frequency_hz, 9) for c in sim.ground_truth.components if c.fault}
        assert injected and injected <= predicted

    def test_injected_brb_appears_at_predicted_frequencies(self):
        spec = make_spec()
        sim = simulate(
            spec,
            SimulationConfig(load_factor=0.8, duration_s=16.0, seed=2),
            faults=[FaultInjection(fault=SimFault.BROKEN_ROTOR_BAR, severity=0.7)],
        )
        spectrum = spectrum_of(sim)
        fmap = build_fault_map(spec, rotor_speed_rpm=sim.ground_truth.rotor_speed_rpm)
        brb = [b for b in fmap.bins if b.fault_class == FaultClass.BROKEN_ROTOR_BAR]
        for b in brb:
            measured = spectrum.tone(b.frequency_hz, search_halfwidth_hz=spectrum.resolution_hz * 2)
            assert measured.frequency_hz == pytest.approx(
                b.frequency_hz, abs=spectrum.resolution_hz
            )
            assert spectrum.db_relative(b.frequency_hz, 60.0, 0.5) > -75.0

    def test_healthy_motor_has_no_brb_sideband_above_noise(self):
        spec = make_spec()
        sim = simulate(spec, SimulationConfig(load_factor=0.8, duration_s=16.0, seed=2))
        spectrum = spectrum_of(sim)
        fmap = build_fault_map(spec, rotor_speed_rpm=sim.ground_truth.rotor_speed_rpm)
        primary = next(b for b in fmap.bins if b.is_primary and b.fault_class == "broken_rotor_bar")
        assert spectrum.db_relative(primary.frequency_hz, 60.0, 0.25) < -70.0

    def test_stator_turn_fault_creates_negative_sequence_current(self):
        spec = make_spec()
        sim = simulate(
            spec,
            SimulationConfig(duration_s=2.0, seed=1),
            faults=[FaultInjection(fault=SimFault.STATOR_TURN_FAULT, severity=0.5)],
        )
        comps = sequence_components(
            [fundamental_phasor(c, sim.sample_rate_hz, 60.0) for c in sim.currents]
        )
        assert comps.unbalance_factor > 0.02

    def test_supply_unbalance_sets_voltage_negative_sequence(self):
        spec = make_spec(locked_rotor_current_ratio=6.0)
        sim = simulate(
            spec,
            SimulationConfig(duration_s=2.0, seed=1),
            confounders=[SupplyUnbalance(voltage_unbalance=0.02)],
        )
        v = sequence_components(
            [fundamental_phasor(c, sim.sample_rate_hz, 60.0) for c in sim.voltages]
        )
        assert v.unbalance_factor == pytest.approx(0.02, rel=0.01)

    def test_supply_unbalance_requires_lrc_ratio(self):
        with pytest.raises(ValueError, match="locked_rotor_current_ratio"):
            simulate(
                make_spec(),
                SimulationConfig(duration_s=1.0),
                confounders=[SupplyUnbalance(voltage_unbalance=0.02)],
            )

    def test_bearing_fault_requires_known_bearing(self):
        with pytest.raises(ValueError, match="NDE"):
            simulate(
                make_spec(bearings=[]),
                SimulationConfig(duration_s=1.0),
                faults=[
                    FaultInjection(
                        fault=SimFault.BEARING_OUTER, severity=0.5, bearing_position="NDE"
                    )
                ],
            )

    def test_simulator_requires_rated_efficiency(self):
        with pytest.raises(ValueError, match="rated_efficiency"):
            simulate(make_spec(rated_efficiency=None), SimulationConfig(duration_s=1.0))
