"""Load estimation tests."""

import numpy as np
import pytest
from pydantic import ValidationError

from backend.energy.load import (
    LossSplit,
    MissingParameterError,
    estimate_load,
    input_power_w,
    loss_breakdown,
)
from backend.simulator.motor_sim import SimulationConfig, simulate
from backend.tests.test_physics import make_spec


class TestInputPower:
    def test_balanced_resistive_load(self):
        fs, f = 10_000.0, 50.0
        t = np.arange(int(fs)) / fs
        shifts = (0.0, -2 * np.pi / 3, 2 * np.pi / 3)
        v = np.array([325.0 * np.cos(2 * np.pi * f * t + s) for s in shifts])
        i = np.array([10.0 * np.cos(2 * np.pi * f * t + s) for s in shifts])
        # 3 * Vrms * Irms = 3 * (325/√2)(10/√2) = 4875 W
        assert input_power_w(v, i) == pytest.approx(4875.0, rel=1e-6)


class TestLossModelAgainstNameplate:
    """Independent of the simulator: the loss model must reproduce the nameplate rated point."""

    def test_losses_at_rated_current_equal_rated_losses(self):
        spec = make_spec()  # 15 kW, eta 0.91 -> 15000 * (1/0.91 - 1) = 1483.5 W
        assert loss_breakdown(spec, spec.rated_current_a).total_w == pytest.approx(1483.5, abs=0.1)

    def test_hand_built_waveforms_give_hand_calculated_load(self):
        """P_in = 3 * 265.58 V * 20 A * cos(0.5) = 13 985 W, built without the simulator.

        Losses at 20 A: fixed 0.30 * 1483.5 = 445.0 W; load-dependent
        0.70 * 1483.5 * (20/22)^2 = 858.2 W -> P_shaft = 12 681.8 W -> load 0.8455.
        """
        spec = make_spec()
        fs = 10_000.0
        t = np.arange(int(fs)) / fs
        v_ph, i_rms, phi = 460 / np.sqrt(3), 20.0, 0.5
        shifts = (0.0, -2 * np.pi / 3, 2 * np.pi / 3)
        w = 2 * np.pi * 60.0 * t
        v = np.array([np.sqrt(2) * v_ph * np.cos(w + s) for s in shifts])
        i = np.array([np.sqrt(2) * i_rms * np.cos(w + s - phi) for s in shifts])
        est = estimate_load(v, i, spec)
        assert est.input_power_w == pytest.approx(13985.0, rel=1e-3)
        assert est.load_factor == pytest.approx(0.8455, abs=1e-3)


class TestLoadEstimate:
    """Round-trip tests. The simulator and estimator share the loss model, so these check
    the inversion, not the loss model itself (see TestLossModelAgainstNameplate)."""

    @pytest.mark.parametrize("load", [0.3, 0.6, 0.9])
    @pytest.mark.parametrize("stator_r", [None, 0.45])
    def test_recovers_simulated_load_within_band(self, load, stator_r):
        spec = make_spec(stator_resistance_ohm=stator_r)
        sim = simulate(spec, SimulationConfig(load_factor=load, duration_s=2.0, seed=4))
        est = estimate_load(sim.voltages, sim.currents, spec)
        assert est.load_factor == pytest.approx(load, abs=0.02)
        assert est.load_factor_low <= load <= est.load_factor_high

    def test_efficiency_is_labelled_indicative(self):
        spec = make_spec()
        sim = simulate(spec, SimulationConfig(load_factor=0.8, duration_s=1.0, seed=4))
        est = estimate_load(sim.voltages, sim.currents, spec)
        assert est.efficiency_is_indicative_only is True
        assert 0.5 < est.efficiency_indicative < 1.0

    def test_unknown_stator_resistance_widens_band_and_is_disclosed(self):
        known = make_spec(stator_resistance_ohm=0.45)
        unknown = make_spec()
        sim = simulate(known, SimulationConfig(load_factor=0.7, duration_s=1.0, seed=4))
        a = estimate_load(sim.voltages, sim.currents, known)
        b = estimate_load(sim.voltages, sim.currents, unknown)
        assert (b.load_factor_high - b.load_factor_low) > (a.load_factor_high - a.load_factor_low)
        assert any("stator_resistance_ohm" in s for s in b.assumptions)

    def test_requires_rated_efficiency(self):
        spec = make_spec()
        sim = simulate(spec, SimulationConfig(load_factor=0.7, duration_s=1.0, seed=4))
        with pytest.raises(MissingParameterError, match="rated_efficiency"):
            estimate_load(sim.voltages, sim.currents, make_spec(rated_efficiency=None))

    def test_loss_split_must_sum_to_one(self):
        with pytest.raises(ValidationError):
            LossSplit(
                stator_copper=0.5, rotor_copper=0.5, core=0.2, friction_windage=0.1, stray=0.1
            )
