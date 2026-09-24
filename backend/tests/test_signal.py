"""Signal engine tests: spectrum, envelope, slip, symmetrical components."""

import cmath
import math

import numpy as np
import pytest

from backend.signal.envelope import envelope, envelope_spectrum
from backend.signal.slip import SlipEstimationUnavailableError, estimate_slip_psh
from backend.signal.spectrum import (
    InsufficientRecordError,
    average_spectra,
    min_record_seconds,
    required_resolution_hz,
    resolution_plan,
    welch_spectrum,
)
from backend.signal.symmetrical import (
    A,
    NegativeSequenceAdmittance,
    fundamental_phasor,
    negative_sequence_admittance_from_nameplate,
    sequence_components,
    unbalance_net_of_supply,
)
from backend.simulator.faults import FaultInjection, SimFault, SupplyUnbalance
from backend.simulator.motor_sim import SimulationConfig, simulate
from backend.tests.test_physics import make_spec

FS = 10_000.0


def tone(freqs_amps, duration=8.0, fs=FS):
    t = np.arange(int(duration * fs)) / fs
    return sum(a * np.cos(2 * np.pi * f * t) for f, a in freqs_amps)


# --- spectrum --------------------------------------------------------------------------


class TestSpectrum:
    def test_pure_tones_recovered(self):
        x = tone([(60.0, 1.0), (180.0, 0.1), (417.3, 0.01)])
        spec = welch_spectrum(x, FS, resolution_hz=0.25)
        for f, a in [(60.0, 1.0), (180.0, 0.1), (417.3, 0.01)]:
            peak = spec.tone(f, search_halfwidth_hz=1.0)
            assert peak.frequency_hz == pytest.approx(f, abs=0.05)
            assert peak.amplitude_peak == pytest.approx(a, rel=0.02)

    def test_resolution_at_least_as_fine_as_requested(self):
        spec = welch_spectrum(tone([(60.0, 1.0)]), FS, resolution_hz=0.3)
        assert spec.resolution_hz <= 0.3
        assert spec.n_segments >= 2

    def test_required_resolution_formula(self):
        assert required_resolution_hz(slip=0.03, supply_frequency_hz=60.0) == pytest.approx(0.9)

    def test_min_record_accounts_for_overlap(self):
        # 0.25 Hz -> 4 s segments; 5 averages at 50 % overlap -> (5 + 1) / 2 * 4 = 12 s
        assert min_record_seconds(0.25, n_averages=5) == pytest.approx(12.0)

    def test_short_record_rejected_with_required_length(self):
        with pytest.raises(InsufficientRecordError, match="s of data"):
            welch_spectrum(tone([(60.0, 1.0)], duration=1.0), FS, resolution_hz=0.25)

    def test_resolution_plan_for_asset(self):
        # rated slip 1/36 at 60 Hz, assessed down to 50 % load -> s = 1/72
        plan = resolution_plan(rated_slip=1 / 36, supply_frequency_hz=60.0, record_seconds=20.0)
        assert plan.required_resolution_hz == pytest.approx(0.5 * 60.0 / 72)
        assert plan.min_window_s == pytest.approx(2.4)
        assert plan.sufficient
        assert not resolution_plan(1 / 36, 60.0, record_seconds=3.0).sufficient

    def test_band_power_of_white_noise_equals_variance(self):
        rng = np.random.default_rng(0)
        x = rng.normal(scale=1.0, size=int(20 * FS))
        spec = welch_spectrum(x, FS, resolution_hz=1.0)
        assert spec.band_power(0.0, FS / 2) == pytest.approx(1.0, rel=0.02)

    def test_tone_near_nyquist_edge_is_rejected_not_truncated(self):
        spec = welch_spectrum(tone([(60.0, 1.0)]), FS, resolution_hz=0.25)
        with pytest.raises(ValueError, match="measurable"):
            spec.tone(FS / 2, search_halfwidth_hz=0.25)

    def test_average_spectra_means_power(self):
        a = welch_spectrum(tone([(60.0, 1.0)]), FS, resolution_hz=0.25)
        b = welch_spectrum(tone([(60.0, 3.0)]), FS, resolution_hz=0.25)
        avg = average_spectra([a, b])
        assert avg.tone(60.0, 0.5).power == pytest.approx((0.5 + 4.5) / 2, rel=1e-3)
        assert avg.n_segments == a.n_segments + b.n_segments

    def test_average_spectra_rejects_mismatched_grids(self):
        a = welch_spectrum(tone([(60.0, 1.0)]), FS, resolution_hz=0.25)
        b = welch_spectrum(tone([(60.0, 1.0)]), FS, resolution_hz=0.5)
        with pytest.raises(ValueError, match="grid"):
            average_spectra([a, b])

    def test_db_relative(self):
        x = tone([(60.0, 1.0), (56.0, 0.01)])
        spec = welch_spectrum(x, FS, resolution_hz=0.25)
        assert spec.db_relative(56.0, 60.0, 0.5) == pytest.approx(-40.0, abs=0.3)


# --- envelope --------------------------------------------------------------------------


class TestEnvelope:
    def test_am_carrier_modulation_frequency_recovered(self):
        t = np.arange(int(8 * FS)) / FS
        x = (1 + 0.2 * np.cos(2 * np.pi * 23.4 * t)) * np.cos(2 * np.pi * 60.0 * t)
        env = envelope(x)
        assert env.mean() == pytest.approx(1.0, rel=0.01)
        spec = envelope_spectrum(x, FS, resolution_hz=0.25)
        peak = spec.tone(23.4, search_halfwidth_hz=2.0)
        assert peak.frequency_hz == pytest.approx(23.4, abs=0.1)
        assert peak.amplitude_peak == pytest.approx(0.2, rel=0.05)
        # strongest envelope line is the modulation, not DC or the carrier
        assert spec.frequencies_hz[np.argmax(spec.power)] == pytest.approx(23.4, abs=0.25)


# --- symmetrical components ------------------------------------------------------------


class TestSymmetrical:
    def test_hand_calculated_negative_sequence(self):
        # Ia = 10∠0, Ib = 8∠-120°, Ic = 9∠120°
        ia, ib, ic = 10, 8 * cmath.exp(-2j * math.pi / 3), 9 * cmath.exp(2j * math.pi / 3)
        i1_hand = (ia + A * ib + A**2 * ic) / 3
        i2_hand = (ia + A**2 * ib + A * ic) / 3
        # direct hand values: I1 = (10 + 8 + 9)/3 = 9, I2 = (10 + 8a + 9a^2)/3
        assert abs(i1_hand) == pytest.approx(9.0)
        assert abs(i2_hand) == pytest.approx(abs(10 + 8 * A + 9 * A**2) / 3)
        assert abs(i2_hand) == pytest.approx(math.sqrt(3) / 3, rel=1e-9)  # |1.5 - j0.866| / 3

        t = np.arange(int(2 * FS)) / FS
        waves = [abs(p) * np.cos(2 * np.pi * 50 * t + cmath.phase(p)) for p in (ia, ib, ic)]
        comps = sequence_components([fundamental_phasor(w, FS, 50.0) for w in waves])
        assert abs(comps.positive) == pytest.approx(9.0, rel=1e-4)
        assert abs(comps.negative) == pytest.approx(math.sqrt(3) / 3, rel=1e-3)
        assert comps.unbalance_factor == pytest.approx((math.sqrt(3) / 3) / 9.0, rel=1e-3)

    def test_fundamental_phasor_recovers_rms_and_phase(self):
        t = np.arange(int(1 * FS)) / FS
        x = 5.0 * np.cos(2 * np.pi * 60 * t - 0.5) + 0.3 * np.cos(2 * np.pi * 300 * t)
        p = fundamental_phasor(x, FS, 60.0)
        assert abs(p) == pytest.approx(5.0, rel=1e-4)
        assert cmath.phase(p) == pytest.approx(-0.5, abs=1e-4)

    def test_hand_built_supply_unbalance_is_removed_independently_of_simulator(self):
        """Waveforms built by hand from I2 = Y2 * V2, not by the simulator.

        Motor: 460 V, 22 A, LRC ratio 6.5 -> |Y2| = 6.5 * 22 / (460/sqrt3) = 0.5384 S.
        Supply: V1 = 265.6 V rms, VUF 2 % -> |V2| = 5.312 V -> |I2| = 2.860 A.
        With I1 = 20 A the raw current unbalance is 2.860 / 20 = 14.3 %.
        """
        spec = make_spec(locked_rotor_current_ratio=6.5)
        v1, i1 = 460 / math.sqrt(3), 20.0
        v2 = 0.02 * v1
        i2 = 6.5 * 22.0 / v1 * v2
        assert i2 == pytest.approx(2.860, abs=1e-3)
        t = np.arange(int(2 * FS)) / FS
        w = 2 * np.pi * 60.0 * t

        def three_phase(pos, neg):
            return np.array(
                [
                    np.real(pos * np.exp(1j * (w - m * 2 * np.pi / 3)))
                    + np.real(neg * np.exp(1j * (w + m * 2 * np.pi / 3)))
                    for m in range(3)
                ]
            )

        voltages = three_phase(math.sqrt(2) * v1, math.sqrt(2) * v2)
        currents = three_phase(
            math.sqrt(2) * i1 * cmath.exp(-0.5j),
            math.sqrt(2) * i2 * cmath.exp(1j * math.radians(-70.0)),
        )
        result = unbalance_net_of_supply(
            currents, voltages, FS, 60.0, negative_sequence_admittance_from_nameplate(spec)
        )
        assert result.raw_current_unbalance == pytest.approx(0.143, abs=1e-3)
        assert result.voltage_unbalance == pytest.approx(0.02, rel=1e-3)
        assert result.net_current_unbalance == pytest.approx(0.0, abs=1e-3)

    def test_admittance_error_leaves_proportional_residual(self):
        """A 20 % admittance error must show up as ~20 % of the supply-driven part."""
        spec = make_spec(locked_rotor_current_ratio=6.5)
        sim = simulate(
            spec,
            SimulationConfig(duration_s=4.0, seed=1),
            confounders=[SupplyUnbalance(voltage_unbalance=0.02)],
        )
        true_y = negative_sequence_admittance_from_nameplate(spec)
        wrong_y = NegativeSequenceAdmittance(
            magnitude_siemens=0.8 * true_y.magnitude_siemens,
            angle_rad=None,
            source="nameplate_lrc_ratio",
        )
        good = unbalance_net_of_supply(sim.currents, sim.voltages, FS, 60.0, true_y)
        bad = unbalance_net_of_supply(sim.currents, sim.voltages, FS, 60.0, wrong_y)
        assert bad.net_current_unbalance == pytest.approx(
            0.2 * good.supply_attributable_unbalance, rel=0.05
        )

    def test_supply_unbalance_not_reported_as_winding_fault(self):
        spec = make_spec(locked_rotor_current_ratio=6.5)
        sim = simulate(
            spec,
            SimulationConfig(duration_s=4.0, seed=1),
            confounders=[SupplyUnbalance(voltage_unbalance=0.02)],
        )
        result = unbalance_net_of_supply(
            sim.currents,
            sim.voltages,
            sim.sample_rate_hz,
            spec.supply_frequency_hz,
            negative_sequence_admittance_from_nameplate(spec),
        )
        assert result.raw_current_unbalance > 0.08  # ~6.5x amplification of 2 % voltage
        assert result.net_current_unbalance is not None
        assert result.net_current_unbalance < 0.01

    def test_calibrated_vector_subtraction_reveals_turn_fault_under_supply_unbalance(self):
        spec = make_spec(locked_rotor_current_ratio=6.5)
        cfg = SimulationConfig(duration_s=4.0, seed=5)
        sim = simulate(
            spec,
            cfg,
            faults=[FaultInjection(fault=SimFault.STATOR_TURN_FAULT, severity=0.5)],
            confounders=[SupplyUnbalance(voltage_unbalance=0.02)],
        )
        nameplate = negative_sequence_admittance_from_nameplate(spec)
        calibrated = NegativeSequenceAdmittance(
            magnitude_siemens=nameplate.magnitude_siemens,
            angle_rad=math.radians(cfg.negative_sequence_angle_deg),
            source="commissioning_calibration",
        )
        result = unbalance_net_of_supply(
            sim.currents, sim.voltages, sim.sample_rate_hz, 60.0, calibrated
        )
        assert result.method == "vector"
        # injected turn-fault negative sequence is 0.10 * 0.5 = 5 % of I1
        assert result.net_current_unbalance == pytest.approx(0.05, abs=0.005)

    def test_net_unbalance_unavailable_without_admittance(self):
        spec = make_spec()
        sim = simulate(spec, SimulationConfig(duration_s=2.0, seed=1))
        result = unbalance_net_of_supply(
            sim.currents, sim.voltages, sim.sample_rate_hz, spec.supply_frequency_hz, None
        )
        assert result.net_current_unbalance is None
        assert "locked_rotor_current_ratio" in result.note

    def test_nameplate_admittance_requires_lrc_ratio(self):
        with pytest.raises(ValueError, match="locked_rotor_current_ratio"):
            negative_sequence_admittance_from_nameplate(make_spec())


# --- slip ------------------------------------------------------------------------------


class TestSlip:
    @pytest.mark.parametrize("load", [0.4, 0.75, 1.0])
    def test_psh_recovers_known_slip_within_0_3_percent(self, load):
        spec = make_spec()
        sim = simulate(spec, SimulationConfig(load_factor=load, duration_s=16.0, seed=3))
        est = estimate_slip_psh(
            welch_spectrum(sim.currents[0], sim.sample_rate_hz, resolution_hz=0.25), spec
        )
        assert est.slip == pytest.approx(sim.ground_truth.slip, abs=0.003)
        assert est.confidence > 0.7

    def test_unknown_rotor_slots_raises(self):
        spec = make_spec(rotor_slots=None)
        sim = simulate(spec, SimulationConfig(duration_s=8.0, seed=3))
        with pytest.raises(SlipEstimationUnavailableError, match="rotor_slots"):
            estimate_slip_psh(welch_spectrum(sim.currents[0], FS, 0.25), spec)

    def test_wrong_slot_count_lowers_confidence(self):
        # Motor truly has 28 slots; operator entered 26 -> PSH search finds nothing credible
        truth = make_spec()
        sim = simulate(truth, SimulationConfig(duration_s=16.0, seed=3))
        wrong = make_spec(rotor_slots=26)
        est = estimate_slip_psh(welch_spectrum(sim.currents[0], FS, 0.25), wrong)
        assert est.confidence < 0.5
