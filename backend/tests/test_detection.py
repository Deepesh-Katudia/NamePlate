"""Detection tests: robust baselining, load buckets, sustained exceedance."""

import pytest

from backend.detection.baseline import (
    BinBaseline,
    DetectorConfig,
    LoadBucket,
    SustainedExceedanceDetector,
    robust_z,
)
from backend.detection.features import analyze_window, find_bin_conflicts
from backend.models.alert import BinMeasurement
from backend.models.fault_map import FaultClass
from backend.physics.fault_map import build_fault_map
from backend.simulator.faults import FaultInjection, SimFault
from backend.simulator.motor_sim import SimulationConfig, simulate
from backend.tests.test_physics import make_spec

CFG = DetectorConfig(min_baseline_windows=6, consecutive_windows=3, z_threshold=4.0)
HEALTHY = [-80.0, -81.0, -79.5, -80.5, -79.0, -80.2, -80.8, -79.7]


def m(key: str, db: float, fault_class=FaultClass.BROKEN_ROTOR_BAR) -> BinMeasurement:
    return BinMeasurement(
        key=key,
        label=key,
        fault_class=fault_class,
        bearing_position=None,
        frequency_hz=57.3,
        equation="f_brb = f_s * (1 +/- 2*k*s)",
        inputs={"f_s": 60.0},
        is_primary=True,
        level_db=db,
    )


def feed(det: SustainedExceedanceDetector, values, load=0.8, key="brb", asset="A1"):
    out = []
    for v in values:
        out.append(det.observe(asset, load, [m(key, v)]))
    return out


class TestRobustStatistics:
    def test_robust_z_uses_median_and_scaled_mad(self):
        values = (1.0, 2.0, 3.0, 4.0, 100.0)  # median 3, MAD 1
        assert robust_z(10.0, values, min_mad=0.0) == pytest.approx(7.0 / 1.4826)

    def test_outlier_does_not_move_baseline(self):
        base = BinBaseline(values=tuple(HEALTHY))
        spiked = BinBaseline(values=(*HEALTHY, -20.0))
        assert spiked.median == pytest.approx(base.median, abs=0.3)

    def test_mad_floor_prevents_division_blowup(self):
        assert robust_z(-79.0, (-80.0,) * 10, min_mad=0.5) == pytest.approx(1 / (1.4826 * 0.5))

    def test_baseline_is_immutable_and_bounded(self):
        b = BinBaseline(values=(1.0, 2.0))
        b2 = b.with_value(3.0, max_history=2)
        assert b.values == (1.0, 2.0)
        assert b2.values == (2.0, 3.0)


class TestLoadBuckets:
    @pytest.mark.parametrize(
        "load,bucket",
        [
            (0.1, LoadBucket.BELOW_25),
            (0.25, LoadBucket.FROM_25_TO_50),
            (0.6, LoadBucket.FROM_50_TO_75),
            (0.75, LoadBucket.ABOVE_75),
            (1.1, LoadBucket.ABOVE_75),
        ],
    )
    def test_bucket_edges(self, load, bucket):
        assert LoadBucket.for_load(load) == bucket

    def test_hysteresis_keeps_bucket_near_edge(self):
        assert LoadBucket.with_hysteresis(0.76, LoadBucket.FROM_50_TO_75, 0.05) == (
            LoadBucket.FROM_50_TO_75
        )
        assert LoadBucket.with_hysteresis(0.81, LoadBucket.FROM_50_TO_75, 0.05) == (
            LoadBucket.ABOVE_75
        )
        assert LoadBucket.with_hysteresis(0.74, None, 0.05) == LoadBucket.FROM_50_TO_75

    def test_load_jitter_across_edge_uses_one_baseline(self):
        det = SustainedExceedanceDetector(CFG)
        for i, v in enumerate(HEALTHY):
            det.observe("A1", 0.74 if i % 2 else 0.77, [m("brb", v)])
        assert not det.observe("A1", 0.76, [m("brb", -80.0)]).learning

    def test_buckets_have_independent_baselines(self):
        det = SustainedExceedanceDetector(CFG)
        feed(det, HEALTHY, load=0.8)
        results = feed(det, [-80.0], load=0.3)
        assert results[-1].learning  # the 25-50 % bucket has no baseline yet


class TestSustainedExceedance:
    def test_learning_until_min_windows(self):
        det = SustainedExceedanceDetector(CFG)
        results = feed(det, HEALTHY[:5])
        assert all(r.learning for r in results)
        assert not any(r.candidates for r in results)

    def test_single_window_spike_rejected(self):
        det = SustainedExceedanceDetector(CFG)
        feed(det, HEALTHY)
        results = feed(det, [-40.0, -80.0, -80.0, -79.0])
        assert not any(r.candidates for r in results)

    def test_two_windows_not_enough(self):
        det = SustainedExceedanceDetector(CFG)
        feed(det, HEALTHY)
        results = feed(det, [-40.0, -40.0, -80.0])
        assert not any(r.candidates for r in results)

    def test_sustained_exceedance_raises_candidate_once_threshold_met(self):
        det = SustainedExceedanceDetector(CFG)
        feed(det, HEALTHY)
        results = feed(det, [-40.0, -40.0, -40.0])
        assert not results[0].candidates and not results[1].candidates
        cand = results[2].candidates
        assert len(cand) == 1
        assert cand[0].fault_class == FaultClass.BROKEN_ROTOR_BAR
        evidence = cand[0].evidence[0]
        assert evidence.consecutive_windows == 3
        assert evidence.z_score > 4.0
        assert evidence.baseline_median_db == pytest.approx(-80.1, abs=0.3)
        assert evidence.equation.startswith("f_brb")

    def test_decrease_in_energy_is_not_a_fault(self):
        det = SustainedExceedanceDetector(CFG)
        feed(det, HEALTHY)
        results = feed(det, [-120.0] * 5)
        assert not any(r.candidates for r in results)

    def test_developing_fault_cannot_inflate_its_own_threshold(self):
        det = SustainedExceedanceDetector(CFG)
        feed(det, HEALTHY)
        feed(det, [-40.0] * 20)
        baseline = det.baseline("A1", LoadBucket.ABOVE_75, "brb")
        assert baseline.median == pytest.approx(-80.1, abs=0.3)
        # and it keeps being reported rather than normalised away
        assert feed(det, [-40.0])[0].candidates

    def test_fault_present_during_learning_is_absorbed_known_limitation(self):
        """Documents the bootstrap limitation surfaced as `baseline_assumption` in the API."""
        det = SustainedExceedanceDetector(CFG)
        results = feed(det, [-40.0] * 12)
        assert not any(r.candidates for r in results)

    def test_assets_are_isolated(self):
        det = SustainedExceedanceDetector(CFG)
        feed(det, HEALTHY, asset="A1")
        assert feed(det, [-80.0], asset="A2")[0].learning


class TestFeatures:
    def test_features_measure_injected_fault_above_healthy(self):
        spec = make_spec()
        cfg = SimulationConfig(load_factor=0.8, duration_s=16.0, seed=11, sample_rate_hz=5000.0)
        healthy = analyze_window(spec, simulate(spec, cfg))
        faulty = analyze_window(
            spec,
            simulate(
                spec, cfg, faults=[FaultInjection(fault=SimFault.BROKEN_ROTOR_BAR, severity=0.5)]
            ),
        )
        key = next(
            b.key
            for b in healthy.measurements
            if b.is_primary and b.fault_class == "broken_rotor_bar"
        )
        h = next(b for b in healthy.measurements if b.key == key)
        f = next(b for b in faulty.measurements if b.key == key)
        assert f.level_db - h.level_db > 30.0
        assert healthy.slip.source == "principal_slot_harmonic"
        assert healthy.load.load_factor == pytest.approx(0.8, abs=0.03)

    def test_slip_falls_back_to_load_scaled_nameplate_without_rotor_slots(self):
        spec = make_spec(rotor_slots=None)
        sim = simulate(spec, SimulationConfig(duration_s=16.0, seed=1, sample_rate_hz=5000.0))
        features = analyze_window(spec, sim)
        assert features.slip.source == "load_scaled_nameplate"
        assert features.slip.confidence < 0.5

    def test_bins_unresolvable_from_fundamental_are_excluded(self):
        # 10 % load -> s = 0.0028, 2*s*f_s = 0.33 Hz: inside the fundamental's main lobe
        spec = make_spec()
        fmap = build_fault_map(spec, rotor_speed_rpm=1800 * (1 - 0.1 * spec.rated_slip))
        conflicts = find_bin_conflicts(fmap, resolution_hz=0.25)
        brb_k1 = [c for c in conflicts.excluded if c.startswith("broken_rotor_bar") and "k=1" in c]
        assert brb_k1

    def test_close_bins_from_different_classes_marked_ambiguous(self):
        fmap = build_fault_map(make_spec())
        conflicts = find_bin_conflicts(fmap, resolution_hz=0.5)
        # NDE FTF upper k=1 (71.13 Hz) vs DE FTF upper k=1 (71.62 Hz) are the same class; BRB
        # upper k=3 at 70.0 Hz sits within 4 bins of both
        brb_key = next(k for k in conflicts.ambiguous if "BRB upper k=3" in k)
        assert any("bearing_cage" in other for other in conflicts.ambiguous[brb_key])

    def test_stator_feature_present_only_with_lrc_ratio(self):
        cfg = SimulationConfig(duration_s=16.0, seed=1, sample_rate_hz=5000.0)
        with_lrc = make_spec(locked_rotor_current_ratio=6.5)
        feats = analyze_window(with_lrc, simulate(with_lrc, cfg))
        assert any(b.fault_class == FaultClass.STATOR_WINDING for b in feats.measurements)
        without = make_spec()
        feats = analyze_window(without, simulate(without, cfg))
        assert not any(b.fault_class == FaultClass.STATOR_WINDING for b in feats.measurements)
