"""Physics engine tests.

Reference values:
- Synchronous speed / slip / BRB sidebands: textbook relations, hand-computed.
- Bearing defect frequencies: Case Western Reserve University Bearing Data Center,
  SKF 6205-2RS JEM (drive end) and 6203-2RS JEM (fan end) defect-frequency tables,
  expressed as multiples of shaft speed.
- Rotor-bar severity bands: Thomson & Fenger, IEEE Industry Applications Magazine, 2001.
"""

import json

import pytest
from pydantic import ValidationError

from backend.models.fault_map import FaultClass, OperatingPointSource
from backend.models.motor import BearingGeometry, BearingPosition, BearingSpec, MotorSpec
from backend.physics.bearings import (
    BEARING_DATABASE,
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
    synchronous_speed_rpm,
)
from backend.physics.fault_map import build_fault_map
from backend.physics.severity import (
    RotorBarSeverity,
    RotorBarSeverityConfig,
    SeverityBand,
    classify_rotor_bar,
    load_rotor_bar_config,
)

DE_6205 = BearingSpec(position=BearingPosition.DRIVE_END, designation="6205")
NDE_6203 = BearingSpec(position=BearingPosition.NON_DRIVE_END, designation="6203")


def make_spec(**overrides) -> MotorSpec:
    base = dict(
        asset_id="M-001",
        rated_power_kw=15.0,
        rated_voltage_v=460.0,
        rated_current_a=22.0,
        supply_frequency_hz=60.0,
        poles=4,
        rated_speed_rpm=1750.0,
        rated_efficiency=0.91,
        rotor_slots=28,
        bearings=[DE_6205, NDE_6203],
    )
    base.update(overrides)
    return MotorSpec(**base)


# --- core relations -------------------------------------------------------------------


class TestCoreRelations:
    def test_four_pole_sixty_hz_synchronous_speed(self):
        assert synchronous_speed_rpm(60.0, 4) == pytest.approx(1800.0)

    def test_slip_at_1750_rpm(self):
        assert per_unit_slip(1800.0, 1750.0) == pytest.approx(0.027778, abs=1e-5)

    def test_rotor_frequency(self):
        assert rotor_frequency_hz(1750.0) == pytest.approx(29.1667, abs=1e-4)

    def test_odd_pole_count_rejected(self):
        with pytest.raises(ValueError, match="even"):
            synchronous_speed_rpm(60.0, 3)

    def test_negative_slip_rejected(self):
        with pytest.raises(ValueError, match="slip"):
            per_unit_slip(1800.0, 1810.0)


# --- broken rotor bar -----------------------------------------------------------------


class TestBrokenRotorBar:
    def test_k1_sidebands_at_1750_rpm(self):
        s = per_unit_slip(1800.0, 1750.0)
        bins = broken_rotor_bar_frequencies(60.0, s)
        k1 = {b.sideband: b.frequency_hz for b in bins if b.harmonic == 1}
        assert k1["lower"] == pytest.approx(56.667, abs=1e-3)
        assert k1["upper"] == pytest.approx(63.333, abs=1e-3)

    def test_returns_both_sidebands_for_k_1_to_3(self):
        bins = broken_rotor_bar_frequencies(60.0, 0.03)
        assert sorted((b.harmonic, b.sideband) for b in bins) == sorted(
            (k, side) for k in (1, 2, 3) for side in ("lower", "upper")
        )

    def test_only_k1_lower_is_primary(self):
        bins = broken_rotor_bar_frequencies(60.0, 0.03)
        primary = [b for b in bins if b.is_primary]
        assert len(primary) == 1
        assert (primary[0].harmonic, primary[0].sideband) == (1, "lower")

    def test_bins_carry_their_equation_and_inputs(self):
        b = broken_rotor_bar_frequencies(50.0, 0.02)[0]
        assert "f_s" in b.equation and "k" in b.equation
        assert b.inputs["f_s"] == 50.0 and b.inputs["s"] == 0.02


# --- eccentricity and PSH -------------------------------------------------------------


class TestEccentricityAndPsh:
    def test_eccentricity_sidebands(self):
        f_r = rotor_frequency_hz(1750.0)
        freqs = {
            (b.harmonic, b.sideband): b.frequency_hz for b in eccentricity_frequencies(60.0, f_r)
        }
        assert freqs[(1, "lower")] == pytest.approx(60.0 - f_r)
        assert freqs[(1, "upper")] == pytest.approx(60.0 + f_r)
        assert freqs[(3, "upper")] == pytest.approx(60.0 + 3 * f_r)
        assert len(freqs) == 6

    def test_principal_slot_harmonic(self):
        # R=28, p=2, s=1/36, f_s=60: (R/p)(1-s) = 14 * 35/36
        bins = principal_slot_harmonic_frequencies(60.0, rotor_slots=28, pole_pairs=2, slip=1 / 36)
        freqs = {b.sideband: b.frequency_hz for b in bins}
        assert freqs["lower"] == pytest.approx(60.0 * (14 * 35 / 36 - 1))
        assert freqs["upper"] == pytest.approx(60.0 * (14 * 35 / 36 + 1))


# --- bearings -------------------------------------------------------------------------

# CWRU published defect frequencies, multiples of shaft speed.
CWRU_6205 = {"BPFI": 5.4152, "BPFO": 3.5848, "FTF": 0.39828, "BSF": 2.3568}
CWRU_6203 = {"BPFI": 4.9469, "BPFO": 3.0530, "FTF": 0.3817, "BSF": 1.9944}


class TestBearings:
    @pytest.mark.parametrize("designation,reference", [("6205", CWRU_6205), ("6203", CWRU_6203)])
    def test_matches_published_cwru_values(self, designation, reference):
        f_r = 1797 / 60  # CWRU 0 hp test speed
        chars = bearing_characteristic_frequencies(lookup_bearing(designation).geometry, f_r)
        for name, multiple in reference.items():
            assert getattr(chars, name.lower()) == pytest.approx(multiple * f_r, rel=2e-3), name

    def test_reference_bearings_are_marked_as_published(self):
        assert lookup_bearing("6205").provenance == GeometryProvenance.PUBLISHED_REFERENCE

    @pytest.mark.parametrize("designation", sorted(BEARING_DATABASE))
    def test_bpfi_exceeds_bpfo_for_every_bearing(self, designation):
        chars = bearing_characteristic_frequencies(lookup_bearing(designation).geometry, 25.0)
        assert chars.bpfi > chars.bpfo

    def test_lookup_normalises_designation_suffixes(self):
        assert lookup_bearing("6205-2RS").geometry == lookup_bearing("6205").geometry
        assert lookup_bearing(" 6205 zz ").geometry == lookup_bearing("6205").geometry

    def test_unknown_bearing_raises_typed_error_naming_parameters(self):
        with pytest.raises(UnknownBearingError) as exc_info:
            lookup_bearing("XYZ-999")
        err = exc_info.value
        assert err.designation == "XYZ-999"
        assert set(err.required_parameters) == {
            "ball_count",
            "ball_diameter_mm",
            "pitch_diameter_mm",
            "contact_angle_rad",
        }
        for param in err.required_parameters:
            assert param in str(err)

    def test_geometry_rejects_ball_larger_than_pitch(self):
        with pytest.raises(ValidationError):
            BearingGeometry(
                ball_count=9, ball_diameter_mm=40.0, pitch_diameter_mm=39.0, contact_angle_rad=0.0
            )

    def test_geometry_rejects_balls_that_cannot_fit_on_pitch_circle(self):
        with pytest.raises(ValidationError, match="cannot fit"):
            BearingGeometry(
                ball_count=20, ball_diameter_mm=10.0, pitch_diameter_mm=15.0, contact_angle_rad=0.0
            )

    def test_geometry_rejects_contact_angle_at_ninety_degrees(self):
        with pytest.raises(ValidationError):
            BearingGeometry(
                ball_count=9,
                ball_diameter_mm=7.94,
                pitch_diameter_mm=39.04,
                contact_angle_rad=1.57080,
            )

    def test_every_database_entry_is_physically_valid(self):
        # construction already validates; re-validate explicitly to guard future edits
        for record in BEARING_DATABASE.values():
            BearingGeometry.model_validate(record.geometry.model_dump())

    def test_current_bins_are_absolute_and_cover_all_characteristics(self):
        chars = bearing_characteristic_frequencies(lookup_bearing("6205").geometry, 29.95)
        bins = bearing_current_frequencies(60.0, chars)
        assert len(bins) == 4 * 2 * 2  # 4 characteristics x k=1,2 x two sidebands
        assert all(b.frequency_hz >= 0 for b in bins)
        outer_k1_lower = next(
            b for b in bins if b.source == "BPFO" and b.harmonic == 1 and b.sideband == "lower"
        )
        assert outer_k1_lower.frequency_hz == pytest.approx(abs(60.0 - chars.bpfo))


# --- MotorSpec validation -------------------------------------------------------------


class TestMotorSpecValidation:
    def test_valid_spec(self):
        spec = make_spec()
        assert spec.synchronous_speed_rpm == pytest.approx(1800.0)
        assert spec.rated_slip == pytest.approx(1 / 36)

    def test_rejects_odd_pole_count(self):
        with pytest.raises(ValidationError, match="even"):
            make_spec(poles=5)

    def test_rejects_rotor_speed_above_synchronous(self):
        with pytest.raises(ValidationError, match="synchronous"):
            make_spec(rated_speed_rpm=1850.0)

    def test_rejects_rotor_speed_equal_to_synchronous(self):
        with pytest.raises(ValidationError, match="synchronous"):
            make_spec(rated_speed_rpm=1800.0)

    @pytest.mark.parametrize("power", [0.0, -5.0])
    def test_rejects_non_positive_rated_power(self, power):
        with pytest.raises(ValidationError):
            make_spec(rated_power_kw=power)

    def test_rejects_efficiency_above_one(self):
        with pytest.raises(ValidationError):
            make_spec(rated_efficiency=1.2)

    def test_rejects_duplicate_bearing_positions(self):
        with pytest.raises(ValidationError, match="position"):
            make_spec(bearings=[DE_6205, DE_6205])

    def test_spec_is_immutable(self):
        spec = make_spec()
        with pytest.raises(ValidationError):
            spec.poles = 6


# --- fault map ------------------------------------------------------------------------


class TestFaultMap:
    def test_full_spec_covers_all_fault_classes(self):
        fmap = build_fault_map(make_spec())
        assert fmap.unavailable == []
        assert fmap.operating_point.source == OperatingPointSource.NAMEPLATE_RATED
        classes = {b.fault_class for b in fmap.bins}
        assert {
            FaultClass.BROKEN_ROTOR_BAR,
            FaultClass.ECCENTRICITY,
            FaultClass.BEARING_OUTER,
            FaultClass.BEARING_INNER,
            FaultClass.BEARING_BALL,
            FaultClass.BEARING_CAGE,
        } <= classes
        assert fmap.slot_harmonics  # rotor slots known -> PSH available

    def test_missing_bearing_still_monitors_rotor_and_eccentricity(self):
        fmap = build_fault_map(make_spec(bearings=[]))
        classes = {b.fault_class for b in fmap.bins}
        assert FaultClass.BROKEN_ROTOR_BAR in classes
        assert FaultClass.ECCENTRICITY in classes
        assert not classes & {FaultClass.BEARING_OUTER, FaultClass.BEARING_INNER}
        unavailable = {u.fault_class for u in fmap.unavailable}
        assert FaultClass.BEARING_OUTER in unavailable
        assert any("bearing" in c.parameter for c in fmap.needs_confirmation)

    def test_unknown_bearing_surfaces_confirmation_not_default(self):
        spec = make_spec(
            bearings=[
                BearingSpec(position=BearingPosition.DRIVE_END, designation="ACME-77"),
                NDE_6203,
            ]
        )
        fmap = build_fault_map(spec)
        assert not any(b.bearing_position == "DE" for b in fmap.bins)
        assert any(b.bearing_position == "NDE" for b in fmap.bins)
        assert {u.bearing_position for u in fmap.unavailable} == {"DE"}
        item = next(c for c in fmap.needs_confirmation if "ACME-77" in c.reason)
        assert "pitch_diameter_mm" in item.reason

    def test_user_supplied_geometry_used_for_unknown_designation(self):
        geometry = BearingGeometry(
            ball_count=9, ball_diameter_mm=7.94, pitch_diameter_mm=39.04, contact_angle_rad=0.0
        )
        spec = make_spec(
            bearings=[
                BearingSpec(
                    position=BearingPosition.DRIVE_END, designation="ACME-77", geometry=geometry
                ),
                NDE_6203,
            ]
        )
        fmap = build_fault_map(spec)
        assert any(b.fault_class == FaultClass.BEARING_OUTER for b in fmap.bins)
        assert fmap.needs_confirmation == []

    def test_supplied_geometry_conflicting_with_database_is_flagged(self):
        typo = BearingGeometry(
            ball_count=8, ball_diameter_mm=7.94, pitch_diameter_mm=39.04, contact_angle_rad=0.0
        )
        spec = make_spec(
            bearings=[
                BearingSpec(position=BearingPosition.DRIVE_END, designation="6205", geometry=typo),
                NDE_6203,
            ]
        )
        fmap = build_fault_map(spec)
        assert any("differs from the database" in c.reason for c in fmap.needs_confirmation)

    def test_missing_rotor_slots_requires_confirmation(self):
        fmap = build_fault_map(make_spec(rotor_slots=None))
        assert fmap.slot_harmonics == []
        assert any(c.parameter == "rotor_slots" for c in fmap.needs_confirmation)

    def test_nominal_geometry_bearing_is_flagged(self):
        spec = make_spec(
            bearings=[BearingSpec(position=BearingPosition.DRIVE_END, designation="6309"), NDE_6203]
        )
        fmap = build_fault_map(spec)
        assert lookup_bearing("6309").provenance == GeometryProvenance.NOMINAL
        assert any("6309" in c.reason for c in fmap.needs_confirmation)

    def test_single_bearing_leaves_other_position_unavailable(self):
        fmap = build_fault_map(make_spec(bearings=[DE_6205]))
        assert {u.bearing_position for u in fmap.unavailable} == {"NDE"}
        assert any(c.parameter == "bearing_designation[NDE]" for c in fmap.needs_confirmation)

    def test_every_bin_traces_to_an_equation(self):
        fmap = build_fault_map(make_spec())
        assert all(b.equation and b.inputs for b in [*fmap.bins, *fmap.slot_harmonics])

    def test_measured_operating_point_overrides_rated(self):
        fmap = build_fault_map(make_spec(), rotor_speed_rpm=1770.0)
        assert fmap.operating_point.source == OperatingPointSource.MEASURED
        assert fmap.operating_point.slip == pytest.approx(30 / 1800)

    def test_measured_speed_above_synchronous_rejected(self):
        with pytest.raises(ValueError, match="slip"):
            build_fault_map(make_spec(), rotor_speed_rpm=1805.0)


# --- severity -------------------------------------------------------------------------


class TestRotorBarSeverity:
    @pytest.mark.parametrize(
        "db,expected",
        [
            (-60.0, RotorBarSeverity.EXCELLENT),
            (-50.0, RotorBarSeverity.GOOD),
            (-45.0, RotorBarSeverity.MODERATE),
            (-40.0, RotorBarSeverity.CRACKED_BAR_OR_HIGH_RESISTANCE_JOINT),
            (-33.0, RotorBarSeverity.BROKEN_BARS),
            (-25.0, RotorBarSeverity.MULTIPLE_BROKEN_BARS),
        ],
    )
    def test_bands(self, db, expected):
        assert classify_rotor_bar(db, load_factor=0.8).state == expected

    @pytest.mark.parametrize(
        "db,expected",
        [
            (-54.0, RotorBarSeverity.EXCELLENT),
            (-48.0, RotorBarSeverity.GOOD),
            (-42.0, RotorBarSeverity.MODERATE),
            (-36.0, RotorBarSeverity.CRACKED_BAR_OR_HIGH_RESISTANCE_JOINT),
            (-30.0, RotorBarSeverity.BROKEN_BARS),
            (0.0, RotorBarSeverity.MULTIPLE_BROKEN_BARS),
        ],
    )
    def test_band_boundaries_are_inclusive_of_threshold(self, db, expected):
        assert classify_rotor_bar(db, load_factor=0.8).state == expected

    def test_load_exactly_at_threshold_is_assessed(self):
        assert classify_rotor_bar(-60.0, 0.5).state == RotorBarSeverity.EXCELLENT

    def test_insufficient_load_below_threshold(self):
        result = classify_rotor_bar(-60.0, load_factor=0.3)
        assert result.state == RotorBarSeverity.INSUFFICIENT_LOAD

    def test_sideband_above_fundamental_rejected(self):
        with pytest.raises(ValueError):
            classify_rotor_bar(3.0, load_factor=0.8)

    def test_default_config_loads_from_json_file(self):
        config = load_rotor_bar_config()
        assert config.min_load_factor == pytest.approx(0.5)
        assert len(config.bands) == 6

    def test_custom_config_changes_classification(self, tmp_path):
        custom = {
            "min_load_factor": 0.4,
            "bands": [
                {"min_db_down": 40.0, "state": "good", "description": "site-calibrated ok"},
                {"min_db_down": 0.0, "state": "broken_bars", "description": "site alarm"},
            ],
        }
        path = tmp_path / "bands.json"
        path.write_text(json.dumps(custom))
        config = load_rotor_bar_config(path)
        assert classify_rotor_bar(-45.0, 0.45, config).state == RotorBarSeverity.GOOD
        assert classify_rotor_bar(-35.0, 0.45, config).state == RotorBarSeverity.BROKEN_BARS

    def test_config_bands_must_end_at_zero(self):
        with pytest.raises(ValidationError):
            RotorBarSeverityConfig(
                min_load_factor=0.5,
                bands=[
                    SeverityBand(min_db_down=40.0, state=RotorBarSeverity.GOOD, description="x")
                ],
            )
