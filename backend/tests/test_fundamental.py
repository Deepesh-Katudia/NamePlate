"""Supply-tone cancellation tests.

Motivation: with a Hann window, an off-bin fundamental leaks a steep skirt tens of dB above
the noise floor for several Hz either side. Rotor-bar sidebands live inside that skirt.
Cancelling the fundamental (and odd supply harmonics) before the Welch estimate removes the
skirt while leaving fault sidebands untouched.
"""

import numpy as np
import pytest

from backend.signal.fundamental import cancel_supply_tones, refine_frequency
from backend.signal.spectrum import welch_spectrum

FS = 5000.0
T = np.arange(int(16 * FS)) / FS
RES = 0.146  # 50 Hz is deliberately off-bin at this resolution


def three(x):
    return np.array([x, x, x])


def level_db(spec, f, ref_power):
    return 10 * np.log10(spec.tone(f, 2 * spec.resolution_hz).power / ref_power)


class TestRefineFrequency:
    @pytest.mark.parametrize("f_true", [50.0, 50.0123, 59.987])
    def test_recovers_frequency_to_1e_4_hz(self, f_true):
        x = np.cos(2 * np.pi * f_true * T + 0.3)
        assert refine_frequency(x, FS, round(f_true)) == pytest.approx(f_true, abs=1e-4)


class TestCancellation:
    def test_leakage_skirt_removed(self):
        rng = np.random.default_rng(0)
        x = np.cos(2 * np.pi * 50.0 * T) + rng.normal(scale=1e-3, size=T.size)
        raw = welch_spectrum(x, FS, RES)
        clean = welch_spectrum(cancel_supply_tones(three(x), FS, 50.0).residual[0], FS, RES)
        ref = 0.5  # fundamental power, A_rms^2
        for f in (51.0, 51.5, 52.0):
            assert level_db(raw, f, ref) > level_db(clean, f, ref) + 15
        # after cancellation every bin near the fundamental sits at the noise floor (about
        # -92 dB here); the remaining spread is Welch estimator variance (3 segments), not a
        # skirt (the raw spectrum spans ~40 dB over the same range)
        clean_levels = [level_db(clean, f, ref) for f in np.arange(51.0, 52.6, 0.1)]
        raw_levels = [level_db(raw, f, ref) for f in np.arange(51.0, 52.6, 0.1)]
        assert max(clean_levels) < -85.0
        assert max(clean_levels) - min(clean_levels) < 8.0
        assert max(raw_levels) - min(raw_levels) > 25.0

    def test_sideband_on_the_skirt_is_preserved(self):
        rng = np.random.default_rng(1)
        sideband = 10 ** (-60 / 20)
        x = (
            np.cos(2 * np.pi * 50.0 * T)
            + sideband * np.cos(2 * np.pi * 51.5 * T)
            + rng.normal(scale=1e-3, size=T.size)
        )
        result = cancel_supply_tones(three(x), FS, 50.0)
        clean = welch_spectrum(result.residual[0], FS, RES)
        assert level_db(clean, 51.5, result.fundamental_power) == pytest.approx(-60.0, abs=0.7)

    def test_fundamental_power_and_frequency_reported(self):
        x = 2.0 * np.cos(2 * np.pi * 50.01 * T)
        result = cancel_supply_tones(three(x), FS, 50.0)
        assert result.fundamental_power == pytest.approx(2.0, rel=1e-3)  # (2/sqrt2)^2
        assert result.fundamental_hz == pytest.approx(50.01, abs=1e-4)

    def test_odd_harmonics_cancelled_even_left(self):
        x = np.cos(2 * np.pi * 50 * T) + 0.03 * np.cos(2 * np.pi * 250 * T)
        x = x + 0.01 * np.cos(2 * np.pi * 100 * T)  # not a supply harmonic we remove
        result = cancel_supply_tones(three(x), FS, 50.0)
        clean = welch_spectrum(result.residual[0], FS, 0.25)
        assert level_db(clean, 250.0, 0.5) < -80
        assert level_db(clean, 100.0, 0.5) == pytest.approx(-40.0, abs=0.5)

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match="shape"):
            cancel_supply_tones(np.zeros(100), FS, 50.0)
