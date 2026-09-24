"""Welch power spectrum with explicit frequency-resolution guarantees.

Scaling is 'spectrum' (A_rms^2 per bin), so a pure tone's power is read directly from its
main lobe: summing the Hann main lobe (+/-2 bins) and dividing by the window's equivalent noise
bandwidth (1.5 bins) recovers the tone's RMS^2 regardless of where it falls between bins.

Resolution requirement: the k=1 rotor-bar sidebands sit 2*s*f_s either side of the
fundamental. Bin spacing finer than 0.5*s*f_s puts at least four bins between the
fundamental and each sideband, clearing the Hann main lobe of the much larger fundamental.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch

HANN_ENBW_BINS = 1.5
MAIN_LOBE_HALF_WIDTH_BINS = 2
WELCH_OVERLAP = 0.5
_POWER_FLOOR = 1e-30


class InsufficientRecordError(ValueError):
    """The record is too short for the requested frequency resolution."""


@dataclass(frozen=True)
class Tone:
    frequency_hz: float
    power: float  # A_rms^2
    bin_index: int

    @property
    def amplitude_rms(self) -> float:
        return math.sqrt(self.power)

    @property
    def amplitude_peak(self) -> float:
        return math.sqrt(2.0 * self.power)


@dataclass(frozen=True)
class Spectrum:
    frequencies_hz: NDArray[np.float64]
    power: NDArray[np.float64]  # A_rms^2 per bin
    resolution_hz: float
    n_segments: int
    sample_rate_hz: float

    def _index(self, frequency_hz: float) -> int:
        return int(round(frequency_hz / self.resolution_hz))

    def tone(self, frequency_hz: float, search_halfwidth_hz: float) -> Tone:
        """Strongest line within +/- halfwidth of `frequency_hz`, with interpolated frequency."""
        # keep a full main lobe (+/-2 bins) inside the array so the power sum is never truncated
        margin = MAIN_LOBE_HALF_WIDTH_BINS
        lo = max(self._index(frequency_hz - search_halfwidth_hz), margin)
        hi = min(self._index(frequency_hz + search_halfwidth_hz), self.power.size - 1 - margin)
        if hi < lo:
            raise ValueError(f"{frequency_hz} Hz is outside the measurable spectrum range")
        k = lo + int(np.argmax(self.power[lo : hi + 1]))
        return Tone(
            frequency_hz=self._interpolate_peak(k), power=self._main_lobe_power(k), bin_index=k
        )

    def _main_lobe_power(self, k: int) -> float:
        w = MAIN_LOBE_HALF_WIDTH_BINS
        lobe = self.power[k - w : k + w + 1]
        return float(np.sum(lobe) / HANN_ENBW_BINS)

    def _interpolate_peak(self, k: int) -> float:
        """Parabolic interpolation on log power across the peak bin and its neighbours."""
        a, b, c = np.log(np.maximum(self.power[k - 1 : k + 2], _POWER_FLOOR))
        denom = a - 2 * b + c
        offset = 0.0 if denom == 0 else 0.5 * (a - c) / denom
        return float((k + np.clip(offset, -0.5, 0.5)) * self.resolution_hz)

    def db_relative(
        self, frequency_hz: float, reference_hz: float, search_halfwidth_hz: float
    ) -> float:
        """Line power at `frequency_hz` relative to the line at `reference_hz`, in dB."""
        num = self.tone(frequency_hz, search_halfwidth_hz).power
        ref = self.tone(reference_hz, search_halfwidth_hz).power
        return 10.0 * math.log10(max(num, _POWER_FLOOR) / max(ref, _POWER_FLOOR))

    def band_power(self, low_hz: float, high_hz: float) -> float:
        """Total power in [low, high] Hz, ENBW-corrected."""
        lo, hi = max(self._index(low_hz), 0), min(self._index(high_hz), self.power.size - 1)
        return float(np.sum(self.power[lo : hi + 1]) / HANN_ENBW_BINS)


def required_resolution_hz(slip: float, supply_frequency_hz: float) -> float:
    """Bin spacing needed to separate rotor-bar sidebands from the fundamental: 0.5*s*f_s."""
    if slip <= 0 or supply_frequency_hz <= 0:
        raise ValueError("slip and supply_frequency_hz must be positive")
    return 0.5 * slip * supply_frequency_hz


def min_record_seconds(resolution_hz: float, n_averages: int = 1) -> float:
    """Record length for `n_averages` Welch segments at 50 % overlap."""
    if resolution_hz <= 0 or n_averages < 1:
        raise ValueError("resolution_hz must be positive and n_averages >= 1")
    return (n_averages + 1) / 2.0 / resolution_hz


@dataclass(frozen=True)
class ResolutionPlan:
    required_resolution_hz: float
    min_window_s: float
    min_record_s: float
    achievable_resolution_hz: float
    sufficient: bool


def resolution_plan(
    rated_slip: float,
    supply_frequency_hz: float,
    record_seconds: float,
    min_load_factor: float = 0.5,
    n_averages: int = 4,
) -> ResolutionPlan:
    """Resolution needed at the lightest load that will be assessed (slip scales with load)."""
    required = required_resolution_hz(rated_slip * min_load_factor, supply_frequency_hz)
    achievable = (n_averages + 1) / 2.0 / record_seconds
    return ResolutionPlan(
        required_resolution_hz=required,
        min_window_s=1.0 / required,
        min_record_s=min_record_seconds(required, n_averages),
        achievable_resolution_hz=achievable,
        sufficient=achievable <= required,
    )


def average_spectra(spectra: list[Spectrum]) -> Spectrum:
    """Mean power across spectra on the same frequency grid (e.g. the three phase currents).

    Averaging phases lowers noise variance and cancels per-phase fundamental shifts caused by
    unbalance, which would otherwise move every fundamental-normalised level together.
    """
    if not spectra:
        raise ValueError("need at least one spectrum")
    first = spectra[0]
    if any(
        s.power.shape != first.power.shape or s.resolution_hz != first.resolution_hz
        for s in spectra
    ):
        raise ValueError("spectra must share frequency grid and resolution")
    return Spectrum(
        frequencies_hz=first.frequencies_hz,
        power=np.mean([s.power for s in spectra], axis=0),
        resolution_hz=first.resolution_hz,
        n_segments=sum(s.n_segments for s in spectra),
        sample_rate_hz=first.sample_rate_hz,
    )


def welch_spectrum(x: NDArray[np.float64], sample_rate_hz: float, resolution_hz: float) -> Spectrum:
    """Hann-windowed Welch spectrum, 50 % overlap, bin spacing <= `resolution_hz`."""
    if resolution_hz <= 0:
        raise ValueError("resolution_hz must be positive")
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"expected a 1-D signal (got shape {x.shape})")
    nperseg = int(math.ceil(sample_rate_hz / resolution_hz))
    if x.size < nperseg:
        raise InsufficientRecordError(
            f"{resolution_hz} Hz resolution requires {nperseg / sample_rate_hz:.2f} s of data; "
            f"record is {x.size / sample_rate_hz:.2f} s"
        )
    noverlap = int(nperseg * WELCH_OVERLAP)
    freqs, power = welch(
        x,
        fs=sample_rate_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="spectrum",
    )
    return Spectrum(
        frequencies_hz=freqs,
        power=power,
        resolution_hz=sample_rate_hz / nperseg,
        n_segments=1 + (x.size - nperseg) // (nperseg - noverlap),
        sample_rate_hz=sample_rate_hz,
    )
