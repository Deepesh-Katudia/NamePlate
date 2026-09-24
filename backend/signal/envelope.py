"""Hilbert envelope and envelope spectrum.

A defect that modulates the current amplitude at f_char puts sidebands at f_s +/- f_char in the
current spectrum; demodulating with the analytic signal moves that energy down to f_char itself,
where it no longer competes with the fundamental's skirt.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import hilbert

from backend.signal.spectrum import Spectrum, welch_spectrum


def envelope(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Instantaneous amplitude |x + j*H{x}|."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"expected a 1-D signal (got shape {x.shape})")
    return np.abs(hilbert(x - np.mean(x)))


def envelope_spectrum(
    x: NDArray[np.float64], sample_rate_hz: float, resolution_hz: float
) -> Spectrum:
    """Welch spectrum of the mean-removed envelope."""
    env = envelope(x)
    return welch_spectrum(env - np.mean(env), sample_rate_hz, resolution_hz)
