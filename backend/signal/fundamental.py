"""Supply-tone cancellation ahead of spectral estimation.

The fundamental is 40-90 dB stronger than the fault sidebands next to it. With a Hann window
an off-bin fundamental leaks a skirt that buries sidebands a few hertz away, and the skirt's
level at a given bin shifts with small changes in where that bin falls. Rather than trade
window sidelobes against resolution, we remove the tone itself:

1. Refine the fundamental frequency from the phase slope of the demodulated signal
   (1 s blocks, linear fit to the unwrapped phase). This reaches ~1e-5 Hz over 16 s,
   which a spectral peak cannot.
2. Jointly least-squares fit cos/sin at the fundamental and odd supply harmonics, plus DC,
   per phase, and subtract.

Fault sidebands are at other frequencies, so over a long record they are nearly orthogonal to
the fitted tones and pass through unchanged. Even harmonics are left alone: they are not
supply harmonics of a balanced drive, and 2-pole eccentricity lines sit near 2*f_s.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

SUPPLY_HARMONICS = (1, 3, 5, 7, 9, 11, 13)
PHASE_BLOCK_S = 1.0
NYQUIST_GUARD = 0.95
REFINE_ITERATIONS = 2


@dataclass(frozen=True)
class CancellationResult:
    residual: NDArray[np.float64]  # shape (3, N)
    fundamental_hz: float
    fundamental_power: float  # mean A_rms^2 across phases
    harmonics_removed: tuple[int, ...]


def refine_frequency(x: NDArray[np.float64], sample_rate_hz: float, f_nominal: float) -> float:
    """Fundamental frequency from the slope of block-wise demodulated phase."""
    x = np.asarray(x, dtype=float)
    block = int(PHASE_BLOCK_S * sample_rate_hz)
    n_blocks = x.size // block
    if n_blocks < 3:
        raise ValueError("need at least three 1 s blocks to refine the frequency")
    f = f_nominal
    t = np.arange(n_blocks * block) / sample_rate_hz
    for _ in range(REFINE_ITERATIONS):
        demod = (x[: n_blocks * block] * np.exp(-2j * np.pi * f * t)).reshape(n_blocks, block)
        phase = np.unwrap(np.angle(demod.mean(axis=1)))
        centres = (np.arange(n_blocks) + 0.5) * PHASE_BLOCK_S
        slope = np.polyfit(centres, phase, 1)[0]
        f += slope / (2 * np.pi)
    return float(f)


def _fit_and_subtract(
    x: NDArray[np.float64], basis: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    coeffs, *_ = np.linalg.lstsq(basis, x, rcond=None)
    return x - basis @ coeffs, coeffs


def cancel_supply_tones(
    currents: NDArray[np.float64],
    sample_rate_hz: float,
    f_nominal: float,
    harmonics: tuple[int, ...] = SUPPLY_HARMONICS,
) -> CancellationResult:
    """Remove the fundamental and odd supply harmonics from each phase."""
    currents = np.asarray(currents, dtype=float)
    if currents.ndim != 2 or currents.shape[0] != 3:
        raise ValueError(f"expected shape (3, N) (got {currents.shape})")
    f1 = refine_frequency(currents[0], sample_rate_hz, f_nominal)
    orders = tuple(h for h in harmonics if h * f1 < NYQUIST_GUARD * sample_rate_hz / 2)
    t = np.arange(currents.shape[1]) / sample_rate_hz
    columns = [np.ones_like(t)]
    for h in orders:
        w = 2 * np.pi * h * f1 * t
        columns += [np.cos(w), np.sin(w)]
    basis = np.column_stack(columns)

    residuals, fundamental_powers = [], []
    for phase in currents:
        residual, coeffs = _fit_and_subtract(phase, basis)
        residuals.append(residual)
        fundamental_powers.append((coeffs[1] ** 2 + coeffs[2] ** 2) / 2)  # A_rms^2
    return CancellationResult(
        residual=np.array(residuals),
        fundamental_hz=f1,
        fundamental_power=float(np.mean(fundamental_powers)),
        harmonics_removed=orders,
    )


def fundamental_rms(result: CancellationResult) -> float:
    return math.sqrt(result.fundamental_power)
