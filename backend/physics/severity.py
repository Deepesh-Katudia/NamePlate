"""Rotor-bar severity classification from lower-sideband amplitude.

Bands are configuration, not code: defaults live in `rotor_bar_severity.json` next to this
module and a site can pass its own calibrated file to `load_rotor_bar_config`.

Sideband amplitude scales with rotor current, so it falls with load. Below `min_load_factor`
the classifier returns INSUFFICIENT_LOAD instead of a verdict: a quiet sideband at light load
is not evidence of a healthy rotor.
"""

from __future__ import annotations

import json
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_CONFIG_PATH = Path(__file__).with_name("rotor_bar_severity.json")


class RotorBarSeverity(StrEnum):
    EXCELLENT = "excellent"
    GOOD = "good"
    MODERATE = "moderate"
    CRACKED_BAR_OR_HIGH_RESISTANCE_JOINT = "cracked_bar_or_high_resistance_joint"
    BROKEN_BARS = "broken_bars"
    MULTIPLE_BROKEN_BARS = "multiple_broken_bars"
    INSUFFICIENT_LOAD = "insufficient_load"


class SeverityBand(BaseModel):
    model_config = ConfigDict(frozen=True)

    min_db_down: float = Field(ge=0, description="Band applies when dB-down >= this value")
    state: RotorBarSeverity
    description: str


class RotorBarSeverityConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    min_load_factor: float = Field(gt=0, le=1)
    bands: list[SeverityBand] = Field(min_length=1)

    @model_validator(mode="after")
    def _bands_descending_and_complete(self) -> RotorBarSeverityConfig:
        thresholds = [b.min_db_down for b in self.bands]
        if thresholds != sorted(thresholds, reverse=True) or len(set(thresholds)) != len(
            thresholds
        ):
            raise ValueError("bands must be ordered by strictly descending min_db_down")
        if thresholds[-1] != 0.0:
            raise ValueError("last band must have min_db_down 0 so every amplitude is classified")
        if any(b.state == RotorBarSeverity.INSUFFICIENT_LOAD for b in self.bands):
            raise ValueError("INSUFFICIENT_LOAD is load-gated and cannot be an amplitude band")
        return self


class SeverityAssessment(BaseModel):
    model_config = ConfigDict(frozen=True)

    state: RotorBarSeverity
    description: str
    sideband_db_relative: float
    load_factor: float


def load_rotor_bar_config(path: Path | str | None = None) -> RotorBarSeverityConfig:
    """Load severity bands from JSON. Keys starting with '_' are treated as comments."""
    source = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw = json.loads(source.read_text(encoding="utf-8"))
    return RotorBarSeverityConfig(**{k: v for k, v in raw.items() if not k.startswith("_")})


@lru_cache(maxsize=1)
def default_rotor_bar_config() -> RotorBarSeverityConfig:
    return load_rotor_bar_config()


def classify_rotor_bar(
    sideband_db_relative: float,
    load_factor: float,
    config: RotorBarSeverityConfig | None = None,
) -> SeverityAssessment:
    """Classify the k=1 lower sideband, given in dB relative to the fundamental (<= 0)."""
    if sideband_db_relative > 0:
        raise ValueError(
            f"sideband amplitude {sideband_db_relative} dB exceeds the fundamental; "
            "check the fundamental estimate before classifying"
        )
    if load_factor < 0:
        raise ValueError(f"load_factor must be non-negative (got {load_factor})")
    cfg = config or default_rotor_bar_config()
    if load_factor < cfg.min_load_factor:
        return SeverityAssessment(
            state=RotorBarSeverity.INSUFFICIENT_LOAD,
            description=(
                f"Load factor {load_factor:.0%} below {cfg.min_load_factor:.0%}: sideband "
                "amplitude is not a reliable rotor indicator at this load"
            ),
            sideband_db_relative=sideband_db_relative,
            load_factor=load_factor,
        )
    db_down = -sideband_db_relative
    band = next(b for b in cfg.bands if db_down >= b.min_db_down)
    return SeverityAssessment(
        state=band.state,
        description=band.description,
        sideband_db_relative=sideband_db_relative,
        load_factor=load_factor,
    )
