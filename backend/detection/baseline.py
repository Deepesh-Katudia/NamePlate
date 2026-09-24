"""Robust per-asset, per-load-bucket baselines and sustained-exceedance detection.

Each monitored bin keeps a bounded history of its level (dB relative to the fundamental)
per asset and per load bucket, because sideband levels depend on load. Scoring uses
median and MAD rather than mean and standard deviation:

    z = (x - median) / (1.4826 * MAD)

The 1.4826 factor makes MAD a consistent estimator of sigma for Gaussian data. Median/MAD has
a 50 % breakdown point, and in addition exceeding windows are never added to the baseline,
so a developing fault cannot drag its own threshold upward. A candidate is raised only when a
bin exceeds the threshold for `consecutive_windows` windows in a row; single-window spikes
are rejected.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from itertools import groupby

from pydantic import BaseModel, ConfigDict, Field

from backend.models.alert import BinEvidence, BinMeasurement, Candidate

MAD_TO_SIGMA = 1.4826


class DetectorConfig(BaseModel):
    """Detection thresholds. Site-tunable; defaults favour few false alarms."""

    model_config = ConfigDict(frozen=True)

    z_threshold: float = Field(default=4.0, gt=0)
    consecutive_windows: int = Field(default=3, ge=2)
    min_baseline_windows: int = Field(default=8, ge=3)
    max_history: int = Field(default=200, ge=10)
    min_mad_db: float = Field(
        default=0.5, gt=0, description="MAD floor: stops a very quiet bin producing huge z"
    )
    bucket_hysteresis: float = Field(
        default=0.05,
        ge=0,
        le=0.1,
        description="Load must cross a bucket edge by this much before the bucket changes",
    )


class LoadBucket(StrEnum):
    BELOW_25 = "<25%"
    FROM_25_TO_50 = "25-50%"
    FROM_50_TO_75 = "50-75%"
    ABOVE_75 = ">75%"

    @classmethod
    def for_load(cls, load_factor: float) -> LoadBucket:
        if load_factor < 0.25:
            return cls.BELOW_25
        if load_factor < 0.5:
            return cls.FROM_25_TO_50
        if load_factor < 0.75:
            return cls.FROM_50_TO_75
        return cls.ABOVE_75

    @property
    def bounds(self) -> tuple[float, float]:
        return _BUCKET_BOUNDS[self]

    @classmethod
    def with_hysteresis(
        cls, load_factor: float, previous: LoadBucket | None, margin: float
    ) -> LoadBucket:
        """Stay in `previous` while load is within `margin` of its edges."""
        if previous is not None:
            low, high = previous.bounds
            if low - margin <= load_factor < high + margin:
                return previous
        return cls.for_load(load_factor)


_BUCKET_BOUNDS: dict[LoadBucket, tuple[float, float]] = {
    LoadBucket.BELOW_25: (float("-inf"), 0.25),
    LoadBucket.FROM_25_TO_50: (0.25, 0.5),
    LoadBucket.FROM_50_TO_75: (0.5, 0.75),
    LoadBucket.ABOVE_75: (0.75, float("inf")),
}


@dataclass(frozen=True)
class BinBaseline:
    values: tuple[float, ...] = ()

    @property
    def median(self) -> float:
        return statistics.median(self.values)

    @property
    def mad(self) -> float:
        med = self.median
        return statistics.median(abs(v - med) for v in self.values)

    def with_value(self, value: float, max_history: int) -> BinBaseline:
        return BinBaseline(values=(*self.values, value)[-max_history:])


def robust_z(x: float, values: Sequence[float], min_mad: float) -> float:
    base = BinBaseline(values=tuple(values))
    return (x - base.median) / (MAD_TO_SIGMA * max(base.mad, min_mad))


@dataclass(frozen=True)
class BinScore:
    key: str
    level_db: float
    baseline_count: int
    median_db: float | None
    mad_db: float | None
    z_score: float | None
    consecutive_windows: int


@dataclass(frozen=True)
class ObservationResult:
    window_index: int
    load_bucket: LoadBucket
    learning: bool
    scores: list[BinScore]
    candidates: list[Candidate] = field(default_factory=list)


_Key = tuple[str, LoadBucket, str]


class SustainedExceedanceDetector:
    """Stateful over windows; each stored baseline is an immutable value replaced on update."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self._baselines: dict[_Key, BinBaseline] = {}
        self._streaks: dict[_Key, int] = {}
        self._window_index: dict[str, int] = {}
        self._bucket: dict[str, LoadBucket] = {}

    def baseline(self, asset_id: str, bucket: LoadBucket, key: str) -> BinBaseline:
        return self._baselines.get((asset_id, bucket, key), BinBaseline())

    def snapshot(self, asset_id: str) -> dict[tuple[LoadBucket, str], BinBaseline]:
        """Immutable copy of one asset's baselines (values are frozen tuples)."""
        return {
            (bucket, key): base
            for (asset, bucket, key), base in list(self._baselines.items())
            if asset == asset_id
        }

    def _score(self, asset_id: str, bucket: LoadBucket, m: BinMeasurement) -> tuple[BinScore, bool]:
        cfg, k = self.config, (asset_id, bucket, m.key)
        base = self._baselines.get(k, BinBaseline())
        x = m.level_db
        if len(base.values) < cfg.min_baseline_windows:
            self._baselines[k] = base.with_value(x, cfg.max_history)
            return BinScore(m.key, x, len(base.values), None, None, None, 0), True
        z = robust_z(x, base.values, cfg.min_mad_db)
        if z > cfg.z_threshold:
            self._streaks[k] = self._streaks.get(k, 0) + 1  # exceeding windows never enter
        else:
            self._streaks[k] = 0
            self._baselines[k] = base.with_value(x, cfg.max_history)
        score = BinScore(m.key, x, len(base.values), base.median, base.mad, z, self._streaks[k])
        return score, False

    def observe(
        self,
        asset_id: str,
        load_factor: float,
        measurements: Sequence[BinMeasurement],
        timestamp: datetime | None = None,
    ) -> ObservationResult:
        bucket = LoadBucket.with_hysteresis(
            load_factor, self._bucket.get(asset_id), self.config.bucket_hysteresis
        )
        self._bucket[asset_id] = bucket
        index = self._window_index.get(asset_id, 0)
        self._window_index[asset_id] = index + 1
        scored = [(m, *self._score(asset_id, bucket, m)) for m in measurements]
        sustained = [
            (m, s) for m, s, _ in scored if s.consecutive_windows >= self.config.consecutive_windows
        ]
        return ObservationResult(
            window_index=index,
            load_bucket=bucket,
            learning=any(learning for *_, learning in scored),
            scores=[s for _, s, _ in scored],
            candidates=self._candidates(
                asset_id, bucket, load_factor, index, sustained, timestamp or datetime.now(UTC)
            ),
        )

    def _candidates(
        self,
        asset_id: str,
        bucket: LoadBucket,
        load_factor: float,
        index: int,
        sustained: list[tuple[BinMeasurement, BinScore]],
        timestamp: datetime,
    ) -> list[Candidate]:
        def group_key(item: tuple[BinMeasurement, BinScore]) -> tuple[str, str]:
            return item[0].fault_class.value, item[0].bearing_position or ""

        candidates = []
        for _, items in groupby(sorted(sustained, key=group_key), key=group_key):
            group = list(items)
            first = group[0][0]
            candidates.append(
                Candidate(
                    asset_id=asset_id,
                    fault_class=first.fault_class,
                    bearing_position=first.bearing_position,
                    load_bucket=bucket.value,
                    load_factor=load_factor,
                    window_index=index,
                    raised_at=timestamp,
                    evidence=[_evidence(m, s) for m, s in group],
                )
            )
        return candidates


def _evidence(m: BinMeasurement, s: BinScore) -> BinEvidence:
    assert s.median_db is not None and s.mad_db is not None and s.z_score is not None
    return BinEvidence(
        key=m.key,
        label=m.label,
        frequency_hz=m.frequency_hz,
        equation=m.equation,
        inputs=m.inputs,
        measured_db=m.level_db,
        baseline_median_db=s.median_db,
        baseline_mad_db=s.mad_db,
        z_score=s.z_score,
        consecutive_windows=s.consecutive_windows,
        ambiguous_with=m.ambiguous_with,
    )
