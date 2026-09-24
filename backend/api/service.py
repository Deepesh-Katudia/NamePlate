"""Monitoring service: commissioning, window processing, alert lifecycle, fleet energy.

Data source in the prototype is the simulator, driven by each asset's `SimulationProfile`.
In deployment `_capture` is the only function that changes: it would read a window of drive
current and voltage instead of synthesising one.
"""

from __future__ import annotations

import math
import threading
import zlib
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from backend.agents.monitoring import MonitoringAgent
from backend.api.repository import Repository
from backend.api.schemas import (
    HEALTH_SEVERITY_ORDER,
    Alert,
    AlertStatus,
    AssetCreate,
    AssetDetail,
    AssetRecord,
    AssetSummary,
    BinScoreView,
    EnergyAsset,
    FleetEnergy,
    HealthState,
    LoadView,
    SimulationProfile,
    SlipView,
    SpectrumView,
    UnbalanceView,
    WindowResultView,
    WindowSnapshot,
)
from backend.api.settings import Settings
from backend.detection.baseline import DetectorConfig, LoadBucket, ObservationResult
from backend.detection.features import WindowFeatures
from backend.energy.load import DEFAULT_LOSS_SPLIT, rated_losses_w
from backend.models.alert import Candidate
from backend.physics.fault_map import build_fault_map
from backend.simulator.motor_sim import SimulationConfig, SimulationResult, simulate

REOPEN_Z_FACTOR = 2.0  # a discarded alert re-opens if its strongest z doubles
RECAPTURE_SEED_OFFSET = 1_000_000

OVERSIZING_LOAD_THRESHOLD = 0.5
RIGHT_SIZED_LOAD = 0.75
HOURS_PER_YEAR = 8760.0
_POWER_FLOOR = 1e-30
BASELINE_ASSUMPTION = (
    "The first {n} steady windows in each load bucket are taken as the healthy "
    "reference. A fault already present at commissioning becomes part of that "
    "reference and will not be flagged; confirm the machine is healthy (e.g. a recent "
    "offline test) before relying on the baseline."
)


class AssetNotFoundError(KeyError):
    pass


@dataclass(frozen=True)
class LatestWindow:
    """Raw capture and features of an asset's most recent window, kept in memory for
    diagnosis. Waveforms are deliberately not persisted through the repository."""

    capture: SimulationResult
    features: WindowFeatures
    load_bucket: LoadBucket | None


class MonitoringService:
    def __init__(
        self,
        repository: Repository,
        settings: Settings | None = None,
        detector_config: DetectorConfig | None = None,
    ) -> None:
        self.repo = repository
        self.settings = settings or Settings()
        self.monitoring = MonitoringAgent(detector_config)
        self.detector = self.monitoring.detector
        self._latest: dict[str, LatestWindow] = {}
        self._state_lock = threading.Lock()  # detector + repository updates
        self._asset_locks: dict[str, threading.Lock] = {}
        self._asset_locks_guard = threading.Lock()

    # --- commissioning ----------------------------------------------------------------

    def commission(self, request: AssetCreate) -> AssetRecord:
        record = AssetRecord(
            spec=request.spec,
            commissioned_at=datetime.now(UTC),
            commissioning_fault_map=build_fault_map(request.spec),
            simulation=request.simulation,
        )
        self.repo.add_asset(record)
        return record

    def get(self, asset_id: str) -> AssetRecord:
        record = self.repo.get_asset(asset_id)
        if record is None:
            raise AssetNotFoundError(asset_id)
        return record

    def set_simulation(self, asset_id: str, profile: SimulationProfile) -> AssetRecord:
        record = self.get(asset_id).model_copy(update={"simulation": profile})
        self.repo.update_asset(record)
        return record

    # --- window processing ------------------------------------------------------------

    def _capture(
        self, record: AssetRecord, duration_s: float | None = None, seed_offset: int = 0
    ) -> SimulationResult:
        sim = record.simulation
        seed = zlib.crc32(record.spec.asset_id.encode()) + record.windows_processed + seed_offset
        rng = np.random.default_rng(seed)
        load = sim.load_factor * (1 + rng.uniform(-sim.load_jitter, sim.load_jitter))
        config = SimulationConfig(
            sample_rate_hz=self.settings.sample_rate_hz,
            duration_s=duration_s or self.settings.window_duration_s,
            load_factor=min(max(load, 0.05), 1.3),
            seed=seed,
        )
        return simulate(record.spec, config, sim.faults, sim.confounders)

    def _asset_lock(self, asset_id: str) -> threading.Lock:
        with self._asset_locks_guard:
            return self._asset_locks.setdefault(asset_id, threading.Lock())

    def run_window(self, asset_id: str) -> WindowResultView:
        """Windows of one asset run in order; different assets analyse in parallel."""
        with self._asset_lock(asset_id):
            record = self.get(asset_id)
            capture = self._capture(record)
            features = self.monitoring.analyze(record.spec, capture)
            now = datetime.now(UTC)
            with self._state_lock:
                return self._record_window(record, capture, features, now)

    def recapture(self, asset_id: str, duration_s: float) -> SimulationResult:
        """A fresh, longer capture for diagnostics (in deployment: a request to the drive)."""
        return self._capture(self.get(asset_id), duration_s, RECAPTURE_SEED_OFFSET)

    def baseline_snapshot(self, asset_id: str) -> dict:
        """Consistent copy of an asset's baselines for a diagnosis session."""
        with self._state_lock:
            return self.detector.snapshot(asset_id)

    def latest(self, asset_id: str) -> LatestWindow:
        """Latest window for diagnosis, processing one first if none is held in memory."""
        if asset_id not in self._latest:
            self.run_window(asset_id)
        return self._latest[asset_id]

    def _record_window(
        self,
        record: AssetRecord,
        capture: SimulationResult,
        features: WindowFeatures,
        now: datetime,
    ) -> WindowResultView:
        asset_id = record.spec.asset_id
        observation = self.monitoring.score(record.spec, features, now)
        self._latest[asset_id] = LatestWindow(
            capture, features, observation.load_bucket if observation else None
        )
        new_alerts = self._update_alerts(observation.candidates if observation else [], now)
        health = self._health(asset_id, observation, record.health)
        self.repo.save_window(
            _snapshot(record, features, observation, now, record.windows_processed)
        )
        self.repo.update_asset(
            record.model_copy(
                update={"health": health, "windows_processed": record.windows_processed + 1}
            )
        )
        return WindowResultView(
            asset_id=asset_id,
            window_index=record.windows_processed,
            scored=observation is not None,
            health=health,
            load_factor=features.load.load_factor,
            slip=features.slip.slip,
            new_alerts=new_alerts,
        )

    def advance(self, asset_id: str, windows: int) -> list[WindowResultView]:
        return [self.run_window(asset_id) for _ in range(windows)]

    def _update_alerts(self, candidates: list[Candidate], now: datetime) -> list[Alert]:
        """One alert per (asset, fault class, bearing position); repeat candidates refresh it."""
        new: list[Alert] = []
        for cand in candidates:
            alert_id = f"{cand.asset_id}:{cand.fault_class.value}:{cand.bearing_position or '-'}"
            existing = next(
                (a for a in self.repo.list_alerts(cand.asset_id) if a.id == alert_id), None
            )
            if existing is not None:
                refreshed = existing.model_copy(
                    update={
                        "last_seen_at": now,
                        "windows_seen": existing.windows_seen + 1,
                        "candidate": cand,
                    }
                )
                if _should_reopen(existing, cand):
                    refreshed = refreshed.model_copy(
                        update={"status": AlertStatus.ACTIVE, "discarded_at_z": None}
                    )
                    new.append(refreshed)
                self.repo.upsert_alert(refreshed)
                continue
            alert = Alert(
                id=alert_id,
                asset_id=cand.asset_id,
                fault_class=cand.fault_class,
                bearing_position=cand.bearing_position,
                status=AlertStatus.ACTIVE,
                first_raised_at=now,
                last_seen_at=now,
                windows_seen=1,
                candidate=cand,
            )
            self.repo.upsert_alert(alert)
            new.append(alert)
        return new

    def _health(
        self, asset_id: str, observation: ObservationResult | None, previous: HealthState
    ) -> HealthState:
        if self.active_alert_count(asset_id):
            return HealthState.ALERT
        if observation is None:
            return previous
        if any(s.consecutive_windows > 0 for s in observation.scores):
            return HealthState.WATCH
        return HealthState.LEARNING if observation.learning else HealthState.HEALTHY

    def active_alert_count(self, asset_id: str) -> int:
        """Open alerts: everything not discarded (confirmed faults stay open until repaired)."""
        return sum(1 for a in self.repo.list_alerts(asset_id) if a.status != AlertStatus.DISCARDED)

    def record_diagnosis(self, alert: Alert) -> None:
        """Persist a diagnosed alert and refresh the asset's health."""
        with self._state_lock:
            self.repo.upsert_alert(alert)
            record = self.get(alert.asset_id)
            health = (
                HealthState.ALERT if self.active_alert_count(alert.asset_id) else HealthState.WATCH
            )
            self.repo.update_asset(record.model_copy(update={"health": health}))

    # --- read models ------------------------------------------------------------------

    def summary(self, record: AssetRecord) -> AssetSummary:
        latest = self.repo.latest_window(record.spec.asset_id)
        fmap = record.commissioning_fault_map
        return AssetSummary(
            asset_id=record.spec.asset_id,
            rated_power_kw=record.spec.rated_power_kw,
            poles=record.spec.poles,
            supply_frequency_hz=record.spec.supply_frequency_hz,
            health=record.health,
            load_factor=latest.load.load_factor if latest else None,
            slip=latest.slip.slip if latest else None,
            active_alerts=self.active_alert_count(record.spec.asset_id),
            unavailable_fault_classes=sorted({u.fault_class for u in fmap.unavailable}),
            needs_confirmation=len(fmap.needs_confirmation),
            windows_processed=record.windows_processed,
        )

    def fleet(self) -> list[AssetSummary]:
        summaries = [self.summary(r) for r in self.repo.list_assets()]
        return sorted(
            summaries,
            key=lambda s: (HEALTH_SEVERITY_ORDER.index(s.health), -s.active_alerts, s.asset_id),
        )

    def detail(self, asset_id: str) -> AssetDetail:
        record = self.get(asset_id)
        fmap = record.commissioning_fault_map
        return AssetDetail(
            summary=self.summary(record),
            spec=record.spec,
            commissioning_fault_map=fmap,
            needs_confirmation=fmap.needs_confirmation,
            unavailable=fmap.unavailable,
            simulation=record.simulation,
            baseline_assumption=BASELINE_ASSUMPTION.format(
                n=self.detector.config.min_baseline_windows
            ),
            latest_window=self.repo.latest_window(asset_id),
        )

    def spectrum(self, asset_id: str, max_hz: float) -> SpectrumView | None:
        record = self.get(asset_id)
        latest = self.repo.latest_window(asset_id)
        if latest is None:
            return None
        n = int(np.searchsorted(latest.spectrum_hz, max_hz, side="right"))
        return SpectrumView(
            asset_id=asset_id,
            window_index=latest.window_index,
            resolution_hz=latest.resolution_hz,
            supply_frequency_hz=record.spec.supply_frequency_hz,
            frequencies_hz=latest.spectrum_hz[:n],
            level_db=latest.spectrum_db[:n],
            bins=[b for b in latest.scores if b.frequency_hz is None or b.frequency_hz <= max_hz],
            excluded_bins=latest.excluded_bins,
            slip=latest.slip,
        )

    def fleet_energy(self) -> FleetEnergy:
        assets = []
        for record in self.repo.list_assets():
            latest = self.repo.latest_window(record.spec.asset_id)
            if latest is not None:
                assets.append(_energy_view(record, latest))
        assets.sort(key=lambda a: a.estimated_waste_kw, reverse=True)
        return FleetEnergy(
            assets=assets,
            oversizing_candidates=[a.asset_id for a in assets if a.oversizing_candidate],
            assumptions=[
                f"Oversizing candidate: upper load bound below {OVERSIZING_LOAD_THRESHOLD:.0%}",
                f"Right-sized rating runs at {RIGHT_SIZED_LOAD:.0%} load",
                "Fixed (core + friction/windage) losses scale linearly with motor rating at the "
                "same efficiency class; waste is the fixed-loss difference only",
                "Annual figure assumes continuous operation at the latest load",
            ],
        )


def max_z(candidate: Candidate) -> float:
    return max(e.z_score for e in candidate.evidence)


def _should_reopen(existing: Alert, candidate: Candidate) -> bool:
    return (
        existing.status == AlertStatus.DISCARDED
        and existing.discarded_at_z is not None
        and max_z(candidate) > REOPEN_Z_FACTOR * existing.discarded_at_z
    )


def _energy_view(record: AssetRecord, latest: WindowSnapshot) -> EnergyAsset:
    spec, load = record.spec, latest.load
    oversized = load.load_factor_high < OVERSIZING_LOAD_THRESHOLD
    waste_kw = 0.0
    if oversized and spec.rated_efficiency is not None and load.shaft_power_kw > 0:
        fixed_kw = (DEFAULT_LOSS_SPLIT.core + DEFAULT_LOSS_SPLIT.friction_windage) * (
            rated_losses_w(spec) / 1000.0
        )
        right_sized_kw = load.shaft_power_kw / RIGHT_SIZED_LOAD
        waste_kw = max(fixed_kw * (1 - right_sized_kw / spec.rated_power_kw), 0.0)
    return EnergyAsset(
        asset_id=spec.asset_id,
        rated_power_kw=spec.rated_power_kw,
        load_factor=load.load_factor,
        load_factor_low=load.load_factor_low,
        load_factor_high=load.load_factor_high,
        shaft_power_kw=load.shaft_power_kw,
        input_power_kw=load.input_power_kw,
        oversizing_candidate=oversized,
        estimated_waste_kw=waste_kw,
        estimated_waste_mwh_per_year=waste_kw * HOURS_PER_YEAR / 1000.0,
    )


def _snapshot(
    record: AssetRecord,
    f: WindowFeatures,
    observation: ObservationResult | None,
    now: datetime,
    index: int,
) -> WindowSnapshot:
    db = 10 * np.log10(np.maximum(f.display_spectrum.power, _POWER_FLOOR) / f.fundamental_power)
    scores = {s.key: s for s in observation.scores} if observation else {}
    bins = []
    for m in f.measurements:
        s = scores.get(m.key)
        bins.append(
            BinScoreView(
                key=m.key,
                label=m.label,
                fault_class=m.fault_class,
                bearing_position=m.bearing_position,
                frequency_hz=m.frequency_hz,
                equation=m.equation,
                level_db=m.level_db,
                baseline_count=s.baseline_count if s else 0,
                baseline_median_db=s.median_db if s else None,
                baseline_mad_db=s.mad_db if s else None,
                z_score=s.z_score if s else None,
                consecutive_windows=s.consecutive_windows if s else 0,
                ambiguous_with=m.ambiguous_with,
            )
        )
    u = f.unbalance
    return WindowSnapshot(
        asset_id=record.spec.asset_id,
        window_index=index,
        captured_at=now,
        stationary=f.stationary,
        stationarity_cv=f.stationarity_cv,
        scored=observation is not None,
        load_bucket=observation.load_bucket.value if observation else None,
        learning=observation.learning if observation else False,
        load=LoadView(
            load_factor=f.load.load_factor,
            load_factor_low=f.load.load_factor_low,
            load_factor_high=f.load.load_factor_high,
            input_power_kw=f.load.input_power_w / 1000.0,
            shaft_power_kw=f.load.shaft_power_w / 1000.0,
            efficiency_indicative=f.load.efficiency_indicative,
            assumptions=f.load.assumptions,
        ),
        slip=SlipView(
            slip=f.slip.slip,
            rotor_speed_rpm=f.slip.rotor_speed_rpm,
            confidence=f.slip.confidence,
            source=f.slip.source,
            notes=f.slip.notes,
        ),
        unbalance=UnbalanceView(**u.__dict__) if u else None,
        resolution_hz=f.spectrum.resolution_hz,
        spectrum_hz=[round(float(x), 4) for x in f.display_spectrum.frequencies_hz],
        spectrum_db=[round(float(x), 2) if math.isfinite(x) else -300.0 for x in db],
        scores=bins,
        excluded_bins=f.conflicts.excluded,
        operating_fault_map=f.fault_map,
    )
