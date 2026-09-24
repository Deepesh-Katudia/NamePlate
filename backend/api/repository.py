"""Persistence boundary.

`Repository` is the only interface the service layer depends on. The prototype ships an
in-memory implementation; moving to a database means writing one class that satisfies this
Protocol (e.g. a SQL-backed repository) and passing it to `create_app`. Nothing else changes.
"""

from __future__ import annotations

import threading
from typing import Protocol

from backend.api.schemas import Alert, AssetRecord, WindowSnapshot


class AssetExistsError(ValueError):
    pass


class Repository(Protocol):
    def add_asset(self, record: AssetRecord) -> None: ...
    def get_asset(self, asset_id: str) -> AssetRecord | None: ...
    def list_assets(self) -> list[AssetRecord]: ...
    def update_asset(self, record: AssetRecord) -> None: ...
    def save_window(self, snapshot: WindowSnapshot) -> None: ...
    def latest_window(self, asset_id: str) -> WindowSnapshot | None: ...
    def upsert_alert(self, alert: Alert) -> None: ...
    def list_alerts(self, asset_id: str | None = None) -> list[Alert]: ...


class InMemoryRepository:
    """Thread-safe dict-backed repository. State is lost on restart."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._assets: dict[str, AssetRecord] = {}
        self._windows: dict[str, WindowSnapshot] = {}
        self._alerts: dict[str, Alert] = {}

    def add_asset(self, record: AssetRecord) -> None:
        with self._lock:
            if record.spec.asset_id in self._assets:
                raise AssetExistsError(f"asset '{record.spec.asset_id}' already commissioned")
            self._assets[record.spec.asset_id] = record

    def get_asset(self, asset_id: str) -> AssetRecord | None:
        with self._lock:
            return self._assets.get(asset_id)

    def list_assets(self) -> list[AssetRecord]:
        with self._lock:
            return list(self._assets.values())

    def update_asset(self, record: AssetRecord) -> None:
        with self._lock:
            if record.spec.asset_id not in self._assets:
                raise KeyError(record.spec.asset_id)
            self._assets[record.spec.asset_id] = record

    def save_window(self, snapshot: WindowSnapshot) -> None:
        with self._lock:
            self._windows[snapshot.asset_id] = snapshot

    def latest_window(self, asset_id: str) -> WindowSnapshot | None:
        with self._lock:
            return self._windows.get(asset_id)

    def upsert_alert(self, alert: Alert) -> None:
        with self._lock:
            self._alerts[alert.id] = alert

    def list_alerts(self, asset_id: str | None = None) -> list[Alert]:
        with self._lock:
            alerts = [a for a in self._alerts.values() if asset_id in (None, a.asset_id)]
        return sorted(alerts, key=lambda a: a.first_raised_at, reverse=True)
