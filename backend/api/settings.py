"""Runtime settings, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:5173"])
    window_duration_s: float = 16.0
    sample_rate_hz: float = 5_000.0
    telemetry_interval_s: float = 0.0  # 0 disables the background window loop
    seed_on_startup: bool = False
    max_advance_windows: int = 50

    @classmethod
    def from_env(cls) -> Settings:
        origins = os.environ.get("CORS_ORIGINS", "http://localhost:5173")
        return cls(
            cors_origins=[o.strip() for o in origins.split(",") if o.strip()],
            window_duration_s=float(os.environ.get("WINDOW_DURATION_S", 16.0)),
            sample_rate_hz=float(os.environ.get("SAMPLE_RATE_HZ", 5_000.0)),
            telemetry_interval_s=float(os.environ.get("TELEMETRY_INTERVAL_S", 0.0)),
            seed_on_startup=_env_bool("SEED_ON_STARTUP", False),
        )
