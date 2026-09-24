"""Commission a twelve-motor demonstration fleet and drive a subset into degradation.

Usage (API must be running):
    python -m backend.scripts.seed_fleet --url http://localhost:8000

Or set SEED_ON_STARTUP=true to seed in-process when the API starts.

Every motor first runs BASELINE_WINDOWS healthy windows so its baseline is established, then
its scenario. Scenarios cover: progressive rotor-bar, bearing and stator faults; an incipient
eccentricity that stops at WATCH (not yet sustained); a supply-voltage unbalance that must NOT
alert; an oversized motor for the energy view; and incomplete nameplates that show
commissioning's "needs confirmation" behaviour.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from typing import Protocol

from backend.api.schemas import AssetCreate, SimulationProfile
from backend.api.service import MonitoringService

logger = logging.getLogger(__name__)

BASELINE_WINDOWS = 10
HTTP_TIMEOUT_S = 600.0


@dataclass(frozen=True)
class Step:
    windows: int
    faults: list[dict] = field(default_factory=list)
    confounders: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class SeedMotor:
    spec: dict
    load_factor: float
    scenario: list[Step] = field(default_factory=list)
    note: str = "healthy"


def _bearings(de: str | None, nde: str | None) -> list[dict]:
    return [
        {"position": pos, "designation": d}
        for pos, d in (("DE", de), ("NDE", nde))
        if d is not None
    ]


def _motor(asset_id, kw, v, a, hz, poles, rpm, eta, lrc, slots, de, nde) -> dict:
    return {
        "asset_id": asset_id,
        "rated_power_kw": kw,
        "rated_voltage_v": v,
        "rated_current_a": a,
        "supply_frequency_hz": hz,
        "poles": poles,
        "rated_speed_rpm": rpm,
        "rated_efficiency": eta,
        "locked_rotor_current_ratio": lrc,
        "rotor_slots": slots,
        "bearings": _bearings(de, nde),
    }


def _progressive(fault: str, severities: list[float], **extra) -> list[Step]:
    return [Step(1, faults=[{"fault": fault, "severity": s, **extra}]) for s in severities]


FLEET: list[SeedMotor] = [
    SeedMotor(_motor("PUMP-101", 15, 460, 22, 60, 4, 1750, 0.91, 6.5, 28, "6205", "6203"), 0.80),
    SeedMotor(_motor("PUMP-102", 30, 400, 55, 50, 4, 1470, 0.93, 7.0, 40, "6309", "6208"), 0.85),
    SeedMotor(_motor("FAN-201", 7.5, 400, 14.5, 50, 2, 2910, 0.89, 7.5, 28, "6206", "6205"), 0.70),
    SeedMotor(
        _motor("CONV-301", 11, 460, 16.5, 60, 4, 1760, 0.915, 6.8, 36, "6206", "6205"),
        0.65,
        _progressive("broken_rotor_bar", [0.15, 0.2, 0.25, 0.3, 0.35]),
        "progressive broken rotor bar",
    ),
    SeedMotor(
        _motor("COMP-401", 55, 400, 98, 50, 4, 1480, 0.945, 7.2, 48, "NU310", "6310"),
        0.85,
        _progressive("bearing_outer", [0.4, 0.5, 0.6, 0.7], bearing_position="DE"),
        "progressive DE outer-race defect",
    ),
    SeedMotor(
        _motor("MIX-501", 22, 400, 41, 50, 6, 975, 0.92, 6.5, 54, "6309", "6208"),
        0.30,
        note="oversized: runs at 30 % load",
    ),
    SeedMotor(
        _motor("PUMP-103", 4, 460, 6.4, 60, 2, 3500, 0.875, 7.6, None, "6205", "6204"),
        0.70,
        note="rotor slots unknown: slip from nameplate scaling",
    ),
    SeedMotor(
        _motor("FAN-202", 18.5, 400, 34, 50, 4, 1475, 0.925, 7.0, 40, "6309", "6208"),
        0.80,
        _progressive("eccentricity", [0.3, 0.3]),
        "incipient eccentricity: two windows, not yet sustained",
    ),
    SeedMotor(
        _motor("CRUSH-601", 75, 690, 76, 50, 4, 1485, 0.95, 7.0, 48, "NU310", "6310"),
        0.85,
        _progressive("stator_turn_fault", [0.3, 0.35, 0.4, 0.45]),
        "progressive stator turn fault",
    ),
    SeedMotor(
        _motor("AGIT-701", 5.5, 400, 11, 50, 4, 1440, 0.88, 6.0, 28, "6206", "6205"),
        0.75,
        [Step(4, confounders=[{"kind": "supply_voltage_unbalance", "voltage_unbalance": 0.02}])],
        "2 % supply voltage unbalance: must not alert",
    ),
    SeedMotor(
        _motor("PUMP-104", 37, 460, 58, 60, 4, 1775, 0.94, 7.0, 40, "7310-BEP", "6310"),
        0.80,
        note="DE angular-contact bearing not in database: DE bearing coverage unavailable",
    ),
    SeedMotor(
        _motor("BLOW-801", 45, 400, 80, 50, 2, 2965, 0.94, None, None, None, None),
        0.45,
        note="minimal nameplate: no bearings, slots or LRC ratio",
    ),
]


class Seeder(Protocol):
    def commission(self, payload: dict) -> None: ...
    def run(self, asset_id: str, step: Step) -> None: ...


class ServiceSeeder:
    def __init__(self, service: MonitoringService) -> None:
        self.service = service

    def commission(self, payload: dict) -> None:
        self.service.commission(AssetCreate.model_validate(payload))

    def run(self, asset_id: str, step: Step) -> None:
        record = self.service.get(asset_id)
        profile = SimulationProfile.model_validate(
            {
                **record.simulation.model_dump(),
                "faults": step.faults,
                "confounders": step.confounders,
            }
        )
        self.service.set_simulation(asset_id, profile)
        self.service.advance(asset_id, step.windows)


class HttpSeeder:
    def __init__(self, base_url: str) -> None:
        import httpx

        self.client = httpx.Client(base_url=base_url, timeout=HTTP_TIMEOUT_S)

    def _post(self, path: str, body: dict) -> None:
        response = self.client.post(path, json=body)
        if response.status_code >= 400:
            raise RuntimeError(f"POST {path} failed ({response.status_code}): {response.text}")

    def commission(self, payload: dict) -> None:
        self._post("/api/assets", payload)

    def run(self, asset_id: str, step: Step) -> None:
        self._post(
            "/api/simulator/inject",
            {
                "asset_id": asset_id,
                "faults": step.faults,
                "confounders": step.confounders,
                "advance_windows": step.windows,
            },
        )


def seed(seeder: Seeder, fleet: list[SeedMotor] = FLEET) -> None:
    for motor in fleet:
        asset_id = motor.spec["asset_id"]
        logger.info("seeding %s (%s)", asset_id, motor.note)
        seeder.commission({"spec": motor.spec, "simulation": {"load_factor": motor.load_factor}})
        seeder.run(asset_id, Step(BASELINE_WINDOWS))
        for step in motor.scenario:
            seeder.run(asset_id, step)


def seed_service(service: MonitoringService) -> None:
    seed(ServiceSeeder(service))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8000", help="Nameplate API base URL")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    seed(HttpSeeder(args.url))
    return 0


if __name__ == "__main__":
    sys.exit(main())
