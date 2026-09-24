"""Simulator control: set an asset's injected faults/confounders and advance windows."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from backend.api.deps import get_hub, get_service
from backend.api.schemas import AdvanceRequest, Envelope, InjectRequest, WindowResultView, ok
from backend.api.service import MonitoringService
from backend.api.telemetry import TelemetryHub

router = APIRouter(prefix="/api/simulator", tags=["simulator"])
Service = Annotated[MonitoringService, Depends(get_service)]
Hub = Annotated[TelemetryHub, Depends(get_hub)]


async def _advance_and_publish(
    service: MonitoringService, hub: TelemetryHub, asset_id: str, windows: int
) -> list[WindowResultView]:
    results = []
    for _ in range(windows):
        result = await run_in_threadpool(service.run_window, asset_id)
        await hub.publish({"type": "window", "data": result.model_dump(mode="json")})
        results.append(result)
    return results


@router.post("/inject", response_model=Envelope[list[WindowResultView]])
async def inject(
    body: InjectRequest, service: Service, hub: Hub
) -> Envelope[list[WindowResultView]]:
    """Replace the asset's injected faults and confounders; optionally advance windows."""
    record = service.get(body.asset_id)
    profile = record.simulation.model_copy(
        update={"faults": body.faults, "confounders": body.confounders}
    )
    service.set_simulation(body.asset_id, profile)
    results = await _advance_and_publish(service, hub, body.asset_id, body.advance_windows)
    return ok(results)


@router.post("/advance", response_model=Envelope[list[WindowResultView]])
async def advance(
    body: AdvanceRequest, service: Service, hub: Hub
) -> Envelope[list[WindowResultView]]:
    service.get(body.asset_id)
    return ok(await _advance_and_publish(service, hub, body.asset_id, body.windows))
