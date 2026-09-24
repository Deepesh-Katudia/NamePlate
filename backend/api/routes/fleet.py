"""Fleet-level endpoints and the telemetry WebSocket."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from backend.api.deps import get_service
from backend.api.schemas import Envelope, FleetEnergy, ok
from backend.api.service import MonitoringService

router = APIRouter(tags=["fleet"])
Service = Annotated[MonitoringService, Depends(get_service)]


@router.get("/api/fleet/energy", response_model=Envelope[FleetEnergy])
def fleet_energy(service: Service) -> Envelope[FleetEnergy]:
    return ok(service.fleet_energy())


@router.websocket("/ws/telemetry")
async def telemetry(websocket: WebSocket) -> None:
    hub = websocket.app.state.hub
    await websocket.accept()
    queue = hub.subscribe()
    try:
        while True:
            await websocket.send_json(await queue.get())
    except WebSocketDisconnect:
        pass
    finally:
        hub.unsubscribe(queue)
