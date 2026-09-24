"""FastAPI application factory.

Run: uvicorn backend.api.main:app --reload   (from the repository root)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from backend.api.repository import AssetExistsError, InMemoryRepository, Repository
from backend.api.routes import assets, fleet, simulator
from backend.api.schemas import ApiError, Envelope
from backend.api.service import AssetNotFoundError, MonitoringService
from backend.api.settings import Settings
from backend.api.telemetry import TelemetryHub
from backend.detection.baseline import DetectorConfig
from backend.models.errors import DomainError

logger = logging.getLogger(__name__)


def _error(status: int, code: str, message: str, details: list | None = None) -> JSONResponse:
    body = Envelope[None](
        success=False, error=ApiError(code=code, message=message, details=details)
    )
    return JSONResponse(status_code=status, content=body.model_dump(mode="json"))


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AssetNotFoundError)
    async def _not_found(_: Request, exc: AssetNotFoundError) -> JSONResponse:
        return _error(404, "asset_not_found", f"asset '{exc.args[0]}' not found")

    @app.exception_handler(AssetExistsError)
    async def _exists(_: Request, exc: AssetExistsError) -> JSONResponse:
        return _error(409, "asset_exists", str(exc))

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()
        ]
        return _error(422, "validation_error", "request failed validation", details)

    @app.exception_handler(HTTPException)
    async def _http(_: Request, exc: HTTPException) -> JSONResponse:
        return _error(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(DomainError)
    async def _domain(request: Request, exc: DomainError) -> JSONResponse:
        # Only DomainError is a user error. Any other exception propagates as a 500 so an
        # internal bug is never disguised as bad input.
        logger.warning("domain error on %s %s: %s", request.method, request.url.path, exc)
        return _error(422, "domain_error", str(exc))


async def _telemetry_loop(app: FastAPI, interval_s: float) -> None:
    service: MonitoringService = app.state.service
    hub: TelemetryHub = app.state.hub
    while True:
        for record in service.repo.list_assets():
            try:
                result = await run_in_threadpool(service.run_window, record.spec.asset_id)
                await hub.publish({"type": "window", "data": result.model_dump(mode="json")})
            except Exception:  # keep monitoring the rest of the fleet
                logger.exception("window failed for %s", record.spec.asset_id)
        await asyncio.sleep(interval_s)


def create_app(
    settings: Settings | None = None,
    repository: Repository | None = None,
    detector_config: DetectorConfig | None = None,
) -> FastAPI:
    cfg = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if cfg.seed_on_startup:
            from backend.scripts.seed_fleet import seed_service

            await run_in_threadpool(seed_service, app.state.service)
        task = (
            asyncio.create_task(_telemetry_loop(app, cfg.telemetry_interval_s))
            if cfg.telemetry_interval_s > 0
            else None
        )
        yield
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app = FastAPI(
        title="Nameplate",
        version="0.3.0",
        description="Physics-derived condition monitoring for VFD-driven induction motors",
        lifespan=lifespan,
    )
    app.state.service = MonitoringService(repository or InMemoryRepository(), cfg, detector_config)
    app.state.hub = TelemetryHub()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    _install_error_handlers(app)
    for router in (assets.router, simulator.router, fleet.router):
        app.include_router(router)

    @app.get("/api/health", tags=["meta"])
    def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
