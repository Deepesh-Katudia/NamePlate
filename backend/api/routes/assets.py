"""Asset endpoints: commissioning, fleet listing, detail, spectrum, alerts."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.api.deps import get_service
from backend.api.schemas import (
    Alert,
    AssetCreate,
    AssetDetail,
    AssetSummary,
    Envelope,
    SpectrumView,
    ok,
)
from backend.api.service import MonitoringService

router = APIRouter(prefix="/api/assets", tags=["assets"])
Service = Annotated[MonitoringService, Depends(get_service)]

DEFAULT_SPECTRUM_MAX_HZ = 500.0


@router.post("", status_code=status.HTTP_201_CREATED, response_model=Envelope[AssetDetail])
def commission_asset(body: AssetCreate, service: Service) -> Envelope[AssetDetail]:
    record = service.commission(body)
    return ok(service.detail(record.spec.asset_id))


@router.get("", response_model=Envelope[list[AssetSummary]])
def list_assets(service: Service) -> Envelope[list[AssetSummary]]:
    fleet = service.fleet()
    return ok(fleet, meta={"total": len(fleet)})


@router.get("/{asset_id}", response_model=Envelope[AssetDetail])
def get_asset(asset_id: str, service: Service) -> Envelope[AssetDetail]:
    return ok(service.detail(asset_id))


@router.get("/{asset_id}/spectrum", response_model=Envelope[SpectrumView])
def get_spectrum(
    asset_id: str,
    service: Service,
    max_hz: Annotated[float, Query(gt=0, le=10_000)] = DEFAULT_SPECTRUM_MAX_HZ,
) -> Envelope[SpectrumView]:
    view = service.spectrum(asset_id, max_hz)
    if view is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no window processed yet for this asset")
    return ok(view)


@router.get("/{asset_id}/alerts", response_model=Envelope[list[Alert]])
def get_alerts(asset_id: str, service: Service) -> Envelope[list[Alert]]:
    service.get(asset_id)
    alerts = service.repo.list_alerts(asset_id)
    return ok(alerts, meta={"total": len(alerts)})


@router.post("/{asset_id}/diagnose", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def diagnose(asset_id: str, service: Service) -> None:
    service.get(asset_id)
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, "Diagnosis agent is not implemented yet")
