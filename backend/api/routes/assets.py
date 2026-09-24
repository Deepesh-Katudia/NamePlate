"""Asset endpoints: commissioning, fleet listing, detail, spectrum, alerts."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from starlette.concurrency import run_in_threadpool

from backend.agents.commissioning import CommissioningResult
from backend.api.deps import get_diagnosis, get_service
from backend.api.diagnosis_service import DiagnosisService
from backend.api.schemas import (
    Alert,
    AssetCreate,
    AssetDetail,
    AssetSummary,
    CommissioningPreviewRequest,
    DiagnoseRequest,
    Envelope,
    SpectrumView,
    ok,
)
from backend.api.service import MonitoringService
from backend.models.diagnosis import Receipt

router = APIRouter(prefix="/api/assets", tags=["assets"])
Service = Annotated[MonitoringService, Depends(get_service)]
Diagnosis = Annotated[DiagnosisService, Depends(get_diagnosis)]

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


@router.post("/{asset_id}/diagnose", response_model=Envelope[Receipt])
async def diagnose(
    asset_id: str, diagnosis: Diagnosis, body: DiagnoseRequest | None = None
) -> Envelope[Receipt]:
    """Run the Diagnosis Agent on an open alert (and the Action Agent if it is confirmed)."""
    alert_id = body.alert_id if body else None
    receipt = await run_in_threadpool(diagnosis.diagnose, asset_id, alert_id)
    return ok(receipt)


preview_router = APIRouter(prefix="/api/commissioning", tags=["commissioning"])


@preview_router.post("/preview", response_model=Envelope[CommissioningResult])
async def commissioning_preview(
    body: CommissioningPreviewRequest, diagnosis: Diagnosis
) -> Envelope[CommissioningResult]:
    """Validate a nameplate (structured or free text) and derive its fault map. Not saved."""
    result = await run_in_threadpool(diagnosis.commissioning_preview, body)
    return ok(result)
