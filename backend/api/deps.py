"""FastAPI dependencies resolving shared state from the application."""

from __future__ import annotations

from fastapi import Request

from backend.api.diagnosis_service import DiagnosisService
from backend.api.service import MonitoringService
from backend.api.telemetry import TelemetryHub


def get_service(request: Request) -> MonitoringService:
    return request.app.state.service


def get_diagnosis(request: Request) -> DiagnosisService:
    return request.app.state.diagnosis


def get_hub(request: Request) -> TelemetryHub:
    return request.app.state.hub
