from __future__ import annotations

from fastapi import APIRouter, Response, status

from agent_service.core.config import get_settings
from agent_service.schemas.health import HealthResponse, ReadinessResponse
from agent_service.services.stream_queue_limiter import stream_queue_limiter
from agent_service.services.stream_task_manager import stream_task_manager

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        service=settings.service_name,
        version=settings.service_version,
    )


@router.get("/ready", response_model=ReadinessResponse)
async def readiness(response: Response) -> ReadinessResponse:
    """Report dependencies required for accepting new streaming work, not mere liveness."""

    settings = get_settings()
    checks = {
        "streamCancellation": stream_task_manager.is_ready,
        "streamQueue": stream_queue_limiter.is_ready,
    }
    ready = all(checks.values())
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        status="ok" if ready else "unavailable",
        service=settings.service_name,
        version=settings.service_version,
        checks=checks,
    )

