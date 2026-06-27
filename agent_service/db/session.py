from __future__ import annotations

from functools import lru_cache
from typing import Any

from agent_service.core.config import get_settings


@lru_cache
def get_engine() -> Any:
    settings = get_settings()
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is not configured")

    try:
        from sqlalchemy import create_engine
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError("sqlalchemy is required for database access") from exc

    return create_engine(settings.database_url, pool_pre_ping=True, future=True)
