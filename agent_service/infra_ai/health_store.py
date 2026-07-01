from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

from agent_service.core.config import get_settings


class HealthState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class ModelHealth:
    state: HealthState = HealthState.CLOSED
    failure_count: int = 0
    open_until: float = 0.0
    half_open_in_flight: int = 0


class ModelHealthStore:
    def __init__(
        self,
        *,
        failure_threshold: int,
        open_seconds: float,
        half_open_max_in_flight: int,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._open_seconds = open_seconds
        self._half_open_max_in_flight = half_open_max_in_flight
        self._health_by_id: dict[str, ModelHealth] = {}

    def is_unavailable(self, model_id: str) -> bool:
        health = self._health_by_id.get(model_id)
        if health is None:
            return False
        now = time.time()
        if health.state == HealthState.OPEN:
            return health.open_until > now
        return health.state == HealthState.HALF_OPEN and (
            health.half_open_in_flight >= self._half_open_max_in_flight
        )

    def allow_call(self, model_id: str) -> bool:
        health = self._health_by_id.setdefault(model_id, ModelHealth())
        now = time.time()
        if health.state == HealthState.OPEN:
            if health.open_until > now:
                return False
            health.state = HealthState.HALF_OPEN
            health.half_open_in_flight = 0
        if health.state == HealthState.HALF_OPEN:
            if health.half_open_in_flight >= self._half_open_max_in_flight:
                return False
            health.half_open_in_flight += 1
        return True

    def record_success(self, model_id: str) -> None:
        self._health_by_id[model_id] = ModelHealth()

    def record_failure(self, model_id: str) -> None:
        health = self._health_by_id.setdefault(model_id, ModelHealth())
        if health.state == HealthState.HALF_OPEN:
            self._open(health)
            return
        health.failure_count += 1
        if health.failure_count >= self._failure_threshold:
            self._open(health)

    def _open(self, health: ModelHealth) -> None:
        health.state = HealthState.OPEN
        health.open_until = time.time() + self._open_seconds
        health.half_open_in_flight = 0


@lru_cache
def get_model_health_store() -> ModelHealthStore:
    settings = get_settings()
    return ModelHealthStore(
        failure_threshold=settings.ai_circuit_failure_threshold,
        open_seconds=settings.ai_circuit_open_seconds,
        half_open_max_in_flight=settings.ai_circuit_half_open_max_in_flight,
    )
