from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

from agent_service.core.errors import ModelUnavailableError
from agent_service.infra_ai.health_store import ModelHealthStore
from agent_service.infra_ai.models import ModelCapability, ModelTarget

logger = logging.getLogger(__name__)

C = TypeVar("C")
T = TypeVar("T")


class ModelRoutingExecutor:
    def __init__(self, health_store: ModelHealthStore) -> None:
        self._health_store = health_store

    async def execute_with_fallback(
        self,
        capability: ModelCapability,
        targets: list[ModelTarget],
        client_resolver: Callable[[ModelTarget], C | None],
        caller: Callable[[C, ModelTarget], Awaitable[T]],
    ) -> T:
        label = capability.value
        if not targets:
            raise ModelUnavailableError(f"no {label} model candidates available")

        last_error: Exception | None = None
        for target in targets:
            client = client_resolver(target)
            if client is None:
                logger.warning(
                    "%s provider client missing provider=%s modelId=%s",
                    label,
                    target.candidate.provider,
                    target.id,
                )
                continue
            if not self._health_store.allow_call(target.id):
                logger.info("%s model target skipped by circuit id=%s", label, target.id)
                continue

            try:
                result = await caller(client, target)
            except Exception as exc:
                self._health_store.record_failure(target.id)
                last_error = exc
                logger.warning(
                    "%s model failed, fallback to next. modelId=%s provider=%s error=%s",
                    label,
                    target.id,
                    target.candidate.provider,
                    exc,
                )
                continue
            self._health_store.record_success(target.id)
            logger.info("%s model target succeeded id=%s", label, target.id)
            return result

        if last_error is not None:
            raise ModelUnavailableError(
                f"all {label} model candidates failed: {last_error}"
            ) from last_error
        raise ModelUnavailableError(f"no available {label} model targets")

    async def execute_targets(
        self,
        targets: list[ModelTarget],
        call: Callable[[ModelTarget], Awaitable[T]],
    ) -> T:
        capability = targets[0].capability if targets else ModelCapability.CHAT
        return await self.execute_with_fallback(
            capability,
            targets,
            lambda target: target,
            lambda target, _: call(target),
        )
