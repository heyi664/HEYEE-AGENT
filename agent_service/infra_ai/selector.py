from __future__ import annotations

import logging
from collections.abc import Iterable

from agent_service.infra_ai.health_store import ModelHealthStore
from agent_service.infra_ai.models import (
    AIModelConfig,
    ModelCandidate,
    ModelCapability,
    ModelGroup,
    ModelTarget,
)

logger = logging.getLogger(__name__)


class ModelSelector:
    def __init__(self, config: AIModelConfig, health_store: ModelHealthStore) -> None:
        self._config = config
        self._health_store = health_store

    def select_chat_candidates(
        self,
        *,
        deep_thinking: bool = False,
        require_tools: bool = False,
    ) -> list[ModelTarget]:
        first_choice = self._resolve_first_choice_model(self._config.chat, deep_thinking)
        return self._select_candidates(
            self._config.chat,
            ModelCapability.CHAT,
            first_choice_model_id=first_choice,
            deep_thinking=deep_thinking,
            require_tools=require_tools,
        )

    def select_embedding_candidates(self) -> list[ModelTarget]:
        return self._select_candidates(
            self._config.embedding,
            ModelCapability.EMBEDDING,
            first_choice_model_id=self._config.embedding.default_model,
        )

    def select_rerank_candidates(self) -> list[ModelTarget]:
        return self._select_candidates(
            self._config.rerank,
            ModelCapability.RERANK,
            first_choice_model_id=self._config.rerank.default_model,
        )

    def _resolve_first_choice_model(self, group: ModelGroup, deep_thinking: bool) -> str | None:
        if deep_thinking and group.deep_thinking_model:
            return group.deep_thinking_model
        return group.default_model

    def _select_candidates(
        self,
        group: ModelGroup,
        capability: ModelCapability,
        *,
        first_choice_model_id: str | None = None,
        deep_thinking: bool = False,
        require_tools: bool = False,
    ) -> list[ModelTarget]:
        candidates = self._filter_and_sort_candidates(
            group.candidates,
            first_choice_model_id=first_choice_model_id,
            deep_thinking=deep_thinking,
            require_tools=require_tools,
        )
        targets: list[ModelTarget] = []
        for candidate in candidates:
            target = self._build_model_target(candidate, capability)
            if target is not None:
                targets.append(target)
        return targets

    def _filter_and_sort_candidates(
        self,
        candidates: Iterable[ModelCandidate],
        *,
        first_choice_model_id: str | None,
        deep_thinking: bool,
        require_tools: bool,
    ) -> list[ModelCandidate]:
        enabled = [
            candidate
            for candidate in candidates
            if candidate.enabled
            and (not deep_thinking or candidate.supports_thinking)
            and (not require_tools or candidate.supports_tools)
        ]
        if deep_thinking and not enabled:
            logger.warning("deep thinking mode has no available model candidates")
        return sorted(
            enabled,
            key=lambda candidate: (
                self._resolve_id(candidate) != first_choice_model_id,
                candidate.priority is None,
                candidate.priority if candidate.priority is not None else 0,
                self._resolve_id(candidate),
            ),
        )

    def _build_model_target(
        self,
        candidate: ModelCandidate,
        capability: ModelCapability,
    ) -> ModelTarget | None:
        model_id = self._resolve_id(candidate)
        if self._health_store.is_unavailable(model_id):
            return None
        provider = self._config.providers.get(candidate.provider)
        if provider is None:
            logger.warning(
                "provider config missing provider=%s modelId=%s",
                candidate.provider,
                model_id,
            )
            return None
        return ModelTarget(
            id=model_id,
            capability=capability,
            candidate=candidate,
            provider=provider,
        )

    def _resolve_id(self, candidate: ModelCandidate) -> str:
        if candidate.id:
            return candidate.id
        provider = candidate.provider or "unknown"
        model = candidate.model or "unknown"
        return f"{provider}::{model}"
