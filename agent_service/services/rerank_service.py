from __future__ import annotations

from typing import cast

from agent_service.infra_ai import get_model_routing_executor, get_model_selector
from agent_service.infra_ai.clients import RerankModelClient, RerankModelClientRegistry
from agent_service.infra_ai.models import ModelCapability, ModelTarget
from agent_service.rag.schemas import RetrievedChunk


class RerankService:
    def __init__(self, client_registry: RerankModelClientRegistry | None = None) -> None:
        self._selector = get_model_selector()
        self._routing_executor = get_model_routing_executor()
        self._client_registry = client_registry or RerankModelClientRegistry()

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_n: int,
    ) -> list[RetrievedChunk]:
        if top_n <= 0 or not candidates:
            return []
        targets = self._selector.select_rerank_candidates()
        return await self._routing_executor.execute_with_fallback(
            cast(ModelCapability, ModelCapability.RERANK),
            targets,
            self._client_registry.resolve,
            lambda client, target: self._rerank_with_client(
                client,
                target,
                query,
                candidates,
                top_n,
            ),
        )

    async def _rerank_with_client(
        self,
        client: RerankModelClient,
        target: ModelTarget,
        query: str,
        candidates: list[RetrievedChunk],
        top_n: int,
    ) -> list[RetrievedChunk]:
        return await client.rerank(target, query, candidates, top_n)
