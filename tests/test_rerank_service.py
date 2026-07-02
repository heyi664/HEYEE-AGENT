from __future__ import annotations

import pytest

from agent_service.infra_ai.models import (
    ModelCandidate,
    ModelCapability,
    ModelTarget,
    ProviderConfig,
)
from agent_service.rag.schemas import RetrievedChunk
from agent_service.services.rerank_service import RerankService


class FakeRerankClient:
    async def rerank(
        self,
        target: ModelTarget,
        query: str,
        candidates: list[RetrievedChunk],
        top_n: int,
    ) -> list[RetrievedChunk]:
        assert target.id == "chosen-rerank"
        assert query == "AirPods"
        assert top_n == 1
        return candidates[:top_n]


class FakeClientRegistry:
    def __init__(self) -> None:
        self.client = FakeRerankClient()

    def resolve(self, target: ModelTarget) -> FakeRerankClient:
        return self.client


class FakeSelector:
    def __init__(self, targets: list[ModelTarget]) -> None:
        self.targets = targets

    def select_rerank_candidates(self) -> list[ModelTarget]:
        return self.targets


class FakeRoutingExecutor:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def execute_with_fallback(self, capability, targets, client_resolver, caller):
        assert capability == ModelCapability.RERANK
        self.calls.append([target.id for target in targets])
        target = targets[0]
        client = client_resolver(target)
        return await caller(client, target)


def _target(model_id: str) -> ModelTarget:
    return ModelTarget(
        id=model_id,
        capability=ModelCapability.RERANK,
        candidate=ModelCandidate(id=model_id, provider="bailian", model="gte-rerank"),
        provider=ProviderConfig(
            name="bailian",
            url="https://dashscope.example.com",
            api_key="sk-test",
            endpoints={"rerank": "/rerank"},
        ),
    )


def _service(targets: list[ModelTarget]) -> tuple[RerankService, FakeRoutingExecutor]:
    executor = FakeRoutingExecutor()
    service = RerankService(client_registry=FakeClientRegistry())
    service._selector = FakeSelector(targets)
    service._routing_executor = executor
    return service, executor


@pytest.mark.asyncio
async def test_rerank_service_routes_candidates_through_model_executor() -> None:
    service, executor = _service([_target("chosen-rerank"), _target("fallback-rerank")])
    candidates = [
        RetrievedChunk(id="A", text="first", score=0.1),
        RetrievedChunk(id="B", text="second", score=0.2),
    ]

    reranked = await service.rerank("AirPods", candidates, top_n=1)

    assert [chunk.id for chunk in reranked] == ["A"]
    assert executor.calls == [["chosen-rerank", "fallback-rerank"]]


@pytest.mark.asyncio
async def test_rerank_service_returns_empty_for_empty_or_non_positive_top_n() -> None:
    service, executor = _service([_target("chosen-rerank")])

    assert await service.rerank("AirPods", [], top_n=3) == []
    assert await service.rerank("AirPods", [RetrievedChunk(id="A", text="first")], top_n=0) == []
    assert executor.calls == []
