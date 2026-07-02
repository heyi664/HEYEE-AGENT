from __future__ import annotations

import pytest

from agent_service.core.errors import ModelUnavailableError
from agent_service.infra_ai.models import (
    ModelCandidate,
    ModelCapability,
    ModelTarget,
    ProviderConfig,
)
from agent_service.services.embedding_service import EmbeddingService


class FakeEmbeddingClient:
    async def embed_batch(self, target: ModelTarget, texts: list[str]) -> list[list[float]]:
        base = 10.0 if target.id == "chosen" else 20.0
        return [[base + float(index)] for index, _ in enumerate(texts)]


class FakeClientRegistry:
    def __init__(self) -> None:
        self.client = FakeEmbeddingClient()

    def resolve(self, target: ModelTarget) -> FakeEmbeddingClient:
        return self.client


class FakeSelector:
    def __init__(self, targets: list[ModelTarget]) -> None:
        self.targets = targets

    def select_embedding_candidates(self) -> list[ModelTarget]:
        return self.targets


class FakeRoutingExecutor:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def execute_with_fallback(self, capability, targets, client_resolver, caller):
        assert capability == ModelCapability.EMBEDDING
        self.calls.append([target.id for target in targets])
        target = targets[0]
        client = client_resolver(target)
        return await caller(client, target)


def _target(model_id: str, dimension: int = 3) -> ModelTarget:
    return ModelTarget(
        id=model_id,
        capability=ModelCapability.EMBEDDING,
        candidate=ModelCandidate(
            id=model_id,
            provider="siliconflow",
            model=model_id,
            dimension=dimension,
        ),
        provider=ProviderConfig(
            name="siliconflow",
            url="https://embedding.example.com",
            api_key="sk-test",
            endpoints={"embedding": "/v1/embeddings"},
        ),
    )


def _service(targets: list[ModelTarget]) -> tuple[EmbeddingService, FakeRoutingExecutor]:
    executor = FakeRoutingExecutor()
    service = EmbeddingService(client_registry=FakeClientRegistry())
    service._selector = FakeSelector(targets)
    service._routing_executor = executor
    return service, executor


@pytest.mark.asyncio
async def test_embedding_service_embed_and_embed_batch_can_target_model_id() -> None:
    service, executor = _service([_target("fallback"), _target("chosen")])

    one = await service.embed("hello", model_id="chosen")
    many = await service.embed_batch(["a", "b"], model_id="chosen")

    assert one == [10.0]
    assert many == [[10.0], [11.0]]
    assert executor.calls == [["chosen"], ["chosen"]]


@pytest.mark.asyncio
async def test_embedding_service_embed_batch_uses_all_candidates_without_model_id() -> None:
    service, executor = _service([_target("fallback"), _target("chosen")])

    vectors = await service.embed_batch(["a", "b"])

    assert vectors == [[20.0], [21.0]]
    assert executor.calls == [["fallback", "chosen"]]


def test_embedding_service_dimension_resolves_model_id() -> None:
    service, _executor = _service([_target("fallback", 5), _target("chosen", 7)])

    assert service.dimension("chosen") == 7


@pytest.mark.asyncio
async def test_embedding_service_rejects_unknown_model_id() -> None:
    service, _executor = _service([_target("fallback")])

    with pytest.raises(ModelUnavailableError, match="Embedding model is unavailable"):
        await service.embed("hello", model_id="missing")
