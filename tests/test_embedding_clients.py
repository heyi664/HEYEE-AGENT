from __future__ import annotations

import json

import httpx
import pytest

from agent_service.core.errors import ModelUnavailableError
from agent_service.infra_ai.clients import (
    OllamaEmbeddingModelClient,
    SiliconFlowEmbeddingModelClient,
)
from agent_service.infra_ai.models import (
    ModelCandidate,
    ModelCapability,
    ModelTarget,
    ProviderConfig,
)


def _embedding_target(
    *,
    provider_name: str = "siliconflow",
    api_key: str | None = "sk-test",
    dimension: int | None = 3,
) -> ModelTarget:
    return ModelTarget(
        id="embed-model",
        capability=ModelCapability.EMBEDDING,
        candidate=ModelCandidate(
            id="embed-model",
            provider=provider_name,
            model="embedding-model",
            dimension=dimension,
        ),
        provider=ProviderConfig(
            name=provider_name,
            url="https://embedding.example.com",
            api_key=api_key,
            endpoints={"embedding": "/v1/embeddings"},
        ),
    )


@pytest.mark.asyncio
async def test_siliconflow_embedding_client_batches_at_32_and_preserves_order() -> None:
    batch_sizes: list[int] = []
    request_bodies: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        request_bodies.append(body)
        inputs = body["input"]
        assert isinstance(inputs, list)
        batch_sizes.append(len(inputs))
        offset = sum(batch_sizes[:-1])
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": index, "embedding": [float(offset + index), 1.0, 2.0]}
                    for index, _ in enumerate(inputs)
                ]
            },
        )

    client = SiliconFlowEmbeddingModelClient(transport=httpx.MockTransport(handler))
    vectors = await client.embed_batch(
        _embedding_target(provider_name="siliconflow", dimension=3),
        [f"chunk-{index}" for index in range(70)],
    )

    assert batch_sizes == [32, 32, 6]
    assert len(vectors) == 70
    assert vectors[0] == [0.0, 1.0, 2.0]
    assert vectors[31] == [31.0, 1.0, 2.0]
    assert vectors[32] == [32.0, 1.0, 2.0]
    assert vectors[69] == [69.0, 1.0, 2.0]
    assert request_bodies[0]["model"] == "embedding-model"
    assert request_bodies[0]["dimensions"] == 3
    assert request_bodies[0]["encoding_format"] == "float"


@pytest.mark.asyncio
async def test_ollama_embedding_client_omits_auth_and_encoding_format() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers.get("authorization")
        captured["json"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]},
        )

    client = OllamaEmbeddingModelClient(transport=httpx.MockTransport(handler))
    vectors = await client.embed_batch(
        _embedding_target(provider_name="ollama", api_key=None, dimension=3),
        ["hello"],
    )

    assert vectors == [[0.1, 0.2, 0.3]]
    assert captured["authorization"] is None
    assert captured["json"] == {
        "model": "embedding-model",
        "input": ["hello"],
        "dimensions": 3,
    }


@pytest.mark.asyncio
async def test_embedding_client_treats_200_error_field_as_provider_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"error": {"code": "invalid_model", "message": "model missing"}},
        )

    client = SiliconFlowEmbeddingModelClient(transport=httpx.MockTransport(handler))

    with pytest.raises(ModelUnavailableError, match="invalid_model"):
        await client.embed_batch(
            _embedding_target(provider_name="siliconflow", dimension=3),
            ["hello"],
        )
