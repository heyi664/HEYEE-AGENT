from __future__ import annotations

import json

import httpx
import pytest

from agent_service.infra_ai.clients import BaiLianRerankClient, NoopRerankClient
from agent_service.infra_ai.models import (
    ModelCandidate,
    ModelCapability,
    ModelTarget,
    ProviderConfig,
)
from agent_service.rag.schemas import RetrievedChunk


def _rerank_target(provider: str = "bailian") -> ModelTarget:
    return ModelTarget(
        id="bailian-rerank",
        capability=ModelCapability.RERANK,
        candidate=ModelCandidate(
            id="bailian-rerank",
            provider=provider,
            model="gte-rerank",
        ),
        provider=ProviderConfig(
            name=provider,
            url="https://dashscope.example.com",
            api_key="sk-test",
            endpoints={"rerank": "/api/v1/services/rerank/text-rerank/text-rerank"},
        ),
    )


def _chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(id="A", text="AirPods Pro 2 warranty is one year.", score=0.2),
        RetrievedChunk(id="B", text="AppleCare extends coverage.", score=0.1),
        RetrievedChunk(id="A", text="duplicated by hybrid retrieval", score=0.3),
        RetrievedChunk(id="C", text="Battery service details.", score=0.05),
        RetrievedChunk(id="D", text="Ear tip replacement details.", score=0.04),
    ]


@pytest.mark.asyncio
async def test_noop_rerank_client_deduplicates_and_limits_candidates() -> None:
    client = NoopRerankClient()

    reranked = await client.rerank(_rerank_target("noop"), "query", _chunks(), 2)

    assert [chunk.id for chunk in reranked] == ["A", "B"]
    assert reranked[0].score == 0.2


@pytest.mark.asyncio
async def test_bailian_rerank_client_posts_protocol_maps_scores_and_backfills() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        body = json.loads(request.content.decode("utf-8"))
        captured["body"] = body
        return httpx.Response(
            200,
            json={
                "output": {
                    "results": [
                        {"index": 1, "relevance_score": 0.88},
                        {"index": 99, "relevance_score": 0.99},
                    ]
                }
            },
        )

    client = BaiLianRerankClient(transport=httpx.MockTransport(handler))

    reranked = await client.rerank(_rerank_target(), "AirPods warranty", _chunks(), 3)

    assert captured["url"] == (
        "https://dashscope.example.com/api/v1/services/rerank/text-rerank/text-rerank"
    )
    assert captured["authorization"] == "Bearer sk-test"
    assert captured["body"] == {
        "model": "gte-rerank",
        "input": {
            "query": "AirPods warranty",
            "documents": [
                "AirPods Pro 2 warranty is one year.",
                "AppleCare extends coverage.",
                "Battery service details.",
                "Ear tip replacement details.",
            ],
        },
        "parameters": {"top_n": 3, "return_documents": True},
    }
    assert [(chunk.id, chunk.score) for chunk in reranked] == [
        ("B", 0.88),
        ("A", 0.2),
        ("C", 0.05),
    ]


@pytest.mark.asyncio
async def test_bailian_rerank_client_skips_http_when_deduped_candidates_fit_top_n() -> None:
    called = False

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500)

    client = BaiLianRerankClient(transport=httpx.MockTransport(handler))

    reranked = await client.rerank(_rerank_target(), "query", _chunks()[:4], 3)

    assert called is False
    assert [chunk.id for chunk in reranked] == ["A", "B", "C"]
