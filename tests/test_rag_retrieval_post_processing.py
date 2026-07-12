from __future__ import annotations

import pytest

from agent_service.rag.retrieval_pipeline import (
    DeduplicationPostProcessor,
    MultiChannelRetriever,
    RerankPostProcessor,
    RetrievalContext,
    SearchChannelResult,
)
from agent_service.rag.schemas import RetrievedChunk, RetrievedSource


def source(
    source_id: str,
    content: str,
    *,
    channel: str | None = None,
    score: float = 0.5,
) -> RetrievedSource:
    return RetrievedSource(
        id=source_id,
        title=f"{source_id}.md",
        content=content,
        score=score,
        channel=channel,
    )


class StaticChannel:
    def __init__(self, name: str, results: list[RetrievedSource] | Exception) -> None:
        self.name = name
        self._results = results

    async def is_enabled(self, context: RetrievalContext) -> bool:
        return True

    async def search(self, context: RetrievalContext) -> SearchChannelResult:
        if isinstance(self._results, Exception):
            raise self._results
        return SearchChannelResult(channel=self.name, sources=self._results, latency_ms=1)


class RecordingRerankService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[RetrievedChunk], int]] = []

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_n: int,
    ) -> list[RetrievedChunk]:
        self.calls.append((query, candidates, top_n))
        return [
            candidates[1].model_copy(update={"score": 0.99}),
            candidates[0].model_copy(update={"score": 0.88}),
        ]


def test_deduplication_keeps_higher_priority_channel_for_same_chunk() -> None:
    processor = DeduplicationPostProcessor()
    context = RetrievalContext(question="return policy", final_top_k=5)
    results = [
        SearchChannelResult(
            channel="intent_directed",
            sources=[source("chunk-1", "policy", channel="intent_directed", score=0.4)],
            latency_ms=1,
        ),
        SearchChannelResult(
            channel="vector_global",
            sources=[source("chunk-1", "policy", channel="vector_global", score=0.9)],
            latency_ms=1,
        ),
    ]

    deduplicated = processor.process([], results, context)

    assert [item.channel for item in deduplicated] == ["intent_directed"]
    assert deduplicated[0].score == 0.4


@pytest.mark.asyncio
async def test_rerank_post_processor_maps_scores_and_applies_global_top_k() -> None:
    service = RecordingRerankService()
    processor = RerankPostProcessor(service)
    context = RetrievalContext(question="return policy", final_top_k=2)
    candidates = [
        source("chunk-1", "first", channel="intent_directed"),
        source("chunk-2", "second", channel="keyword"),
        source("chunk-3", "third", channel="vector_global"),
    ]

    reranked = await processor.process(candidates, [], context)

    assert [item.id for item in reranked] == ["chunk-2", "chunk-1"]
    assert reranked[0].score == 0.99
    assert service.calls[0][0] == "return policy"
    assert service.calls[0][2] == 2


@pytest.mark.asyncio
async def test_multi_channel_retriever_keeps_healthy_results_when_one_channel_fails() -> None:
    retriever = MultiChannelRetriever(
        channels=[
            StaticChannel("intent_directed", [source("chunk-1", "policy")]),
            StaticChannel("keyword", RuntimeError("keyword unavailable")),
        ],
        post_processors=[DeduplicationPostProcessor()],
    )

    result = await retriever.retrieve(RetrievalContext(question="return policy", final_top_k=5))

    assert [item.id for item in result.sources] == ["chunk-1"]
    assert [item.channel for item in result.channel_results] == ["intent_directed"]
