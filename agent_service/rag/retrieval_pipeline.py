from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent_service.core.observability import record_stage
from agent_service.rag.schemas import RetrievedChunk, RetrievedSource
from agent_service.services.rerank_service import RerankService

logger = logging.getLogger(__name__)

CHANNEL_PRIORITIES = {
    "intent_directed": 1,
    "keyword": 2,
    "vector_global": 3,
}


@dataclass(frozen=True)
class RetrievalContext:
    question: str
    sub_intents: list[Any] = field(default_factory=list)
    fallback_queries: list[str] = field(default_factory=list)
    candidate_top_k: int = 10
    final_top_k: int = 5


@dataclass(frozen=True)
class SearchChannelResult:
    channel: str
    sources: list[RetrievedSource]
    latency_ms: int


@dataclass(frozen=True)
class RetrievalResult:
    sources: list[RetrievedSource]
    channel_results: list[SearchChannelResult]


class SearchChannel(Protocol):
    name: str

    async def is_enabled(self, context: RetrievalContext) -> bool: ...

    async def search(self, context: RetrievalContext) -> SearchChannelResult: ...


class SearchResultPostProcessor(Protocol):
    name: str
    order: int

    def process(
        self,
        sources: list[RetrievedSource],
        results: list[SearchChannelResult],
        context: RetrievalContext,
    ) -> list[RetrievedSource] | object: ...


class DeduplicationPostProcessor:
    name = "deduplication"
    order = 1

    def process(
        self,
        sources: list[RetrievedSource],
        results: list[SearchChannelResult],
        context: RetrievalContext,
    ) -> list[RetrievedSource]:
        del sources, context
        deduplicated: dict[str, RetrievedSource] = {}
        for result in sorted(
            results,
            key=lambda item: CHANNEL_PRIORITIES.get(item.channel, 99),
        ):
            for source in result.sources:
                normalized = source.model_copy(update={"channel": source.channel or result.channel})
                deduplicated.setdefault(_source_key(normalized), normalized)
        return list(deduplicated.values())


class RerankPostProcessor:
    name = "rerank"
    order = 10

    def __init__(self, rerank_service: RerankService | object | None = None) -> None:
        self._rerank_service = rerank_service or RerankService()

    async def process(
        self,
        sources: list[RetrievedSource],
        results: list[SearchChannelResult],
        context: RetrievalContext,
    ) -> list[RetrievedSource]:
        del results
        if not sources or context.final_top_k <= 0:
            return []

        source_by_id: dict[str, RetrievedSource] = {}
        candidates: list[RetrievedChunk] = []
        for source in sources:
            source_id = source.id or _source_key(source)
            if source_id in source_by_id:
                continue
            source_by_id[source_id] = source
            candidates.append(
                RetrievedChunk(id=source_id, text=source.content, score=source.score)
            )

        started_at = time.perf_counter()
        try:
            reranked = await self._rerank_service.rerank(
                context.question,
                candidates,
                context.final_top_k,
            )
        except Exception:
            record_stage(
                "rerank_llm",
                elapsed_ms=_elapsed_ms(started_at),
                candidateCount=len(candidates),
                outputCount=0,
                status="failed",
            )
            raise
        record_stage(
            "rerank_llm",
            elapsed_ms=_elapsed_ms(started_at),
            candidateCount=len(candidates),
            outputCount=len(reranked),
            status="success",
        )
        reranked_sources: list[RetrievedSource] = []
        seen: set[str] = set()
        for chunk in reranked:
            source = source_by_id.get(chunk.id)
            if source is None or chunk.id in seen:
                continue
            seen.add(chunk.id)
            reranked_sources.append(source.model_copy(update={"score": chunk.score}))
            if len(reranked_sources) >= context.final_top_k:
                return reranked_sources

        for source_id, source in source_by_id.items():
            if source_id in seen:
                continue
            reranked_sources.append(source)
            if len(reranked_sources) >= context.final_top_k:
                break
        return reranked_sources


class MultiChannelRetriever:
    def __init__(
        self,
        *,
        channels: list[SearchChannel],
        post_processors: list[SearchResultPostProcessor],
    ) -> None:
        self._channels = channels
        self._post_processors = sorted(post_processors, key=lambda item: item.order)

    async def retrieve(self, context: RetrievalContext) -> RetrievalResult:
        enabled_channels = await self._enabled_channels(context)
        raw_results = await asyncio.gather(
            *(self._search_channel(channel, context) for channel in enabled_channels),
            return_exceptions=True,
        )
        channel_results: list[SearchChannelResult] = []
        for channel, raw_result in zip(enabled_channels, raw_results, strict=True):
            if isinstance(raw_result, Exception):
                logger.warning(
                    "search channel failed channel=%s error=%s",
                    channel.name,
                    raw_result,
                )
                continue
            channel_results.append(raw_result)
            record_stage(
                "retrieval_channel",
                elapsed_ms=raw_result.latency_ms,
                channel=raw_result.channel,
                sourceCount=len(raw_result.sources),
                status="success",
            )
            logger.info(
                "search channel completed channel=%s sources=%s latencyMs=%s",
                raw_result.channel,
                len(raw_result.sources),
                raw_result.latency_ms,
            )

        sources = [source for result in channel_results for source in result.sources]
        record_stage(
            "retrieval_aggregation",
            elapsed_ms=0,
            channelCount=len(channel_results),
            candidateCount=len(sources),
        )
        for processor in self._post_processors:
            before_size = len(sources)
            processor_started_at = time.perf_counter()
            try:
                processed = processor.process(sources, channel_results, context)
                sources = await processed if inspect.isawaitable(processed) else processed
                logger.info(
                    "retrieval post processor completed processor=%s before=%s after=%s",
                    processor.name,
                    before_size,
                    len(sources),
                )
                record_stage(
                    "retrieval_post_processor",
                    elapsed_ms=_elapsed_ms(processor_started_at),
                    afterCount=len(sources),
                    beforeCount=before_size,
                    processor=processor.name,
                    status="success",
                )
            except Exception:
                logger.exception(
                    "retrieval post processor failed; skipped processor=%s",
                    processor.name,
                )
                record_stage(
                    "retrieval_post_processor",
                    elapsed_ms=_elapsed_ms(processor_started_at),
                    afterCount=before_size,
                    beforeCount=before_size,
                    processor=processor.name,
                    status="failed",
                )
        return RetrievalResult(
            sources=sources[: max(context.final_top_k, 0)],
            channel_results=channel_results,
        )

    async def _search_channel(
        self,
        channel: SearchChannel,
        context: RetrievalContext,
    ) -> SearchChannelResult:
        started_at = time.perf_counter()
        try:
            return await channel.search(context)
        except Exception:
            record_stage(
                "retrieval_channel",
                elapsed_ms=_elapsed_ms(started_at),
                channel=channel.name,
                sourceCount=0,
                status="failed",
            )
            raise

    async def _enabled_channels(self, context: RetrievalContext) -> list[SearchChannel]:
        enabled_flags = await asyncio.gather(
            *(channel.is_enabled(context) for channel in self._channels),
            return_exceptions=True,
        )
        enabled: list[SearchChannel] = []
        for channel, value in zip(self._channels, enabled_flags, strict=True):
            if isinstance(value, Exception):
                logger.warning(
                    "search channel enablement failed channel=%s error=%s",
                    channel.name,
                    value,
                )
                continue
            if value:
                enabled.append(channel)
        return enabled


def _source_key(source: RetrievedSource) -> str:
    if source.id:
        return source.id
    return f"{source.url or ''}|{source.title}|{source.content}".strip()


def _elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)
