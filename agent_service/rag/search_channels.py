from __future__ import annotations

import time

from agent_service.core.config import get_settings
from agent_service.rag.intent_directed_retriever import IntentDirectedRetriever
from agent_service.rag.keyword_retriever import PostgresKeywordRetriever
from agent_service.rag.pgvector_retriever import PgvectorRetriever
from agent_service.rag.retrieval_pipeline import RetrievalContext, SearchChannelResult


class IntentDirectedSearchChannel:
    name = "intent_directed"

    def __init__(self, retriever: IntentDirectedRetriever | None = None) -> None:
        self._retriever = retriever or IntentDirectedRetriever(PgvectorRetriever())

    async def is_enabled(self, context: RetrievalContext) -> bool:
        return bool(context.sub_intents)

    async def search(self, context: RetrievalContext) -> SearchChannelResult:
        started_at = time.perf_counter()
        sources = await self._retriever.search(
            context.sub_intents,
            fallback_queries=context.fallback_queries,
            top_k=context.candidate_top_k,
        )
        return SearchChannelResult(
            channel=self.name,
            sources=sources,
            latency_ms=int((time.perf_counter() - started_at) * 1000),
        )


class KeywordSearchChannel:
    name = "keyword"

    def __init__(self, retriever: PostgresKeywordRetriever | None = None) -> None:
        self._retriever = retriever or PostgresKeywordRetriever()

    async def is_enabled(self, context: RetrievalContext) -> bool:
        del context
        return get_settings().rag_retrieval_keyword_enabled

    async def search(self, context: RetrievalContext) -> SearchChannelResult:
        started_at = time.perf_counter()
        sources = await self._retriever.search(context.question, context.candidate_top_k)
        return SearchChannelResult(
            channel=self.name,
            sources=sources,
            latency_ms=int((time.perf_counter() - started_at) * 1000),
        )


class GlobalVectorSearchChannel:
    name = "vector_global"

    def __init__(self, retriever: PgvectorRetriever | None = None) -> None:
        self._retriever = retriever or PgvectorRetriever()

    async def is_enabled(self, context: RetrievalContext) -> bool:
        del context
        return get_settings().rag_retrieval_global_vector_enabled

    async def search(self, context: RetrievalContext) -> SearchChannelResult:
        started_at = time.perf_counter()
        sources = await self._retriever.search_global(context.question, context.candidate_top_k)
        return SearchChannelResult(
            channel=self.name,
            sources=sources,
            latency_ms=int((time.perf_counter() - started_at) * 1000),
        )
