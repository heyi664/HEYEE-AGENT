from __future__ import annotations

import asyncio

from agent_service.rag.schemas import RetrievedSource


class Retriever:
    async def search(self, query: str, top_k: int = 5) -> list[RetrievedSource]:
        return []

    async def search_many(self, queries: list[str], top_k: int = 5) -> list[RetrievedSource]:
        cleaned_queries = [query.strip() for query in queries if query.strip()]
        query_results = await asyncio.gather(
            *(self.search(query, top_k=top_k) for query in cleaned_queries)
        )
        return [source for results in query_results for source in results]
