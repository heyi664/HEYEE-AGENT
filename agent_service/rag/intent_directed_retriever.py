from __future__ import annotations

import asyncio

from agent_service.rag.intent_models import SubQuestionIntent
from agent_service.rag.retriever import Retriever
from agent_service.rag.schemas import RetrievedSource


class IntentDirectedRetriever:
    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    async def search(
        self,
        sub_intents: list[SubQuestionIntent],
        *,
        fallback_queries: list[str],
        top_k: int = 5,
    ) -> list[RetrievedSource]:
        tasks = []
        for sub_intent in sub_intents:
            for score in sub_intent.node_scores:
                node = score.node
                if not node.is_kb() or not node.collection_name:
                    continue
                tasks.append(
                    self._retriever.search(
                        sub_intent.sub_question,
                        top_k=node.top_k or top_k,
                        collection_name=node.collection_name,
                    )
                )
        if not tasks:
            return await self._retriever.search_many(fallback_queries, top_k=top_k)
        query_results = await asyncio.gather(*tasks)
        return [source for results in query_results for source in results]
