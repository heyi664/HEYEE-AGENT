from __future__ import annotations

import pytest

from agent_service.rag.retriever import Retriever
from agent_service.rag.schemas import RetrievedSource


class FakeRetriever(Retriever):
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def search(
        self,
        query: str,
        top_k: int = 5,
        collection_name: str | None = None,
    ) -> list[RetrievedSource]:
        self.queries.append(query)
        return [RetrievedSource(title=query, content="content", score=1.0)]


@pytest.mark.asyncio
async def test_search_many_searches_each_sub_question() -> None:
    retriever = FakeRetriever()

    results = await retriever.search_many(["问题 A", "问题 B"], top_k=3)

    assert retriever.queries == ["问题 A", "问题 B"]
    assert [result.title for result in results] == ["问题 A", "问题 B"]


@pytest.mark.asyncio
async def test_search_many_skips_blank_questions() -> None:
    retriever = FakeRetriever()

    await retriever.search_many(["问题 A", " ", "问题 B"], top_k=3)

    assert retriever.queries == ["问题 A", "问题 B"]
