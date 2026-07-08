from __future__ import annotations

import pytest

from agent_service.rag.intent_directed_retriever import IntentDirectedRetriever
from agent_service.rag.intent_models import (
    IntentKind,
    IntentLevel,
    IntentNode,
    NodeScore,
    SubQuestionIntent,
)
from agent_service.rag.retriever import Retriever
from agent_service.rag.schemas import RetrievedSource


class FakeRetriever(Retriever):
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str | None]] = []
        self.search_many_calls: list[tuple[list[str], int]] = []

    async def search(
        self,
        query: str,
        top_k: int = 5,
        collection_name: str | None = None,
    ) -> list[RetrievedSource]:
        self.calls.append((query, top_k, collection_name))
        return [RetrievedSource(title=query, content=collection_name or "fallback", score=1.0)]

    async def search_many(self, queries: list[str], top_k: int = 5) -> list[RetrievedSource]:
        self.search_many_calls.append((queries, top_k))
        return [RetrievedSource(title=query, content="fallback", score=1.0) for query in queries]


def score(
    node_id: str,
    *,
    collection_name: str | None,
    top_k: int | None = None,
    kind: IntentKind = IntentKind.KB,
) -> NodeScore:
    return NodeScore(
        node=IntentNode(
            id=node_id,
            name=node_id,
            level=IntentLevel.TOPIC,
            kind=kind,
            collection_name=collection_name,
            top_k=top_k,
        ),
        score=0.9,
    )


@pytest.mark.asyncio
async def test_directed_retriever_searches_kb_intent_collections() -> None:
    base = FakeRetriever()
    retriever = IntentDirectedRetriever(base)
    sub_intents = [
        SubQuestionIntent(
            "退换政策是什么？",
            [
                score("3c-return", collection_name="kb_3c_return", top_k=2),
                score("mcp-order", collection_name=None, kind=IntentKind.MCP),
            ],
        )
    ]

    results = await retriever.search(sub_intents, fallback_queries=["退换政策是什么？"], top_k=5)

    assert base.calls == [("退换政策是什么？", 2, "kb_3c_return")]
    assert base.search_many_calls == []
    assert results[0].content == "kb_3c_return"


@pytest.mark.asyncio
async def test_directed_retriever_falls_back_without_kb_collections() -> None:
    base = FakeRetriever()
    retriever = IntentDirectedRetriever(base)

    results = await retriever.search(
        [SubQuestionIntent("查订单", [score("order", collection_name=None, kind=IntentKind.MCP)])],
        fallback_queries=["查订单"],
        top_k=3,
    )

    assert base.calls == []
    assert base.search_many_calls == [(["查订单"], 3)]
    assert results[0].title == "查订单"
