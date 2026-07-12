from __future__ import annotations

import pytest

from agent_service.rag.pgvector_retriever import (
    KnowledgeBaseEmbedding,
    PgvectorRetriever,
)
from agent_service.rag.schemas import RetrievedSource


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    async def embed(self, text: str, model_id: str | None = None) -> list[float]:
        self.calls.append((text, model_id))
        return [0.1, 0.2, 0.3]


class FakeVectorRepository:
    def __init__(self) -> None:
        self.collection: KnowledgeBaseEmbedding | None = KnowledgeBaseEmbedding(
            kb_id="kb-return",
            collection_name="test111",
            embedding_model="BAAI/bge-m3",
        )
        self.calls: list[tuple[str, list[float], int]] = []

    def find_knowledge_base(self, collection_name: str) -> KnowledgeBaseEmbedding | None:
        if self.collection is None or collection_name != self.collection.collection_name:
            return None
        return self.collection

    def search(
        self,
        *,
        knowledge_base: KnowledgeBaseEmbedding,
        query_vector: list[float],
        top_k: int,
    ) -> list[RetrievedSource]:
        self.calls.append((knowledge_base.kb_id, query_vector, top_k))
        return [
            RetrievedSource(
                title="return-policy.md",
                content="Supports returns for eligible items.",
                score=0.91,
                source_type="knowledge_base",
                url="rustfs://test111/return-policy.md",
                collection_name=knowledge_base.collection_name,
            )
        ]


@pytest.mark.asyncio
async def test_pgvector_retriever_uses_collection_embedding_model_and_top_k() -> None:
    embedding_service = FakeEmbeddingService()
    repository = FakeVectorRepository()
    retriever = PgvectorRetriever(
        embedding_service=embedding_service,
        repository=repository,
    )

    sources = await retriever.search(
        "return policy",
        collection_name="test111",
        top_k=3,
    )

    assert embedding_service.calls == [("return policy", "BAAI/bge-m3")]
    assert repository.calls == [("kb-return", [0.1, 0.2, 0.3], 3)]
    assert sources[0].collection_name == "test111"


@pytest.mark.asyncio
async def test_pgvector_retriever_skips_unknown_collection_without_embedding() -> None:
    embedding_service = FakeEmbeddingService()
    repository = FakeVectorRepository()
    retriever = PgvectorRetriever(
        embedding_service=embedding_service,
        repository=repository,
    )

    sources = await retriever.search("return policy", collection_name="missing")

    assert sources == []
    assert embedding_service.calls == []
    assert repository.calls == []
