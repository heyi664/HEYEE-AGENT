from __future__ import annotations

import asyncio
from dataclasses import dataclass

from agent_service.db.session import get_engine
from agent_service.rag.retriever import Retriever
from agent_service.rag.schemas import RetrievedSource
from agent_service.services.embedding_service import EmbeddingService


@dataclass(frozen=True)
class KnowledgeBaseEmbedding:
    kb_id: str
    collection_name: str
    embedding_model: str | None


class PgvectorKnowledgeRepository:
    def find_knowledge_base(self, collection_name: str) -> KnowledgeBaseEmbedding | None:
        from sqlalchemy import text

        with get_engine().connect() as connection:
            row = (
                connection.execute(
                    text(
                        """
                        SELECT id, collection_name, embedding_model
                        FROM t_knowledge_base
                        WHERE collection_name = :collection_name AND deleted = 0
                        LIMIT 1
                        """
                    ),
                    {"collection_name": collection_name},
                )
                .mappings()
                .first()
            )
        if row is None:
            return None
        return KnowledgeBaseEmbedding(
            kb_id=str(row["id"]),
            collection_name=str(row["collection_name"]),
            embedding_model=(
                str(row["embedding_model"]).strip()
                if row["embedding_model"] is not None and str(row["embedding_model"]).strip()
                else None
            ),
        )

    def search(
        self,
        *,
        knowledge_base: KnowledgeBaseEmbedding,
        query_vector: list[float],
        top_k: int,
    ) -> list[RetrievedSource]:
        from sqlalchemy import text

        vector_literal = _pgvector_literal(query_vector)
        with get_engine().connect() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT
                        d.doc_name,
                        d.file_url,
                        d.source_type,
                        c.content,
                        1 - (v.embedding <=> CAST(:query_vector AS vector)) AS score
                    FROM t_knowledge_vector v
                    JOIN t_knowledge_chunk c
                      ON c.id = v.chunk_id
                     AND c.deleted = 0
                     AND c.enabled = 1
                    JOIN t_knowledge_document d
                      ON d.id = v.doc_id
                     AND d.deleted = 0
                     AND d.enabled = 1
                     AND d.status = 'SUCCESS'
                    WHERE v.kb_id = :kb_id
                      AND v.deleted = 0
                      AND v.enabled = 1
                    ORDER BY v.embedding <=> CAST(:query_vector AS vector)
                    LIMIT :top_k
                    """
                ),
                {
                    "kb_id": knowledge_base.kb_id,
                    "query_vector": vector_literal,
                    "top_k": top_k,
                },
            ).mappings()
            return [
                RetrievedSource(
                    title=str(row["doc_name"]),
                    content=str(row["content"]),
                    score=float(row["score"]) if row["score"] is not None else None,
                    source_type=(
                        str(row["source_type"])
                        if row["source_type"] is not None
                        else "knowledge_base"
                    ),
                    url=str(row["file_url"]) if row["file_url"] is not None else None,
                    collection_name=knowledge_base.collection_name,
                )
                for row in rows
            ]


class PgvectorRetriever(Retriever):
    def __init__(
        self,
        *,
        embedding_service: EmbeddingService | object | None = None,
        repository: PgvectorKnowledgeRepository | object | None = None,
    ) -> None:
        self._embedding_service = embedding_service or EmbeddingService()
        self._repository = repository or PgvectorKnowledgeRepository()

    async def search(
        self,
        query: str,
        top_k: int = 5,
        collection_name: str | None = None,
    ) -> list[RetrievedSource]:
        cleaned_query = query.strip()
        cleaned_collection = (collection_name or "").strip()
        if not cleaned_query or not cleaned_collection or top_k <= 0:
            return []

        knowledge_base = await asyncio.to_thread(
            self._repository.find_knowledge_base,
            cleaned_collection,
        )
        if knowledge_base is None:
            return []

        query_vector = await self._embedding_service.embed(
            cleaned_query,
            model_id=knowledge_base.embedding_model,
        )
        return await asyncio.to_thread(
            self._repository.search,
            knowledge_base=knowledge_base,
            query_vector=query_vector,
            top_k=top_k,
        )


def _pgvector_literal(vector: list[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in vector) + "]"
