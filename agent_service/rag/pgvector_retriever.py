from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from agent_service.core.observability import record_stage
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
                        v.chunk_id,
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
                    id=str(row["chunk_id"]),
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

    def list_knowledge_bases(self) -> list[KnowledgeBaseEmbedding]:
        from sqlalchemy import text

        with get_engine().connect() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT id, collection_name, embedding_model
                    FROM t_knowledge_base
                    WHERE deleted = 0
                    ORDER BY id
                    """
                )
            ).mappings()
            return [
                KnowledgeBaseEmbedding(
                    kb_id=str(row["id"]),
                    collection_name=str(row["collection_name"]),
                    embedding_model=(
                        str(row["embedding_model"]).strip()
                        if row["embedding_model"] is not None
                        and str(row["embedding_model"]).strip()
                        else None
                    ),
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

        return await self._search_knowledge_base(
            cleaned_query,
            knowledge_base,
            top_k,
            stage="pgvector_directed_search",
        )

    async def search_global(self, query: str, top_k: int = 5) -> list[RetrievedSource]:
        cleaned_query = query.strip()
        if not cleaned_query or top_k <= 0:
            return []
        knowledge_bases = await asyncio.to_thread(self._repository.list_knowledge_bases)
        tasks = [
            self._search_knowledge_base(
                cleaned_query,
                item,
                top_k,
                stage="pgvector_global_search",
            )
            for item in knowledge_bases
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        sources = [
            source
            for result in results
            if not isinstance(result, Exception)
            for source in result
        ]
        return sorted(sources, key=lambda item: item.score or 0.0, reverse=True)[:top_k]

    async def _search_knowledge_base(
        self,
        query: str,
        knowledge_base: KnowledgeBaseEmbedding,
        top_k: int,
        *,
        stage: str,
    ) -> list[RetrievedSource]:
        started_at = time.perf_counter()
        try:
            query_vector = await self._embedding_service.embed(
                query,
                model_id=knowledge_base.embedding_model,
            )
            sources = await asyncio.to_thread(
                self._repository.search,
                knowledge_base=knowledge_base,
                query_vector=query_vector,
                top_k=top_k,
            )
        except Exception:
            record_stage(
                stage,
                elapsed_ms=_elapsed_ms(started_at),
                collectionName=knowledge_base.collection_name,
                embeddingModel=knowledge_base.embedding_model,
                sourceCount=0,
                status="failed",
            )
            raise
        record_stage(
            stage,
            elapsed_ms=_elapsed_ms(started_at),
            collectionName=knowledge_base.collection_name,
            embeddingModel=knowledge_base.embedding_model,
            sourceCount=len(sources),
            status="success",
        )
        return sources


def _pgvector_literal(vector: list[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in vector) + "]"


def _elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)
