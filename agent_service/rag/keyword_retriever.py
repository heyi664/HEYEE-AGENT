from __future__ import annotations

import asyncio

from agent_service.db.session import get_engine
from agent_service.rag.schemas import RetrievedSource


class PostgresKeywordRetriever:
    async def search(self, query: str, top_k: int) -> list[RetrievedSource]:
        cleaned_query = query.strip()
        if not cleaned_query or top_k <= 0:
            return []
        return await asyncio.to_thread(self._search_sync, cleaned_query, top_k)

    def _search_sync(self, query: str, top_k: int) -> list[RetrievedSource]:
        from sqlalchemy import text

        with get_engine().connect() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT
                        c.id AS chunk_id,
                        c.content,
                        d.doc_name,
                        d.file_url,
                        d.source_type,
                        kb.collection_name,
                        CASE
                            WHEN c.content ILIKE '%' || :query || '%' THEN 1.0
                            ELSE ts_rank_cd(
                                to_tsvector('simple', c.content),
                                websearch_to_tsquery('simple', :query)
                            )
                        END AS score
                    FROM t_knowledge_chunk c
                    JOIN t_knowledge_document d
                      ON d.id = c.doc_id
                     AND d.deleted = 0
                     AND d.enabled = 1
                     AND d.status = 'SUCCESS'
                    JOIN t_knowledge_base kb
                      ON kb.id = c.kb_id
                     AND kb.deleted = 0
                    WHERE c.deleted = 0
                      AND c.enabled = 1
                      AND (
                        c.content ILIKE '%' || :query || '%'
                        OR to_tsvector('simple', c.content)
                           @@ websearch_to_tsquery('simple', :query)
                      )
                    ORDER BY score DESC, c.chunk_index ASC
                    LIMIT :top_k
                    """
                ),
                {"query": query, "top_k": top_k},
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
                    collection_name=str(row["collection_name"]),
                )
                for row in rows
            ]
