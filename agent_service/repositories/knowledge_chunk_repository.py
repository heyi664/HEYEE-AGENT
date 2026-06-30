from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from agent_service.db.session import get_engine
from agent_service.schemas.chunking import TextChunk, VectorChunk


@dataclass(frozen=True)
class KnowledgeDocumentForChunk:
    id: str
    kb_id: str
    doc_name: str
    file_url: str
    process_mode: str
    status: str
    chunk_strategy: str
    chunk_config: dict[str, Any]
    created_by: str


@dataclass(frozen=True)
class ChunkLogStart:
    id: str
    doc_id: str
    message_id: str | None
    process_mode: str
    chunk_strategy: str
    pipeline_id: str


class KnowledgeChunkRepository:
    def find_document_for_chunk(self, doc_id: str) -> KnowledgeDocumentForChunk | None:
        from sqlalchemy import text

        engine = get_engine()
        with engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT
                            id,
                            kb_id,
                            doc_name,
                            file_url,
                            process_mode,
                            status,
                            chunk_strategy,
                            chunk_config,
                            created_by
                        FROM t_knowledge_document
                        WHERE id = :doc_id AND deleted = 0
                        LIMIT 1
                        """
                    ),
                    {"doc_id": doc_id},
                )
                .mappings()
                .first()
            )
        if row is None:
            return None
        return KnowledgeDocumentForChunk(
            id=str(row["id"]),
            kb_id=str(row["kb_id"]),
            doc_name=str(row["doc_name"]),
            file_url=str(row["file_url"]),
            process_mode=str(row["process_mode"] or "CHUNK"),
            status=str(row["status"]),
            chunk_strategy=str(row["chunk_strategy"]),
            chunk_config=dict(row["chunk_config"] or {}),
            created_by=str(row["created_by"] or "agent"),
        )

    def insert_chunk_log(self, record: ChunkLogStart) -> None:
        from sqlalchemy import text

        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO t_knowledge_document_chunk_log (
                        id,
                        doc_id,
                        status,
                        process_mode,
                        chunk_strategy,
                        pipeline_id,
                        message_id,
                        start_time,
                        create_time,
                        update_time
                    ) VALUES (
                        :id,
                        :doc_id,
                        'RUNNING',
                        :process_mode,
                        :chunk_strategy,
                        :pipeline_id,
                        :message_id,
                        CURRENT_TIMESTAMP,
                        CURRENT_TIMESTAMP,
                        CURRENT_TIMESTAMP
                    )
                    """
                ),
                {
                    "id": record.id,
                    "doc_id": record.doc_id,
                    "process_mode": record.process_mode,
                    "chunk_strategy": record.chunk_strategy,
                    "pipeline_id": record.pipeline_id,
                    "message_id": record.message_id,
                },
            )

    def mark_chunk_log_success(
        self,
        *,
        log_id: str,
        extract_duration: int,
        chunk_duration: int,
        embed_duration: int,
        persist_duration: int,
        total_duration: int,
        chunk_count: int,
    ) -> None:
        from sqlalchemy import text

        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE t_knowledge_document_chunk_log
                    SET status = 'SUCCESS',
                        extract_duration = :extract_duration,
                        chunk_duration = :chunk_duration,
                        embed_duration = :embed_duration,
                        persist_duration = :persist_duration,
                        total_duration = :total_duration,
                        chunk_count = :chunk_count,
                        end_time = CURRENT_TIMESTAMP,
                        update_time = CURRENT_TIMESTAMP
                    WHERE id = :log_id
                    """
                ),
                {
                    "log_id": log_id,
                    "extract_duration": extract_duration,
                    "chunk_duration": chunk_duration,
                    "embed_duration": embed_duration,
                    "persist_duration": persist_duration,
                    "total_duration": total_duration,
                    "chunk_count": chunk_count,
                },
            )

    def mark_chunk_log_failed(
        self, *, log_id: str, error_message: str, total_duration: int
    ) -> None:
        from sqlalchemy import text

        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE t_knowledge_document_chunk_log
                    SET status = 'FAILED',
                        error_message = :error_message,
                        total_duration = :total_duration,
                        end_time = CURRENT_TIMESTAMP,
                        update_time = CURRENT_TIMESTAMP
                    WHERE id = :log_id
                    """
                ),
                {
                    "log_id": log_id,
                    "error_message": error_message[:4000],
                    "total_duration": total_duration,
                },
            )

    def mark_document_failed(self, *, doc_id: str, updated_by: str) -> None:
        from sqlalchemy import text

        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE t_knowledge_document
                    SET status = 'FAILED',
                        updated_by = :updated_by,
                        update_time = CURRENT_TIMESTAMP
                    WHERE id = :doc_id AND deleted = 0
                    """
                ),
                {"doc_id": doc_id, "updated_by": updated_by},
            )

    def replace_chunks_and_vectors_atomic(
        self,
        *,
        doc: KnowledgeDocumentForChunk,
        chunks: list[TextChunk],
        vectors: list[VectorChunk],
        updated_by: str,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        from sqlalchemy import text

        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    DELETE FROM t_knowledge_chunk
                    WHERE doc_id = :doc_id
                    """
                ),
                {"doc_id": doc.id},
            )
            conn.execute(
                text(
                    """
                    DELETE FROM t_knowledge_vector
                    WHERE doc_id = :doc_id
                    """
                ),
                {"doc_id": doc.id},
            )
            if chunks:
                conn.execute(
                    text(
                        """
                        INSERT INTO t_knowledge_chunk (
                            id,
                            kb_id,
                            doc_id,
                            chunk_index,
                            content,
                            content_hash,
                            char_count,
                            token_count,
                            enabled,
                            created_by,
                            updated_by,
                            create_time,
                            update_time,
                            deleted
                        ) VALUES (
                            :id,
                            :kb_id,
                            :doc_id,
                            :chunk_index,
                            :content,
                            :content_hash,
                            :char_count,
                            :token_count,
                            1,
                            :created_by,
                            :updated_by,
                            CURRENT_TIMESTAMP,
                            CURRENT_TIMESTAMP,
                            0
                        )
                        """
                    ),
                    [
                        {
                            "id": chunk.chunk_id,
                            "kb_id": doc.kb_id,
                            "doc_id": doc.id,
                            "chunk_index": chunk.chunk_index,
                            "content": chunk.content,
                            "content_hash": chunk.content_hash,
                            "char_count": chunk.char_count,
                            "token_count": chunk.token_count,
                            "created_by": updated_by,
                            "updated_by": updated_by,
                        }
                        for chunk in chunks
                    ],
                )
                conn.execute(
                    text(
                        """
                        INSERT INTO t_knowledge_vector (
                            id,
                            kb_id,
                            doc_id,
                            chunk_id,
                            chunk_index,
                            embedding,
                            metadata,
                            enabled,
                            created_by,
                            updated_by,
                            create_time,
                            update_time,
                            deleted
                        ) VALUES (
                            :id,
                            :kb_id,
                            :doc_id,
                            :chunk_id,
                            :chunk_index,
                            CAST(:embedding AS vector),
                            CAST(:metadata AS jsonb),
                            1,
                            :created_by,
                            :updated_by,
                            CURRENT_TIMESTAMP,
                            CURRENT_TIMESTAMP,
                            0
                        )
                        """
                    ),
                    [
                        {
                            "id": uuid.uuid4().hex[:20],
                            "kb_id": doc.kb_id,
                            "doc_id": doc.id,
                            "chunk_id": vector.chunk_id,
                            "chunk_index": vector.chunk_index,
                            "embedding": _pgvector_literal(vector.vector),
                            "metadata": json.dumps(vector.metadata, ensure_ascii=False),
                            "created_by": updated_by,
                            "updated_by": updated_by,
                        }
                        for vector in vectors
                    ],
                )
            conn.execute(
                text(
                    """
                    UPDATE t_knowledge_document
                    SET status = 'SUCCESS',
                        chunk_count = :chunk_count,
                        updated_by = :updated_by,
                        update_time = CURRENT_TIMESTAMP
                    WHERE id = :doc_id AND deleted = 0
                    """
                ),
                {"doc_id": doc.id, "chunk_count": len(chunks), "updated_by": updated_by},
            )


def _pgvector_literal(vector: list[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in vector) + "]"
