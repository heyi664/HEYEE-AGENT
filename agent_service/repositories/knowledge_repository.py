from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agent_service.db.session import get_engine
from agent_service.schemas.knowledge import KnowledgeBaseSummary


@dataclass(frozen=True)
class KnowledgeBaseRecord:
    id: str
    name: str
    embedding_model: str
    collection_name: str
    created_by: str

@dataclass(frozen=True)
class KnowledgeDocumentRecord:
    id: str
    kb_id: str
    doc_name: str
    file_url: str
    file_type: str
    file_size: int
    source_type: str
    source_location: str | None
    chunk_strategy: str
    chunk_config: dict[str, Any]
    created_by: str


@dataclass(frozen=True)
class KnowledgeDocumentChunkTarget:
    id: str
    kb_id: str
    doc_name: str
    file_url: str
    file_type: str
    file_size: int
    source_type: str
    source_location: str | None
    chunk_strategy: str
    chunk_config: dict[str, Any]
    status: str


@dataclass(frozen=True)
class KnowledgeDocumentChunkStatusRecord:
    id: str
    kb_id: str
    doc_name: str
    status: str
    chunk_count: int
    log_status: str | None
    message_id: str | None
    error_message: str | None
    total_duration: int | None
    extract_duration: int | None
    chunk_duration: int | None
    embed_duration: int | None
    persist_duration: int | None
    log_create_time: str | None
    log_end_time: str | None

class KnowledgeRepository:
    def list_knowledge_bases(self) -> list[KnowledgeBaseSummary]:
        from sqlalchemy import text

        engine = get_engine()
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT id, name, collection_name
                    FROM t_knowledge_base
                    WHERE deleted = 0
                    ORDER BY name ASC
                    """
                )
            ).mappings()
            return [
                KnowledgeBaseSummary(
                    id=str(row["id"]),
                    name=str(row["name"]),
                    collectionName=str(row["collection_name"]),
                )
                for row in rows
            ]

    def find_knowledge_base_by_name(self, name: str) -> KnowledgeBaseSummary | None:
        from sqlalchemy import text

        engine = get_engine()
        with engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT id, name, collection_name
                        FROM t_knowledge_base
                        WHERE name = :name AND deleted = 0
                        LIMIT 1
                        """
                    ),
                    {"name": name},
                )
                .mappings()
                .first()
            )
        if row is None:
            return None
        return KnowledgeBaseSummary(
                    id=str(row["id"]),
                    name=str(row["name"]),
                    collectionName=str(row["collection_name"]),
                )


    def find_knowledge_base_by_collection_name(
        self,
        collection_name: str,
    ) -> KnowledgeBaseSummary | None:
        from sqlalchemy import text

        engine = get_engine()
        with engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT id, name, collection_name
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
        return KnowledgeBaseSummary(
                    id=str(row["id"]),
                    name=str(row["name"]),
                    collectionName=str(row["collection_name"]),
                )

    def insert_knowledge_base(self, record: KnowledgeBaseRecord) -> None:
        from sqlalchemy import text

        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO t_knowledge_base (
                        id,
                        name,
                        embedding_model,
                        collection_name,
                        created_by,
                        updated_by,
                        create_time,
                        update_time,
                        deleted
                    ) VALUES (
                        :id,
                        :name,
                        :embedding_model,
                        :collection_name,
                        :created_by,
                        :created_by,
                        CURRENT_TIMESTAMP,
                        CURRENT_TIMESTAMP,
                        0
                    )
                    """
                ),
                {
                    "id": record.id,
                    "name": record.name,
                    "embedding_model": record.embedding_model,
                    "collection_name": record.collection_name,
                    "created_by": record.created_by,
                },
            )
    def insert_document(self, record: KnowledgeDocumentRecord) -> None:
        from sqlalchemy import text

        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO t_knowledge_document (
                        id,
                        kb_id,
                        doc_name,
                        enabled,
                        chunk_count,
                        file_url,
                        file_type,
                        file_size,
                        process_mode,
                        status,
                        source_type,
                        source_location,
                        schedule_enabled,
                        schedule_cron,
                        chunk_strategy,
                        chunk_config,
                        created_by,
                        updated_by,
                        create_time,
                        update_time,
                        deleted
                    ) VALUES (
                        :id,
                        :kb_id,
                        :doc_name,
                        1,
                        0,
                        :file_url,
                        :file_type,
                        :file_size,
                        'CHUNK',
                        'PENDING',
                        :source_type,
                        :source_location,
                        0,
                        NULL,
                        :chunk_strategy,
                        CAST(:chunk_config AS jsonb),
                        :created_by,
                        :created_by,
                        CURRENT_TIMESTAMP,
                        CURRENT_TIMESTAMP,
                        0
                    )
                    """
                ),
                {
                    "id": record.id,
                    "kb_id": record.kb_id,
                    "doc_name": record.doc_name,
                    "file_url": record.file_url,
                    "file_type": record.file_type,
                    "file_size": record.file_size,
                    "source_type": record.source_type,
                    "source_location": record.source_location,
                    "chunk_strategy": record.chunk_strategy,
                    "chunk_config": json.dumps(record.chunk_config, ensure_ascii=False),
                    "created_by": record.created_by,
                },
            )


    def find_document_chunk_status(self, doc_id: str) -> KnowledgeDocumentChunkStatusRecord | None:
        from sqlalchemy import text

        engine = get_engine()
        with engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT
                            d.id,
                            d.kb_id,
                            d.doc_name,
                            d.status,
                            d.chunk_count,
                            l.status AS log_status,
                            l.message_id,
                            l.error_message,
                            l.total_duration,
                            l.extract_duration,
                            l.chunk_duration,
                            l.embed_duration,
                            l.persist_duration,
                            l.create_time AS log_create_time,
                            l.end_time AS log_end_time
                        FROM t_knowledge_document d
                        LEFT JOIN LATERAL (
                            SELECT
                                status,
                                message_id,
                                error_message,
                                total_duration,
                                extract_duration,
                                chunk_duration,
                                embed_duration,
                                persist_duration,
                                create_time,
                                end_time
                            FROM t_knowledge_document_chunk_log
                            WHERE doc_id = d.id
                            ORDER BY create_time DESC
                            LIMIT 1
                        ) l ON TRUE
                        WHERE d.id = :doc_id AND d.deleted = 0
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
        return KnowledgeDocumentChunkStatusRecord(
            id=str(row["id"]),
            kb_id=str(row["kb_id"]),
            doc_name=str(row["doc_name"]),
            status=str(row["status"]),
            chunk_count=int(row["chunk_count"] or 0),
            log_status=str(row["log_status"]) if row["log_status"] is not None else None,
            message_id=str(row["message_id"]) if row["message_id"] is not None else None,
            error_message=str(row["error_message"]) if row["error_message"] is not None else None,
            total_duration=int(row["total_duration"]) if row["total_duration"] is not None else None,
            extract_duration=int(row["extract_duration"]) if row["extract_duration"] is not None else None,
            chunk_duration=int(row["chunk_duration"]) if row["chunk_duration"] is not None else None,
            embed_duration=int(row["embed_duration"]) if row["embed_duration"] is not None else None,
            persist_duration=int(row["persist_duration"]) if row["persist_duration"] is not None else None,
            log_create_time=str(row["log_create_time"]) if row["log_create_time"] is not None else None,
            log_end_time=str(row["log_end_time"]) if row["log_end_time"] is not None else None,
        )
    def mark_document_chunk_running_cas(
        self,
        *,
        doc_id: str,
        updated_by: str,
    ) -> KnowledgeDocumentChunkTarget | None:
        from sqlalchemy import text

        engine = get_engine()
        with engine.begin() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        UPDATE t_knowledge_document
                        SET status = 'RUNNING',
                            updated_by = :updated_by,
                            update_time = CURRENT_TIMESTAMP
                        WHERE id = :doc_id
                          AND deleted = 0
                          AND status = 'PENDING'
                        RETURNING
                            id,
                            kb_id,
                            doc_name,
                            file_url,
                            file_type,
                            file_size,
                            source_type,
                            source_location,
                            chunk_strategy,
                            chunk_config,
                            status
                        """
                    ),
                    {"doc_id": doc_id, "updated_by": updated_by},
                )
                .mappings()
                .first()
            )
        if row is None:
            return None
        return KnowledgeDocumentChunkTarget(
            id=str(row["id"]),
            kb_id=str(row["kb_id"]),
            doc_name=str(row["doc_name"]),
            file_url=str(row["file_url"]),
            file_type=str(row["file_type"]),
            file_size=int(row["file_size"] or 0),
            source_type=str(row["source_type"]),
            source_location=(
                str(row["source_location"]) if row["source_location"] is not None else None
            ),
            chunk_strategy=str(row["chunk_strategy"]),
            chunk_config=dict(row["chunk_config"] or {}),
            status=str(row["status"]),
        )