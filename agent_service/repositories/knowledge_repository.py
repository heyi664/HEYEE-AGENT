from __future__ import annotations

import json
import uuid
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



@dataclass(frozen=True)
class KnowledgeDocumentListItem:
    id: str
    kb_id: str
    knowledge_base_name: str | None
    doc_name: str
    enabled: bool
    status: str
    chunk_count: int
    file_url: str | None
    file_type: str | None
    file_size: int
    source_type: str | None
    source_location: str | None
    chunk_strategy: str
    chunk_config: dict[str, Any]
    create_time: str | None
    update_time: str | None


@dataclass(frozen=True)
class KnowledgeDocumentDetail:
    id: str
    kb_id: str
    doc_name: str
    file_url: str
    status: str
    enabled: bool
    chunk_strategy: str
    chunk_config: dict[str, Any]
    collection_name: str | None


@dataclass(frozen=True)
class KnowledgeVectorChunkSource:
    chunk_id: str
    kb_id: str
    doc_id: str
    chunk_index: int
    content: str
    content_hash: str
    char_count: int
    token_count: int

class KnowledgeRepository:

    def list_documents(self) -> list[KnowledgeDocumentListItem]:
        from sqlalchemy import text

        engine = get_engine()
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT
                        d.id,
                        d.kb_id,
                        kb.name AS knowledge_base_name,
                        d.doc_name,
                        d.enabled,
                        d.status,
                        d.chunk_count,
                        d.file_url,
                        d.file_type,
                        d.file_size,
                        d.source_type,
                        d.source_location,
                        d.chunk_strategy,
                        d.chunk_config,
                        d.create_time,
                        d.update_time
                    FROM t_knowledge_document d
                    LEFT JOIN t_knowledge_base kb ON kb.id = d.kb_id AND kb.deleted = 0
                    WHERE d.deleted = 0
                    ORDER BY d.update_time DESC, d.create_time DESC
                    """
                )
            ).mappings()
            return [
                KnowledgeDocumentListItem(
                    id=str(row["id"]),
                    kb_id=str(row["kb_id"]),
                    knowledge_base_name=(
                        str(row["knowledge_base_name"])
                        if row["knowledge_base_name"] is not None
                        else None
                    ),
                    doc_name=str(row["doc_name"]),
                    enabled=bool(row["enabled"]),
                    status=str(row["status"]),
                    chunk_count=int(row["chunk_count"] or 0),
                    file_url=str(row["file_url"]) if row["file_url"] is not None else None,
                    file_type=str(row["file_type"]) if row["file_type"] is not None else None,
                    file_size=int(row["file_size"] or 0),
                    source_type=str(row["source_type"]) if row["source_type"] is not None else None,
                    source_location=(
                        str(row["source_location"]) if row["source_location"] is not None else None
                    ),
                    chunk_strategy=str(row["chunk_strategy"]),
                    chunk_config=_json_dict(row["chunk_config"]),
                    create_time=str(row["create_time"]) if row["create_time"] is not None else None,
                    update_time=str(row["update_time"]) if row["update_time"] is not None else None,
                )
                for row in rows
            ]
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
            total_duration=(
                int(row["total_duration"]) if row["total_duration"] is not None else None
            ),
            extract_duration=(
                int(row["extract_duration"]) if row["extract_duration"] is not None else None
            ),
            chunk_duration=(
                int(row["chunk_duration"]) if row["chunk_duration"] is not None else None
            ),
            embed_duration=(
                int(row["embed_duration"]) if row["embed_duration"] is not None else None
            ),
            persist_duration=(
                int(row["persist_duration"]) if row["persist_duration"] is not None else None
            ),
            log_create_time=(
                str(row["log_create_time"]) if row["log_create_time"] is not None else None
            ),
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
    def find_document_detail(self, doc_id: str) -> KnowledgeDocumentDetail | None:
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
                            d.file_url,
                            d.status,
                            d.enabled,
                            d.chunk_strategy,
                            d.chunk_config,
                            kb.collection_name
                        FROM t_knowledge_document d
                        LEFT JOIN t_knowledge_base kb ON kb.id = d.kb_id AND kb.deleted = 0
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
        return KnowledgeDocumentDetail(
            id=str(row["id"]),
            kb_id=str(row["kb_id"]),
            doc_name=str(row["doc_name"]),
            file_url=str(row["file_url"]),
            status=str(row["status"]),
            enabled=bool(row["enabled"]),
            chunk_strategy=str(row["chunk_strategy"]),
            chunk_config=_json_dict(row["chunk_config"]),
            collection_name=(
                str(row["collection_name"]) if row["collection_name"] is not None else None
            ),
        )

    def update_document_config(
        self,
        *,
        doc_id: str,
        doc_name: str,
        chunk_strategy: str,
        chunk_config: dict[str, Any],
        updated_by: str,
    ) -> None:
        from sqlalchemy import text

        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE t_knowledge_document
                    SET doc_name = :doc_name,
                        chunk_strategy = :chunk_strategy,
                        chunk_config = CAST(:chunk_config AS jsonb),
                        updated_by = :updated_by,
                        update_time = CURRENT_TIMESTAMP
                    WHERE id = :doc_id AND deleted = 0
                    """
                ),
                {
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "chunk_strategy": chunk_strategy,
                    "chunk_config": json.dumps(chunk_config, ensure_ascii=False),
                    "updated_by": updated_by,
                },
            )

    def delete_document_atomic(self, *, doc_id: str, updated_by: str) -> None:
        from sqlalchemy import text

        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM t_knowledge_chunk WHERE doc_id = :doc_id"),
                {"doc_id": doc_id},
            )
            conn.execute(
                text("DELETE FROM t_knowledge_document_chunk_log WHERE doc_id = :doc_id"),
                {"doc_id": doc_id},
            )
            conn.execute(
                text(
                    """
                    UPDATE t_knowledge_document
                    SET deleted = 1,
                        updated_by = :updated_by,
                        update_time = CURRENT_TIMESTAMP
                    WHERE id = :doc_id AND deleted = 0
                    """
                ),
                {"doc_id": doc_id, "updated_by": updated_by},
            )
            conn.execute(
                text("DELETE FROM t_knowledge_vector WHERE doc_id = :doc_id"),
                {"doc_id": doc_id},
            )

    def list_vector_chunk_sources(self, doc_id: str) -> list[KnowledgeVectorChunkSource]:
        from sqlalchemy import text

        engine = get_engine()
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT
                        id,
                        kb_id,
                        doc_id,
                        chunk_index,
                        content,
                        content_hash,
                        char_count,
                        token_count
                    FROM t_knowledge_chunk
                    WHERE doc_id = :doc_id AND deleted = 0
                    ORDER BY chunk_index ASC
                    """
                ),
                {"doc_id": doc_id},
            ).mappings()
            return [
                KnowledgeVectorChunkSource(
                    chunk_id=str(row["id"]),
                    kb_id=str(row["kb_id"]),
                    doc_id=str(row["doc_id"]),
                    chunk_index=int(row["chunk_index"]),
                    content=str(row["content"]),
                    content_hash=str(row["content_hash"]),
                    char_count=int(row["char_count"] or 0),
                    token_count=int(row["token_count"] or 0),
                )
                for row in rows
            ]

    def set_document_enabled_atomic(
        self,
        *,
        doc_id: str,
        enabled: bool,
        vectors: list[Any],
        updated_by: str,
    ) -> None:
        from sqlalchemy import text

        enabled_value = 1 if enabled else 0
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE t_knowledge_document
                    SET enabled = :enabled,
                        updated_by = :updated_by,
                        update_time = CURRENT_TIMESTAMP
                    WHERE id = :doc_id AND deleted = 0
                    """
                ),
                {"doc_id": doc_id, "enabled": enabled_value, "updated_by": updated_by},
            )
            conn.execute(
                text(
                    """
                    UPDATE t_knowledge_chunk
                    SET enabled = :enabled,
                        updated_by = :updated_by,
                        update_time = CURRENT_TIMESTAMP
                    WHERE doc_id = :doc_id AND deleted = 0
                    """
                ),
                {"doc_id": doc_id, "enabled": enabled_value, "updated_by": updated_by},
            )
            conn.execute(
                text("DELETE FROM t_knowledge_vector WHERE doc_id = :doc_id"),
                {"doc_id": doc_id},
            )
            if enabled and vectors:
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
                            "kb_id": vector.kb_id,
                            "doc_id": vector.doc_id,
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

def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        loaded = json.loads(value)
        if isinstance(loaded, dict):
            return loaded
    return {}


def _pgvector_literal(vector: list[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in vector) + "]"
