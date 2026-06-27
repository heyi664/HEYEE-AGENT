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


class KnowledgeRepository:
    def list_knowledge_bases(self) -> list[KnowledgeBaseSummary]:
        from sqlalchemy import text

        engine = get_engine()
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT id, name
                    FROM t_knowledge_base
                    WHERE deleted = 0
                    ORDER BY name ASC
                    """
                )
            ).mappings()
            return [
                KnowledgeBaseSummary(id=str(row["id"]), name=str(row["name"]))
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
                        SELECT id, name
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
        return KnowledgeBaseSummary(id=str(row["id"]), name=str(row["name"]))


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
                        SELECT id, name
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
        return KnowledgeBaseSummary(id=str(row["id"]), name=str(row["name"]))

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
