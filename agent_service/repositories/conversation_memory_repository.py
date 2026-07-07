from __future__ import annotations

from typing import Any, cast

from agent_service.db.session import get_engine
from agent_service.memory.models import ConversationRecord, MemoryMessage, MemoryRole, MemorySummary


class ConversationMemoryRepository:
    def __init__(self, engine: Any | None = None) -> None:
        self._engine = engine or get_engine()

    def create_or_update_conversation(
        self,
        *,
        row_id: str,
        conversation_id: str,
        user_id: str,
        title: str,
    ) -> None:
        from sqlalchemy import text

        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO t_conversation (id, conversation_id, user_id, title, last_time)
                    VALUES (:id, :conversation_id, :user_id, :title, CURRENT_TIMESTAMP)
                    ON CONFLICT (conversation_id, user_id)
                    DO UPDATE SET last_time = CURRENT_TIMESTAMP, update_time = CURRENT_TIMESTAMP
                    """
                ),
                {
                    "id": row_id,
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "title": title[:128],
                },
            )

    def append_message(
        self,
        *,
        message_id: str,
        conversation_id: str,
        user_id: str,
        role: str,
        content: str,
    ) -> None:
        from sqlalchemy import text

        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO t_message (id, conversation_id, user_id, role, content)
                    VALUES (:id, :conversation_id, :user_id, :role, :content)
                    """
                ),
                {
                    "id": message_id,
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "role": role,
                    "content": content,
                },
            )

    def get_latest_summary(self, conversation_id: str, user_id: str) -> MemorySummary | None:
        from sqlalchemy import text

        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT id, conversation_id, user_id, last_message_id, content
                        FROM t_conversation_summary
                        WHERE conversation_id = :conversation_id
                          AND user_id = :user_id
                          AND deleted = 0
                        ORDER BY update_time DESC
                        LIMIT 1
                        """
                    ),
                    {"conversation_id": conversation_id, "user_id": user_id},
                )
                .mappings()
                .first()
            )
        if row is None:
            return None
        return MemorySummary(
            id=str(row["id"]),
            conversation_id=str(row["conversation_id"]),
            user_id=str(row["user_id"]),
            last_message_id=str(row["last_message_id"]),
            content=str(row["content"]),
        )

    def list_recent_turn_messages(
        self,
        conversation_id: str,
        user_id: str,
        keep_turns: int,
    ) -> list[MemoryMessage]:
        from sqlalchemy import text

        with self._engine.connect() as conn:
            rows = (
                conn.execute(
                    text(
                        """
                        WITH recent_users AS (
                            SELECT create_time
                            FROM t_message
                            WHERE conversation_id = :conversation_id
                              AND user_id = :user_id
                              AND role = 'user'
                              AND deleted = 0
                            ORDER BY create_time DESC
                            LIMIT :keep_turns
                        ), cutoff AS (
                            SELECT MIN(create_time) AS cutoff_time FROM recent_users
                        )
                        SELECT id, role, content, create_time
                        FROM t_message
                        WHERE conversation_id = :conversation_id
                          AND user_id = :user_id
                          AND role IN ('user', 'assistant')
                          AND deleted = 0
                          AND create_time >= COALESCE((SELECT cutoff_time FROM cutoff), create_time)
                        ORDER BY create_time ASC
                        """
                    ),
                    {
                        "conversation_id": conversation_id,
                        "user_id": user_id,
                        "keep_turns": keep_turns,
                    },
                )
                .mappings()
                .all()
            )
        return [self._row_to_message(row) for row in rows]

    def list_messages_after(
        self,
        conversation_id: str,
        user_id: str,
        after_message_id: str | None,
    ) -> list[MemoryMessage]:
        from sqlalchemy import text

        after_clause = ""
        if after_message_id:
            after_clause = """
              AND create_time > (
                  SELECT create_time FROM t_message
                  WHERE id = :after_message_id
                    AND conversation_id = :conversation_id
                    AND user_id = :user_id
                    AND deleted = 0
                  LIMIT 1
              )
            """

        with self._engine.connect() as conn:
            rows = (
                conn.execute(
                    text(
                        f"""
                        SELECT id, role, content, create_time
                        FROM t_message
                        WHERE conversation_id = :conversation_id
                          AND user_id = :user_id
                          AND role IN ('user', 'assistant')
                          AND deleted = 0
                          {after_clause}
                        ORDER BY create_time ASC
                        """
                    ),
                    {
                        "conversation_id": conversation_id,
                        "user_id": user_id,
                        "after_message_id": after_message_id,
                    },
                )
                .mappings()
                .all()
            )
        return [self._row_to_message(row) for row in rows]

    def list_conversations(self, user_id: str, limit: int = 50) -> list[ConversationRecord]:
        from sqlalchemy import text

        with self._engine.connect() as conn:
            rows = (
                conn.execute(
                    text(
                        """
                        SELECT conversation_id, title, last_time, update_time
                        FROM t_conversation
                        WHERE user_id = :user_id
                          AND deleted = 0
                        ORDER BY last_time DESC, update_time DESC
                        LIMIT :limit
                        """
                    ),
                    {"user_id": user_id, "limit": limit},
                )
                .mappings()
                .all()
            )
        return [
            ConversationRecord(
                conversation_id=str(row["conversation_id"]),
                title=str(row["title"]) if row.get("title") is not None else None,
                last_time=row.get("last_time"),
                update_time=row.get("update_time"),
            )
            for row in rows
        ]

    def list_conversation_messages(self, conversation_id: str, user_id: str) -> list[MemoryMessage]:
        from sqlalchemy import text

        with self._engine.connect() as conn:
            rows = (
                conn.execute(
                    text(
                        """
                        SELECT id, role, content, create_time
                        FROM t_message
                        WHERE conversation_id = :conversation_id
                          AND user_id = :user_id
                          AND role IN ('user', 'assistant')
                          AND deleted = 0
                        ORDER BY create_time ASC
                        """
                    ),
                    {"conversation_id": conversation_id, "user_id": user_id},
                )
                .mappings()
                .all()
            )
        return [self._row_to_message(row) for row in rows]

    def delete_conversation(self, conversation_id: str, user_id: str) -> None:
        from sqlalchemy import text

        params = {"conversation_id": conversation_id, "user_id": user_id}
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE t_conversation
                    SET deleted = 1, update_time = CURRENT_TIMESTAMP
                    WHERE conversation_id = :conversation_id AND user_id = :user_id
                    """
                ),
                params,
            )
            conn.execute(
                text(
                    """
                    UPDATE t_message
                    SET deleted = 1, update_time = CURRENT_TIMESTAMP
                    WHERE conversation_id = :conversation_id AND user_id = :user_id
                    """
                ),
                params,
            )
            conn.execute(
                text(
                    """
                    UPDATE t_conversation_summary
                    SET deleted = 1, update_time = CURRENT_TIMESTAMP
                    WHERE conversation_id = :conversation_id AND user_id = :user_id
                    """
                ),
                params,
            )

    def clear_conversations(self, user_id: str) -> None:
        from sqlalchemy import text

        params = {"user_id": user_id}
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE t_conversation
                    SET deleted = 1, update_time = CURRENT_TIMESTAMP
                    WHERE user_id = :user_id
                    """
                ),
                params,
            )
            conn.execute(
                text(
                    """
                    UPDATE t_message
                    SET deleted = 1, update_time = CURRENT_TIMESTAMP
                    WHERE user_id = :user_id
                    """
                ),
                params,
            )
            conn.execute(
                text(
                    """
                    UPDATE t_conversation_summary
                    SET deleted = 1, update_time = CURRENT_TIMESTAMP
                    WHERE user_id = :user_id
                    """
                ),
                params,
            )

    def upsert_summary(
        self,
        *,
        summary_id: str,
        conversation_id: str,
        user_id: str,
        last_message_id: str,
        content: str,
    ) -> None:
        from sqlalchemy import text

        with self._engine.begin() as conn:
            result = conn.execute(
                text(
                    """
                    UPDATE t_conversation_summary
                    SET last_message_id = :last_message_id,
                        content = :content,
                        update_time = CURRENT_TIMESTAMP
                    WHERE conversation_id = :conversation_id
                      AND user_id = :user_id
                      AND deleted = 0
                    """
                ),
                {
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "last_message_id": last_message_id,
                    "content": content,
                },
            )
            if result.rowcount:
                return
            conn.execute(
                text(
                    """
                    INSERT INTO t_conversation_summary (
                        id, conversation_id, user_id, last_message_id, content
                    ) VALUES (
                        :id, :conversation_id, :user_id, :last_message_id, :content
                    )
                    """
                ),
                {
                    "id": summary_id,
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "last_message_id": last_message_id,
                    "content": content,
                },
            )

    def _row_to_message(self, row: Any) -> MemoryMessage:
        return MemoryMessage(
            id=str(row["id"]),
            role=cast(MemoryRole, str(row["role"])),
            content=str(row["content"]),
            create_time=row["create_time"],
        )
