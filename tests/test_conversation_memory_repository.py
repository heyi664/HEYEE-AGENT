from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agent_service.repositories.conversation_memory_repository import (
    ConversationMemoryRepository,
)


@dataclass
class RecordedStatement:
    sql: str
    params: dict[str, Any]


class FakeResult:
    def __init__(self, rows: list[dict[str, Any]] | None = None, rowcount: int = 0) -> None:
        self._rows = rows or []
        self.rowcount = rowcount

    def mappings(self) -> FakeResult:
        return self

    def first(self) -> dict[str, Any] | None:
        return self._rows[0] if self._rows else None

    def all(self) -> list[dict[str, Any]]:
        return self._rows


class FakeConnection:
    def __init__(self, rows: list[dict[str, Any]] | None = None, rowcount: int = 0) -> None:
        self.rows = rows or []
        self.rowcount = rowcount
        self.statements: list[RecordedStatement] = []

    def execute(self, statement: Any, params: dict[str, Any] | None = None) -> FakeResult:
        self.statements.append(RecordedStatement(str(statement), params or {}))
        return FakeResult(self.rows, self.rowcount)

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


@dataclass
class FakeEngine:
    connection: FakeConnection = field(default_factory=FakeConnection)

    def begin(self) -> FakeConnection:
        return self.connection

    def connect(self) -> FakeConnection:
        return self.connection


def test_append_message_inserts_into_t_message() -> None:
    engine = FakeEngine()
    repository = ConversationMemoryRepository(engine)

    repository.append_message(
        message_id="msg_1",
        conversation_id="conv_1",
        user_id="user_1",
        role="user",
        content="hello",
    )

    statement = engine.connection.statements[0]
    assert "INSERT INTO t_message" in statement.sql
    assert statement.params["id"] == "msg_1"
    assert statement.params["role"] == "user"


def test_get_latest_summary_maps_row() -> None:
    engine = FakeEngine(
        FakeConnection(
            rows=[
                {
                    "id": "sum_1",
                    "conversation_id": "conv_1",
                    "user_id": "user_1",
                    "last_message_id": "msg_1",
                    "content": "摘要",
                }
            ]
        )
    )
    repository = ConversationMemoryRepository(engine)

    summary = repository.get_latest_summary("conv_1", "user_1")

    assert summary is not None
    assert summary.content == "摘要"
    assert "FROM t_conversation_summary" in engine.connection.statements[0].sql


def test_list_recent_turn_messages_uses_user_cutoff_window() -> None:
    created = datetime(2026, 7, 3, 12, 0, 0)
    engine = FakeEngine(
        FakeConnection(
            rows=[
                {"id": "msg_1", "role": "user", "content": "u", "create_time": created},
                {"id": "msg_2", "role": "assistant", "content": "a", "create_time": created},
            ]
        )
    )
    repository = ConversationMemoryRepository(engine)

    messages = repository.list_recent_turn_messages("conv_1", "user_1", 8)

    assert [message.content for message in messages] == ["u", "a"]
    statement = engine.connection.statements[0]
    assert "WITH recent_users" in statement.sql
    assert statement.params["keep_turns"] == 8

def test_list_conversations_queries_user_conversations() -> None:
    created = datetime(2026, 7, 3, 12, 0, 0)
    engine = FakeEngine(
        FakeConnection(
            rows=[
                {
                    "conversation_id": "conv_1",
                    "title": "hello",
                    "last_time": created,
                    "update_time": created,
                }
            ]
        )
    )
    repository = ConversationMemoryRepository(engine)

    conversations = repository.list_conversations("user_1", limit=20)

    assert conversations[0].conversation_id == "conv_1"
    statement = engine.connection.statements[0]
    assert "FROM t_conversation" in statement.sql
    assert statement.params["user_id"] == "user_1"
    assert statement.params["limit"] == 20


def test_delete_conversation_soft_deletes_conversation_messages_and_summaries() -> None:
    engine = FakeEngine()
    repository = ConversationMemoryRepository(engine)

    repository.delete_conversation("conv_1", "user_1")

    sql = "\n".join(statement.sql for statement in engine.connection.statements)
    assert "UPDATE t_conversation" in sql
    assert "UPDATE t_message" in sql
    assert "UPDATE t_conversation_summary" in sql


def test_clear_conversations_soft_deletes_all_user_memory() -> None:
    engine = FakeEngine()
    repository = ConversationMemoryRepository(engine)

    repository.clear_conversations("user_1")

    sql = "\n".join(statement.sql for statement in engine.connection.statements)
    assert "UPDATE t_conversation" in sql
    assert "UPDATE t_message" in sql
    assert "UPDATE t_conversation_summary" in sql
    assert all(
        statement.params["user_id"] == "user_1"
        for statement in engine.connection.statements
    )


