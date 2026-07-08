from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_service.repositories.intent_node_repository import IntentNodeRepository


@dataclass
class RecordedStatement:
    sql: str
    params: dict[str, Any]


class FakeResult:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def mappings(self) -> FakeResult:
        return self

    def __iter__(self):
        return iter(self._rows)


class FakeConnection:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.statements: list[RecordedStatement] = []

    def execute(self, statement: Any, params: dict[str, Any] | None = None) -> FakeResult:
        self.statements.append(RecordedStatement(str(statement), params or {}))
        return FakeResult(self.rows)

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


@dataclass
class FakeEngine:
    connection: FakeConnection = field(default_factory=lambda: FakeConnection([]))

    def connect(self) -> FakeConnection:
        return self.connection


def test_intent_node_repository_maps_enabled_nodes() -> None:
    engine = FakeEngine(
        FakeConnection(
            [
                {
                    "id": "1",
                    "kb_id": "kb1",
                    "intent_code": "return-policy",
                    "name": "退换政策",
                    "level": 2,
                    "parent_code": "3c",
                    "description": "退换货",
                    "examples": '["退货政策是什么？"]',
                    "collection_name": "kb_3c_return",
                    "top_k": 4,
                    "mcp_tool_id": None,
                    "kind": 0,
                    "prompt_snippet": None,
                    "prompt_template": None,
                    "param_prompt_template": None,
                    "sort_order": 10,
                }
            ]
        )
    )
    repository = IntentNodeRepository(engine)

    records = repository.list_enabled_nodes()

    assert records[0].intent_code == "return-policy"
    assert records[0].examples == ["退货政策是什么？"]
    assert records[0].collection_name == "kb_3c_return"
    statement = engine.connection.statements[0].sql
    assert "FROM t_intent_node" in statement
    assert "deleted = 0" in statement
    assert "enabled = 1" in statement


def test_intent_node_repository_tolerates_bad_examples_json() -> None:
    engine = FakeEngine(
        FakeConnection(
            [
                {
                    "id": "1",
                    "kb_id": None,
                    "intent_code": "broken",
                    "name": "坏样例",
                    "level": 2,
                    "parent_code": None,
                    "description": None,
                    "examples": "{not json}",
                    "collection_name": None,
                    "top_k": None,
                    "mcp_tool_id": None,
                    "kind": 0,
                    "prompt_snippet": None,
                    "prompt_template": None,
                    "param_prompt_template": None,
                    "sort_order": 0,
                }
            ]
        )
    )
    repository = IntentNodeRepository(engine)

    assert repository.list_enabled_nodes()[0].examples == []
