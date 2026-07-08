from __future__ import annotations

import json
from typing import Any

from agent_service.db.session import get_engine
from agent_service.rag.intent_models import IntentNodeRecord


class IntentNodeRepository:
    def __init__(self, engine: Any | None = None) -> None:
        self._engine = engine

    def list_enabled_nodes(self) -> list[IntentNodeRecord]:
        from sqlalchemy import text

        engine = self._engine or get_engine()
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT
                        id,
                        kb_id,
                        intent_code,
                        name,
                        level,
                        parent_code,
                        description,
                        examples,
                        collection_name,
                        top_k,
                        mcp_tool_id,
                        kind,
                        prompt_snippet,
                        prompt_template,
                        param_prompt_template,
                        sort_order
                    FROM t_intent_node
                    WHERE deleted = 0 AND enabled = 1
                    ORDER BY sort_order ASC, create_time ASC
                    """
                )
            ).mappings()
            return [_map_row(row) for row in rows]


def _map_row(row: Any) -> IntentNodeRecord:
    return IntentNodeRecord(
        id=str(row["id"]),
        kb_id=_optional_str(row["kb_id"]),
        intent_code=str(row["intent_code"]),
        name=str(row["name"]),
        level=int(row["level"]),
        parent_code=_optional_str(row["parent_code"]),
        description=_optional_str(row["description"]),
        examples=_parse_examples(row["examples"]),
        collection_name=_optional_str(row["collection_name"]),
        top_k=int(row["top_k"]) if row["top_k"] is not None else None,
        mcp_tool_id=_optional_str(row["mcp_tool_id"]),
        kind=int(row["kind"]),
        prompt_snippet=_optional_str(row["prompt_snippet"]),
        prompt_template=_optional_str(row["prompt_template"]),
        param_prompt_template=_optional_str(row["param_prompt_template"]),
        sort_order=int(row["sort_order"] or 0),
    )


def _optional_str(value: Any) -> str | None:
    return str(value) if value is not None else None


def _parse_examples(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if not isinstance(value, str):
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item).strip()]
