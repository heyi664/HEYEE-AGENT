from __future__ import annotations

import pytest

from agent_service.rag.llm_response_cleaner import parse_json_object


def test_parse_json_object_accepts_fenced_json() -> None:
    assert parse_json_object(
        '```json\n{"rewritten_question": "A", "sub_questions": ["A"]}\n```'
    ) == {
        "rewritten_question": "A",
        "sub_questions": ["A"],
    }


def test_parse_json_object_accepts_plain_json_with_surrounding_text() -> None:
    assert parse_json_object('结果：{"rewritten_question": "A"}') == {
        "rewritten_question": "A",
    }


def test_parse_json_object_rejects_non_object() -> None:
    with pytest.raises(ValueError):
        parse_json_object('["A"]')
