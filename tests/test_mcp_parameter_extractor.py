from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from agent_service.mcp.parameter_extractor import McpParameterExtractor, build_tool_definition
from agent_service.tools.registry import ToolDefinition


@dataclass(frozen=True)
class Reply:
    reply: str


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.messages: list[dict[str, str]] = []

    async def complete(self, messages: list[dict[str, str]], use_tools: bool = True) -> Reply:
        assert use_tools is False
        self.messages = messages
        return Reply(self.reply)


async def _handler(arguments: dict[str, Any]) -> str:
    return str(arguments)


def _tool() -> ToolDefinition:
    return ToolDefinition(
        name="sales_query",
        description="Query sales data.",
        handler=_handler,
        input_schema={
            "type": "object",
            "properties": {
                "region": {"type": "string", "enum": ["华东", "华南"]},
                "period": {"type": "string", "default": "本月"},
                "queryType": {"type": "string", "default": "summary"},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["region"],
            "additionalProperties": False,
        },
    )


@pytest.mark.asyncio
async def test_extractor_allow_lists_converts_and_fills_defaults() -> None:
    llm = FakeLLM(
        '```json\n{"region":"华东","limit":5.0,"ignored":"value"}\n```'
    )
    extractor = McpParameterExtractor(llm)

    result = await extractor.extract("华东本月的销售额", _tool())

    assert result.ready is True
    assert result.arguments == {
        "region": "华东",
        "period": "本月",
        "queryType": "summary",
        "limit": 5,
    }
    assert result.used_defaults is True
    assert "untrusted user question" in llm.messages[1]["content"].lower()
    assert "ignored" not in result.arguments


@pytest.mark.asyncio
async def test_extractor_falls_back_to_defaults_and_requests_missing_required_values() -> None:
    extractor = McpParameterExtractor(FakeLLM("not JSON"))

    result = await extractor.extract("查销售", _tool())

    assert result.fallback_used is True
    assert result.arguments == {"period": "本月", "queryType": "summary", "limit": 10}
    assert result.missing_required == ["region"]
    assert result.ready is False


@pytest.mark.asyncio
async def test_extractor_rejects_invalid_enum_and_non_integer_number() -> None:
    extractor = McpParameterExtractor(FakeLLM('{"region":"华北","limit":1.5}'))

    result = await extractor.extract("查销售", _tool())

    assert result.invalid_fields == ["region", "limit"]
    assert result.arguments == {"period": "本月", "queryType": "summary", "limit": 10}
    assert result.missing_required == ["region"]


def test_tool_definition_includes_schema_constraints() -> None:
    definition = build_tool_definition(_tool())

    assert "Tool ID: sales_query" in definition
    assert "region (type=string, required)" in definition
    assert "default=10" in definition
    assert 'enum=["华东", "华南"]' in definition
