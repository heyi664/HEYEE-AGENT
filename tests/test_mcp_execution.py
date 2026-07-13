from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from agent_service.mcp.execution import McpExecutionService
from agent_service.mcp.parameter_extractor import McpParameterExtractor
from agent_service.rag.intent_models import (
    IntentKind,
    IntentLevel,
    IntentNode,
    NodeScore,
    SubQuestionIntent,
)
from agent_service.tools.registry import ToolDefinition, ToolRegistry


@dataclass(frozen=True)
class Reply:
    reply: str


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    async def complete(self, messages, use_tools=True) -> Reply:
        del messages
        assert use_tools is False
        return Reply(self.reply)


def _sub_intent() -> list[SubQuestionIntent]:
    node = IntentNode(
        id="order-progress",
        name="order progress",
        level=IntentLevel.TOPIC,
        kind=IntentKind.MCP,
        mcp_tool_id="order_query",
    )
    return [SubQuestionIntent("订单 2024112801 到哪里了？", [NodeScore(node, 0.9)])]


@pytest.mark.asyncio
async def test_execution_extracts_parameters_calls_registered_tool_and_formats_context() -> None:
    calls: list[dict[str, Any]] = []

    async def handler(arguments: dict[str, Any]) -> str:
        calls.append(arguments)
        return '{"status":"运输中"}'

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="order_query",
            description="Query one order.",
            handler=handler,
            input_schema={
                "type": "object",
                "properties": {"orderNo": {"type": "string"}},
                "required": ["orderNo"],
            },
        )
    )
    service = McpExecutionService(
        registry=registry,
        parameter_extractor=McpParameterExtractor(FakeLLM('{"orderNo":"2024112801"}')),
    )

    result = await service.execute(_sub_intent())

    assert calls == [{"orderNo": "2024112801"}]
    assert result.tool_calls == ["order_query"]
    assert "Real-time result" in result.context
    assert "运输中" in result.context


@pytest.mark.asyncio
async def test_execution_does_not_call_tool_when_a_required_parameter_is_missing() -> None:
    called = False

    async def handler(arguments: dict[str, Any]) -> str:
        nonlocal called
        del arguments
        called = True
        return "unexpected"

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="order_query",
            description="Query one order.",
            handler=handler,
            input_schema={
                "type": "object",
                "properties": {"orderNo": {"type": "string"}},
                "required": ["orderNo"],
            },
        )
    )
    service = McpExecutionService(
        registry=registry,
        parameter_extractor=McpParameterExtractor(FakeLLM("{}")),
    )

    result = await service.execute(_sub_intent())

    assert called is False
    assert result.tool_calls == []
    assert "required parameters are missing: orderNo" in result.context
