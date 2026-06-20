from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from agent_service.services.function_call_service import (
    FunctionCallRequest,
    FunctionCallService,
    FunctionCallTurn,
)
from agent_service.tools.registry import ToolDefinition, ToolRegistry


class StubFunctionCallService(FunctionCallService):
    def __init__(
        self,
        registry: ToolRegistry,
        turns: list[FunctionCallTurn],
    ) -> None:
        super().__init__(registry)
        self._turns: Iterator[FunctionCallTurn] = iter(turns)
        self.received_schemas: list[list[dict[str, Any]]] = []
        self.received_messages: list[list[dict[str, Any]]] = []

    async def _generate_turn(
        self,
        messages: list[dict[str, Any]],
        schemas: list[dict[str, Any]],
    ) -> FunctionCallTurn:
        self.received_messages.append([dict(message) for message in messages])
        self.received_schemas.append(schemas)
        return next(self._turns)


@pytest.mark.asyncio
async def test_function_call_executes_tool_and_returns_answer() -> None:
    registry = ToolRegistry()

    async def search_shop(arguments: dict[str, Any]) -> str:
        return f"shop:{arguments['keyword']}"

    registry.register(
        ToolDefinition(
            name="java_search_shop",
            description="Search shops from Java.",
            handler=search_shop,
            input_schema={
                "type": "object",
                "properties": {"keyword": {"type": "string"}},
                "required": ["keyword"],
            },
        )
    )
    service = StubFunctionCallService(
        registry,
        [
            FunctionCallTurn(
                content=None,
                tool_calls=[
                    FunctionCallRequest(
                        call_id="call_1",
                        name="java_search_shop",
                        arguments={"keyword": "hotpot"},
                    )
                ],
                assistant_message={"role": "assistant", "tool_calls": []},
            ),
            FunctionCallTurn(
                content="推荐查询到的火锅店。",
                tool_calls=[],
                assistant_message={
                    "role": "assistant",
                    "content": "推荐查询到的火锅店。",
                },
            ),
        ],
    )

    result = await service.complete(
        [{"role": "user", "content": "推荐火锅"}],
        max_steps=3,
    )

    assert result.reply == "推荐查询到的火锅店。"
    assert result.tool_calls == [
        'java_search_shop({"keyword": "hotpot"}) -> shop:hotpot'
    ]
    function = service.received_schemas[0][0]["function"]
    assert function["name"] == "java_search_shop"
    assert function["parameters"]["required"] == ["keyword"]
    assert service.received_messages[1][-1]["role"] == "tool"