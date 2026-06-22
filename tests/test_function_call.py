from __future__ import annotations

import logging
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
async def test_function_call_executes_tool_and_returns_answer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
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
    assert "LLM function call step=1 tool=java_search_shop" in caplog.text
    assert '"keyword": "hotpot"' in caplog.text


@pytest.mark.asyncio
async def test_function_call_can_select_different_tools_across_iterations() -> None:
    registry = ToolRegistry()

    async def search_shops(arguments: dict[str, Any]) -> str:
        return '{"shops":[{"id":42,"name":"Hotpot House"}]}'

    async def get_shop_detail(arguments: dict[str, Any]) -> str:
        return '{"id":42,"name":"Hotpot House","score":4.8}'

    registry.register(
        ToolDefinition(
            name="search_shops",
            description="Search platform shops using the user's requirements.",
            handler=search_shops,
            input_schema={
                "type": "object",
                "properties": {"keyword": {"type": "string"}},
                "required": ["keyword"],
            },
        )
    )
    registry.register(
        ToolDefinition(
            name="get_shop_detail",
            description="Get details for a platform shop by ID.",
            handler=get_shop_detail,
            input_schema={
                "type": "object",
                "properties": {"shop_id": {"type": "integer"}},
                "required": ["shop_id"],
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
                        call_id="call_search",
                        name="search_shops",
                        arguments={"keyword": "hotpot"},
                    )
                ],
                assistant_message={"role": "assistant", "tool_calls": []},
            ),
            FunctionCallTurn(
                content=None,
                tool_calls=[
                    FunctionCallRequest(
                        call_id="call_detail",
                        name="get_shop_detail",
                        arguments={"shop_id": 42},
                    )
                ],
                assistant_message={"role": "assistant", "tool_calls": []},
            ),
            FunctionCallTurn(
                content="Hotpot House has a score of 4.8.",
                tool_calls=[],
                assistant_message={
                    "role": "assistant",
                    "content": "Hotpot House has a score of 4.8.",
                },
            ),
        ],
    )

    result = await service.complete(
        [{"role": "user", "content": "Find a hotpot shop and check its score."}],
        max_steps=4,
    )

    assert result.reply == "Hotpot House has a score of 4.8."
    assert [summary.split("(", 1)[0] for summary in result.tool_calls] == [
        "search_shops",
        "get_shop_detail",
    ]
    assert len(service.received_messages) == 3
    assert service.received_messages[1][-1]["role"] == "tool"
    assert service.received_messages[2][-1]["role"] == "tool"
    assert all(
        {schema["function"]["name"] for schema in schemas}
        == {"search_shops", "get_shop_detail"}
        for schemas in service.received_schemas
    )
