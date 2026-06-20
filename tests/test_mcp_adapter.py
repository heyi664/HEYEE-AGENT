from __future__ import annotations

from typing import Any

import pytest

from agent_service.mcp.adapter import register_mcp_tools
from agent_service.mcp.contracts import McpToolDefinition
from agent_service.tools.registry import ToolRegistry


class FakeJavaMcpClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> list[McpToolDefinition]:
        return [
            McpToolDefinition(
                name="search_shops",
                description="Search shops by keyword.",
                input_schema={
                    "type": "object",
                    "properties": {"keyword": {"type": "string"}},
                    "required": ["keyword"],
                },
            )
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        return "java-result"


@pytest.mark.asyncio
async def test_mcp_tools_are_adapted_to_function_registry() -> None:
    client = FakeJavaMcpClient()
    registry = ToolRegistry()

    registered = await register_mcp_tools(client, registry)

    assert registered == ["search_shops"]
    tool = registry.get("search_shops")
    assert tool is not None
    assert tool.to_function_schema()["function"]["parameters"]["required"] == ["keyword"]

    observation = await tool.handler({"keyword": "hotpot"})

    assert observation == "java-result"
    assert client.calls == [("search_shops", {"keyword": "hotpot"})]