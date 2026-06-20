from __future__ import annotations

from typing import Any

from agent_service.mcp.contracts import McpClientProtocol
from agent_service.tools.registry import ToolDefinition, ToolHandler, ToolRegistry


def _build_handler(client: McpClientProtocol, remote_name: str) -> ToolHandler:
    async def handler(arguments: dict[str, Any]) -> str:
        return await client.call_tool(remote_name, arguments)

    return handler


async def register_mcp_tools(
    client: McpClientProtocol,
    registry: ToolRegistry,
    name_prefix: str = "java_",
) -> list[str]:
    registered: list[str] = []
    for remote_tool in await client.list_tools():
        local_name = f"{name_prefix}{remote_tool.name}"
        registry.register(
            ToolDefinition(
                name=local_name,
                description=f"Java MCP tool: {remote_tool.description}",
                handler=_build_handler(client, remote_tool.name),
                input_schema=remote_tool.input_schema,
            )
        )
        registered.append(local_name)
    return registered