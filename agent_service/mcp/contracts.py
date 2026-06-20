from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class McpToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]


class McpClientProtocol(Protocol):
    async def list_tools(self) -> list[McpToolDefinition]:
        """Return tools exposed by the Java MCP server."""

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call one Java MCP tool and return a text observation."""