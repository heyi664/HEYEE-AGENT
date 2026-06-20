from __future__ import annotations

from datetime import datetime
from typing import Any

from agent_service.tools.registry import ToolDefinition, ToolRegistry


async def get_current_time(arguments: dict[str, Any]) -> str:
    del arguments
    return datetime.now().astimezone().isoformat(timespec="seconds")


def register_builtin_tools(registry: ToolRegistry) -> None:
    if registry.get("get_current_time") is None:
        registry.register(
            ToolDefinition(
                name="get_current_time",
                description=(
                    "Return the current server time. Action Input must be an empty JSON object."
                ),
                handler=get_current_time,
            )
        )
