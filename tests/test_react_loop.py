from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from agent_service.core.config import get_settings
from agent_service.services.llm_service import LLMService
from agent_service.tools.registry import ToolDefinition, ToolRegistry


class StubReActLLMService(LLMService):
    def __init__(self, outputs: list[str], registry: ToolRegistry) -> None:
        super().__init__(registry)
        self._outputs: Iterator[str] = iter(outputs)
        self.received_messages: list[list[dict[str, str]]] = []

    async def _generate(self, messages: list[dict[str, str]]) -> str:
        self.received_messages.append(messages)
        return next(self._outputs)


@pytest.mark.asyncio
async def test_react_executes_tool_and_returns_final_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_MOCK_MODE", "false")
    monkeypatch.setenv("AGENT_TOOL_MODE", "react")
    get_settings.cache_clear()
    registry = ToolRegistry()

    async def lookup(arguments: dict[str, Any]) -> str:
        return f"result for {arguments['query']}"

    registry.register(ToolDefinition("lookup", "Look up a query.", lookup))
    service = StubReActLLMService(
        [
            (
                "Thought: I should search.\n"
                "Action: lookup\n"
                "Action Input: {\"query\": \"hotpot\"}\n"
                "Final Answer: placeholder"
            ),
            "Thought: I have enough information.\nFinal Answer: 推荐这家火锅店。",
        ],
        registry,
    )

    result = await service.complete([{"role": "user", "content": "推荐火锅"}])

    assert result.reply == "推荐这家火锅店。"
    assert result.tool_calls == ['lookup({"query": "hotpot"}) -> result for hotpot']
    assert "Observation: result for hotpot" in service.received_messages[1][-1]["content"]


@pytest.mark.asyncio
async def test_react_accepts_direct_final_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_MOCK_MODE", "false")
    monkeypatch.setenv("AGENT_TOOL_MODE", "react")
    get_settings.cache_clear()
    service = StubReActLLMService(
        ["Thought: No tool is needed.\nFinal Answer: 你好。"],
        ToolRegistry(),
    )

    result = await service.complete([{"role": "user", "content": "你好"}])

    assert result.reply == "你好。"
    assert result.tool_calls == []
