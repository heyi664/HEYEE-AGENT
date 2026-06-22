from __future__ import annotations

from dataclasses import dataclass, field

from agent_service.core.config import get_settings
from agent_service.core.errors import ModelUnavailableError
from agent_service.services.function_call_service import (
    FunctionCallingUnavailable,
    FunctionCallService,
)
from agent_service.tools.builtin import register_builtin_tools
from agent_service.tools.registry import ToolRegistry, tool_registry


@dataclass(frozen=True)
class LLMResult:
    reply: str
    tool_calls: list[str] = field(default_factory=list)


class LLMService:
    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self._registry = registry or tool_registry

    async def complete(self, messages: list[dict[str, str]]) -> LLMResult:
        settings = get_settings()
        if settings.agent_mock_mode:
            user_message = messages[-1]["content"] if messages else ""
            return LLMResult(
                reply=(
                    "这是 HYEEE AI 的本地测试回复。"
                    f"我已收到你的问题：{user_message}。"
                    "接入真实模型后，这里会返回模型生成的回答。"
                )
            )

        try:
            result = await FunctionCallService(self._registry).complete(
                messages,
                settings.agent_max_steps,
            )
        except FunctionCallingUnavailable as exc:
            raise ModelUnavailableError(
                "The configured model does not support the Agent tool loop: "
                f"{exc}"
            ) from exc
        return LLMResult(result.reply, result.tool_calls)


def get_llm_service() -> LLMService:
    register_builtin_tools(tool_registry)
    return LLMService(tool_registry)
