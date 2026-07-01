from __future__ import annotations

import json
import logging
from typing import Any

from agent_service.core.config import get_settings
from agent_service.core.errors import ModelUnavailableError
from agent_service.infra_ai import get_model_routing_executor, get_model_selector
from agent_service.infra_ai.clients import (
    ChatModelClient,
    ChatModelClientRegistry,
    ToolCallingUnavailable,
)
from agent_service.infra_ai.clients import (
    ChatTurn as FunctionCallTurn,
)
from agent_service.infra_ai.clients import (
    ToolCallRequest as FunctionCallRequest,
)
from agent_service.infra_ai.models import ModelCapability, ModelTarget
from agent_service.tools.registry import ToolRegistry

FunctionCallingUnavailable = ToolCallingUnavailable

logger = logging.getLogger(__name__)


class FunctionCallResult:
    def __init__(self, reply: str, tool_calls: list[str]) -> None:
        self.reply = reply
        self.tool_calls = tool_calls


class FunctionCallService:
    def __init__(
        self,
        registry: ToolRegistry,
        client_registry: ChatModelClientRegistry | None = None,
    ) -> None:
        self._registry = registry
        self._selector = get_model_selector()
        self._routing_executor = get_model_routing_executor()
        self._client_registry = client_registry or ChatModelClientRegistry()

    async def complete(
        self,
        messages: list[dict[str, str]],
        max_steps: int,
    ) -> FunctionCallResult:
        conversation: list[dict[str, Any]] = [dict(message) for message in messages]
        schemas = self._registry.function_schemas()
        summaries: list[str] = []

        if schemas:
            policy = (
                "\n\nUse the provided tools whenever the answer depends on current, "
                "user-specific, shop, voucher, review, or other external business data. "
                "After each tool result, reassess the remaining task and call any "
                "additional tool needed before answering. Never guess or fabricate "
                "data that a tool can retrieve."
            )
            if conversation and conversation[0].get("role") == "system":
                conversation[0]["content"] = str(conversation[0].get("content", "")) + policy
            else:
                conversation.insert(
                    0,
                    {"role": "system", "content": policy.strip()},
                )

        # Every turn sees the updated observations and all schemas, so the model can
        # select a different tool for the next part of a multi-step task.
        for step in range(1, max_steps + 1):
            turn = await self._generate_turn(conversation, schemas)
            if not turn.tool_calls:
                if turn.content and turn.content.strip():
                    return FunctionCallResult(turn.content.strip(), summaries)
                raise ModelUnavailableError("function call model returned empty reply")

            conversation.append(turn.assistant_message)
            for call in turn.tool_calls:
                logger.info(
                    "LLM function call step=%s tool=%s arguments=%s",
                    step,
                    call.name,
                    json.dumps(call.arguments, ensure_ascii=False, sort_keys=True),
                )
                observation, summary = await self._execute(call)
                summaries.append(summary)
                conversation.append(self._observation_message(call, observation))
                logger.info(
                    "LLM function call completed step=%s tool=%s",
                    step,
                    call.name,
                )

        final_turn = await self._generate_turn(conversation, [])
        if not final_turn.content or not final_turn.content.strip():
            raise ModelUnavailableError("function call loop reached its limit")
        return FunctionCallResult(final_turn.content.strip(), summaries)

    async def _generate_turn(
        self,
        messages: list[dict[str, Any]],
        schemas: list[dict[str, Any]],
    ) -> FunctionCallTurn:
        targets = self._selector.select_chat_candidates(require_tools=bool(schemas))
        return await self._routing_executor.execute_with_fallback(
            ModelCapability.CHAT,
            targets,
            self._client_registry.resolve,
            lambda client, target: self._call_chat_client(client, target, messages, schemas),
        )

    async def _call_chat_client(
        self,
        client: ChatModelClient,
        target: ModelTarget,
        messages: list[dict[str, Any]],
        schemas: list[dict[str, Any]],
    ) -> FunctionCallTurn:
        return await client.complete_turn(target, messages, schemas)

    async def _execute(self, call: FunctionCallRequest) -> tuple[str, str]:
        tool = self._registry.get(call.name)
        arguments = json.dumps(call.arguments, ensure_ascii=False, sort_keys=True)
        if tool is None:
            available = ", ".join(self._registry.list_names()) or "none"
            observation = (
                f"Tool '{call.name}' is unavailable. Available tools: {available}."
            )
            return observation, f"{call.name}({arguments}) -> unavailable"

        try:
            observation = await tool.handler(call.arguments)
            return observation, f"{call.name}({arguments}) -> {observation}"
        except Exception as exc:
            logger.exception("function tool failed tool=%s", call.name)
            observation = f"Tool '{call.name}' failed: {exc}"
            return observation, f"{call.name}({arguments}) -> failed"

    def _observation_message(
        self,
        call: FunctionCallRequest,
        observation: str,
    ) -> dict[str, Any]:
        if get_settings().ai_provider.lower() == "ollama":
            return {
                "role": "tool",
                "content": observation,
                "tool_name": call.name,
            }
        return {
            "role": "tool",
            "content": observation,
            "tool_call_id": call.call_id,
        }
