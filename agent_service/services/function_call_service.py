from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, cast

import httpx

from agent_service.core.config import get_settings
from agent_service.core.errors import ModelUnavailableError
from agent_service.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class FunctionCallingUnavailable(RuntimeError):
    """The configured model or provider cannot process native tools."""


@dataclass(frozen=True)
class FunctionCallRequest:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class FunctionCallTurn:
    content: str | None
    tool_calls: list[FunctionCallRequest]
    assistant_message: dict[str, Any]


@dataclass(frozen=True)
class FunctionCallResult:
    reply: str
    tool_calls: list[str]


class FunctionCallService:
    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

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
        if get_settings().ai_provider.lower() == "ollama":
            return await self._ollama_turn(messages, schemas)
        return await self._openai_turn(messages, schemas)

    async def _ollama_turn(
        self,
        messages: list[dict[str, Any]],
        schemas: list[dict[str, Any]],
    ) -> FunctionCallTurn:
        settings = get_settings()
        payload: dict[str, Any] = {
            "model": settings.ai_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0},
        }
        if schemas:
            payload["tools"] = schemas

        try:
            async with httpx.AsyncClient(
                timeout=settings.ai_timeout_seconds,
                trust_env=False,
            ) as client:
                response = await client.post(
                    f"{settings.ai_base_url.rstrip('/')}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            if schemas and exc.response.status_code in {400, 404, 422, 500}:
                raise FunctionCallingUnavailable(
                    f"Ollama model '{settings.ai_model}' does not support tools"
                ) from exc
            raise ModelUnavailableError(str(exc)) from exc
        except Exception as exc:
            raise ModelUnavailableError(str(exc)) from exc

        message = cast(dict[str, Any], data.get("message") or {})
        calls: list[FunctionCallRequest] = []
        for index, raw_call in enumerate(message.get("tool_calls") or []):
            function = raw_call.get("function") or {}
            name = str(function.get("name") or "").strip()
            if name:
                calls.append(
                    FunctionCallRequest(
                        call_id=str(raw_call.get("id") or f"ollama_call_{index}"),
                        name=name,
                        arguments=self._normalize_arguments(function.get("arguments")),
                    )
                )

        return FunctionCallTurn(
            content=self._optional_text(message.get("content")),
            tool_calls=calls,
            assistant_message=message,
        )
    async def _openai_turn(
        self,
        messages: list[dict[str, Any]],
        schemas: list[dict[str, Any]],
    ) -> FunctionCallTurn:
        settings = get_settings()
        if not settings.ai_api_key:
            raise ModelUnavailableError("AI_API_KEY is not configured")

        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=settings.ai_api_key,
                base_url=settings.ai_base_url,
                timeout=settings.ai_timeout_seconds,
            )
            request: dict[str, Any] = {
                "model": settings.ai_model,
                "messages": messages,
                "temperature": 0.2,
            }
            if schemas:
                request["tools"] = schemas
                request["tool_choice"] = "auto"
            response = await client.chat.completions.create(**cast(Any, request))
            if not response.choices:
                raise ModelUnavailableError("model returned no choices")
            message = response.choices[0].message
        except ModelUnavailableError:
            raise
        except Exception as exc:
            error_text = str(exc).lower()
            if schemas and ("tool" in error_text or "function" in error_text):
                raise FunctionCallingUnavailable(str(exc)) from exc
            raise ModelUnavailableError(str(exc)) from exc

        calls = [
            FunctionCallRequest(
                call_id=tool_call.id,
                name=tool_call.function.name,
                arguments=self._normalize_arguments(tool_call.function.arguments),
            )
            for tool_call in (message.tool_calls or [])
        ]
        return FunctionCallTurn(
            content=self._optional_text(message.content),
            tool_calls=calls,
            assistant_message=cast(dict[str, Any], message.model_dump(exclude_none=True)),
        )

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

    def _normalize_arguments(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return cast(dict[str, Any], value)
        if value is None:
            return {}
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                return {"input": value.strip()}
            if isinstance(decoded, dict):
                return cast(dict[str, Any], decoded)
            return {"input": decoded}
        return {"input": value}

    def _optional_text(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None