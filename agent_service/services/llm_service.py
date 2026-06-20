from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, cast

import httpx

from agent_service.core.config import get_settings
from agent_service.core.errors import ModelUnavailableError
from agent_service.services.function_call_service import (
    FunctionCallingUnavailable,
    FunctionCallService,
)
from agent_service.tools.builtin import register_builtin_tools
from agent_service.tools.registry import ToolRegistry, tool_registry

logger = logging.getLogger(__name__)

FINAL_ANSWER_PATTERN = re.compile(r"Final Answer\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)
ACTION_PATTERN = re.compile(r"Action\s*:\s*([^\r\n]+)", re.IGNORECASE)
ACTION_INPUT_PATTERN = re.compile(r"Action Input\s*:\s*([^\r\n]+)", re.IGNORECASE)


@dataclass(frozen=True)
class LLMResult:
    reply: str
    tool_calls: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReActDecision:
    final_answer: str | None = None
    action: str | None = None
    action_input: dict[str, Any] = field(default_factory=dict)


class LLMService:
    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self._registry = registry or tool_registry

    async def complete(self, messages: list[dict[str, str]]) -> LLMResult:
        settings = get_settings()
        if settings.agent_mock_mode:
            return await self._complete_with_react(messages)

        mode = settings.agent_tool_mode.lower()
        if mode not in {"auto", "function_call", "react"}:
            raise ModelUnavailableError(f"Unsupported AGENT_TOOL_MODE: {mode}")

        if mode in {"auto", "function_call"}:
            try:
                result = await FunctionCallService(self._registry).complete(
                    messages,
                    settings.react_max_steps,
                )
                return LLMResult(result.reply, result.tool_calls)
            except FunctionCallingUnavailable as exc:
                if mode == "function_call":
                    raise ModelUnavailableError(str(exc)) from exc
                logger.info(
                    "native function calling unavailable; falling back to ReAct: %s",
                    exc,
                )

        return await self._complete_with_react(messages)

    async def _complete_with_react(
        self,
        messages: list[dict[str, str]],
    ) -> LLMResult:
        settings = get_settings()
        if settings.agent_mock_mode:
            user_message = messages[-1]["content"] if messages else ""
            return LLMResult(
                reply=(
                    "这是 HYEEE AI 的本地测试回复。"
                    f"我已收到你的问题：{user_message}。"
                    "真实模型接入后，这里会返回大模型生成的回答。"
                )
            )

        react_messages = self._with_react_instructions(messages)
        tool_calls: list[str] = []

        for step in range(1, settings.react_max_steps + 1):
            model_output = await self._generate(react_messages)
            decision = self._parse_react_output(model_output)

            if decision.final_answer:
                return LLMResult(reply=decision.final_answer, tool_calls=tool_calls)

            if not decision.action:
                logger.info("react completed without protocol step=%s", step)
                return LLMResult(reply=model_output.strip(), tool_calls=tool_calls)

            logger.info(
                "LLM ReAct tool call step=%s tool=%s arguments=%s",
                step,
                decision.action,
                json.dumps(
                    decision.action_input,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )
            observation, call_summary = await self._execute_tool(
                decision.action,
                decision.action_input,
            )
            tool_calls.append(call_summary)
            react_messages.extend(
                [
                    {"role": "assistant", "content": model_output},
                    {"role": "user", "content": f"Observation: {observation}"},
                ]
            )
            logger.info("react tool executed step=%s tool=%s", step, decision.action)

        final_output = await self._generate(
            react_messages
            + [
                {
                    "role": "user",
                    "content": "Stop using tools and provide Final Answer now.",
                }
            ]
        )
        final_decision = self._parse_react_output(final_output)
        reply = final_decision.final_answer or final_output.strip()
        if not reply:
            raise ModelUnavailableError("ReAct loop returned empty reply")
        return LLMResult(reply=reply, tool_calls=tool_calls)

    def _with_react_instructions(
        self,
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        tools = self._registry.list_tools()
        tool_lines = "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)
        if not tool_lines:
            tool_lines = "- No tools are currently available."

        react_prompt = (
            "\n\n你必须使用 ReAct 协议回答。可用工具：\n"
            f"{tool_lines}\n\n"
            "当问题涉及当前时间等实时信息时，必须先调用对应工具，"
            "绝对不能依靠记忆、猜测、示例值或编造 Observation。\n"
            "需要调用工具时，只输出：\n"
            "Thought: 简短说明为什么需要该工具\n"
            "Action: 工具名称\n"
            "Action Input: JSON 对象\n\n"
            "系统返回 Observation 后再继续。能够回答时，只输出：\n"
            "Thought: 简短说明已有足够信息\n"
            "Final Answer: 给用户看的最终答案\n\n"
            "示例：用户问当前时间时，第一步必须输出：\n"
            "Thought: 需要查询真实的服务器时间\n"
            "Action: get_current_time\n"
            "Action Input: {}\n"
            "不要解释 ReAct 协议，不要展示虚构的 JSON 示例，"
            "不要调用未提供的工具。"
        )

        result = [dict(message) for message in messages]
        if result and result[0].get("role") == "system":
            result[0]["content"] += react_prompt
        else:
            result.insert(0, {"role": "system", "content": react_prompt.strip()})
        return result

    def _parse_react_output(self, output: str) -> ReActDecision:
        action_match = ACTION_PATTERN.search(output)
        final_match = FINAL_ANSWER_PATTERN.search(output)

        if action_match and (not final_match or action_match.start() < final_match.start()):
            action = action_match.group(1).strip().strip(chr(96))
            input_match = ACTION_INPUT_PATTERN.search(output)
            raw_input = input_match.group(1).strip() if input_match else "{}"
            return ReActDecision(
                action=action,
                action_input=self._parse_action_input(raw_input),
            )

        if final_match:
            answer = final_match.group(1).strip()
            return ReActDecision(final_answer=answer or None)

        return ReActDecision()

    def _parse_action_input(self, raw_input: str) -> dict[str, Any]:
        cleaned = raw_input.strip()
        if cleaned.startswith(chr(96) * 3):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            value = json.loads(cleaned)
        except json.JSONDecodeError:
            return {"input": cleaned}
        if isinstance(value, dict):
            return cast(dict[str, Any], value)
        return {"input": value}

    async def _execute_tool(
        self,
        action: str,
        action_input: dict[str, Any],
    ) -> tuple[str, str]:
        tool = self._registry.get(action)
        arguments = json.dumps(action_input, ensure_ascii=False, sort_keys=True)
        if tool is None:
            available = ", ".join(self._registry.list_names()) or "none"
            observation = f"Tool '{action}' is unavailable. Available tools: {available}."
            return observation, f"{action}({arguments}) -> unavailable"

        try:
            observation = await tool.handler(action_input)
            summary = f"{action}({arguments}) -> {observation}"
            return observation, summary
        except Exception as exc:  # pragma: no cover - defensive tool boundary
            logger.exception("react tool failed tool=%s", action)
            observation = f"Tool '{action}' failed: {exc}"
            return observation, f"{action}({arguments}) -> failed"

    async def _generate(self, messages: list[dict[str, str]]) -> str:
        settings = get_settings()
        if settings.ai_provider.lower() == "ollama":
            return await self._complete_with_ollama(messages)
        return await self._complete_with_openai_compatible(messages)

    async def _complete_with_ollama(self, messages: list[dict[str, str]]) -> str:
        settings = get_settings()
        base_url = settings.ai_base_url.rstrip("/")
        try:
            async with httpx.AsyncClient(
                timeout=settings.ai_timeout_seconds,
                trust_env=False,
            ) as client:
                response = await client.post(
                    f"{base_url}/api/chat",
                    json={
                        "model": settings.ai_model,
                        "messages": messages,
                        "stream": False,
                        "options": {"temperature": 0},
                    },
                )
                response.raise_for_status()
                data = response.json()
        except Exception as exc:  # pragma: no cover - external provider path
            raise ModelUnavailableError(str(exc)) from exc

        reply = data.get("message", {}).get("content")
        if not reply or not str(reply).strip():
            raise ModelUnavailableError("ollama returned empty reply")
        return str(reply).strip()

    async def _complete_with_openai_compatible(self, messages: list[dict[str, str]]) -> str:
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
            response: Any = await client.chat.completions.create(
                model=settings.ai_model,
                messages=cast(Any, messages),
                temperature=0.2,
            )
            reply = cast(
                str | None,
                response.choices[0].message.content if response.choices else None,
            )
        except Exception as exc:  # pragma: no cover - external provider path
            raise ModelUnavailableError(str(exc)) from exc

        if not reply or not reply.strip():
            raise ModelUnavailableError("model returned empty reply")
        return str(reply).strip()


def get_llm_service() -> LLMService:
    register_builtin_tools(tool_registry)
    return LLMService(tool_registry)
