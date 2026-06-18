from __future__ import annotations

from typing import Any, cast

import httpx

from agent_service.core.config import get_settings
from agent_service.core.errors import ModelUnavailableError


class LLMService:
    async def complete(self, messages: list[dict[str, str]]) -> str:
        settings = get_settings()
        if settings.agent_mock_mode:
            user_message = messages[-1]["content"] if messages else ""
            return (
                "这是 HYEEE AI 的本地测试回复。"
                f"我已收到你的问题：{user_message}。"
                "真实模型接入后，这里会返回大模型生成的回答。"
            )

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
                temperature=0.7,
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
    return LLMService()
