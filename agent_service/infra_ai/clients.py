from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, cast

import httpx

from agent_service.core.config import get_settings
from agent_service.core.errors import ModelUnavailableError
from agent_service.infra_ai.models import ModelTarget
from agent_service.infra_ai.url_resolver import resolve_model_url


class ToolCallingUnavailable(RuntimeError):
    """The target model or provider cannot process native tools."""


@dataclass(frozen=True)
class ToolCallRequest:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ChatTurn:
    content: str | None
    tool_calls: list[ToolCallRequest]
    assistant_message: dict[str, Any]


class ChatModelClient(Protocol):
    def supports(self, target: ModelTarget) -> bool: ...

    async def complete_turn(
        self,
        target: ModelTarget,
        messages: list[dict[str, Any]],
        schemas: list[dict[str, Any]],
    ) -> ChatTurn: ...


class EmbeddingModelClient(Protocol):
    def supports(self, target: ModelTarget) -> bool: ...

    async def embed_batch(
        self,
        target: ModelTarget,
        texts: list[str],
    ) -> list[list[float]]: ...


class ChatModelClientRegistry:
    def __init__(self, clients: list[ChatModelClient] | None = None) -> None:
        self._clients = clients or [OllamaChatModelClient(), OpenAICompatibleChatModelClient()]

    def resolve(self, target: ModelTarget) -> ChatModelClient | None:
        return next((client for client in self._clients if client.supports(target)), None)


class EmbeddingModelClientRegistry:
    def __init__(self, clients: list[EmbeddingModelClient] | None = None) -> None:
        self._clients = clients or [OpenAICompatibleEmbeddingModelClient()]

    def resolve(self, target: ModelTarget) -> EmbeddingModelClient | None:
        return next((client for client in self._clients if client.supports(target)), None)


class OllamaChatModelClient:
    def supports(self, target: ModelTarget) -> bool:
        return (
            target.provider.name.lower() == "ollama"
            or target.candidate.provider.lower() == "ollama"
        )

    async def complete_turn(
        self,
        target: ModelTarget,
        messages: list[dict[str, Any]],
        schemas: list[dict[str, Any]],
    ) -> ChatTurn:
        settings = get_settings()
        payload: dict[str, Any] = {
            "model": target.candidate.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0},
        }
        if schemas:
            payload["tools"] = schemas

        try:
            async with httpx.AsyncClient(
                timeout=target.candidate.timeout_seconds or settings.ai_timeout_seconds,
                trust_env=False,
            ) as client:
                response = await client.post(
                    f"{(target.provider.url or settings.ai_base_url).rstrip('/')}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            if schemas and exc.response.status_code in {400, 404, 422, 500}:
                raise ToolCallingUnavailable(
                    f"Ollama model '{target.candidate.model}' does not support tools"
                ) from exc
            raise ModelUnavailableError(str(exc)) from exc
        except Exception as exc:
            raise ModelUnavailableError(str(exc)) from exc

        message = cast(dict[str, Any], data.get("message") or {})
        calls: list[ToolCallRequest] = []
        for index, raw_call in enumerate(message.get("tool_calls") or []):
            function = raw_call.get("function") or {}
            name = str(function.get("name") or "").strip()
            if name:
                calls.append(
                    ToolCallRequest(
                        call_id=str(raw_call.get("id") or f"ollama_call_{index}"),
                        name=name,
                        arguments=_normalize_arguments(function.get("arguments")),
                    )
                )
        return ChatTurn(
            content=_optional_text(message.get("content")),
            tool_calls=calls,
            assistant_message=message,
        )


class OpenAICompatibleChatModelClient:
    def supports(self, target: ModelTarget) -> bool:
        return not OllamaChatModelClient().supports(target)

    async def complete_turn(
        self,
        target: ModelTarget,
        messages: list[dict[str, Any]],
        schemas: list[dict[str, Any]],
    ) -> ChatTurn:
        settings = get_settings()
        if not target.provider.api_key:
            raise ModelUnavailableError("AI provider API key is not configured")
        if not target.provider.url:
            raise ModelUnavailableError("AI provider base URL is not configured")

        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=target.provider.api_key,
                base_url=target.provider.url,
                timeout=target.candidate.timeout_seconds or settings.ai_timeout_seconds,
            )
            request: dict[str, Any] = {
                "model": target.candidate.model,
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
                raise ToolCallingUnavailable(str(exc)) from exc
            raise ModelUnavailableError(str(exc)) from exc

        calls = [
            ToolCallRequest(
                call_id=tool_call.id,
                name=tool_call.function.name,
                arguments=_normalize_arguments(tool_call.function.arguments),
            )
            for tool_call in (message.tool_calls or [])
        ]
        return ChatTurn(
            content=_optional_text(message.content),
            tool_calls=calls,
            assistant_message=cast(dict[str, Any], message.model_dump(exclude_none=True)),
        )


class OpenAICompatibleEmbeddingModelClient:
    def supports(self, target: ModelTarget) -> bool:
        return True

    async def embed_batch(
        self,
        target: ModelTarget,
        texts: list[str],
    ) -> list[list[float]]:
        if not target.provider.api_key:
            raise ModelUnavailableError("Embedding provider API key is not configured")
        url = resolve_model_url(target)
        payload = {"model": target.candidate.model, "input": texts}
        headers = {
            "Authorization": f"Bearer {target.provider.api_key}",
            "Content-Type": "application/json",
        }
        timeout = target.candidate.timeout_seconds or get_settings().embedding_timeout_seconds
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
        if response.is_error:
            raise ModelUnavailableError(
                f"Embedding API failed status={response.status_code} body={response.text[:500]}"
            )
        data = response.json()
        rows = data.get("data")
        if not isinstance(rows, list):
            raise ModelUnavailableError("Embedding API response missing data")
        rows = sorted(
            rows,
            key=lambda row: int(row.get("index", 0)) if isinstance(row, dict) else 0,
        )
        embeddings: list[list[float]] = []
        for row in rows:
            embedding = row.get("embedding") if isinstance(row, dict) else None
            if not isinstance(embedding, list):
                raise ModelUnavailableError("Embedding API response missing embedding")
            embeddings.append([float(value) for value in embedding])
        return embeddings


def _normalize_arguments(value: Any) -> dict[str, Any]:
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


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
