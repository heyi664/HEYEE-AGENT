from __future__ import annotations

import asyncio
import inspect
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


@dataclass(frozen=True)
class StreamEvent:
    content: str | None = None
    reasoning: str | None = None
    completed: bool = False


class StreamCancellationHandle:
    def __init__(self, task: asyncio.Task[None], cancelled: asyncio.Event) -> None:
        self._task = task
        self._cancelled = cancelled

    def cancel(self) -> None:
        if self._cancelled.is_set():
            return
        self._cancelled.set()
        self._task.cancel()

    async def wait(self) -> None:
        try:
            await self._task
        except asyncio.CancelledError:
            return


class StreamCallback(Protocol):
    def on_content(self, content: str) -> object: ...

    def on_complete(self) -> object: ...

    def on_error(self, error: Exception) -> object: ...


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


class AbstractOpenAIStyleChatModelClient:
    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._transport = transport

    async def complete_turn(
        self,
        target: ModelTarget,
        messages: list[dict[str, Any]],
        schemas: list[dict[str, Any]],
    ) -> ChatTurn:
        return await self.do_chat(target, messages, schemas)

    async def do_chat(
        self,
        target: ModelTarget,
        messages: list[dict[str, Any]],
        schemas: list[dict[str, Any]],
    ) -> ChatTurn:
        settings = get_settings()
        self._require_provider(target)
        if self.requires_api_key():
            self._require_api_key(target)

        payload = self.build_request_body(target, messages, schemas, stream=False)
        headers = self.new_authorized_headers(target)
        timeout = target.candidate.timeout_seconds or settings.ai_timeout_seconds
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                transport=self._transport,
                trust_env=False,
            ) as client:
                response = await client.post(
                    resolve_model_url(target),
                    json=payload,
                    headers=headers,
                )
        except httpx.HTTPError as exc:
            raise ModelUnavailableError(str(exc)) from exc
        except ValueError as exc:
            raise ModelUnavailableError(str(exc)) from exc

        if response.is_error:
            if schemas and response.status_code in {400, 404, 422, 500}:
                raise ToolCallingUnavailable(
                    f"{target.candidate.model} does not support tools: "
                    f"HTTP {response.status_code}"
                )
            raise ModelUnavailableError(
                f"{target.provider.name} chat API failed: "
                f"HTTP {response.status_code} body={response.text[:500]}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise ModelUnavailableError("chat API returned invalid JSON") from exc
        return self.extract_chat_turn(data)

    def requires_api_key(self) -> bool:
        return True

    def build_request_body(
        self,
        target: ModelTarget,
        messages: list[dict[str, Any]],
        schemas: list[dict[str, Any]],
        stream: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": target.candidate.model,
            "messages": messages,
            "temperature": 0.2,
        }
        if stream:
            body["stream"] = True
        if schemas:
            body["tools"] = schemas
            body["tool_choice"] = "auto"
        self.customize_request_body(body, target, messages, schemas)
        return body

    def customize_request_body(
        self,
        body: dict[str, Any],
        target: ModelTarget,
        messages: list[dict[str, Any]],
        schemas: list[dict[str, Any]],
    ) -> None:
        return None

    def new_authorized_headers(self, target: ModelTarget) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.requires_api_key():
            headers["Authorization"] = f"Bearer {target.provider.api_key}"
        return headers

    def extract_chat_turn(self, data: Any) -> ChatTurn:
        if not isinstance(data, dict):
            raise ModelUnavailableError("chat API response must be a JSON object")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ModelUnavailableError("model returned no choices")
        choice0 = choices[0]
        if not isinstance(choice0, dict):
            raise ModelUnavailableError("model returned invalid choice")
        message = choice0.get("message")
        if not isinstance(message, dict):
            raise ModelUnavailableError("model returned invalid message")

        calls: list[ToolCallRequest] = []
        for index, raw_call in enumerate(message.get("tool_calls") or []):
            if not isinstance(raw_call, dict):
                continue
            function = raw_call.get("function") or {}
            if not isinstance(function, dict):
                continue
            name = str(function.get("name") or "").strip()
            if name:
                calls.append(
                    ToolCallRequest(
                        call_id=str(raw_call.get("id") or f"tool_call_{index}"),
                        name=name,
                        arguments=_normalize_arguments(function.get("arguments")),
                    )
                )

        return ChatTurn(
            content=_optional_text(message.get("content")),
            tool_calls=calls,
            assistant_message=cast(dict[str, Any], message),
        )

    def _require_provider(self, target: ModelTarget) -> None:
        if not target.provider.url and not target.candidate.url:
            raise ModelUnavailableError("AI provider base URL is not configured")

    def _require_api_key(self, target: ModelTarget) -> None:
        if not target.provider.api_key:
            raise ModelUnavailableError("AI provider API key is not configured")

    async def do_stream_chat(
        self,
        target: ModelTarget,
        messages: list[dict[str, Any]],
        schemas: list[dict[str, Any]],
        callback: StreamCallback,
        *,
        reasoning_enabled: bool = False,
    ) -> StreamCancellationHandle:
        settings = get_settings()
        self._require_provider(target)
        if self.requires_api_key():
            self._require_api_key(target)

        payload = self.build_request_body(target, messages, schemas, stream=True)
        headers = self.new_authorized_headers(target)
        headers["Accept"] = "text/event-stream"
        timeout = target.candidate.timeout_seconds or settings.ai_timeout_seconds
        cancelled = asyncio.Event()
        task = asyncio.create_task(
            self._run_stream(
                target,
                payload,
                headers,
                timeout,
                callback,
                cancelled,
                reasoning_enabled,
            )
        )
        return StreamCancellationHandle(task, cancelled)

    async def _run_stream(
        self,
        target: ModelTarget,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
        callback: StreamCallback,
        cancelled: asyncio.Event,
        reasoning_enabled: bool,
    ) -> None:
        completed = False
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                transport=self._transport,
                trust_env=False,
            ) as client:
                async with client.stream(
                    "POST",
                    resolve_model_url(target),
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.is_error:
                        error_body = (await response.aread()).decode(
                            "utf-8",
                            errors="replace",
                        )[:500]
                        raise ModelUnavailableError(
                            f"{target.provider.name} stream chat API failed: "
                            f"HTTP {response.status_code} body={error_body}"
                        )
                    async for line in response.aiter_lines():
                        if cancelled.is_set():
                            return
                        try:
                            event = self.parse_sse_line(line, reasoning_enabled)
                        except Exception:
                            continue
                        if event.reasoning:
                            await _notify_callback(callback, "on_thinking", event.reasoning)
                        if event.content:
                            await _notify_callback(callback, "on_content", event.content)
                        if event.completed:
                            completed = True
                            await _notify_callback(callback, "on_complete")
                            return
            if not completed and not cancelled.is_set():
                raise ModelUnavailableError("chat stream ended before completion")
        except asyncio.CancelledError:
            return
        except Exception as exc:
            if not cancelled.is_set():
                await _notify_callback(callback, "on_error", exc)

    def parse_sse_line(self, line: str, reasoning_enabled: bool) -> StreamEvent:
        if not line or not line.strip():
            return StreamEvent()
        payload = line.strip()
        if payload.startswith("data:"):
            payload = payload[len("data:") :].strip()
        if payload.lower() == "[done]":
            return StreamEvent(completed=True)

        data = json.loads(payload)
        if not isinstance(data, dict):
            return StreamEvent()
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return StreamEvent()
        choice0 = choices[0]
        if not isinstance(choice0, dict):
            return StreamEvent()

        content = self._extract_stream_text(choice0, "content")
        reasoning = (
            self._extract_stream_text(choice0, "reasoning_content")
            if reasoning_enabled
            else None
        )
        completed = choice0.get("finish_reason") is not None
        return StreamEvent(content=content, reasoning=reasoning, completed=completed)

    def _extract_stream_text(self, choice: dict[str, Any], field_name: str) -> str | None:
        for container_name in ("delta", "message"):
            container = choice.get(container_name)
            if isinstance(container, dict):
                value = container.get(field_name)
                if value is not None:
                    text = str(value)
                    return text if text else None
        return None


class OpenAICompatibleChatModelClient(AbstractOpenAIStyleChatModelClient):
    def supports(self, target: ModelTarget) -> bool:
        return not OllamaChatModelClient().supports(target)


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



async def _notify_callback(callback: object, method_name: str, *args: object) -> None:
    method = getattr(callback, method_name, None)
    if method is None:
        return
    result = method(*args)
    if inspect.isawaitable(result):
        await result

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
