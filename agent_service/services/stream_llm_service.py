from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any

from agent_service.core.config import get_settings
from agent_service.core.errors import ModelUnavailableError
from agent_service.infra_ai import get_model_health_store, get_model_selector
from agent_service.infra_ai.clients import (
    ChatModelClientRegistry,
    StreamCallback,
    StreamCancellationHandle,
)
from agent_service.infra_ai.models import ModelTarget

logger = logging.getLogger(__name__)


class StreamLLMService:
    """Select a model and start one cancellable stream for a prepared answer prompt."""

    def __init__(self, client_registry: ChatModelClientRegistry | None = None) -> None:
        self._selector = get_model_selector()
        self._health_store = get_model_health_store()
        self._client_registry = client_registry or ChatModelClientRegistry()

    async def start(
        self,
        messages: list[dict[str, str]],
        callback: StreamCallback,
        *,
        temperature: float,
        top_p: float | None,
        deep_thinking: bool = False,
    ) -> StreamCancellationHandle:
        if get_settings().agent_mock_mode:
            return self._start_mock_stream(messages, callback)

        targets = self._selector.select_chat_candidates(deep_thinking=deep_thinking)
        last_error: Exception | None = None
        for target in targets:
            client = self._client_registry.resolve(target)
            stream = getattr(client, "do_stream_chat", None) if client is not None else None
            if not callable(stream):
                continue
            routed_callback = _RoutedStreamCallback(callback, self._health_store, target)
            try:
                handle = await stream(
                    target,
                    messages,
                    [],
                    routed_callback,
                    reasoning_enabled=deep_thinking and target.candidate.supports_thinking,
                    temperature=temperature,
                    top_p=top_p,
                )
            except Exception as exc:
                self._health_store.record_failure(target.id)
                last_error = exc
                logger.warning("stream model start failed modelId=%s error=%s", target.id, exc)
                continue
            return handle

        if last_error is not None:
            raise ModelUnavailableError(f"all chat stream candidates failed: {last_error}")
        raise ModelUnavailableError("no stream-capable chat model candidates available")

    def _start_mock_stream(
        self,
        messages: list[dict[str, str]],
        callback: StreamCallback,
    ) -> StreamCancellationHandle:
        cancelled = asyncio.Event()

        async def run() -> None:
            try:
                question = messages[-1]["content"] if messages else ""
                reply = f"这是 HYEEE AI 的本地流式测试回复。已收到你的问题：{question}"
                for chunk in _split_text(reply, chunk_size=24):
                    if cancelled.is_set():
                        return
                    await _invoke(callback, "on_content", chunk)
                    await asyncio.sleep(0)
                if not cancelled.is_set():
                    await _invoke(callback, "on_complete")
            except Exception as exc:
                if not cancelled.is_set():
                    await _invoke(callback, "on_error", exc)

        return StreamCancellationHandle(asyncio.create_task(run()), cancelled)


class _RoutedStreamCallback:
    def __init__(self, delegate: StreamCallback, health_store: Any, target: ModelTarget) -> None:
        self._delegate = delegate
        self._health_store = health_store
        self._target = target
        self._completed = False

    async def on_content(self, content: str) -> None:
        await _invoke(self._delegate, "on_content", content)

    async def on_thinking(self, content: str) -> None:
        await _invoke(self._delegate, "on_thinking", content)

    async def on_complete(self) -> None:
        self._completed = True
        self._health_store.record_success(self._target.id)
        await _invoke(self._delegate, "on_complete")

    async def on_error(self, error: Exception) -> None:
        if not self._completed:
            self._health_store.record_failure(self._target.id)
        await _invoke(self._delegate, "on_error", error)


async def _invoke(callback: object, method_name: str, *args: object) -> None:
    method = getattr(callback, method_name, None)
    if method is None:
        return
    result = method(*args)
    if inspect.isawaitable(result):
        await result


def _split_text(text: str, *, chunk_size: int) -> list[str]:
    return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]
