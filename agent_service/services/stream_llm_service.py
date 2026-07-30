from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import Any

from agent_service.core.config import get_settings
from agent_service.core.errors import ModelUnavailableError
from agent_service.core.observability import record_stage
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
        settings = get_settings()
        if settings.agent_mock_mode:
            return self._start_mock_stream(messages, callback)

        targets = self._selector.select_chat_candidates(deep_thinking=deep_thinking)
        last_error: Exception | None = None
        for candidate_index, target in enumerate(targets):
            client = self._client_registry.resolve(target)
            stream = getattr(client, "do_stream_chat", None) if client is not None else None
            if not callable(stream):
                continue
            routed_callback = _FirstTokenStreamCallback(callback, self._health_store, target)
            started_at = time.perf_counter()
            handle: StreamCancellationHandle | None = None
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
                status, error = await routed_callback.await_first_packet(
                    settings.ai_stream_first_token_timeout_seconds
                )
            except asyncio.CancelledError:
                # The task manager can cancel while this method is waiting for a first
                # packet.  Do not leave that model request consuming tokens in the
                # background merely because the public handle has not been returned yet.
                routed_callback.suppress()
                if handle is not None:
                    handle.cancel()
                raise
            except Exception as exc:
                routed_callback.suppress()
                if handle is not None:
                    handle.cancel()
                self._health_store.record_failure(target.id)
                last_error = exc
                record_stage(
                    "llm_first_token",
                    elapsed_ms=int((time.perf_counter() - started_at) * 1000),
                    candidateIndex=candidate_index,
                    modelId=target.id,
                    status="start_error",
                )
                logger.warning("stream model start failed modelId=%s error=%s", target.id, exc)
                continue

            if status == "first_token":
                if not routed_callback.failed_after_visible_output:
                    self._health_store.record_success(target.id)
                record_stage(
                    "llm_first_token",
                    elapsed_ms=int((time.perf_counter() - started_at) * 1000),
                    candidateIndex=candidate_index,
                    modelId=target.id,
                    status="success",
                )
                return handle

            # No visible output has reached the caller, so a different candidate can
            # safely take over.  Suppression happens before cancellation because a client
            # may race an error/empty completion callback with this branch.
            routed_callback.suppress()
            if handle is not None:
                handle.cancel()
            self._health_store.record_failure(target.id)
            last_error = error or RuntimeError(f"stream ended before first token ({status})")
            record_stage(
                "llm_first_token",
                elapsed_ms=int((time.perf_counter() - started_at) * 1000),
                candidateIndex=candidate_index,
                modelId=target.id,
                status=status,
            )
            logger.warning(
                "stream first-token attempt failed modelId=%s status=%s error=%s",
                target.id,
                status,
                last_error,
            )

        if last_error is not None:
            raise ModelUnavailableError(f"all chat stream candidates failed: {last_error}")
        raise ModelUnavailableError("no stream-capable chat model candidates available")

    def _start_mock_stream(
        self,
        messages: list[dict[str, str]],
        callback: StreamCallback,
    ) -> StreamCancellationHandle:
        cancelled = asyncio.Event()
        started_at = time.perf_counter()

        async def run() -> None:
            try:
                question = messages[-1]["content"] if messages else ""
                reply = f"这是 HYEEE AI 的本地流式测试回复。已收到你的问题：{question}"
                first_token_recorded = False
                for chunk in _split_text(reply, chunk_size=24):
                    if cancelled.is_set():
                        return
                    if not first_token_recorded:
                        first_token_recorded = True
                        record_stage(
                            "llm_first_token",
                            elapsed_ms=int((time.perf_counter() - started_at) * 1000),
                            candidateIndex=0,
                            modelId="mock",
                            status="success",
                        )
                    await _invoke(callback, "on_content", chunk)
                    await asyncio.sleep(0)
                if not cancelled.is_set():
                    await _invoke(callback, "on_complete")
            except Exception as exc:
                if not cancelled.is_set():
                    await _invoke(callback, "on_error", exc)

        return StreamCancellationHandle(asyncio.create_task(run()), cancelled)


class _FirstTokenStreamCallback:
    """Delay forwarding a candidate until it proves it can produce a stream.

    A callback error before the first visible chunk is normally indistinguishable from a
    failed connection or a stalled provider.  It is therefore held locally so the caller can
    try the next model.  After the first content/reasoning chunk is sent downstream, fallback
    is deliberately disabled: continuing with a second model would create one mixed answer.
    """

    def __init__(self, delegate: StreamCallback, health_store: Any, target: ModelTarget) -> None:
        self._delegate = delegate
        self._health_store = health_store
        self._target = target
        self._first_packet = asyncio.Event()
        self._first_packet_status: str | None = None
        self._first_packet_error: Exception | None = None
        self._visible_output = False
        self._suppressed = False
        self._completed = False
        self._failed_after_output = False

    async def on_content(self, content: str) -> None:
        if not content:
            return
        self._mark_first_packet("first_token")
        if not self._suppressed:
            await _invoke(self._delegate, "on_content", content)

    async def on_thinking(self, content: str) -> None:
        if not content:
            return
        self._mark_first_packet("first_token")
        if not self._suppressed:
            await _invoke(self._delegate, "on_thinking", content)

    async def on_complete(self) -> None:
        self._completed = True
        if not self._visible_output:
            self._mark_first_packet("empty")
            return
        if not self._suppressed:
            await _invoke(self._delegate, "on_complete")

    async def on_error(self, error: Exception) -> None:
        if not self._visible_output:
            self._mark_first_packet("error", error)
            return
        if not self._completed and not self._failed_after_output:
            self._health_store.record_failure(self._target.id)
            self._failed_after_output = True
        if not self._suppressed:
            await _invoke(self._delegate, "on_error", error)

    async def await_first_packet(self, timeout_seconds: float) -> tuple[str, Exception | None]:
        try:
            await asyncio.wait_for(self._first_packet.wait(), timeout=timeout_seconds)
        except TimeoutError:
            return "timeout", None
        return self._first_packet_status or "empty", self._first_packet_error

    def suppress(self) -> None:
        self._suppressed = True

    @property
    def failed_after_visible_output(self) -> bool:
        return self._failed_after_output

    def _mark_first_packet(self, status: str, error: Exception | None = None) -> None:
        if self._first_packet_status is not None:
            return
        self._first_packet_status = status
        self._first_packet_error = error
        if status == "first_token":
            self._visible_output = True
        self._first_packet.set()


async def _invoke(callback: object, method_name: str, *args: object) -> None:
    method = getattr(callback, method_name, None)
    if method is None:
        return
    result = method(*args)
    if inspect.isawaitable(result):
        await result


def _split_text(text: str, *, chunk_size: int) -> list[str]:
    return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]
