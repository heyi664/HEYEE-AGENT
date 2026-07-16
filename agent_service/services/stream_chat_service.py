from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from agent_service.schemas.chat import ChatRequest
from agent_service.services.chat_service import ChatPreparation, ChatService, get_chat_service
from agent_service.services.stream_llm_service import StreamLLMService
from agent_service.services.stream_queue_limiter import (
    QueueAcquireStatus,
    StreamQueueLimiter,
    StreamQueuePermit,
    stream_queue_limiter,
)
from agent_service.services.stream_task_manager import StreamTaskManager, stream_task_manager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StreamEvent:
    name: str
    data: dict[str, Any]


class StreamChatService:
    """Own the stream lifecycle, including cancellation and partial-answer persistence."""

    def __init__(
        self,
        chat_service: ChatService | None = None,
        llm_service: StreamLLMService | None = None,
        task_manager: StreamTaskManager | None = None,
        queue_limiter: StreamQueueLimiter | None = None,
    ) -> None:
        self._chat_service = chat_service or get_chat_service()
        self._llm_service = llm_service or StreamLLMService()
        self._task_manager = task_manager or stream_task_manager
        self._queue_limiter = queue_limiter or stream_queue_limiter

    async def stream(
        self,
        request: ChatRequest,
        *,
        task_id: str | None = None,
        is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    ) -> AsyncIterator[bytes]:
        task_id = task_id or f"task_{uuid4().hex[:16]}"
        registered = await self._task_manager.register(task_id)
        if not registered:
            yield _encode_event("error", {"message": "taskId is already active"})
            yield _encode_event("done", {"taskId": task_id})
            return

        preparation: ChatPreparation | None = None
        answer_started_at: float | None = None
        response_parts: list[str] = []
        handle: Any | None = None
        queue_permit: StreamQueuePermit | None = None
        lease_task: asyncio.Task[None] | None = None
        cancellation_persisted = False

        async def cancelled_payload() -> dict[str, Any]:
            nonlocal cancellation_persisted
            payload: dict[str, Any] = {
                "taskId": task_id,
                "conversationId": preparation.conversation_id if preparation else None,
                "messageId": None,
                "title": request.message[:128],
                "partial": False,
            }
            reply = "".join(response_parts).strip()
            if (
                preparation is not None
                and answer_started_at is not None
                and reply
                and not cancellation_persisted
            ):
                cancellation_persisted = True
                try:
                    response = await self._chat_service.complete_preparation(
                        preparation,
                        reply=reply,
                        tool_calls=[],
                        answer_started_at=answer_started_at,
                        interrupted=True,
                    )
                    payload.update(
                        {
                            "conversationId": response.conversationId,
                            "messageId": response.messageId,
                            "title": request.message[:128],
                            "partial": True,
                            "sources": [item.model_dump(mode="json") for item in response.sources],
                            "toolCalls": response.toolCalls,
                            "ragIntent": response.ragIntent,
                        }
                    )
                except Exception:
                    # Cancellation must still terminate the stream when persistence is down.
                    logger.exception("failed to persist interrupted reply taskId=%s", task_id)
            return payload

        try:
            # Send the task identifier before admission and slow RAG preparation so a user can
            # cancel whether the request is queued, rewriting or streaming an answer.
            yield _encode_event("meta", {"taskId": task_id, "phase": "queued"})
            if self._task_manager.is_cancelled(task_id):
                yield _encode_event("cancel", await cancelled_payload())
                yield _encode_event("done", {"taskId": task_id})
                return

            admission = await self._queue_limiter.acquire(
                task_id,
                should_cancel=lambda: self._should_cancel(task_id, is_disconnected),
            )
            if admission.status == QueueAcquireStatus.CANCELLED:
                yield _encode_event("cancel", await cancelled_payload())
                yield _encode_event("done", {"taskId": task_id})
                return
            if admission.status == QueueAcquireStatus.TIMED_OUT:
                yield _encode_event(
                    "error",
                    {
                        "message": "Server is busy. Please try again shortly.",
                        "code": "queue_timeout",
                    },
                )
                yield _encode_event("done", {"taskId": task_id})
                return
            if admission.status == QueueAcquireStatus.UNAVAILABLE or admission.permit is None:
                yield _encode_event(
                    "error",
                    {
                        "message": "Stream admission service is unavailable.",
                        "code": "queue_unavailable",
                    },
                )
                yield _encode_event("done", {"taskId": task_id})
                return
            queue_permit = admission.permit
            lease_task = asyncio.create_task(
                self._maintain_queue_lease(queue_permit, task_id),
                name=f"stream-permit-lease-{task_id}",
            )
            yield _encode_event("meta", {"taskId": task_id, "phase": "preparing"})

            preparation_task = asyncio.create_task(
                self._chat_service.prepare(request),
                name=f"stream-prepare-{task_id}",
            )
            if not self._task_manager.bind(task_id, _AsyncTaskCancellationHandle(preparation_task)):
                yield _encode_event("cancel", await cancelled_payload())
                yield _encode_event("done", {"taskId": task_id})
                return
            preparation = await self._await_preparation(
                preparation_task,
                task_id=task_id,
                is_disconnected=is_disconnected,
            )
            if self._task_manager.is_cancelled(task_id):
                yield _encode_event("cancel", await cancelled_payload())
                yield _encode_event("done", {"taskId": task_id})
                return

            yield _encode_event(
                "meta",
                {
                    "taskId": task_id,
                    "phase": "answering",
                    "conversationId": preparation.conversation_id,
                    "ragIntent": preparation.rag_intent,
                    "sources": [
                        source.model_dump(mode="json") for source in self._sources(preparation)
                    ],
                },
            )

            queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
            callback = _QueueStreamCallback(queue)
            answer_started_at = time.perf_counter()
            handle = await self._llm_service.start(
                preparation.messages,
                callback,
                temperature=preparation.prompt_plan.temperature,
                top_p=preparation.prompt_plan.top_p,
                deep_thinking=preparation.deep_thinking,
            )
            if not self._task_manager.bind(task_id, handle):
                # A cancellation can win the race between model start and handle binding.
                yield _encode_event("cancel", await cancelled_payload())
                yield _encode_event("done", {"taskId": task_id})
                return

            while True:
                if is_disconnected is not None and await is_disconnected():
                    await self._task_manager.cancel(task_id)
                    await cancelled_payload()
                    return
                if self._task_manager.is_cancelled(task_id):
                    yield _encode_event("cancel", await cancelled_payload())
                    yield _encode_event("done", {"taskId": task_id})
                    return
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.15)
                except TimeoutError:
                    continue
                if event.name == "thinking":
                    yield _encode_event("message", {"type": "think", "delta": event.data["delta"]})
                    continue
                if event.name == "content":
                    chunk = str(event.data["delta"])
                    response_parts.append(chunk)
                    yield _encode_event("message", {"type": "response", "delta": chunk})
                    continue
                if event.name == "error":
                    if self._task_manager.is_cancelled(task_id):
                        yield _encode_event("cancel", await cancelled_payload())
                    else:
                        self._chat_service.record_answer_failure(preparation, answer_started_at)
                        yield _encode_event("error", {"message": str(event.data["error"])})
                    yield _encode_event("done", {"taskId": task_id})
                    return
                if event.name == "complete":
                    reply = "".join(response_parts).strip()
                    if not reply:
                        self._chat_service.record_answer_failure(preparation, answer_started_at)
                        yield _encode_event("error", {"message": "model returned empty stream"})
                    else:
                        response = await self._chat_service.complete_preparation(
                            preparation,
                            reply=reply,
                            tool_calls=[],
                            answer_started_at=answer_started_at,
                        )
                        yield _encode_event(
                            "finish",
                            {
                                "conversationId": response.conversationId,
                                "messageId": response.messageId,
                                "title": request.message[:128],
                                "sources": [
                                    item.model_dump(mode="json") for item in response.sources
                                ],
                                "toolCalls": response.toolCalls,
                                "ragIntent": response.ragIntent,
                            },
                        )
                    yield _encode_event("done", {"taskId": task_id})
                    return
        except asyncio.CancelledError:
            # The preparation task is deliberately cancelled when the user stops
            # before answer generation starts.  It is a normal terminal path,
            # not an SSE error.
            if self._task_manager.is_cancelled(task_id):
                yield _encode_event("cancel", await cancelled_payload())
                yield _encode_event("done", {"taskId": task_id})
                return
            raise
        except Exception as exc:
            logger.exception("stream chat failed taskId=%s", task_id)
            if self._task_manager.is_cancelled(task_id):
                yield _encode_event("cancel", await cancelled_payload())
            elif preparation is not None and answer_started_at is not None:
                self._chat_service.record_answer_failure(preparation, answer_started_at)
                yield _encode_event(
                    "error",
                    {"message": "AI stream is unavailable", "detail": str(exc)},
                )
            else:
                yield _encode_event(
                    "error",
                    {"message": "AI stream preparation failed", "detail": str(exc)},
                )
            yield _encode_event("done", {"taskId": task_id})
        finally:
            if handle is not None and self._task_manager.is_cancelled(task_id):
                handle.cancel()
            if lease_task is not None:
                lease_task.cancel()
                try:
                    await lease_task
                except asyncio.CancelledError:
                    pass
            await self._queue_limiter.release(queue_permit)
            await self._task_manager.finalize(task_id)

    async def cancel(self, task_id: str) -> bool:
        cancelled = await self._task_manager.cancel(task_id)
        await self._queue_limiter.cancel(task_id)
        return cancelled

    def _sources(self, preparation: ChatPreparation) -> list[Any]:
        return [self._chat_service._to_chat_source(item) for item in preparation.retrieved_sources]

    async def _should_cancel(
        self,
        task_id: str,
        is_disconnected: Callable[[], Awaitable[bool]] | None,
    ) -> bool:
        if self._task_manager.is_cancelled(task_id):
            return True
        if is_disconnected is not None and await is_disconnected():
            await self.cancel(task_id)
            return True
        return False

    async def _maintain_queue_lease(self, permit: StreamQueuePermit, task_id: str) -> None:
        if not permit.distributed:
            return
        interval_seconds = max(1.0, min(self._queue_limiter.lease_seconds / 3, 60.0))
        try:
            while True:
                await asyncio.sleep(interval_seconds)
                if not await self._queue_limiter.renew(permit):
                    logger.error("stream permit lease was lost taskId=%s", task_id)
                    await self.cancel(task_id)
                    return
        except asyncio.CancelledError:
            raise

    async def _await_preparation(
        self,
        task: asyncio.Task[ChatPreparation],
        *,
        task_id: str,
        is_disconnected: Callable[[], Awaitable[bool]] | None,
    ) -> ChatPreparation:
        """Wait in short intervals so client disconnects can cancel pre-stream LLM work."""
        while True:
            if is_disconnected is not None and await is_disconnected():
                await self._task_manager.cancel(task_id)
            if self._task_manager.is_cancelled(task_id):
                task.cancel()
            try:
                return await asyncio.wait_for(asyncio.shield(task), timeout=0.15)
            except TimeoutError:
                continue


class _AsyncTaskCancellationHandle:
    """Adapts a pre-stream asyncio task to the same cancellation protocol as an LLM stream."""

    def __init__(self, task: asyncio.Task[ChatPreparation]) -> None:
        self._task = task

    def cancel(self) -> None:
        if not self._task.done():
            self._task.cancel()


class _QueueStreamCallback:
    def __init__(self, queue: asyncio.Queue[StreamEvent]) -> None:
        self._queue = queue

    def on_thinking(self, content: str) -> None:
        self._queue.put_nowait(StreamEvent("thinking", {"delta": content}))

    def on_content(self, content: str) -> None:
        self._queue.put_nowait(StreamEvent("content", {"delta": content}))

    def on_complete(self) -> None:
        self._queue.put_nowait(StreamEvent("complete", {}))

    def on_error(self, error: Exception) -> None:
        self._queue.put_nowait(StreamEvent("error", {"error": error}))


def _encode_event(name: str, data: dict[str, Any]) -> bytes:
    body = json.dumps(data, ensure_ascii=False, default=str, separators=(",", ":"))
    return f"event: {name}\ndata: {body}\n\n".encode()


def get_stream_chat_service() -> StreamChatService:
    return StreamChatService()
