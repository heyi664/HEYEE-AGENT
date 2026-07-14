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
from agent_service.services.stream_task_manager import StreamTaskManager, stream_task_manager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StreamEvent:
    name: str
    data: dict[str, Any]


class StreamChatService:
    """Owns the answer-stream lifecycle from prepared prompt to SSE terminal events."""

    def __init__(
        self,
        chat_service: ChatService | None = None,
        llm_service: StreamLLMService | None = None,
        task_manager: StreamTaskManager | None = None,
    ) -> None:
        self._chat_service = chat_service or get_chat_service()
        self._llm_service = llm_service or StreamLLMService()
        self._task_manager = task_manager or stream_task_manager

    async def stream(
        self,
        request: ChatRequest,
        *,
        task_id: str | None = None,
        is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    ) -> AsyncIterator[bytes]:
        task_id = task_id or f"task_{uuid4().hex[:16]}"
        try:
            preparation = await self._chat_service.prepare(request)
        except Exception as exc:
            logger.exception("stream chat preparation failed taskId=%s", task_id)
            yield _encode_event(
                "error",
                {"message": "AI stream preparation failed", "detail": str(exc)},
            )
            yield _encode_event("done", {"taskId": task_id})
            return
        yield _encode_event(
            "meta",
            {
                "taskId": task_id,
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
        handle = None
        response_parts: list[str] = []
        terminal = False
        try:
            handle = await self._llm_service.start(
                preparation.messages,
                callback,
                temperature=preparation.prompt_plan.temperature,
                top_p=preparation.prompt_plan.top_p,
                deep_thinking=preparation.deep_thinking,
            )
            if not self._task_manager.bind(task_id, handle):
                handle.cancel()
                yield _encode_event("error", {"message": "taskId is already active"})
                yield _encode_event("done", {"taskId": task_id})
                return

            while True:
                if is_disconnected is not None and await is_disconnected():
                    self._task_manager.cancel(task_id)
                    terminal = True
                    return
                if self._task_manager.is_cancelled(task_id):
                    terminal = True
                    yield _encode_event("cancel", {"taskId": task_id})
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
                    terminal = True
                    self._chat_service.record_answer_failure(
                        preparation,
                        answer_started_at,
                    )
                    yield _encode_event("error", {"message": str(event.data["error"])})
                    yield _encode_event("done", {"taskId": task_id})
                    return
                if event.name == "complete":
                    terminal = True
                    reply = "".join(response_parts).strip()
                    if not reply:
                        self._chat_service.record_answer_failure(
                            preparation,
                            answer_started_at,
                        )
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
        except Exception as exc:
            terminal = True
            logger.exception("stream chat failed taskId=%s", task_id)
            self._chat_service.record_answer_failure(preparation, answer_started_at)
            yield _encode_event(
                "error",
                {"message": "AI stream is unavailable", "detail": str(exc)},
            )
            yield _encode_event("done", {"taskId": task_id})
        finally:
            if handle is not None and not terminal:
                handle.cancel()
            self._task_manager.unbind(task_id)

    def cancel(self, task_id: str) -> bool:
        return self._task_manager.cancel(task_id)

    def _sources(self, preparation: ChatPreparation) -> list[Any]:
        return [self._chat_service._to_chat_source(item) for item in preparation.retrieved_sources]


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
