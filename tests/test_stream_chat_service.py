from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from agent_service.schemas.chat import ChatRequest, ChatResponse
from agent_service.services.chat_service import ChatPreparation
from agent_service.services.prompt_service import PromptBuildPlan, PromptScene
from agent_service.services.stream_chat_service import StreamChatService
from agent_service.services.stream_queue_limiter import StreamQueueLimiter
from agent_service.services.stream_task_manager import StreamTaskManager


class FakeChatService:
    def __init__(self) -> None:
        self.failures = 0
        self.completed: list[tuple[str, bool]] = []
        self.queue_rejections: list[tuple[str, str]] = []
        self.prepare_calls = 0

    async def prepare(self, request: ChatRequest) -> ChatPreparation:
        self.prepare_calls += 1
        return ChatPreparation(
            started_at=time.perf_counter(),
            settings=object(),
            conversation_id="conv_stream",
            user_id="0",
            deep_thinking=False,
            memory_service=None,
            memory_enabled=False,
            rag_intent=None,
            retrieved_sources=[],
            mcp_result=type("McpResult", (), {"tool_calls": []})(),
            messages=[{"role": "user", "content": request.message}],
            prompt_plan=PromptBuildPlan(
                scene=PromptScene.EMPTY,
                system_prompt="system",
                user_content=request.message,
                temperature=0.7,
                top_p=None,
            ),
        )

    def record_answer_failure(self, preparation: ChatPreparation, answer_started_at: float) -> None:
        self.failures += 1

    async def complete_preparation(
        self,
        preparation: ChatPreparation,
        *,
        reply: str,
        tool_calls: list[str],
        answer_started_at: float,
        interrupted: bool = False,
    ) -> ChatResponse:
        self.completed.append((reply, interrupted))
        return ChatResponse(
            conversationId=preparation.conversation_id,
            messageId="msg_stream",
            interrupted=interrupted,
            reply=reply,
            sources=[],
            toolCalls=tool_calls,
            ragIntent=None,
        )

    def _to_chat_source(self, source: object) -> object:
        return source

    async def record_queue_rejection(
        self,
        request: ChatRequest,
        *,
        reply: str,
    ) -> SimpleNamespace:
        self.queue_rejections.append((request.message, reply))
        return SimpleNamespace(
            conversation_id=request.conversationId or "conv_rejected",
            message_id="msg_rejected",
            title=request.message[:128],
        )


class BlockingPreparationChatService(FakeChatService):
    def __init__(self) -> None:
        super().__init__()
        self.prepare_started = asyncio.Event()
        self.prepare_cancelled = False

    async def prepare(self, request: ChatRequest) -> ChatPreparation:
        self.prepare_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.prepare_cancelled = True
            raise
        raise AssertionError("blocking preparation unexpectedly completed")


class FailingInterruptedPersistenceChatService(FakeChatService):
    async def complete_preparation(
        self,
        preparation: ChatPreparation,
        *,
        reply: str,
        tool_calls: list[str],
        answer_started_at: float,
        interrupted: bool = False,
    ) -> ChatResponse:
        if interrupted:
            raise RuntimeError("database unavailable")
        return await super().complete_preparation(
            preparation,
            reply=reply,
            tool_calls=tool_calls,
            answer_started_at=answer_started_at,
            interrupted=interrupted,
        )


class FakeHandle:
    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class FakeDistributedCancellationBackend:
    def __init__(self) -> None:
        self.cancelled: set[str] = set()
        self.operations: list[str] = []
        self.on_cancel: Any = None

    async def start(self, on_cancel: Any) -> bool:
        self.on_cancel = on_cancel
        return True

    async def mark_cancelled(self, task_id: str) -> bool:
        self.operations.append(f"mark:{task_id}")
        self.cancelled.add(task_id)
        return True

    async def publish_cancel(self, task_id: str) -> bool:
        self.operations.append(f"publish:{task_id}")
        if self.on_cancel is not None:
            self.on_cancel(task_id)
        return True

    async def is_cancelled(self, task_id: str) -> bool:
        self.operations.append(f"read:{task_id}")
        return task_id in self.cancelled

    async def clear(self, task_id: str) -> None:
        self.operations.append(f"clear:{task_id}")
        self.cancelled.discard(task_id)

    async def close(self) -> None:
        return None


class FakeStreamLLMService:
    def __init__(self, events: list[tuple[str, Any]]) -> None:
        self._events = events
        self.handle = FakeHandle()

    async def start(
        self,
        messages: list[dict[str, str]],
        callback: Any,
        **kwargs: object,
    ) -> FakeHandle:
        for name, value in self._events:
            method = getattr(callback, f"on_{name}")
            method(value) if value else method()
        return self.handle


def _parse_events(chunks: list[bytes]) -> list[tuple[str, dict[str, Any]]]:
    parsed: list[tuple[str, dict[str, Any]]] = []
    for chunk in chunks:
        text = chunk.decode()
        event_name = next(line[7:] for line in text.splitlines() if line.startswith("event: "))
        data = next(line[6:] for line in text.splitlines() if line.startswith("data: "))
        parsed.append((event_name, json.loads(data)))
    return parsed


@pytest.mark.asyncio
async def test_stream_chat_emits_meta_deltas_finish_and_done() -> None:
    chat = FakeChatService()
    service = StreamChatService(
        chat_service=chat,
        llm_service=FakeStreamLLMService(
            [
                ("thinking", "analyzing"),
                ("content", "Hello"),
                ("content", " world"),
                ("complete", ""),
            ]
        ),
        task_manager=StreamTaskManager(),
    )

    chunks = [
        chunk
        async for chunk in service.stream(ChatRequest(message="hello"), task_id="task_1")
    ]
    events = _parse_events(chunks)

    assert [name for name, _ in events] == [
        "meta",
        "meta",
        "meta",
        "message",
        "message",
        "message",
        "finish",
        "done",
    ]
    assert events[0][1]["taskId"] == "task_1"
    assert events[0][1]["phase"] == "queued"
    assert events[0][1]["traceId"].startswith("trace_")
    assert events[1][1] == {"taskId": "task_1", "phase": "preparing"}
    assert events[2][1]["conversationId"] == "conv_stream"
    assert events[3][1] == {"type": "think", "delta": "analyzing"}
    assert events[4][1] == {"type": "response", "delta": "Hello"}
    assert events[6][1]["messageId"] == "msg_stream"
    assert chat.completed == [("Hello world", False)]


@pytest.mark.asyncio
async def test_stream_chat_cancellation_before_preparation_emits_cancel() -> None:
    manager = StreamTaskManager()
    chat = FakeChatService()
    service = StreamChatService(
        chat_service=chat,
        llm_service=FakeStreamLLMService([]),
        task_manager=manager,
    )
    stream: AsyncIterator[bytes] = service.stream(
        ChatRequest(message="hello"),
        task_id="task_cancel",
    )

    meta = await anext(stream)
    assert _parse_events([meta])[0][0] == "meta"
    assert await service.cancel("task_cancel") is True
    remaining = [chunk async for chunk in stream]

    assert [name for name, _ in _parse_events(remaining)] == ["cancel", "done"]
    assert chat.completed == []
    assert manager.contains("task_cancel") is True
    assert manager.is_cancelled("task_cancel") is True


@pytest.mark.asyncio
async def test_stream_chat_times_out_before_preparation_when_queue_is_full() -> None:
    limiter = StreamQueueLimiter(
        max_concurrent=1,
        max_wait_seconds=0.03,
        poll_interval_seconds=0.01,
    )

    async def not_cancelled() -> bool:
        return False

    held = await limiter.acquire("occupying-task", should_cancel=not_cancelled)
    chat = FakeChatService()
    service = StreamChatService(
        chat_service=chat,
        llm_service=FakeStreamLLMService([]),
        task_manager=StreamTaskManager(),
        queue_limiter=limiter,
    )

    events = _parse_events(
        [
            chunk
            async for chunk in service.stream(
                ChatRequest(message="hello"),
                task_id="timed-out-task",
            )
        ]
    )

    assert [name for name, _ in events] == ["meta", "reject", "finish", "done"]
    assert events[0][1]["phase"] == "queued"
    assert events[1][1]["code"] == "queue_timeout"
    assert events[1][1]["delta"] == "系统繁忙，请稍后重试。"
    assert events[2][1]["messageId"] == "msg_rejected"
    assert events[2][1]["rejected"] is True
    assert chat.prepare_calls == 0
    assert chat.queue_rejections == [("hello", "系统繁忙，请稍后重试。")]
    await limiter.release(held.permit)


@pytest.mark.asyncio
async def test_stream_chat_cancellation_interrupts_preparation_task() -> None:
    manager = StreamTaskManager()
    chat = BlockingPreparationChatService()
    service = StreamChatService(
        chat_service=chat,
        llm_service=FakeStreamLLMService([]),
        task_manager=manager,
    )
    stream = service.stream(ChatRequest(message="hello"), task_id="task_preparing")

    assert _parse_events([await anext(stream)])[0][1]["phase"] == "queued"
    assert _parse_events([await anext(stream)])[0][1]["phase"] == "preparing"
    next_event = asyncio.create_task(anext(stream))
    await asyncio.wait_for(chat.prepare_started.wait(), timeout=1)

    assert await service.cancel("task_preparing") is True
    cancel_event = _parse_events([await next_event])[0]
    done_event = _parse_events([await anext(stream)])[0]

    assert cancel_event[0] == "cancel"
    assert done_event[0] == "done"
    assert chat.prepare_cancelled is True
    assert chat.completed == []


@pytest.mark.asyncio
async def test_stream_chat_cancellation_persists_non_empty_partial_reply() -> None:
    manager = StreamTaskManager()
    chat = FakeChatService()
    service = StreamChatService(
        chat_service=chat,
        llm_service=FakeStreamLLMService([("content", "partial answer")]),
        task_manager=manager,
    )
    stream = service.stream(ChatRequest(message="hello"), task_id="task_partial")

    assert _parse_events([await anext(stream)])[0][0] == "meta"
    assert _parse_events([await anext(stream)])[0][0] == "meta"
    assert _parse_events([await anext(stream)])[0][0] == "meta"
    message = await anext(stream)
    assert _parse_events([message])[0][1]["delta"] == "partial answer"

    assert await service.cancel("task_partial") is True
    remaining = _parse_events([chunk async for chunk in stream])

    assert [name for name, _ in remaining] == ["cancel", "done"]
    assert remaining[0][1]["partial"] is True
    assert remaining[0][1]["messageId"] == "msg_stream"
    assert chat.completed == [("partial answer", True)]


@pytest.mark.asyncio
async def test_stream_chat_cancellation_still_finishes_when_partial_persistence_fails() -> None:
    manager = StreamTaskManager()
    chat = FailingInterruptedPersistenceChatService()
    service = StreamChatService(
        chat_service=chat,
        llm_service=FakeStreamLLMService([("content", "partial answer")]),
        task_manager=manager,
    )
    stream = service.stream(ChatRequest(message="hello"), task_id="task_partial_failure")

    await anext(stream)
    await anext(stream)
    await anext(stream)
    await anext(stream)
    assert await service.cancel("task_partial_failure") is True
    remaining = _parse_events([chunk async for chunk in stream])

    assert [name for name, _ in remaining] == ["cancel", "done"]
    assert remaining[0][1]["partial"] is False
    assert remaining[0][1]["messageId"] is None


@pytest.mark.asyncio
async def test_task_manager_uses_persistent_marker_to_close_cancel_before_register_race() -> None:
    backend = FakeDistributedCancellationBackend()
    manager = StreamTaskManager(backend=backend)
    await manager.start()

    # This simulates a stop request landing on node A before node B has registered the task.
    assert await manager.cancel("task_race") is True
    assert backend.operations[:2] == ["mark:task_race", "publish:task_race"]

    assert await manager.register("task_race") is True
    handle = FakeHandle()
    assert manager.bind("task_race", handle) is False
    assert handle.cancelled is True
    assert manager.is_cancelled("task_race") is True

    await manager.finalize("task_race")
    assert manager.contains("task_race") is True


@pytest.mark.asyncio
async def test_stream_chat_emits_error_and_done_when_the_model_stream_fails() -> None:
    chat = FakeChatService()
    service = StreamChatService(
        chat_service=chat,
        llm_service=FakeStreamLLMService([("error", RuntimeError("provider failed"))]),
        task_manager=StreamTaskManager(),
    )

    chunks = [
        chunk
        async for chunk in service.stream(ChatRequest(message="hello"), task_id="task_error")
    ]
    events = _parse_events(chunks)

    assert [name for name, _ in events] == ["meta", "meta", "meta", "error", "done"]
    assert events[3][1]["message"] == "AI stream is unavailable"
    assert chat.failures == 1
    assert chat.completed == []


@pytest.mark.asyncio
async def test_stream_chat_cancels_model_when_delivery_buffer_overflows() -> None:
    chat = FakeChatService()
    llm = FakeStreamLLMService([("content", str(index)) for index in range(6)])
    service = StreamChatService(
        chat_service=chat,
        llm_service=llm,
        task_manager=StreamTaskManager(),
        callback_queue_max_events=2,
    )

    events = _parse_events(
        [chunk async for chunk in service.stream(ChatRequest(message="hello"), task_id="task_full")]
    )

    assert [name for name, _ in events][-2:] == ["error", "done"]
    assert events[-2][1]["code"] == "stream_backpressure"
    assert llm.handle.cancelled is True


@pytest.mark.asyncio
async def test_task_manager_rejects_cancel_from_another_owner() -> None:
    manager = StreamTaskManager()
    assert await manager.register("owned-task", owner_id="user-a") is True

    assert await manager.cancel("owned-task", owner_id="user-b") is False
    assert manager.is_cancelled("owned-task") is False
    assert await manager.cancel("owned-task", owner_id="user-a") is True
