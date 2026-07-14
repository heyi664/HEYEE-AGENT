from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest

from agent_service.schemas.chat import ChatRequest, ChatResponse
from agent_service.services.chat_service import ChatPreparation
from agent_service.services.prompt_service import PromptBuildPlan, PromptScene
from agent_service.services.stream_chat_service import StreamChatService
from agent_service.services.stream_task_manager import StreamTaskManager


class FakeChatService:
    def __init__(self) -> None:
        self.failures = 0
        self.completed: list[str] = []

    async def prepare(self, request: ChatRequest) -> ChatPreparation:
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
    ) -> ChatResponse:
        self.completed.append(reply)
        return ChatResponse(
            conversationId=preparation.conversation_id,
            messageId="msg_stream",
            reply=reply,
            sources=[],
            toolCalls=tool_calls,
            ragIntent=None,
        )

    def _to_chat_source(self, source: object) -> object:
        return source


class FakeHandle:
    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


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
        "message",
        "message",
        "message",
        "finish",
        "done",
    ]
    assert events[0][1]["conversationId"] == "conv_stream"
    assert events[1][1] == {"type": "think", "delta": "analyzing"}
    assert events[2][1] == {"type": "response", "delta": "Hello"}
    assert events[4][1]["messageId"] == "msg_stream"
    assert chat.completed == ["Hello world"]


@pytest.mark.asyncio
async def test_stream_chat_cancellation_emits_cancel_and_does_not_persist_reply() -> None:
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
    next_chunk = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    assert service.cancel("task_cancel") is True
    remaining = [await next_chunk, *[chunk async for chunk in stream]]

    assert [name for name, _ in _parse_events(remaining)] == ["cancel", "done"]
    assert chat.completed == []
    assert manager.contains("task_cancel") is False


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

    assert [name for name, _ in events] == ["meta", "error", "done"]
    assert "provider failed" in events[1][1]["message"]
    assert chat.failures == 1
    assert chat.completed == []
