from __future__ import annotations

import asyncio

import pytest

from agent_service.memory.models import MemoryContext, MemoryMessage
from agent_service.schemas.chat import ChatRequest
from agent_service.services.chat_service import ChatService
from agent_service.services.llm_service import LLMResult


class FakeLLMService:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    async def complete(self, messages: list[dict[str, str]]) -> LLMResult:
        self.messages = messages
        return LLMResult(reply="assistant reply")


class FakeMemoryService:
    def __init__(self) -> None:
        self.appended: list[tuple[str, str, str, str]] = []
        self.compressed: list[tuple[str, str]] = []

    def load_and_append(self, conversation_id: str, user_id: str, content: str) -> MemoryContext:
        self.appended.append((conversation_id, user_id, "user", content))
        return MemoryContext(
            messages=[MemoryMessage(id="msg_1", role="user", content="history question")]
        )

    def append(self, conversation_id: str, user_id: str, role: str, content: str) -> str:
        self.appended.append((conversation_id, user_id, role, content))
        return "msg_2"

    def compress_if_needed(self, conversation_id: str, user_id: str) -> None:
        self.compressed.append((conversation_id, user_id))


@pytest.mark.asyncio
async def test_chat_service_loads_memory_and_appends_assistant_reply() -> None:
    llm_service = FakeLLMService()
    memory_service = FakeMemoryService()
    service = ChatService(llm_service=llm_service, memory_service=memory_service)

    response = await service.chat(
        ChatRequest(
            userId=7,
            conversationId="conv_1",
            message="current question",
            history=[],
        )
    )

    assert response.reply == "assistant reply"
    assert memory_service.appended == [
        ("conv_1", "7", "user", "current question"),
        ("conv_1", "7", "assistant", "assistant reply"),
    ]
    assert memory_service.compressed == [("conv_1", "7")]
    assert llm_service.messages[-1] == {"role": "user", "content": "current question"}

class SlowAsyncMemoryService(FakeMemoryService):
    def __init__(self) -> None:
        super().__init__()
        self.compression_started = asyncio.Event()
        self.compression_released = asyncio.Event()

    async def compress_if_needed(self, conversation_id: str, user_id: str) -> None:
        self.compressed.append((conversation_id, user_id))
        self.compression_started.set()
        await self.compression_released.wait()


@pytest.mark.asyncio
async def test_chat_service_does_not_wait_for_async_memory_compression() -> None:
    llm_service = FakeLLMService()
    memory_service = SlowAsyncMemoryService()
    service = ChatService(llm_service=llm_service, memory_service=memory_service)

    response = await asyncio.wait_for(
        service.chat(
            ChatRequest(
                userId=7,
                conversationId="conv_1",
                message="current question",
                history=[],
            )
        ),
        timeout=0.1,
    )

    assert response.reply == "assistant reply"
    await asyncio.wait_for(memory_service.compression_started.wait(), timeout=0.1)
    memory_service.compression_released.set()

