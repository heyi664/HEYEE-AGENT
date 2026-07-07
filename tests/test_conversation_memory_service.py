from __future__ import annotations

import pytest

from agent_service.core.config import Settings
from agent_service.memory.conversation_memory_service import ConversationMemoryService
from agent_service.memory.models import MemoryContext, MemoryMessage, MemorySummary


def test_memory_settings_defaults() -> None:
    settings = Settings()

    assert settings.memory_enabled is True
    assert settings.memory_history_keep_turns == 8
    assert settings.memory_summary_enabled is False
    assert settings.memory_summary_batch_size == 3
    assert settings.memory_summary_max_chars == 300
    assert settings.memory_async_compress is True


def test_memory_context_to_prompt_messages_injects_summary_first() -> None:
    context = MemoryContext(
        summary=MemorySummary(
            id="sum_1",
            conversation_id="conv_1",
            user_id="user_1",
            last_message_id="msg_1",
            content="budget constraints.",
        ),
        messages=[
            MemoryMessage(id="msg_2", role="user", content="what is my budget?"),
            MemoryMessage(id="msg_3", role="assistant", content="your budget is 5000."),
        ],
    )

    messages = context.to_prompt_messages()

    assert messages[0]["role"] == "system"
    assert messages[0]["content"].endswith("budget constraints.")
    assert messages[1:] == [
        {"role": "user", "content": "what is my budget?"},
        {"role": "assistant", "content": "your budget is 5000."},
    ]


class FakeMemoryRepository:
    def __init__(self) -> None:
        self.summary = MemorySummary(
            id="sum_1",
            conversation_id="conv_1",
            user_id="user_1",
            last_message_id="msg_1",
            content="early summary",
        )
        self.messages_after = [
            MemoryMessage(id="msg_2", role="user", content="important budget is 5000"),
            MemoryMessage(id="msg_3", role="assistant", content="noted"),
            MemoryMessage(id="msg_4", role="user", content="recent question"),
            MemoryMessage(id="msg_5", role="assistant", content="recent answer"),
        ]
        self.recent_messages = [
            MemoryMessage(id="msg_4", role="user", content="recent question"),
            MemoryMessage(id="msg_5", role="assistant", content="recent answer"),
        ]

    def get_latest_summary(self, conversation_id: str, user_id: str) -> MemorySummary | None:
        return self.summary

    def list_messages_after(
        self,
        conversation_id: str,
        user_id: str,
        after_message_id: str | None,
    ) -> list[MemoryMessage]:
        return self.messages_after

    def list_recent_turn_messages(
        self,
        conversation_id: str,
        user_id: str,
        keep_turns: int,
    ) -> list[MemoryMessage]:
        return self.recent_messages


def test_load_keeps_pending_messages_after_summary_watermark_when_summary_enabled() -> None:
    service = ConversationMemoryService(
        repository=FakeMemoryRepository(),
        history_keep_turns=1,
        summary_enabled=True,
    )

    context = service.load("conv_1", "user_1")

    assert context.summary is not None
    assert [message.content for message in context.messages] == [
        "important budget is 5000",
        "noted",
        "recent question",
        "recent answer",
    ]


def test_load_uses_recent_window_when_summary_disabled() -> None:
    service = ConversationMemoryService(
        repository=FakeMemoryRepository(),
        history_keep_turns=1,
        summary_enabled=False,
    )

    context = service.load("conv_1", "user_1")

    assert context.summary is None
    assert [message.content for message in context.messages] == ["recent question", "recent answer"]


def test_normalize_history_removes_leading_assistant_messages() -> None:
    service = ConversationMemoryService(repository=FakeMemoryRepository())

    messages = service.normalize_history(
        [
            MemoryMessage(id="msg_1", role="assistant", content="orphan"),
            MemoryMessage(id="msg_2", role="user", content="user"),
            MemoryMessage(id="msg_3", role="assistant", content="assistant"),
        ]
    )

    assert [message.content for message in messages] == ["user", "assistant"]


def test_pending_user_turn_count_counts_only_user_messages() -> None:
    service = ConversationMemoryService(repository=FakeMemoryRepository())

    count = service.count_pending_user_turns(
        [
            MemoryMessage(id="msg_2", role="user", content="u1"),
            MemoryMessage(id="msg_3", role="assistant", content="a1"),
            MemoryMessage(id="msg_4", role="user", content="u2"),
        ]
    )

    assert count == 2


class FakeSummaryService:
    def __init__(self) -> None:
        self.calls: list[tuple[str | None, list[MemoryMessage]]] = []

    async def summarize(
        self,
        *,
        existing_summary: str | None,
        pending_messages: list[MemoryMessage],
    ) -> str:
        self.calls.append((existing_summary, pending_messages))
        return "merged summary"


class CompressRepository(FakeMemoryRepository):
    def __init__(self) -> None:
        super().__init__()
        self.upserts: list[tuple[str, str, str]] = []

    def upsert_summary(
        self,
        *,
        summary_id: str,
        conversation_id: str,
        user_id: str,
        last_message_id: str,
        content: str,
    ) -> None:
        self.upserts.append((conversation_id, last_message_id, content))


@pytest.mark.asyncio
async def test_compress_if_needed_merges_pending_and_updates_watermark() -> None:
    repository = CompressRepository()
    summary_service = FakeSummaryService()
    service = ConversationMemoryService(
        repository=repository,
        summary_service=summary_service,
        summary_enabled=True,
        summary_batch_size=2,
    )

    await service.compress_if_needed("conv_1", "user_1")

    assert summary_service.calls[0][0] == "early summary"
    assert [message.id for message in summary_service.calls[0][1]] == [
        "msg_2",
        "msg_3",
        "msg_4",
        "msg_5",
    ]
    assert repository.upserts == [("conv_1", "msg_5", "merged summary")]

class DenyingCompressionLock:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def acquire(self, conversation_id: str, user_id: str) -> bool:
        self.calls.append((conversation_id, user_id))
        return False

    def release(self, conversation_id: str, user_id: str) -> None:
        raise AssertionError("release should not be called when acquire fails")


@pytest.mark.asyncio
async def test_compress_if_needed_skips_when_compression_lock_is_not_acquired() -> None:
    repository = CompressRepository()
    summary_service = FakeSummaryService()
    lock = DenyingCompressionLock()
    service = ConversationMemoryService(
        repository=repository,
        summary_service=summary_service,
        summary_enabled=True,
        summary_batch_size=2,
        compression_lock=lock,
    )

    await service.compress_if_needed("conv_1", "user_1")

    assert lock.calls == [("conv_1", "user_1")]
    assert summary_service.calls == []
    assert repository.upserts == []


def test_load_trims_context_to_budget_and_keeps_user_start() -> None:
    repository = FakeMemoryRepository()
    repository.messages_after = [
        MemoryMessage(id="msg_1", role="assistant", content="orphan assistant"),
        MemoryMessage(id="msg_2", role="user", content="first important user message"),
        MemoryMessage(id="msg_3", role="assistant", content="assistant answer that can be trimmed"),
        MemoryMessage(id="msg_4", role="user", content="latest user message"),
    ]
    service = ConversationMemoryService(
        repository=repository,
        summary_enabled=True,
        max_context_chars=35,
    )

    context = service.load("conv_1", "user_1")

    assert context.messages[0].role == "user"
    assert [message.content for message in context.messages] == ["latest user message"]

