from __future__ import annotations

import logging
from uuid import uuid4

from agent_service.core.config import get_settings
from agent_service.memory.conversation_summary_service import ConversationSummaryService
from agent_service.memory.models import MemoryContext, MemoryMessage
from agent_service.memory.summary_lock import RedisSummaryCompressionLock, SummaryCompressionLock
from agent_service.repositories.conversation_memory_repository import (
    ConversationMemoryRepository,
)
from agent_service.services.token_counter_service import (
    HeuristicTokenCounterService,
    TokenCounterService,
)

logger = logging.getLogger(__name__)


class ConversationMemoryService:
    def __init__(
        self,
        repository: ConversationMemoryRepository | object | None = None,
        summary_service: ConversationSummaryService | object | None = None,
        history_keep_turns: int | None = None,
        summary_enabled: bool | None = None,
        summary_batch_size: int | None = None,
        token_counter: TokenCounterService | None = None,
        max_prompt_tokens: int | None = None,
        summary_lock: SummaryCompressionLock | None = None,
    ) -> None:
        settings = get_settings()
        self._repository = repository or ConversationMemoryRepository()
        self._summary_service = summary_service or ConversationSummaryService(
            max_chars=settings.memory_summary_max_chars
        )
        self._history_keep_turns = history_keep_turns or settings.memory_history_keep_turns
        self._summary_enabled = (
            settings.memory_summary_enabled if summary_enabled is None else summary_enabled
        )
        self._summary_batch_size = summary_batch_size or settings.memory_summary_batch_size
        self._token_counter = token_counter or HeuristicTokenCounterService()
        self._max_prompt_tokens = max_prompt_tokens or settings.memory_max_prompt_tokens
        self._summary_lock = summary_lock or self._build_summary_lock(settings)

    def load(self, conversation_id: str, user_id: str) -> MemoryContext:
        if not self._summary_enabled:
            messages = self._repository.list_recent_turn_messages(
                conversation_id,
                user_id,
                self._history_keep_turns,
            )
            normalized_messages = self.normalize_history(messages)
            return MemoryContext(
                messages=self._trim_to_token_budget(None, normalized_messages)
            )

        summary = self._repository.get_latest_summary(conversation_id, user_id)
        messages = self._repository.list_messages_after(
            conversation_id,
            user_id,
            summary.last_message_id if summary else None,
        )
        normalized_messages = self.normalize_history(messages)
        return MemoryContext(
            summary=summary,
            messages=self._trim_to_token_budget(
                summary.content if summary else None,
                normalized_messages,
            ),
        )

    def load_and_append(
        self,
        conversation_id: str,
        user_id: str,
        content: str,
    ) -> MemoryContext:
        context = self.load(conversation_id, user_id)
        self.append(conversation_id, user_id, "user", content)
        return context

    def append(self, conversation_id: str, user_id: str, role: str, content: str) -> str:
        message_id = self._new_id("msg")
        self._repository.create_or_update_conversation(
            row_id=self._new_id("convrow"),
            conversation_id=conversation_id,
            user_id=user_id,
            title=content[:128] or "new conversation",
        )
        self._repository.append_message(
            message_id=message_id,
            conversation_id=conversation_id,
            user_id=user_id,
            role=role,
            content=content,
        )
        return message_id

    async def compress_if_needed(self, conversation_id: str, user_id: str) -> None:
        if not self._summary_enabled:
            return

        lock_token = self._acquire_summary_lock(conversation_id, user_id)
        if self._summary_lock is not None and lock_token is None:
            logger.info(
                "conversation summary compression skipped because lock is held "
                "conversationId=%s userId=%s",
                conversation_id,
                user_id,
            )
            return

        try:
            summary = self._repository.get_latest_summary(conversation_id, user_id)
            pending_messages = self._repository.list_messages_after(
                conversation_id,
                user_id,
                summary.last_message_id if summary else None,
            )
            pending_messages = self.normalize_history(pending_messages)
            if not pending_messages or not self.should_compress(pending_messages):
                return

            content = await self._summary_service.summarize(
                existing_summary=summary.content if summary else None,
                pending_messages=pending_messages,
            )
            if not content:
                return
            self._repository.upsert_summary(
                summary_id=self._new_id("sum"),
                conversation_id=conversation_id,
                user_id=user_id,
                last_message_id=pending_messages[-1].id,
                content=content,
            )
        except Exception:
            logger.exception(
                "conversation summary compression failed conversationId=%s userId=%s",
                conversation_id,
                user_id,
            )
        finally:
            if self._summary_lock is not None and lock_token is not None:
                self._summary_lock.release(lock_token)

    def normalize_history(self, messages: list[MemoryMessage]) -> list[MemoryMessage]:
        start = 0
        while start < len(messages) and messages[start].role == "assistant":
            start += 1
        return messages[start:]

    def count_pending_user_turns(self, messages: list[MemoryMessage]) -> int:
        return sum(1 for message in messages if message.role == "user")

    def should_compress(self, messages: list[MemoryMessage]) -> bool:
        return self.count_pending_user_turns(messages) >= self._summary_batch_size

    def _trim_to_token_budget(
        self,
        summary_content: str | None,
        messages: list[MemoryMessage],
    ) -> list[MemoryMessage]:
        if not messages:
            return messages

        remaining = self._max_prompt_tokens - self._token_counter.count_tokens(summary_content)
        if remaining <= 0:
            return messages[-1:]

        kept: list[MemoryMessage] = []
        used = 0
        for message in reversed(messages):
            message_tokens = self._token_counter.count_tokens(message.content)
            if kept and used + message_tokens > remaining:
                break
            kept.append(message)
            used += message_tokens
            if used >= remaining:
                break
        return self.normalize_history(list(reversed(kept)))

    def _build_summary_lock(self, settings: object) -> SummaryCompressionLock | None:
        if not getattr(settings, "memory_summary_lock_enabled", False):
            return None
        redis_url = getattr(settings, "memory_summary_lock_redis_url", None)
        if not redis_url:
            logger.warning("memory summary lock enabled but redis url is empty; lock disabled")
            return None
        return RedisSummaryCompressionLock(
            redis_url,
            ttl_seconds=getattr(settings, "memory_summary_lock_ttl_seconds", 120),
        )

    def _acquire_summary_lock(self, conversation_id: str, user_id: str) -> str | None:
        if self._summary_lock is None:
            return None
        return self._summary_lock.acquire(conversation_id, user_id)

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}_{uuid4().hex[:14]}"[:20]
