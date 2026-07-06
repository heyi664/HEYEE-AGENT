from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Awaitable
from typing import cast
from uuid import uuid4

from agent_service.core.config import get_settings
from agent_service.memory.conversation_memory_service import ConversationMemoryService
from agent_service.memory.models import MemoryContext, MemoryMessage
from agent_service.schemas.chat import ChatRequest, ChatResponse
from agent_service.services.llm_service import LLMService, get_llm_service
from agent_service.services.prompt_service import build_messages

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        llm_service: LLMService,
        memory_service: ConversationMemoryService | object | None = None,
    ) -> None:
        self._llm_service = llm_service
        self._memory_service = memory_service

    async def chat(self, request: ChatRequest) -> ChatResponse:
        settings = get_settings()
        started_at = time.perf_counter()
        conversation_id = request.conversationId or f"conv_{uuid4().hex[:12]}"
        user_id = str(request.userId or 0)

        memory_service = self._resolve_memory_service(
            settings.database_url,
            settings.agent_mock_mode,
        )
        if settings.memory_enabled and memory_service is not None:
            memory_context = memory_service.load_and_append(
                conversation_id,
                user_id,
                request.message,
            )
        else:
            memory_context = self._context_from_request_history(request)

        messages = build_messages(memory_context, request.message)
        result = await self._llm_service.complete(messages)

        if settings.memory_enabled and memory_service is not None:
            memory_service.append(conversation_id, user_id, "assistant", result.reply)
            compression_result = memory_service.compress_if_needed(conversation_id, user_id)
            await self._handle_compression_result(
                compression_result,
                async_compress=settings.memory_async_compress,
            )

        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        logger.info(
            "chat completed conversationId=%s userId=%s elapsedMs=%s",
            conversation_id,
            request.userId,
            elapsed_ms,
        )
        return ChatResponse(
            conversationId=conversation_id,
            reply=result.reply,
            sources=[],
            toolCalls=result.tool_calls,
        )

    def _resolve_memory_service(
        self,
        database_url: str | None,
        agent_mock_mode: bool,
    ) -> ConversationMemoryService | object | None:
        if self._memory_service is not None:
            return self._memory_service
        if agent_mock_mode or not database_url:
            return None
        self._memory_service = ConversationMemoryService()
        return self._memory_service

    def _context_from_request_history(self, request: ChatRequest) -> MemoryContext:
        messages = [
            MemoryMessage(id=f"request_{index}", role=item.role, content=item.content)
            for index, item in enumerate(request.history[-10:])
        ]
        return MemoryContext(messages=messages)

    async def _handle_compression_result(
        self,
        compression_result: object,
        *,
        async_compress: bool,
    ) -> None:
        if not inspect.isawaitable(compression_result):
            return

        awaitable = cast(Awaitable[object], compression_result)
        if async_compress:
            asyncio.create_task(self._log_compression_failure(awaitable))
            return
        await self._log_compression_failure(awaitable)

    async def _log_compression_failure(self, awaitable: Awaitable[object]) -> None:
        try:
            await awaitable
        except Exception:
            logger.exception("conversation memory async compression failed")


def get_chat_service() -> ChatService:
    return ChatService(get_llm_service())
