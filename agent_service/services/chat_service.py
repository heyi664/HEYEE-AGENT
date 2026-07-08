from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Coroutine
from typing import Any, cast
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
        intent_recognition_pipeline: object | None = None,
    ) -> None:
        self._llm_service = llm_service
        self._memory_service: Any = memory_service
        self._intent_recognition_pipeline: Any = intent_recognition_pipeline

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

        rag_intent = None
        if settings.rag_intent_enabled:
            rag_intent = await self._recognize_intent(request.message, memory_context.messages)

        messages = build_messages(memory_context, request.message)
        result = await self._llm_service.complete(messages)

        if settings.memory_enabled and memory_service is not None:
            memory_service.append(conversation_id, user_id, "assistant", result.reply)
            compression_result = memory_service.compress_if_needed(conversation_id, user_id)
            if inspect.isawaitable(compression_result):
                if settings.memory_async_compress:
                    self._run_background_compression(compression_result)
                else:
                    await compression_result

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
            ragIntent=rag_intent,
        )

    async def _recognize_intent(
        self,
        question: str,
        history: list[MemoryMessage],
    ) -> dict[str, Any] | None:
        pipeline = self._resolve_intent_recognition_pipeline()
        if pipeline is None:
            return None
        try:
            result = await pipeline.recognize(question, history=history)
        except Exception:
            logger.exception("rag intent recognition failed; continue normal chat")
            return None
        to_response_dict = getattr(result, "to_response_dict", None)
        if callable(to_response_dict):
            return cast(dict[str, Any], to_response_dict())
        return cast(dict[str, Any] | None, result)

    def _resolve_intent_recognition_pipeline(self) -> Any | None:
        if self._intent_recognition_pipeline is not None:
            return self._intent_recognition_pipeline
        try:
            from agent_service.rag.intent_recognition_pipeline import (
                get_intent_recognition_pipeline,
            )

            self._intent_recognition_pipeline = get_intent_recognition_pipeline()
        except Exception:
            logger.exception("rag intent recognition pipeline initialization failed")
            return None
        return self._intent_recognition_pipeline

    def _resolve_memory_service(
        self,
        database_url: str | None,
        agent_mock_mode: bool,
    ) -> Any | None:
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

    def _run_background_compression(self, compression_result: object) -> None:
        coroutine = cast(Coroutine[Any, Any, object], compression_result)
        task: asyncio.Task[object] = asyncio.create_task(coroutine)
        task.add_done_callback(self._log_background_compression_result)

    def _log_background_compression_result(self, task: asyncio.Task[object]) -> None:
        try:
            task.result()
        except Exception:
            logger.exception("background conversation summary compression failed")


def get_chat_service() -> ChatService:
    return ChatService(get_llm_service())
