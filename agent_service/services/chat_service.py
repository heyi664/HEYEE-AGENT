from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Coroutine
from typing import Any, cast
from uuid import uuid4

from agent_service.core.config import get_settings
from agent_service.core.observability import bind_conversation, record_stage
from agent_service.memory.conversation_memory_service import ConversationMemoryService
from agent_service.memory.models import MemoryContext, MemoryMessage
from agent_service.rag.schemas import RetrievedSource
from agent_service.schemas.chat import ChatRequest, ChatResponse, ChatSource
from agent_service.services.llm_service import LLMService, get_llm_service
from agent_service.services.prompt_service import build_messages

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        llm_service: LLMService,
        memory_service: ConversationMemoryService | object | None = None,
        intent_recognition_pipeline: object | None = None,
        retrieval_pipeline: object | None = None,
    ) -> None:
        self._llm_service = llm_service
        self._memory_service: Any = memory_service
        self._intent_recognition_pipeline: Any = intent_recognition_pipeline
        self._retrieval_pipeline: Any = retrieval_pipeline

    async def chat(self, request: ChatRequest) -> ChatResponse:
        settings = get_settings()
        started_at = time.perf_counter()
        conversation_id = request.conversationId or f"conv_{uuid4().hex[:12]}"
        user_id = str(request.userId or 0)

        with bind_conversation(conversation_id):
            memory_started_at = time.perf_counter()
            memory_service = self._resolve_memory_service(
                settings.database_url,
                settings.agent_mock_mode,
            )
            if settings.memory_enabled and memory_service is not None:
                try:
                    memory_context = memory_service.load_and_append(
                        conversation_id,
                        user_id,
                        request.message,
                    )
                    memory_mode = "persistent"
                except Exception:
                    record_stage(
                        "memory_context",
                        elapsed_ms=_elapsed_ms(memory_started_at),
                        messageCount=0,
                        mode="persistent",
                        status="failed",
                    )
                    raise
            else:
                memory_context = self._context_from_request_history(request)
                memory_mode = "request_history"
            record_stage(
                "memory_context",
                elapsed_ms=_elapsed_ms(memory_started_at),
                messageCount=len(memory_context.messages),
                mode=memory_mode,
                status="success",
            )

            rag_intent = None
            intent_result = None
            intent_started_at = time.perf_counter()
            if settings.rag_intent_enabled:
                intent_result = await self._run_intent_recognition(
                    request.message,
                    memory_context.messages,
                )
                rag_intent = self._to_rag_intent_response(intent_result)
            record_stage(
                "intent_recognition",
                elapsed_ms=_elapsed_ms(intent_started_at),
                enabled=settings.rag_intent_enabled,
                kbIntentCount=len(getattr(intent_result, "kb_intents", []) or []),
                status=("success" if intent_result is not None else "fallback"),
                subQuestionCount=len(getattr(intent_result, "sub_questions", []) or []),
            )

            retrieval_started_at = time.perf_counter()
            retrieved_sources = await self._retrieve_knowledge(intent_result)
            record_stage(
                "knowledge_retrieval",
                elapsed_ms=_elapsed_ms(retrieval_started_at),
                sourceCount=len(retrieved_sources),
                attempted=self._has_kb_intents(intent_result),
            )

            messages = build_messages(
                memory_context,
                request.message,
                retrieved_sources=retrieved_sources,
                retrieval_attempted=self._has_kb_intents(intent_result),
            )
            answer_started_at = time.perf_counter()
            try:
                result = await self._llm_service.complete(messages, use_tools=False)
            except Exception:
                record_stage(
                    "answer_generation",
                    elapsed_ms=_elapsed_ms(answer_started_at),
                    promptMessageCount=len(messages),
                    replyChars=0,
                    sourceCount=len(retrieved_sources),
                    status="failed",
                    toolCallCount=0,
                )
                raise
            record_stage(
                "answer_generation",
                elapsed_ms=_elapsed_ms(answer_started_at),
                promptMessageCount=len(messages),
                replyChars=len(result.reply),
                sourceCount=len(retrieved_sources),
                status="success",
                toolCallCount=len(result.tool_calls),
            )

            if settings.memory_enabled and memory_service is not None:
                persist_started_at = time.perf_counter()
                memory_service.append(conversation_id, user_id, "assistant", result.reply)
                compression_result = memory_service.compress_if_needed(conversation_id, user_id)
                if inspect.isawaitable(compression_result):
                    if settings.memory_async_compress:
                        self._run_background_compression(compression_result)
                        compression_mode = "async"
                    else:
                        await compression_result
                        compression_mode = "sync"
                else:
                    compression_mode = "not_needed"
                record_stage(
                    "memory_persist",
                    elapsed_ms=_elapsed_ms(persist_started_at),
                    compressionMode=compression_mode,
                )

            elapsed_ms = _elapsed_ms(started_at)
            record_stage(
                "chat_total",
                elapsed_ms=elapsed_ms,
                sourceCount=len(retrieved_sources),
                toolCallCount=len(result.tool_calls),
            )
            logger.info(
                "chat completed conversationId=%s userId=%s elapsedMs=%s",
                conversation_id,
                request.userId,
                elapsed_ms,
            )
            return ChatResponse(
                conversationId=conversation_id,
                reply=result.reply,
                sources=[self._to_chat_source(source) for source in retrieved_sources],
                toolCalls=result.tool_calls,
                ragIntent=rag_intent,
            )

    async def _run_intent_recognition(
        self,
        question: str,
        history: list[MemoryMessage],
    ) -> Any | None:
        pipeline = self._resolve_intent_recognition_pipeline()
        if pipeline is None:
            return None
        try:
            result = await pipeline.recognize(question, history=history)
        except Exception:
            logger.exception("rag intent recognition failed; continue normal chat")
            return None
        return result

    def _to_rag_intent_response(self, result: Any | None) -> dict[str, Any] | None:
        if result is None:
            return None
        to_response_dict = getattr(result, "to_response_dict", None)
        if callable(to_response_dict):
            return cast(dict[str, Any], to_response_dict())
        return cast(dict[str, Any] | None, result)

    async def _retrieve_knowledge(self, intent_result: Any | None) -> list[RetrievedSource]:
        if not self._has_kb_intents(intent_result):
            return []
        pipeline = self._resolve_retrieval_pipeline()
        if pipeline is None:
            return []
        rewrite_result = getattr(intent_result, "rewrite_result", None)
        sub_intents = getattr(intent_result, "sub_intents", None)
        if rewrite_result is None or not sub_intents:
            return []
        fallback_queries = list(getattr(rewrite_result, "sub_questions", []) or [])
        if not fallback_queries:
            fallback_queries = [str(getattr(rewrite_result, "rewritten_question", ""))]
        try:
            from agent_service.rag.retrieval_pipeline import RetrievalContext

            settings = get_settings()
            result = await pipeline.retrieve(
                RetrievalContext(
                    question=str(getattr(rewrite_result, "rewritten_question", "")).strip(),
                    sub_intents=sub_intents,
                    fallback_queries=fallback_queries,
                    candidate_top_k=settings.rag_retrieval_candidate_top_k,
                    final_top_k=settings.rag_retrieval_final_top_k,
                )
            )
            return result.sources
        except Exception:
            logger.exception("knowledge retrieval pipeline failed")
            return []

    def _has_kb_intents(self, intent_result: Any | None) -> bool:
        return bool(getattr(intent_result, "kb_intents", None))

    def _resolve_retrieval_pipeline(self) -> Any | None:
        if self._retrieval_pipeline is not None:
            return self._retrieval_pipeline
        try:
            from agent_service.rag.retrieval_service import get_multi_channel_retriever

            self._retrieval_pipeline = get_multi_channel_retriever()
        except Exception:
            logger.exception("knowledge retrieval pipeline initialization failed")
            return None
        return self._retrieval_pipeline

    def _to_chat_source(self, source: RetrievedSource) -> ChatSource:
        return ChatSource(
            id=source.id,
            title=source.title,
            content=source.content,
            score=source.score,
            sourceType=source.source_type,
            url=source.url,
            collectionName=source.collection_name,
            channel=source.channel,
        )

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


def _elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)
