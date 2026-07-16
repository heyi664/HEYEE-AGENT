from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any, cast
from uuid import uuid4

from agent_service.core.config import get_settings
from agent_service.core.observability import bind_conversation, record_stage
from agent_service.memory.conversation_memory_service import ConversationMemoryService
from agent_service.memory.models import MemoryContext, MemoryMessage
from agent_service.rag.schemas import RetrievedSource
from agent_service.schemas.chat import ChatRequest, ChatResponse, ChatSource
from agent_service.services.llm_service import LLMService, get_llm_service
from agent_service.services.prompt_service import (
    PromptBuildPlan,
    build_messages_from_plan,
    build_prompt_plan,
)

logger = logging.getLogger(__name__)


@dataclass
class ChatPreparation:
    started_at: float
    settings: Any
    conversation_id: str
    user_id: str
    deep_thinking: bool
    memory_service: Any | None
    memory_enabled: bool
    rag_intent: dict[str, Any] | None
    retrieved_sources: list[RetrievedSource]
    mcp_result: Any
    messages: list[dict[str, str]]
    prompt_plan: PromptBuildPlan


class ChatService:
    def __init__(
        self,
        llm_service: LLMService,
        memory_service: ConversationMemoryService | object | None = None,
        intent_recognition_pipeline: object | None = None,
        retrieval_pipeline: object | None = None,
        mcp_execution_service: object | None = None,
    ) -> None:
        self._llm_service = llm_service
        self._memory_service: Any = memory_service
        self._intent_recognition_pipeline: Any = intent_recognition_pipeline
        self._retrieval_pipeline: Any = retrieval_pipeline
        self._mcp_execution_service: Any = mcp_execution_service

    async def chat(self, request: ChatRequest) -> ChatResponse:
        preparation = await self.prepare(request)
        answer_started_at = time.perf_counter()
        try:
            result = await self._complete_answer(
                preparation.messages,
                preparation.prompt_plan,
            )
        except Exception:
            self.record_answer_failure(preparation, answer_started_at)
            raise
        return await self.complete_preparation(
            preparation,
            reply=result.reply,
            tool_calls=result.tool_calls,
            answer_started_at=answer_started_at,
        )

    async def prepare(self, request: ChatRequest) -> ChatPreparation:
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
            retrieved_sources, mcp_result = await asyncio.gather(
                self._retrieve_knowledge(intent_result),
                self._execute_mcp_tools(intent_result),
            )
            record_stage(
                "knowledge_retrieval",
                elapsed_ms=_elapsed_ms(retrieval_started_at),
                sourceCount=len(retrieved_sources),
                attempted=self._has_kb_intents(intent_result),
            )

            prompt_nodes = self._prompt_nodes(intent_result)
            has_evidence = bool(retrieved_sources or mcp_result.context.strip())
            prompt_plan = build_prompt_plan(
                request.message,
                retrieved_sources=retrieved_sources,
                retrieval_attempted=self._has_kb_intents(intent_result),
                mcp_context=mcp_result.context,
                mcp_attempted=self._has_mcp_intents(intent_result),
                sub_questions=self._prompt_sub_questions(intent_result),
                prompt_template=(
                    str(getattr(prompt_nodes[0], "prompt_template", "") or "")
                    if has_evidence and len(prompt_nodes) == 1
                    else None
                ),
                prompt_snippets=(
                    [str(getattr(node, "prompt_snippet", "") or "") for node in prompt_nodes]
                    if has_evidence
                    else []
                ),
            )
            messages = build_messages_from_plan(memory_context, prompt_plan)
            return ChatPreparation(
                started_at=started_at,
                settings=settings,
                conversation_id=conversation_id,
                user_id=user_id,
                deep_thinking=request.deepThinking,
                memory_service=memory_service,
                memory_enabled=settings.memory_enabled,
                rag_intent=rag_intent,
                retrieved_sources=retrieved_sources,
                mcp_result=mcp_result,
                messages=messages,
                prompt_plan=prompt_plan,
            )

    def record_answer_failure(
        self,
        preparation: ChatPreparation,
        answer_started_at: float,
    ) -> None:
        record_stage(
            "answer_generation",
            elapsed_ms=_elapsed_ms(answer_started_at),
            promptMessageCount=len(preparation.messages),
            replyChars=0,
            sourceCount=len(preparation.retrieved_sources),
            status="failed",
            toolCallCount=len(preparation.mcp_result.tool_calls),
        )

    async def complete_preparation(
        self,
        preparation: ChatPreparation,
        *,
        reply: str,
        tool_calls: list[str],
        answer_started_at: float,
        interrupted: bool = False,
    ) -> ChatResponse:
        record_stage(
            "answer_generation",
            elapsed_ms=_elapsed_ms(answer_started_at),
            promptMessageCount=len(preparation.messages),
            replyChars=len(reply),
            sourceCount=len(preparation.retrieved_sources),
            status="cancelled" if interrupted else "success",
            toolCallCount=len(preparation.mcp_result.tool_calls) + len(tool_calls),
        )
        message_id = await self._persist_assistant_reply(preparation, reply)
        elapsed_ms = _elapsed_ms(preparation.started_at)
        record_stage(
            "chat_total",
            elapsed_ms=elapsed_ms,
            sourceCount=len(preparation.retrieved_sources),
            toolCallCount=len(preparation.mcp_result.tool_calls) + len(tool_calls),
            status="cancelled" if interrupted else "success",
        )
        logger.info(
            "chat %s conversationId=%s userId=%s elapsedMs=%s",
            "cancelled" if interrupted else "completed",
            preparation.conversation_id,
            preparation.user_id,
            elapsed_ms,
        )
        response = ChatResponse(
            conversationId=preparation.conversation_id,
            messageId=message_id,
            interrupted=interrupted,
            reply=reply,
            sources=[self._to_chat_source(source) for source in preparation.retrieved_sources],
            toolCalls=preparation.mcp_result.tool_calls + tool_calls,
            ragIntent=preparation.rag_intent,
        )
        return response

    async def _persist_assistant_reply(
        self,
        preparation: ChatPreparation,
        reply: str,
    ) -> str | None:
        if not preparation.memory_enabled or preparation.memory_service is None:
            return None
        persist_started_at = time.perf_counter()
        message_id = preparation.memory_service.append(
            preparation.conversation_id,
            preparation.user_id,
            "assistant",
            reply,
        )
        compression_result = preparation.memory_service.compress_if_needed(
            preparation.conversation_id,
            preparation.user_id,
        )
        if inspect.isawaitable(compression_result):
            if preparation.settings.memory_async_compress:
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
        return str(message_id) if message_id else None

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

    def _has_mcp_intents(self, intent_result: Any | None) -> bool:
        return bool(getattr(intent_result, "mcp_intents", None))

    def _prompt_sub_questions(self, intent_result: Any | None) -> list[str]:
        rewrite_result = getattr(intent_result, "rewrite_result", None)
        return list(getattr(rewrite_result, "sub_questions", []) or [])

    def _prompt_nodes(self, intent_result: Any | None) -> list[Any]:
        nodes: list[Any] = []
        seen: set[str] = set()
        for sub_intent in list(getattr(intent_result, "sub_intents", []) or []):
            for node_score in list(getattr(sub_intent, "node_scores", []) or []):
                node = getattr(node_score, "node", None)
                node_id = str(getattr(node, "id", "") or "")
                if node is not None and node_id not in seen:
                    nodes.append(node)
                    seen.add(node_id)
        return nodes

    async def _complete_answer(
        self,
        messages: list[dict[str, str]],
        prompt_plan: PromptBuildPlan,
    ) -> Any:
        complete = self._llm_service.complete
        parameters = inspect.signature(complete).parameters.values()
        supports_generation_options = any(
            parameter.name in {"temperature", "top_p"}
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
        if supports_generation_options:
            return await complete(
                messages,
                use_tools=False,
                temperature=prompt_plan.temperature,
                top_p=prompt_plan.top_p,
            )
        return await complete(messages, use_tools=False)

    async def _execute_mcp_tools(self, intent_result: Any | None) -> Any:
        from agent_service.mcp.execution import McpExecutionResult

        if not self._has_mcp_intents(intent_result):
            return McpExecutionResult.empty()
        sub_intents = list(getattr(intent_result, "sub_intents", []) or [])
        if not sub_intents:
            return McpExecutionResult.empty()
        service = self._resolve_mcp_execution_service()
        if service is None:
            return McpExecutionResult.empty()
        try:
            return await service.execute(sub_intents)
        except Exception:
            logger.exception("MCP execution pipeline failed")
            return McpExecutionResult.empty()

    def _resolve_mcp_execution_service(self) -> Any | None:
        if self._mcp_execution_service is not None:
            return self._mcp_execution_service
        try:
            from agent_service.mcp.execution import McpExecutionService
            from agent_service.mcp.parameter_extractor import McpParameterExtractor

            settings = get_settings()
            self._mcp_execution_service = McpExecutionService(
                parameter_extractor=McpParameterExtractor(self._llm_service),
                max_context_chars=settings.mcp_context_max_chars,
            )
        except Exception:
            logger.exception("MCP execution pipeline initialization failed")
            return None
        return self._mcp_execution_service

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
