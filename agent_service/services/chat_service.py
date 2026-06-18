from __future__ import annotations

import logging
import time
from uuid import uuid4

from agent_service.schemas.chat import ChatRequest, ChatResponse
from agent_service.services.llm_service import LLMService, get_llm_service
from agent_service.services.prompt_service import build_messages

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, llm_service: LLMService) -> None:
        self._llm_service = llm_service

    async def chat(self, request: ChatRequest) -> ChatResponse:
        started_at = time.perf_counter()
        conversation_id = request.conversationId or f"conv_{uuid4().hex[:12]}"
        messages = build_messages(request.history, request.message)
        reply = await self._llm_service.complete(messages)
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)

        logger.info(
            "chat completed conversationId=%s userId=%s elapsedMs=%s",
            conversation_id,
            request.userId,
            elapsed_ms,
        )
        return ChatResponse(
            conversationId=conversation_id,
            reply=reply,
            sources=[],
            toolCalls=[],
        )


def get_chat_service() -> ChatService:
    return ChatService(get_llm_service())

