from __future__ import annotations

import pytest

from agent_service.rag.schemas import RewriteResult
from agent_service.schemas.chat import ChatRequest
from agent_service.services.chat_service import ChatService
from agent_service.services.llm_service import LLMResult


class FakeLLMService:
    async def complete(
        self,
        messages: list[dict[str, str]],
        use_tools: bool = True,
    ) -> LLMResult:
        assert use_tools is False
        return LLMResult(reply="assistant reply")


class FakeIntentPipeline:
    async def recognize(self, question: str, history=None):
        return type(
            "IntentResult",
            (),
            {
                "rewrite_result": RewriteResult(
                    original_question=question,
                    rewritten_question="改写后的问题",
                    sub_questions=["子问题 A", "子问题 B"],
                ),
                "to_response_dict": lambda self: {
                    "originalQuestion": question,
                    "rewrittenQuestion": "改写后的问题",
                    "subQuestions": ["子问题 A", "子问题 B"],
                    "subIntents": [],
                    "kbIntents": [],
                    "mcpIntents": [],
                    "isSystemOnly": False,
                    "guidance": {"action": "NONE", "prompt": None},
                },
            },
        )()


@pytest.mark.asyncio
async def test_chat_service_outputs_rag_intent_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("RAG_INTENT_ENABLED", "true")
    service = ChatService(
        llm_service=FakeLLMService(),
        memory_service=None,
        intent_recognition_pipeline=FakeIntentPipeline(),
    )

    response = await service.chat(
        ChatRequest(
            userId=7,
            conversationId="conv_1",
            message="current question",
            history=[],
        )
    )

    assert response.ragIntent is not None
    assert response.ragIntent.rewrittenQuestion == "改写后的问题"
    assert response.ragIntent.subQuestions == ["子问题 A", "子问题 B"]
