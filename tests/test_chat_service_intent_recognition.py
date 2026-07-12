from __future__ import annotations

import pytest

from agent_service.rag.intent_models import (
    GuidanceDecision,
    IntentKind,
    IntentLevel,
    IntentNode,
    NodeScore,
    SubQuestionIntent,
)
from agent_service.rag.intent_recognition_pipeline import IntentRecognitionResult
from agent_service.rag.schemas import RetrievedSource, RewriteResult
from agent_service.schemas.chat import ChatRequest
from agent_service.services.chat_service import ChatService
from agent_service.services.llm_service import LLMResult


class FakeLLMService:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    async def complete(
        self,
        messages: list[dict[str, str]],
        use_tools: bool = True,
    ) -> LLMResult:
        assert use_tools is False
        self.messages = messages
        return LLMResult(reply="assistant reply")


class FakeIntentPipeline:
    async def recognize(self, question: str, history=None) -> IntentRecognitionResult:
        return IntentRecognitionResult(
            rewrite_result=RewriteResult(
                original_question=question,
                rewritten_question="rewritten question",
                sub_questions=["sub question A", "sub question B"],
            ),
            sub_intents=[],
            guidance=GuidanceDecision.none(),
            kb_intents=[],
            mcp_intents=[],
            is_system_only=False,
        )


class FakeIntentDirectedRetriever:
    def __init__(self) -> None:
        self.calls = []

    async def search(self, sub_intents, *, fallback_queries, top_k=5):
        self.calls.append((sub_intents, fallback_queries, top_k))
        return [
            RetrievedSource(
                title="return-policy.md",
                content="Eligible items may be returned within seven days.",
                score=0.93,
                source_type="knowledge_base",
                url="rustfs://test111/return-policy.md",
                collection_name="test111",
            )
        ]


class FakeKbIntentPipeline:
    async def recognize(self, question: str, history=None) -> IntentRecognitionResult:
        node_score = NodeScore(
            node=IntentNode(
                id="general-return-policy",
                name="return policy",
                level=IntentLevel.TOPIC,
                kind=IntentKind.KB,
                collection_name="test111",
                top_k=3,
            ),
            score=0.95,
        )
        return IntentRecognitionResult(
            rewrite_result=RewriteResult(
                original_question=question,
                rewritten_question="return policy",
                sub_questions=["return policy"],
            ),
            sub_intents=[SubQuestionIntent("return policy", [node_score])],
            guidance=GuidanceDecision.none(),
            kb_intents=[node_score],
            mcp_intents=[],
            is_system_only=False,
        )


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
    assert response.ragIntent.rewrittenQuestion == "rewritten question"
    assert response.ragIntent.subQuestions == ["sub question A", "sub question B"]


@pytest.mark.asyncio
async def test_chat_service_uses_kb_intents_to_ground_prompt_and_sources(monkeypatch) -> None:
    monkeypatch.setenv("RAG_INTENT_ENABLED", "true")
    llm_service = FakeLLMService()
    directed_retriever = FakeIntentDirectedRetriever()
    service = ChatService(
        llm_service=llm_service,
        memory_service=None,
        intent_recognition_pipeline=FakeKbIntentPipeline(),
        intent_directed_retriever=directed_retriever,
    )

    response = await service.chat(
        ChatRequest(
            userId=7,
            conversationId="conv_1",
            message="return policy",
            history=[],
        )
    )

    assert directed_retriever.calls[0][1] == ["return policy"]
    assert "Eligible items may be returned within seven days." in llm_service.messages[0]["content"]
    assert response.sources[0].title == "return-policy.md"
    assert response.sources[0].collectionName == "test111"
