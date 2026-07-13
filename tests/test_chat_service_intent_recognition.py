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


class FailingLLMService:
    async def complete(self, messages: list[dict[str, str]], use_tools: bool = True) -> LLMResult:
        del messages, use_tools
        raise RuntimeError("model timeout")


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


class FakeRetrievalPipeline:
    def __init__(self) -> None:
        self.calls = []

    async def retrieve(self, context):
        self.calls.append(context)
        return type(
            "RetrievalResult",
            (),
            {
                "sources": [
                    RetrievedSource(
                        id="chunk-1",
                        title="return-policy.md",
                        content="Eligible items may be returned within seven days.",
                        score=0.93,
                        source_type="knowledge_base",
                        url="rustfs://test111/return-policy.md",
                        collection_name="test111",
                        channel="intent_directed",
                    )
                ]
            },
        )()


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
    retrieval_pipeline = FakeRetrievalPipeline()
    service = ChatService(
        llm_service=llm_service,
        memory_service=None,
        intent_recognition_pipeline=FakeKbIntentPipeline(),
        retrieval_pipeline=retrieval_pipeline,
    )

    response = await service.chat(
        ChatRequest(
            userId=7,
            conversationId="conv_1",
            message="return policy",
            history=[],
        )
    )

    assert retrieval_pipeline.calls[0].question == "return policy"
    assert retrieval_pipeline.calls[0].fallback_queries == ["return policy"]
    assert "Eligible items may be returned within seven days." in llm_service.messages[0]["content"]
    assert response.sources[0].title == "return-policy.md"
    assert response.sources[0].id == "chunk-1"
    assert response.sources[0].collectionName == "test111"
    assert response.sources[0].channel == "intent_directed"


@pytest.mark.asyncio
async def test_chat_service_records_stage_metrics(monkeypatch, caplog) -> None:
    monkeypatch.setenv("RAG_INTENT_ENABLED", "true")
    service = ChatService(
        llm_service=FakeLLMService(),
        memory_service=None,
        intent_recognition_pipeline=FakeIntentPipeline(),
    )

    with caplog.at_level("INFO", logger="agent_service.metrics"):
        await service.chat(
            ChatRequest(
                userId=7,
                conversationId="conv_metrics",
                message="current question",
                history=[],
            )
        )

    assert "stage=memory_context" in caplog.text
    assert "stage=intent_recognition" in caplog.text
    assert "stage=answer_generation" in caplog.text
    assert "stage=chat_total" in caplog.text
    assert "conversationId=conv_metrics" in caplog.text


@pytest.mark.asyncio
async def test_chat_service_records_failed_answer_generation_metric(monkeypatch, caplog) -> None:
    monkeypatch.setenv("RAG_INTENT_ENABLED", "false")
    service = ChatService(llm_service=FailingLLMService(), memory_service=None)

    with caplog.at_level("INFO", logger="agent_service.metrics"):
        with pytest.raises(RuntimeError, match="model timeout"):
            await service.chat(
                ChatRequest(
                    userId=7,
                    conversationId="conv_fail",
                    message="current question",
                    history=[],
                )
            )

    assert "stage=answer_generation" in caplog.text
    assert "conversationId=conv_fail" in caplog.text
    assert "status=failed" in caplog.text
