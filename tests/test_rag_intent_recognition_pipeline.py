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
from agent_service.rag.intent_recognition_pipeline import IntentRecognitionPipeline
from agent_service.rag.schemas import RewriteResult


class FakeRewriteService:
    async def rewrite(self, question: str, history=None) -> RewriteResult:
        return RewriteResult(
            original_question=question,
            rewritten_question="订单物流和退换政策",
            sub_questions=["订单物流", "退换政策"],
        )


class FakeResolver:
    def __init__(self) -> None:
        self.kb = NodeScore(
            IntentNode(
                id="return-policy",
                name="退换政策",
                level=IntentLevel.TOPIC,
                kind=IntentKind.KB,
                collection_name="kb_return",
            ),
            0.91,
            "问退换政策",
        )
        self.mcp = NodeScore(
            IntentNode(
                id="order-query",
                name="订单查询",
                level=IntentLevel.TOPIC,
                kind=IntentKind.MCP,
                mcp_tool_id="order.query",
            ),
            0.88,
            "问订单",
        )

    async def resolve(self, rewrite_result: RewriteResult) -> list[SubQuestionIntent]:
        return [
            SubQuestionIntent("订单物流", [self.mcp]),
            SubQuestionIntent("退换政策", [self.kb]),
        ]

    def merge_intent_group(self, sub_intents):
        return type("Group", (), {"kb_intents": [self.kb], "mcp_intents": [self.mcp]})()

    def is_system_only(self, sub_intents) -> bool:
        return False


class FakeGuidance:
    async def detect_ambiguity(self, question: str, sub_intents: list[SubQuestionIntent]):
        return GuidanceDecision.none()


@pytest.mark.asyncio
async def test_intent_recognition_pipeline_outputs_rewrite_split_and_intents() -> None:
    pipeline = IntentRecognitionPipeline(
        rewrite_service=FakeRewriteService(),
        intent_resolver=FakeResolver(),
        guidance_service=FakeGuidance(),
    )

    result = await pipeline.recognize("帮我查订单物流，退换政策是什么？")

    assert result.original_question == "帮我查订单物流，退换政策是什么？"
    assert result.rewritten_question == "订单物流和退换政策"
    assert result.sub_questions == ["订单物流", "退换政策"]
    assert result.is_system_only is False
    assert [item.sub_question for item in result.sub_intents] == ["订单物流", "退换政策"]
    assert [score.node.id for score in result.kb_intents] == ["return-policy"]
    assert [score.node.id for score in result.mcp_intents] == ["order-query"]
    assert result.to_response_dict()["subIntents"][0]["intents"][0]["kind"] == "MCP"
