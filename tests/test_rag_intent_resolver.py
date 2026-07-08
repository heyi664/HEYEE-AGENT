from __future__ import annotations

import pytest

from agent_service.rag.intent_models import IntentKind, IntentLevel, IntentNode, NodeScore
from agent_service.rag.intent_resolver import IntentResolver
from agent_service.rag.schemas import RewriteResult


class FakeClassifier:
    def __init__(self, scores_by_question: dict[str, list[NodeScore]]) -> None:
        self.scores_by_question = scores_by_question

    async def classify_targets(self, question: str) -> list[NodeScore]:
        if question == "raise":
            raise RuntimeError("boom")
        return self.scores_by_question.get(question, [])


def node(node_id: str, score: float, kind: IntentKind = IntentKind.KB) -> NodeScore:
    return NodeScore(
        node=IntentNode(id=node_id, name=node_id, level=IntentLevel.TOPIC, kind=kind),
        score=score,
    )


@pytest.mark.asyncio
async def test_resolver_filters_limits_and_caps_total_intents() -> None:
    resolver = IntentResolver(
        FakeClassifier(
            {
                "A": [node("a1", 0.9), node("a2", 0.6), node("a3", 0.2)],
                "B": [node("b1", 0.88), node("b2", 0.7)],
                "C": [node("c1", 0.86), node("c2", 0.4)],
            }
        ),
        min_score=0.35,
        max_intent_count=3,
    )

    result = await resolver.resolve(
        RewriteResult(
            original_question="q",
            rewritten_question="q",
            sub_questions=["A", "B", "C"],
        )
    )

    simplified = [
        (item.sub_question, [score.node.id for score in item.node_scores])
        for item in result
    ]
    assert simplified == [
        ("A", ["a1"]),
        ("B", ["b1"]),
        ("C", ["c1"]),
    ]


@pytest.mark.asyncio
async def test_resolver_degrades_failed_sub_question_to_empty_list() -> None:
    resolver = IntentResolver(
        FakeClassifier({"A": [node("a1", 0.9)]}),
        min_score=0.35,
        max_intent_count=3,
    )

    result = await resolver.resolve(
        RewriteResult(
            original_question="q",
            rewritten_question="q",
            sub_questions=["A", "raise"],
        )
    )

    assert [item.sub_question for item in result] == ["A", "raise"]
    assert result[0].node_scores
    assert result[1].node_scores == []


def test_merge_intent_group_and_system_only() -> None:
    resolver = IntentResolver(FakeClassifier({}))
    system = node("greeting", 0.9, IntentKind.SYSTEM)
    kb = node("return", 0.8, IntentKind.KB)
    mcp = node("order", 0.7, IntentKind.MCP)

    group = resolver.merge_intent_group(
        [
            ("hello", [system]),
            ("mixed", [kb, mcp]),
        ]
    )

    assert [score.node.id for score in group.kb_intents] == ["return"]
    assert [score.node.id for score in group.mcp_intents] == ["order"]
    assert resolver.is_system_only([("hello", [system])])
    assert not resolver.is_system_only([("mixed", [system, kb])])
