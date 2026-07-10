from __future__ import annotations

import pytest

from agent_service.rag.intent_classifier import IntentClassifier
from agent_service.rag.intent_models import IntentKind, IntentLevel, IntentNode, IntentTreeData
from agent_service.services.llm_service import LLMResult


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.messages: list[list[dict[str, str]]] = []

    async def complete(
        self,
        messages: list[dict[str, str]],
        use_tools: bool = True,
    ) -> LLMResult:
        assert use_tools is False
        self.messages.append(messages)
        return LLMResult(reply=self.reply)


class FakeTreeSource:
    def __init__(self, data: IntentTreeData) -> None:
        self.data = data

    def load(self) -> IntentTreeData:
        return self.data


def make_tree() -> IntentTreeData:
    root = IntentNode(id="domain", name="商品服务", level=IntentLevel.DOMAIN)
    category = IntentNode(id="category", name="3C 数码", level=IntentLevel.CATEGORY)
    leaf = IntentNode(
        id="return-policy",
        name="退换政策",
        level=IntentLevel.TOPIC,
        description="退换货规则",
        examples=["退货政策是什么？"],
        collection_name="kb_3c_return",
        kind=IntentKind.KB,
    )
    tool = IntentNode(
        id="order-query",
        name="订单查询",
        level=IntentLevel.TOPIC,
        kind=IntentKind.MCP,
        mcp_tool_id="order.query",
    )
    root.children = [category]
    category.parent = root
    category.children = [leaf, tool]
    leaf.parent = category
    tool.parent = category
    root.full_path = "商品服务"
    category.full_path = "商品服务 > 3C 数码"
    leaf.full_path = "商品服务 > 3C 数码 > 退换政策"
    tool.full_path = "商品服务 > 3C 数码 > 订单查询"
    all_nodes = [root, category, leaf, tool]
    return IntentTreeData(
        roots=[root],
        all_nodes=all_nodes,
        leaf_nodes=[leaf, tool],
        id_to_node={node.id: node for node in all_nodes},
    )


@pytest.mark.asyncio
async def test_classifier_prompts_with_leaf_nodes_only_and_sorts_scores() -> None:
    llm = FakeLLM(
        """
        ```json
        {"results": [
          {"id": "unknown", "score": 1},
          {"id": "order-query", "score": 0.4, "reason": "查订单"},
          {"id": "return-policy", "score": 0.92, "reason": "退货"}
        ]}
        ```
        """
    )
    classifier = IntentClassifier(FakeTreeSource(make_tree()), llm_service=llm)

    scores = await classifier.classify_targets("退货政策是什么？")

    assert [score.node.id for score in scores] == ["return-policy", "order-query"]
    assert scores[0].reason == "退货"
    system_prompt = llm.messages[0][0]["content"]
    assert "id=return-policy" in system_prompt
    assert "type=KB" in system_prompt
    assert "type=MCP" in system_prompt
    assert "toolId=order.query" in system_prompt
    assert "id=category" not in system_prompt


@pytest.mark.asyncio
async def test_classifier_skips_items_missing_required_fields() -> None:
    llm = FakeLLM(
        '[{"id": "return-policy"}, {"score": 0.8}, {"id": "return-policy", "score": 0.8}]'
    )
    classifier = IntentClassifier(FakeTreeSource(make_tree()), llm_service=llm)

    scores = await classifier.classify_targets("退货政策是什么？")

    assert len(scores) == 1
    assert scores[0].score == 0.8
