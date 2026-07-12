from __future__ import annotations

from pathlib import Path

from agent_service.rag.intent_models import IntentKind, IntentLevel, IntentNodeRecord
from agent_service.rag.intent_tree import build_intent_tree


def test_build_intent_tree_links_children_and_full_paths() -> None:
    tree = build_intent_tree(
        [
            IntentNodeRecord(
                id="1",
                intent_code="product",
                name="商品服务",
                level=0,
                parent_code=None,
                kind=0,
            ),
            IntentNodeRecord(
                id="2",
                intent_code="digital",
                name="3C 数码",
                level=1,
                parent_code="product",
                kind=0,
            ),
            IntentNodeRecord(
                id="3",
                intent_code="return-policy",
                name="退换政策",
                level=2,
                parent_code="digital",
                kind=0,
                collection_name="kb_3c_return",
                top_k=4,
                examples=["退货政策是什么？"],
            ),
        ]
    )

    assert [node.id for node in tree.roots] == ["product"]
    assert tree.id_to_node["digital"].parent is tree.id_to_node["product"]
    assert tree.id_to_node["return-policy"].full_path == "商品服务 > 3C 数码 > 退换政策"
    assert [node.id for node in tree.leaf_nodes] == ["return-policy"]
    assert tree.id_to_node["return-policy"].is_leaf()
    assert tree.id_to_node["return-policy"].is_kb()
    assert tree.id_to_node["return-policy"].level is IntentLevel.TOPIC
    assert tree.id_to_node["return-policy"].kind is IntentKind.KB


def test_build_intent_tree_keeps_orphan_as_root() -> None:
    tree = build_intent_tree(
        [
            IntentNodeRecord(
                id="1",
                intent_code="orphan-topic",
                name="孤儿节点",
                level=2,
                parent_code="missing",
                kind=1,
            )
        ]
    )

    assert [node.id for node in tree.roots] == ["orphan-topic"]
    assert tree.id_to_node["orphan-topic"].parent is None
    assert tree.id_to_node["orphan-topic"].full_path == "孤儿节点"
    assert tree.id_to_node["orphan-topic"].is_system()


def test_return_policy_seed_defines_a_routable_kb_leaf() -> None:
    seed = (
        Path(__file__).resolve().parents[1]
        / "sql"
        / "20260711_seed_intent_node_demo.sql"
    ).read_text(encoding="utf-8")

    assert "product-service" in seed
    assert "general-products" in seed
    assert "general-return-policy" in seed
    assert "test111" in seed
    assert "退换政策是什么" in seed
    assert "create_by" not in seed
    assert "update_by" not in seed
    assert "created_by" not in seed
    assert "updated_by" not in seed
