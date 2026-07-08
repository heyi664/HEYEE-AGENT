from __future__ import annotations

from agent_service.rag.intent_models import (
    IntentKind,
    IntentLevel,
    IntentNode,
    IntentNodeRecord,
    IntentTreeData,
)


def build_intent_tree(records: list[IntentNodeRecord]) -> IntentTreeData:
    sorted_records = sorted(records, key=lambda item: item.sort_order)
    nodes = [_record_to_node(record) for record in sorted_records]
    id_to_node = {node.id: node for node in nodes}
    roots: list[IntentNode] = []

    for node in nodes:
        parent = id_to_node.get(node.parent_id or "")
        if parent is None:
            roots.append(node)
            continue
        node.parent = parent
        parent.children.append(node)

    for root in roots:
        _fill_full_path(root, [])

    leaf_nodes = [node for node in nodes if node.is_leaf()]
    return IntentTreeData(
        roots=roots,
        all_nodes=nodes,
        leaf_nodes=leaf_nodes,
        id_to_node=id_to_node,
    )


def _record_to_node(record: IntentNodeRecord) -> IntentNode:
    return IntentNode(
        id=record.intent_code,
        kb_id=record.kb_id,
        name=record.name,
        description=record.description,
        level=_level(record.level),
        parent_id=record.parent_code,
        examples=list(record.examples),
        kind=_kind(record.kind),
        collection_name=record.collection_name,
        mcp_tool_id=record.mcp_tool_id,
        top_k=record.top_k,
        prompt_snippet=record.prompt_snippet,
        prompt_template=record.prompt_template,
        param_prompt_template=record.param_prompt_template,
        sort_order=record.sort_order,
    )


def _fill_full_path(node: IntentNode, ancestors: list[str]) -> None:
    path = [*ancestors, node.name]
    node.full_path = " > ".join(path)
    for child in node.children:
        _fill_full_path(child, path)


def _level(value: int) -> IntentLevel:
    try:
        return IntentLevel(value)
    except ValueError:
        return IntentLevel.TOPIC


def _kind(value: int) -> IntentKind:
    try:
        return IntentKind(value)
    except ValueError:
        return IntentKind.KB
