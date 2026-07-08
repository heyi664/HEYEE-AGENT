from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class IntentLevel(IntEnum):
    DOMAIN = 0
    CATEGORY = 1
    TOPIC = 2


class IntentKind(IntEnum):
    KB = 0
    SYSTEM = 1
    MCP = 2


@dataclass(frozen=True)
class IntentNodeRecord:
    id: str
    intent_code: str
    name: str
    level: int
    parent_code: str | None
    kind: int
    kb_id: str | None = None
    description: str | None = None
    examples: list[str] = field(default_factory=list)
    collection_name: str | None = None
    top_k: int | None = None
    mcp_tool_id: str | None = None
    prompt_snippet: str | None = None
    prompt_template: str | None = None
    param_prompt_template: str | None = None
    sort_order: int = 0


@dataclass
class IntentNode:
    id: str
    name: str
    level: IntentLevel
    kb_id: str | None = None
    description: str | None = None
    parent_id: str | None = None
    examples: list[str] = field(default_factory=list)
    children: list[IntentNode] = field(default_factory=list)
    full_path: str = ""
    kind: IntentKind = IntentKind.KB
    collection_name: str | None = None
    mcp_tool_id: str | None = None
    top_k: int | None = None
    prompt_snippet: str | None = None
    prompt_template: str | None = None
    param_prompt_template: str | None = None
    sort_order: int = 0
    parent: IntentNode | None = field(default=None, repr=False, compare=False)

    def is_leaf(self) -> bool:
        return not self.children

    def is_kb(self) -> bool:
        return self.kind is IntentKind.KB

    def is_mcp(self) -> bool:
        return self.kind is IntentKind.MCP

    def is_system(self) -> bool:
        return self.kind is IntentKind.SYSTEM


@dataclass(frozen=True)
class NodeScore:
    node: IntentNode
    score: float
    reason: str | None = None


@dataclass(frozen=True)
class SubQuestionIntent:
    sub_question: str
    node_scores: list[NodeScore]


@dataclass(frozen=True)
class IntentCandidate:
    sub_question_index: int
    node_score: NodeScore


@dataclass(frozen=True)
class IntentGroup:
    mcp_intents: list[NodeScore]
    kb_intents: list[NodeScore]


@dataclass(frozen=True)
class IntentTreeData:
    roots: list[IntentNode]
    all_nodes: list[IntentNode]
    leaf_nodes: list[IntentNode]
    id_to_node: dict[str, IntentNode]


class GuidanceAction(IntEnum):
    NONE = 0
    PROMPT = 1


@dataclass(frozen=True)
class GuidanceDecision:
    action: GuidanceAction
    prompt: str | None = None

    @classmethod
    def none(cls) -> GuidanceDecision:
        return cls(GuidanceAction.NONE)

    @classmethod
    def prompt_user(cls, prompt: str) -> GuidanceDecision:
        return cls(GuidanceAction.PROMPT, prompt)

    def is_prompt(self) -> bool:
        return self.action is GuidanceAction.PROMPT
