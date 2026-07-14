from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from html import escape

from agent_service.memory.models import MemoryContext
from agent_service.rag.prompt_template_loader import PromptTemplateLoader
from agent_service.rag.schemas import RetrievedSource


class PromptScene(StrEnum):
    EMPTY = "EMPTY"
    KB_ONLY = "KB_ONLY"
    MCP_ONLY = "MCP_ONLY"
    MIXED = "MIXED"


@dataclass(frozen=True)
class PromptBuildPlan:
    scene: PromptScene
    system_prompt: str
    user_content: str
    temperature: float
    top_p: float | None


_TEMPLATES = {
    PromptScene.EMPTY: "answer-chat-general.st",
    PromptScene.KB_ONLY: "answer-chat-kb.st",
    PromptScene.MCP_ONLY: "answer-chat-mcp.st",
    PromptScene.MIXED: "answer-chat-mixed.st",
}


def build_prompt_plan(
    message: str,
    *,
    retrieved_sources: list[RetrievedSource] | None = None,
    retrieval_attempted: bool = False,
    mcp_context: str = "",
    mcp_attempted: bool = False,
    sub_questions: list[str] | None = None,
    prompt_template: str | None = None,
    prompt_snippets: list[str] | None = None,
    template_loader: PromptTemplateLoader | None = None,
) -> PromptBuildPlan:
    """Build a scene-specific prompt without mixing dynamic evidence into system text."""

    sources = retrieved_sources or []
    scene = _resolve_scene(bool(sources), bool(mcp_context.strip()))
    rules = _render_rules(prompt_snippets or [])
    loader = template_loader or PromptTemplateLoader()
    template = (prompt_template or "").strip() or loader.load(_TEMPLATES[scene])
    if rules and "{rules}" not in template:
        template += "\n\n{rules}"
    system_prompt = loader.render_text(template, {"rules": rules})
    user_content = _build_user_content(
        message,
        sources=sources,
        retrieval_attempted=retrieval_attempted,
        mcp_context=mcp_context,
        mcp_attempted=mcp_attempted,
        sub_questions=sub_questions or [],
    )
    temperature, top_p = _generation_parameters(scene)
    return PromptBuildPlan(
        scene=scene,
        system_prompt=system_prompt,
        user_content=user_content,
        temperature=temperature,
        top_p=top_p,
    )


def build_messages(
    memory_context: MemoryContext,
    message: str,
    *,
    retrieved_sources: list[RetrievedSource] | None = None,
    retrieval_attempted: bool = False,
    mcp_context: str = "",
    mcp_attempted: bool = False,
    sub_questions: list[str] | None = None,
    prompt_template: str | None = None,
    prompt_snippets: list[str] | None = None,
) -> list[dict[str, str]]:
    """Create [system, history..., user] with evidence next to the question."""

    plan = build_prompt_plan(
        message,
        retrieved_sources=retrieved_sources,
        retrieval_attempted=retrieval_attempted,
        mcp_context=mcp_context,
        mcp_attempted=mcp_attempted,
        sub_questions=sub_questions,
        prompt_template=prompt_template,
        prompt_snippets=prompt_snippets,
    )
    return build_messages_from_plan(memory_context, plan)


def build_messages_from_plan(
    memory_context: MemoryContext,
    plan: PromptBuildPlan,
) -> list[dict[str, str]]:
    """Add memory around a prebuilt plan while keeping evidence in the final user turn."""

    messages: list[dict[str, str]] = [{"role": "system", "content": plan.system_prompt}]
    messages.extend(memory_context.to_prompt_messages())
    messages.append({"role": "user", "content": plan.user_content})
    return messages


def _resolve_scene(has_kb: bool, has_mcp: bool) -> PromptScene:
    if has_kb and has_mcp:
        return PromptScene.MIXED
    if has_kb:
        return PromptScene.KB_ONLY
    if has_mcp:
        return PromptScene.MCP_ONLY
    return PromptScene.EMPTY


def _generation_parameters(scene: PromptScene) -> tuple[float, float | None]:
    """Favor fidelity for KB, constrained synthesis for MCP, and natural chat otherwise."""

    if scene is PromptScene.KB_ONLY:
        return 0.0, 1.0
    if scene in {PromptScene.MCP_ONLY, PromptScene.MIXED}:
        return 0.3, 0.8
    return 0.7, None


def _build_user_content(
    message: str,
    *,
    sources: list[RetrievedSource],
    retrieval_attempted: bool,
    mcp_context: str,
    mcp_attempted: bool,
    sub_questions: list[str],
) -> str:
    evidence_sections: list[str] = []
    if mcp_context.strip():
        evidence_sections.append(f"<tool-data>\n{mcp_context.strip()}\n</tool-data>")
    elif mcp_attempted:
        evidence_sections.append(
            "<tool-status>Real-time data could not be obtained. "
            "Do not infer or fabricate it.</tool-status>"
        )

    if sources:
        evidence_sections.append(_format_kb_evidence(sources))
    elif retrieval_attempted:
        evidence_sections.append(
            "<kb-status>No usable knowledge-base evidence was retrieved "
            "for this request.</kb-status>"
        )

    question_section = _format_questions(message, sub_questions)
    if not evidence_sections:
        return message
    return "\n\n".join([*evidence_sections, question_section])


def _format_kb_evidence(sources: list[RetrievedSource]) -> str:
    documents: list[str] = []
    for index, source in enumerate(sources, start=1):
        attributes = [f'index="{index}"', f'title="{escape(source.title or "", quote=True)}"']
        if source.url:
            attributes.append(f'source="{escape(source.url, quote=True)}"')
        if source.collection_name:
            attributes.append(
                f'collection="{escape(source.collection_name, quote=True)}"'
            )
        documents.append(
            f"<document {' '.join(attributes)}>\n{source.content.strip()}\n</document>"
        )
    return "<documents>\n" + "\n\n".join(documents) + "\n</documents>"


def _format_questions(message: str, sub_questions: list[str]) -> str:
    questions = [item.strip() for item in sub_questions if item and item.strip()]
    if len(questions) <= 1:
        question = questions[0] if questions else message.strip()
        return f"<question>{question}</question>"
    numbered = "\n".join(
        f"{index}. {question}" for index, question in enumerate(questions, start=1)
    )
    return f"<questions>\n{numbered}\n</questions>"


def _render_rules(snippets: list[str]) -> str:
    unique: list[str] = []
    for snippet in snippets:
        normalized = snippet.strip()
        if normalized and normalized not in unique:
            unique.append(normalized)
    if not unique:
        return ""
    numbered = "\n".join(f"{index}. {item}" for index, item in enumerate(unique, start=1))
    return f"<rules>\n{numbered}\n</rules>"
