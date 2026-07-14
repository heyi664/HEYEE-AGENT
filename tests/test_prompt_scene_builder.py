from __future__ import annotations

from agent_service.memory.models import MemoryContext, MemoryMessage
from agent_service.rag.schemas import RetrievedSource
from agent_service.services.prompt_service import PromptScene, build_messages, build_prompt_plan


def _source() -> RetrievedSource:
    return RetrievedSource(
        id="chunk_1",
        title="return-policy.md",
        content="Returns are accepted within seven days.",
        url="https://example.test/return-policy",
        collection_name="return-policy",
    )


def test_kb_scene_keeps_evidence_in_the_last_user_message_after_history() -> None:
    messages = build_messages(
        MemoryContext(messages=[MemoryMessage(id="1", role="user", content="history")]),
        "What is the return policy?",
        retrieved_sources=[_source()],
    )

    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "history"}
    assert "<documents>" in messages[-1]["content"]
    assert "Returns are accepted within seven days." in messages[-1]["content"]
    assert messages[-1]["content"].endswith("<question>What is the return policy?</question>")
    assert "Returns are accepted within seven days." not in messages[0]["content"]

    plan = build_prompt_plan("What is the return policy?", retrieved_sources=[_source()])
    assert (plan.temperature, plan.top_p) == (0.0, 1.0)


def test_mixed_scene_declares_live_mcp_data_as_preferred_when_facts_conflict() -> None:
    plan = build_prompt_plan(
        "What is my refund status?",
        retrieved_sources=[_source()],
        mcp_context='[MCP tool: refund_query]\nReal-time result:\n{"status":"PROCESSING"}',
    )

    assert plan.scene is PromptScene.MIXED
    assert "<tool-data> value takes precedence" in plan.system_prompt
    assert "<tool-data>" in plan.user_content
    assert "<documents>" in plan.user_content
    assert (plan.temperature, plan.top_p) == (0.3, 0.8)


def test_multi_question_prompt_numbers_questions_and_deduplicates_rule_snippets() -> None:
    plan = build_prompt_plan(
        "original user question",
        retrieved_sources=[_source()],
        sub_questions=["What is the policy?", "What is the current status?"],
        prompt_snippets=["Use YYYY-MM-DD dates.", "Use YYYY-MM-DD dates.", "Keep two decimals."],
    )

    assert "<questions>" in plan.user_content
    assert "1. What is the policy?" in plan.user_content
    assert "2. What is the current status?" in plan.user_content
    assert plan.system_prompt.count("Use YYYY-MM-DD dates.") == 1
    assert "2. Keep two decimals." in plan.system_prompt


def test_single_intent_template_receives_snippet_rules_when_it_has_no_rules_slot() -> None:
    plan = build_prompt_plan(
        "question",
        retrieved_sources=[_source()],
        prompt_template="Custom answer policy.",
        prompt_snippets=["Never disclose phone numbers."],
    )

    assert plan.system_prompt == (
        "Custom answer policy.\n\n<rules>\n1. Never disclose phone numbers.\n</rules>"
    )


def test_general_chat_uses_a_more_expressive_generation_profile() -> None:
    plan = build_prompt_plan("Tell me about your capabilities.")

    assert plan.scene is PromptScene.EMPTY
    assert (plan.temperature, plan.top_p) == (0.7, None)
