from __future__ import annotations

from agent_service.rag.schemas import RewriteResult


def test_rewrite_result_defaults_sub_questions_to_rewritten_question() -> None:
    result = RewriteResult(original_question="它保修多久？", rewritten_question="iPhone 保修多久？")

    assert result.sub_questions == ["iPhone 保修多久？"]


def test_rewrite_result_removes_blank_sub_questions() -> None:
    result = RewriteResult(
        original_question="问 A；问 B",
        rewritten_question="问 A；问 B",
        sub_questions=["问 A", " ", "问 B"],
    )

    assert result.sub_questions == ["问 A", "问 B"]
