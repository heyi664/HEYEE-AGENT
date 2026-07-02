from __future__ import annotations

from agent_service.services.llm_response_cleaner import LLMResponseCleaner
from agent_service.services.token_counter_service import HeuristicTokenCounterService


def test_heuristic_token_counter_counts_ascii_cjk_and_other_chars() -> None:
    service = HeuristicTokenCounterService()

    assert service.count_tokens("Hello \u4f60\u597d\u4e16\u754c! \U0001f44b") == 7
    assert service.count_tokens("   ") == 0
    assert service.count_tokens("a") == 1


def test_llm_response_cleaner_strips_markdown_code_fence_idempotently() -> None:
    fenced = '```json\n{"intent": "knowledge_retrieval", "confidence": 0.95}\n```'
    plain = '{"intent": "knowledge_retrieval"}'

    assert LLMResponseCleaner.strip_markdown_code_fence(fenced) == (
        '{"intent": "knowledge_retrieval", "confidence": 0.95}'
    )
    assert LLMResponseCleaner.strip_markdown_code_fence(plain) == plain
    assert LLMResponseCleaner.strip_markdown_code_fence(None) is None
