from __future__ import annotations

from agent_service.core.config import Settings


def test_rag_query_rewrite_settings_defaults() -> None:
    settings = Settings()

    assert settings.rag_query_rewrite_enabled is False
    assert settings.rag_query_rewrite_history_turns == 2
    assert settings.rag_query_rewrite_max_sub_questions == 5
    assert settings.rag_term_mapping_cache_ttl_seconds == 300
    assert settings.rag_query_rewrite_history_max_chars == 1500
    assert settings.rag_query_rewrite_history_message_max_chars == 500
