from __future__ import annotations

from agent_service.memory.models import MemoryContext
from agent_service.rag.schemas import RetrievedSource
from agent_service.services.prompt_service import build_messages


def test_build_messages_adds_retrieved_sources_to_the_system_prompt() -> None:
    messages = build_messages(
        MemoryContext(),
        "return policy",
        retrieved_sources=[
            RetrievedSource(
                title="return-policy.md",
                content="Eligible items may be returned within seven days.",
            )
        ],
        retrieval_attempted=True,
    )

    assert "Answer only from this context." in messages[0]["content"]
    assert "Eligible items may be returned within seven days." in messages[0]["content"]


def test_build_messages_requires_insufficient_knowledge_response_when_search_is_empty() -> None:
    messages = build_messages(
        MemoryContext(),
        "return policy",
        retrieval_attempted=True,
    )

    assert "does not provide enough information" in messages[0]["content"]
    assert "Do not answer from general knowledge" in messages[0]["content"]
