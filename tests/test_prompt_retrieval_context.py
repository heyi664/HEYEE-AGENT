from __future__ import annotations

from agent_service.memory.models import MemoryContext
from agent_service.rag.schemas import RetrievedSource
from agent_service.services.prompt_service import build_messages


def test_build_messages_adds_retrieved_sources_next_to_the_user_question() -> None:
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

    assert "only source of business facts" in messages[0]["content"]
    assert "Eligible items may be returned within seven days." in messages[-1]["content"]
    assert messages[-1]["content"].endswith("<question>return policy</question>")


def test_build_messages_requires_insufficient_knowledge_response_when_search_is_empty() -> None:
    messages = build_messages(
        MemoryContext(),
        "return policy",
        retrieval_attempted=True,
    )

    assert "<kb-status>" in messages[-1]["content"]
    assert "No usable knowledge-base evidence" in messages[-1]["content"]
