from __future__ import annotations

from agent_service.rag.schemas import RetrievedSource


class Retriever:
    async def search(self, query: str, top_k: int = 5) -> list[RetrievedSource]:
        return []

