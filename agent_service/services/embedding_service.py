from __future__ import annotations

import httpx

from agent_service.core.config import get_settings
from agent_service.core.errors import ModelUnavailableError
from agent_service.schemas.chunking import TextChunk, VectorChunk


class EmbeddingService:
    async def embed_chunks(
        self,
        *,
        doc_id: str,
        kb_id: str,
        chunks: list[TextChunk],
    ) -> list[VectorChunk]:
        settings = get_settings()
        if not settings.embedding_api_key:
            raise ModelUnavailableError("EMBEDDING_API_KEY is not configured")

        vectors: list[VectorChunk] = []
        for batch in _batches(chunks, settings.embedding_batch_size):
            embeddings = await self._embed_texts([chunk.content for chunk in batch])
            if len(embeddings) != len(batch):
                raise ModelUnavailableError("Embedding API returned an unexpected result count")
            for chunk, vector in zip(batch, embeddings, strict=True):
                if len(vector) != settings.embedding_dimension:
                    raise ModelUnavailableError(
                        "Embedding dimension mismatch: "
                        f"expected {settings.embedding_dimension}, got {len(vector)}"
                    )
                vectors.append(
                    VectorChunk(
                        chunk_id=chunk.chunk_id,
                        doc_id=doc_id,
                        kb_id=kb_id,
                        chunk_index=chunk.chunk_index,
                        content=chunk.content,
                        vector=vector,
                        metadata=chunk.metadata,
                    )
                )
        return vectors

    async def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        settings = get_settings()
        url = settings.embedding_base_url.rstrip("/") + "/embeddings"
        payload = {"model": settings.embedding_model, "input": texts}
        headers = {
            "Authorization": f"Bearer {settings.embedding_api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=settings.embedding_timeout_seconds) as client:
            response = await client.post(url, json=payload, headers=headers)
        if response.is_error:
            raise ModelUnavailableError(
                f"Embedding API failed status={response.status_code} body={response.text[:500]}"
            )
        data = response.json()
        rows = data.get("data")
        if not isinstance(rows, list):
            raise ModelUnavailableError("Embedding API response missing data")
        rows = sorted(
            rows,
            key=lambda row: int(row.get("index", 0)) if isinstance(row, dict) else 0,
        )
        embeddings: list[list[float]] = []
        for row in rows:
            embedding = row.get("embedding") if isinstance(row, dict) else None
            if not isinstance(embedding, list):
                raise ModelUnavailableError("Embedding API response missing embedding")
            embeddings.append([float(value) for value in embedding])
        return embeddings


def _batches(items: list[TextChunk], size: int) -> list[list[TextChunk]]:
    return [items[index : index + size] for index in range(0, len(items), size)]
