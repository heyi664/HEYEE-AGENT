from __future__ import annotations

from typing import cast

from agent_service.core.config import get_settings
from agent_service.core.errors import ModelUnavailableError
from agent_service.infra_ai import get_model_routing_executor, get_model_selector
from agent_service.infra_ai.clients import EmbeddingModelClient, EmbeddingModelClientRegistry
from agent_service.infra_ai.models import ModelCapability, ModelTarget
from agent_service.schemas.chunking import TextChunk, VectorChunk


class EmbeddingService:
    def __init__(self, client_registry: EmbeddingModelClientRegistry | None = None) -> None:
        self._selector = get_model_selector()
        self._routing_executor = get_model_routing_executor()
        self._client_registry = client_registry or EmbeddingModelClientRegistry()

    async def embed(self, text: str, model_id: str | None = None) -> list[float]:
        vectors = await self.embed_batch([text], model_id=model_id)
        if not vectors:
            raise ModelUnavailableError("Embedding API returned no vectors")
        return vectors[0]

    async def embed_batch(
        self,
        texts: list[str],
        model_id: str | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        embeddings, _expected_dimension = await self._embed_texts(texts, model_id=model_id)
        return embeddings

    def dimension(self, model_id: str | None = None) -> int:
        settings = get_settings()
        if model_id is None:
            targets = self._selector.select_embedding_candidates()
            if targets:
                return targets[0].candidate.dimension or settings.embedding_dimension
            return settings.embedding_dimension
        return self._resolve_target(model_id).candidate.dimension or settings.embedding_dimension

    async def embed_chunks(
        self,
        *,
        doc_id: str,
        kb_id: str,
        chunks: list[TextChunk],
    ) -> list[VectorChunk]:
        settings = get_settings()

        vectors: list[VectorChunk] = []
        for batch in _batches(chunks, settings.embedding_batch_size):
            embeddings, expected_dimension = await self._embed_texts(
                [chunk.content for chunk in batch]
            )
            if len(embeddings) != len(batch):
                raise ModelUnavailableError("Embedding API returned an unexpected result count")
            for chunk, vector in zip(batch, embeddings, strict=True):
                if len(vector) != expected_dimension:
                    raise ModelUnavailableError(
                        "Embedding dimension mismatch: "
                        f"expected {expected_dimension}, got {len(vector)}"
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

    async def _embed_texts(
        self,
        texts: list[str],
        model_id: str | None = None,
    ) -> tuple[list[list[float]], int]:
        settings = get_settings()
        targets = (
            [self._resolve_target(model_id)]
            if model_id is not None
            else self._selector.select_embedding_candidates()
        )
        return await self._routing_executor.execute_with_fallback(
            cast(ModelCapability, ModelCapability.EMBEDDING),
            targets,
            self._client_registry.resolve,
            lambda client, target: self._embed_with_client(
                client,
                target,
                texts,
                settings.embedding_dimension,
            ),
        )

    async def _embed_with_client(
        self,
        client: EmbeddingModelClient,
        target: ModelTarget,
        texts: list[str],
        fallback_dimension: int,
    ) -> tuple[list[list[float]], int]:
        embeddings = await client.embed_batch(target, texts)
        return embeddings, target.candidate.dimension or fallback_dimension

    def _resolve_target(self, model_id: str) -> ModelTarget:
        if not model_id.strip():
            raise ModelUnavailableError("Embedding model ID must not be empty")
        for target in self._selector.select_embedding_candidates():
            if target.id == model_id:
                return target
        raise ModelUnavailableError(f"Embedding model is unavailable: {model_id}")


def _batches(items: list[TextChunk], size: int) -> list[list[TextChunk]]:
    return [items[index : index + size] for index in range(0, len(items), size)]
