from __future__ import annotations

from agent_service.schemas.chunking import KnowledgeChunkMessage
from agent_service.services.knowledge_chunk_pipeline import (
    ChunkPipelineResult,
    KnowledgeChunkPipeline,
)


class KnowledgeChunkConsumer:
    def __init__(self, pipeline: KnowledgeChunkPipeline | None = None) -> None:
        self.pipeline = pipeline or KnowledgeChunkPipeline()

    async def process_mock_chunk_message(self, payload: dict[str, object]) -> ChunkPipelineResult:
        message = KnowledgeChunkMessage.from_dict(payload)
        return await self.pipeline.process_message(message)
