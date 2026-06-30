from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from agent_service.consumers.knowledge_chunk_consumer import KnowledgeChunkConsumer
from agent_service.services.knowledge_chunk_pipeline import ChunkPipelineResult

router = APIRouter(tags=["knowledge-chunk-dev"])


@router.post("/knowledge-documents/chunks/mock-consume")
async def consume_mock_chunk_message(payload: dict[str, Any]) -> ChunkPipelineResult:
    return await KnowledgeChunkConsumer().process_mock_chunk_message(payload)
