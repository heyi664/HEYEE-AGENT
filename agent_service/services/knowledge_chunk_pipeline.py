from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agent_service.core.config import get_settings
from agent_service.repositories.knowledge_chunk_repository import (
    ChunkLogStart,
    KnowledgeChunkRepository,
)
from agent_service.schemas.chunking import KnowledgeChunkMessage, VectorChunk
from agent_service.services.chunking_service import ChunkingService
from agent_service.services.document_object_reader import DocumentObjectReader
from agent_service.services.embedding_service import EmbeddingService
from agent_service.services.tika_text_extractor import TikaTextExtractor


@dataclass(frozen=True)
class ChunkPipelineResult:
    doc_id: str
    kb_id: str
    log_id: str
    pipeline_id: str
    status: str
    chunk_count: int
    vectors: list[VectorChunk]


class KnowledgeChunkPipeline:
    def __init__(
        self,
        *,
        repository: KnowledgeChunkRepository | None = None,
        object_reader: DocumentObjectReader | None = None,
        text_extractor: TikaTextExtractor | None = None,
        chunking_service: ChunkingService | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self.repository = repository or KnowledgeChunkRepository()
        self.object_reader = object_reader or DocumentObjectReader()
        self.text_extractor = text_extractor or TikaTextExtractor()
        self.chunking_service = chunking_service or ChunkingService()
        self.embedding_service = embedding_service or EmbeddingService()

    async def process_message(self, message: KnowledgeChunkMessage) -> ChunkPipelineResult:
        started = _now_ms()
        doc = self.repository.find_document_for_chunk(message.doc_id)
        if doc is None:
            raise ValueError(f"document not found: {message.doc_id}")
        if doc.status == "SUCCESS":
            return ChunkPipelineResult(
                doc_id=doc.id,
                kb_id=doc.kb_id,
                log_id="",
                pipeline_id="",
                status="SKIPPED",
                chunk_count=0,
                vectors=[],
            )
        if doc.status != "RUNNING":
            raise ValueError(f"document status must be RUNNING, got {doc.status}")

        settings = get_settings()
        requested_by = message.requested_by or settings.upload_created_by
        pipeline_id = uuid.uuid4().hex[:20]
        log_id = uuid.uuid4().hex[:20]
        self.repository.insert_chunk_log(
            ChunkLogStart(
                id=log_id,
                doc_id=doc.id,
                message_id=message.message_id,
                process_mode=doc.process_mode,
                chunk_strategy=doc.chunk_strategy,
                pipeline_id=pipeline_id,
            )
        )

        extract_duration = 0
        chunk_duration = 0
        embed_duration = 0
        persist_duration = 0
        try:
            stage = _now_ms()
            file_bytes = await _with_retry(
                lambda: self.object_reader.read(message.file_url or doc.file_url),
                retries=settings.chunk_pipeline_max_retries,
                backoff_seconds=settings.chunk_pipeline_retry_backoff_seconds,
            )
            text = await _with_retry(
                lambda: self.text_extractor.extract_text(file_bytes, doc.doc_name),
                retries=settings.chunk_pipeline_max_retries,
                backoff_seconds=settings.chunk_pipeline_retry_backoff_seconds,
            )
            extract_duration = _elapsed_ms(stage)

            strategy = message.chunk_strategy or doc.chunk_strategy
            config = message.chunk_config or doc.chunk_config
            stage = _now_ms()
            chunks = self.chunking_service.split(text, strategy, config)
            chunk_duration = _elapsed_ms(stage)

            stage = _now_ms()
            vectors = await _with_retry(
                lambda: self.embedding_service.embed_chunks(
                    doc_id=doc.id,
                    kb_id=doc.kb_id,
                    chunks=chunks,
                ),
                retries=settings.chunk_pipeline_max_retries,
                backoff_seconds=settings.chunk_pipeline_retry_backoff_seconds,
            )
            embed_duration = _elapsed_ms(stage)

            stage = _now_ms()
            await _with_retry(
                lambda: self.repository.replace_chunks_and_vectors_atomic(
                    doc=doc,
                    chunks=chunks,
                    vectors=vectors,
                    updated_by=requested_by,
                ),
                retries=settings.chunk_pipeline_max_retries,
                backoff_seconds=settings.chunk_pipeline_retry_backoff_seconds,
            )
            persist_duration = _elapsed_ms(stage)
            total_duration = _elapsed_ms(started)
            self.repository.mark_chunk_log_success(
                log_id=log_id,
                extract_duration=extract_duration,
                chunk_duration=chunk_duration,
                embed_duration=embed_duration,
                persist_duration=persist_duration,
                total_duration=total_duration,
                chunk_count=len(chunks),
            )
            return ChunkPipelineResult(
                doc_id=doc.id,
                kb_id=doc.kb_id,
                log_id=log_id,
                pipeline_id=pipeline_id,
                status="SUCCESS",
                chunk_count=len(chunks),
                vectors=vectors,
            )
        except Exception as exc:
            total_duration = _elapsed_ms(started)
            self.repository.mark_chunk_log_failed(
                log_id=log_id,
                error_message=str(exc),
                total_duration=total_duration,
            )
            self.repository.mark_document_failed(doc_id=doc.id, updated_by=requested_by)
            raise


def _now_ms() -> int:
    return time.perf_counter_ns() // 1_000_000


def _elapsed_ms(started_ms: int) -> int:
    return max(0, _now_ms() - started_ms)


async def _with_retry(
    operation: Callable[[], Any],
    *,
    retries: int,
    backoff_seconds: float,
) -> Any:
    attempt = 0
    while True:
        try:
            result = operation()
            if inspect.isawaitable(result):
                return await result
            return result
        except Exception:
            if attempt >= retries:
                raise
            attempt += 1
            if backoff_seconds > 0:
                await asyncio.sleep(backoff_seconds * attempt)
