from __future__ import annotations

import pytest

from agent_service.consumers.knowledge_chunk_consumer import KnowledgeChunkConsumer
from agent_service.repositories.knowledge_chunk_repository import (
    ChunkLogStart,
    KnowledgeDocumentForChunk,
)
from agent_service.schemas.chunking import TextChunk, VectorChunk
from agent_service.services.knowledge_chunk_pipeline import KnowledgeChunkPipeline


class FakeChunkPipelineRepository:
    def __init__(self, doc: KnowledgeDocumentForChunk | None) -> None:
        self.doc = doc
        self.logs: list[ChunkLogStart] = []
        self.success_log = None
        self.failed_log = None
        self.failed_doc = None
        self.persisted = None

    def find_document_for_chunk(self, doc_id: str):
        assert doc_id == "doc-1"
        return self.doc

    def insert_chunk_log(self, record: ChunkLogStart) -> None:
        self.logs.append(record)

    def mark_chunk_log_success(self, **kwargs) -> None:
        self.success_log = kwargs

    def mark_chunk_log_failed(self, **kwargs) -> None:
        self.failed_log = kwargs

    def mark_document_failed(self, **kwargs) -> None:
        self.failed_doc = kwargs

    def replace_chunks_and_vectors_atomic(self, **kwargs) -> None:
        self.persisted = kwargs


class FakeObjectReader:
    def read(self, file_url: str) -> bytes:
        assert file_url == "s3://kb/doc.txt"
        return b"fake file"


class FakeTextExtractor:
    async def extract_text(self, content: bytes, file_name: str | None = None) -> str:
        assert content == b"fake file"
        assert file_name == "manual.txt"
        return "alpha beta gamma delta"


class FakeEmbeddingService:
    async def embed_chunks(self, *, doc_id: str, kb_id: str, chunks: list[TextChunk]):
        return [
            VectorChunk(
                chunk_id=chunk.chunk_id,
                doc_id=doc_id,
                kb_id=kb_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                vector=[0.1] * 1024,
                metadata=chunk.metadata,
            )
            for chunk in chunks
        ]


@pytest.mark.asyncio
async def test_mock_chunk_consumer_runs_pipeline_successfully() -> None:
    doc = KnowledgeDocumentForChunk(
        id="doc-1",
        kb_id="kb-1",
        doc_name="manual.txt",
        file_url="s3://kb/doc.txt",
        process_mode="CHUNK",
        status="RUNNING",
        chunk_strategy="fixed_size",
        chunk_config={"targetChars": 8, "overlapChars": 0},
        created_by="agent",
    )
    repository = FakeChunkPipelineRepository(doc)
    pipeline = KnowledgeChunkPipeline(
        repository=repository,
        object_reader=FakeObjectReader(),
        text_extractor=FakeTextExtractor(),
        embedding_service=FakeEmbeddingService(),
    )
    consumer = KnowledgeChunkConsumer(pipeline)

    result = await consumer.process_mock_chunk_message(
        {
            "docId": "doc-1",
            "messageId": "msg-1",
            "requestedBy": "tester",
        }
    )

    assert result.status == "SUCCESS"
    assert result.doc_id == "doc-1"
    assert result.kb_id == "kb-1"
    assert result.chunk_count >= 1
    assert len(result.vectors) == result.chunk_count
    assert repository.logs[0].message_id == "msg-1"
    assert repository.logs[0].doc_id == "doc-1"
    assert repository.success_log is not None
    assert repository.success_log["chunk_count"] == result.chunk_count
    assert repository.failed_log is None
    assert repository.persisted is not None
    assert repository.persisted["updated_by"] == "tester"
    assert repository.persisted["doc"].id == "doc-1"
    assert len(repository.persisted["chunks"]) == result.chunk_count
    assert len(repository.persisted["vectors"]) == result.chunk_count


@pytest.mark.asyncio
async def test_mock_chunk_consumer_marks_log_and_document_failed() -> None:
    doc = KnowledgeDocumentForChunk(
        id="doc-1",
        kb_id="kb-1",
        doc_name="manual.txt",
        file_url="s3://kb/doc.txt",
        process_mode="CHUNK",
        status="RUNNING",
        chunk_strategy="fixed_size",
        chunk_config={"targetChars": 8},
        created_by="agent",
    )
    repository = FakeChunkPipelineRepository(doc)

    class FailingReader:
        def read(self, file_url: str) -> bytes:
            raise RuntimeError("storage down")

    pipeline = KnowledgeChunkPipeline(
        repository=repository,
        object_reader=FailingReader(),
        text_extractor=FakeTextExtractor(),
        embedding_service=FakeEmbeddingService(),
    )
    consumer = KnowledgeChunkConsumer(pipeline)

    with pytest.raises(RuntimeError, match="storage down"):
        await consumer.process_mock_chunk_message({"docId": "doc-1", "messageId": "msg-1"})

    assert repository.failed_log is not None
    assert repository.failed_log["error_message"] == "storage down"
    assert repository.failed_doc == {"doc_id": "doc-1", "updated_by": "agent"}
    assert repository.persisted is None
