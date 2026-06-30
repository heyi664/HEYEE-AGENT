from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from agent_service.main import create_app
from agent_service.repositories.knowledge_repository import (
    KnowledgeDocumentDetail,
    KnowledgeDocumentListItem,
    KnowledgeVectorChunkSource,
)
from agent_service.schemas.chunking import TextChunk, VectorChunk
from agent_service.services.knowledge_document_service import KnowledgeDocumentService

DEFAULT_CHUNK_CONFIG = '{"targetChars":1400,"maxChars":1800,"minChars":600,"overlapChars":0}'


class FakeDocumentManagementRepository:
    def __init__(self, document: KnowledgeDocumentDetail | None) -> None:
        self.document = document
        self.updated_config = None
        self.deleted_doc = None
        self.listed_doc_ids: list[str] = []
        self.chunk_sources: list[KnowledgeVectorChunkSource] = []
        self.enabled_atomic = None
        self.documents: list[KnowledgeDocumentListItem] = []

    def list_documents(self):
        return self.documents

    def find_document_detail(self, doc_id: str):
        assert doc_id == "doc-1"
        return self.document

    def update_document_config(self, **kwargs) -> None:
        self.updated_config = kwargs

    def delete_document_atomic(self, **kwargs) -> None:
        self.deleted_doc = kwargs

    def list_vector_chunk_sources(self, doc_id: str):
        self.listed_doc_ids.append(doc_id)
        return self.chunk_sources

    def set_document_enabled_atomic(self, **kwargs) -> None:
        self.enabled_atomic = kwargs


class FakeStorage:
    def __init__(self) -> None:
        self.deleted_file_url = None
        self.deleted_bucket_name = None

    def delete_file_url(self, file_url: str, bucket_name: str | None = None) -> None:
        self.deleted_file_url = file_url
        self.deleted_bucket_name = bucket_name


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.calls = []

    async def embed_chunks(self, *, doc_id: str, kb_id: str, chunks: list[TextChunk]):
        self.calls.append({"doc_id": doc_id, "kb_id": kb_id, "chunks": chunks})
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


def _document(*, status: str = "SUCCESS", enabled: bool = True) -> KnowledgeDocumentDetail:
    return KnowledgeDocumentDetail(
        id="doc-1",
        kb_id="kb-1",
        doc_name="manual.pdf",
        file_url="s3://support_kb/manual.pdf",
        status=status,
        enabled=enabled,
        chunk_strategy="fixed_size",
        chunk_config={"targetChars": 1200},
        collection_name="support_kb",
    )


def test_update_document_config_persists_name_strategy_and_config() -> None:
    repository = FakeDocumentManagementRepository(_document())
    service = KnowledgeDocumentService(repository=repository, storage=FakeStorage())

    result = service.update_document_config(
        "doc-1",
        doc_name="manual-v2.pdf",
        chunk_strategy="fixed_size",
        chunk_config=DEFAULT_CHUNK_CONFIG,
    )

    assert result.id == "doc-1"
    assert result.success is True
    assert repository.updated_config == {
        "doc_id": "doc-1",
        "doc_name": "manual-v2.pdf",
        "chunk_strategy": "fixed_size",
        "chunk_config": {
            "targetChars": 1400,
            "maxChars": 1800,
            "minChars": 600,
            "overlapChars": 0,
        },
        "updated_by": "agent",
    }


def test_update_document_config_rejects_running_document() -> None:
    repository = FakeDocumentManagementRepository(_document(status="RUNNING"))
    service = KnowledgeDocumentService(repository=repository, storage=FakeStorage())

    with pytest.raises(HTTPException) as exc_info:
        service.update_document_config(
            "doc-1",
            doc_name="manual-v2.pdf",
            chunk_strategy="fixed_size",
            chunk_config=DEFAULT_CHUNK_CONFIG,
        )

    assert exc_info.value.status_code == 409
    assert repository.updated_config is None


def test_delete_document_removes_database_records_and_storage_object() -> None:
    storage = FakeStorage()
    repository = FakeDocumentManagementRepository(_document())
    service = KnowledgeDocumentService(repository=repository, storage=storage)

    result = service.delete_document("doc-1")

    assert result.id == "doc-1"
    assert result.success is True
    assert repository.deleted_doc == {
        "doc_id": "doc-1",
        "updated_by": "agent",
    }
    assert storage.deleted_file_url == "s3://support_kb/manual.pdf"
    assert storage.deleted_bucket_name == "support_kb"


@pytest.mark.asyncio
async def test_disable_document_updates_document_and_chunks_and_removes_vectors() -> None:
    repository = FakeDocumentManagementRepository(_document(enabled=True))
    service = KnowledgeDocumentService(repository=repository, storage=FakeStorage())

    result = await service.set_document_enabled("doc-1", False)

    assert result.id == "doc-1"
    assert result.enabled is False
    assert result.success is True
    assert repository.enabled_atomic == {
        "doc_id": "doc-1",
        "enabled": False,
        "vectors": [],
        "updated_by": "agent",
    }
    assert repository.listed_doc_ids == []


@pytest.mark.asyncio
async def test_enable_document_embeds_existing_chunks_before_atomic_write() -> None:
    repository = FakeDocumentManagementRepository(_document(enabled=False))
    repository.chunk_sources = [
        KnowledgeVectorChunkSource(
            chunk_id="chunk-1",
            kb_id="kb-1",
            doc_id="doc-1",
            chunk_index=0,
            content="hello world",
            content_hash="hash-1",
            char_count=11,
            token_count=2,
        )
    ]
    embedding = FakeEmbeddingService()
    service = KnowledgeDocumentService(
        repository=repository,
        storage=FakeStorage(),
        embedding_service=embedding,
    )

    result = await service.set_document_enabled("doc-1", True)

    assert result.id == "doc-1"
    assert result.enabled is True
    assert result.success is True
    assert repository.listed_doc_ids == ["doc-1"]
    assert embedding.calls[0]["doc_id"] == "doc-1"
    assert embedding.calls[0]["chunks"][0].content == "hello world"
    assert len(repository.enabled_atomic["vectors"]) == 1
    assert repository.enabled_atomic["vectors"][0].chunk_id == "chunk-1"


@pytest.mark.asyncio
async def test_enable_document_same_target_state_returns_without_work() -> None:
    repository = FakeDocumentManagementRepository(_document(enabled=True))
    embedding = FakeEmbeddingService()
    service = KnowledgeDocumentService(
        repository=repository,
        storage=FakeStorage(),
        embedding_service=embedding,
    )

    result = await service.set_document_enabled("doc-1", True)

    assert result.id == "doc-1"
    assert result.enabled is True
    assert result.success is True
    assert repository.enabled_atomic is None
    assert repository.listed_doc_ids == []
    assert embedding.calls == []


def test_list_documents_returns_repository_records() -> None:
    repository = FakeDocumentManagementRepository(_document())
    repository.documents = [
        KnowledgeDocumentListItem(
            id="doc-1",
            kb_id="kb-1",
            knowledge_base_name="support-kb",
            doc_name="manual.pdf",
            enabled=True,
            status="SUCCESS",
            chunk_count=3,
            file_url="s3://support_kb/manual.pdf",
            file_type="pdf",
            file_size=2048,
            source_type="FILE",
            source_location=None,
            chunk_strategy="fixed_size",
            chunk_config={"targetChars": 1400},
            create_time="2026-06-30 10:00:00",
            update_time="2026-06-30 10:10:00",
        )
    ]
    service = KnowledgeDocumentService(repository=repository, storage=FakeStorage())

    result = service.list_documents()

    assert len(result) == 1
    assert result[0].id == "doc-1"
    assert result[0].knowledgeBaseName == "support-kb"
    assert result[0].enabled is True
    assert result[0].chunkConfig == {"targetChars": 1400}


def test_knowledge_document_management_page_is_served() -> None:
    client = TestClient(create_app())

    response = client.get("/ui/knowledge-documents.html")

    assert response.status_code == 200
    assert "/v1/knowledge-base/docs" in response.text
    assert "deleteDocument" in response.text
    assert "updateDocument" in response.text
    assert "disableDocument" in response.text
    assert "enableDocument" in response.text
    assert ":disabled=\"scope.row.enabled === false" in response.text
    assert ":disabled=\"scope.row.enabled === true" in response.text
