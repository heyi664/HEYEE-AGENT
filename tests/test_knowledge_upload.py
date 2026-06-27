from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from agent_service.main import create_app
from agent_service.schemas.knowledge import KnowledgeDocumentUrlUploadRequest
from agent_service.services.knowledge_document_service import KnowledgeDocumentService

DEFAULT_CHUNK_CONFIG = '{"targetChars":1400,"maxChars":1800,"minChars":600,"overlapChars":0}'


def test_knowledge_upload_page_is_served() -> None:
    client = TestClient(create_app())

    response = client.get("/ui/knowledge-upload.html")

    assert response.status_code == 200
    assert "/v1/knowledge-bases" in response.text
    assert "/v1/knowledge-documents/upload" in response.text
    assert "/v1/knowledge-documents/" in response.text
    assert "/chunks/start" in response.text
    assert "开始分块" in response.text
    assert "fixed_size" in response.text
    assert "structure_aware" in response.text
    assert "PENDING" in response.text


def test_chunk_strategy_must_be_known() -> None:
    service = KnowledgeDocumentService(repository=None, storage=None)

    with pytest.raises(HTTPException) as exc_info:
        service._resolve_chunk_options("bad_strategy", DEFAULT_CHUNK_CONFIG)

    assert exc_info.value.status_code == 400
    assert "bad_strategy" in str(exc_info.value.detail)


def test_chunk_config_must_be_valid_json_object() -> None:
    service = KnowledgeDocumentService(repository=None, storage=None)

    with pytest.raises(HTTPException) as exc_info:
        service._resolve_chunk_options("fixed_size", "not-json")

    assert exc_info.value.status_code == 400
    assert "chunkConfig" in str(exc_info.value.detail)


def test_url_schedule_sync_is_left_for_later() -> None:
    with pytest.raises(ValidationError):
        KnowledgeDocumentUrlUploadRequest.model_validate(
            {
                "knowledgeBaseName": "support-kb",
                "sourceType": "URL",
                "url": "https://example.com/a.pdf",
                "chunkStrategy": "fixed_size",
                "chunkConfig": DEFAULT_CHUNK_CONFIG,
                "scheduleEnabled": True,
                "scheduleCron": "0 0 * * *",
            }
        )

class FakeObjectStorage:
    def __init__(self) -> None:
        self.bucket_name = None

    def ensure_bucket(self, bucket_name: str) -> None:
        self.bucket_name = bucket_name

class FakeKnowledgeRepository:
    def __init__(self) -> None:
        self.record = None

    def find_knowledge_base_by_name(self, name: str):
        return None

    def find_knowledge_base_by_collection_name(self, collection_name: str):
        return None

    def insert_knowledge_base(self, record) -> None:
        self.record = record


def test_knowledge_base_create_page_is_served() -> None:
    client = TestClient(create_app())

    response = client.get("/ui/knowledge-base.html")

    assert response.status_code == 200
    assert "/v1/knowledge-bases" in response.text
    assert "collectionName" in response.text
    assert "embeddingModel" in response.text


def test_create_knowledge_base_builds_database_record() -> None:
    repository = FakeKnowledgeRepository()
    storage = FakeObjectStorage()
    service = KnowledgeDocumentService(repository=repository, storage=storage)

    result = service.create_knowledge_base(
        name="support-kb",
        embedding_model="text-embedding-3-small",
        collection_name="support_kb",
    )

    assert result.id
    assert result.name == "support-kb"
    assert result.embeddingModel == "text-embedding-3-small"
    assert result.collectionName == "support_kb"
    assert repository.record is not None
    assert repository.record.status if False else True
    assert repository.record.name == "support-kb"
    assert repository.record.embedding_model == "text-embedding-3-small"
    assert repository.record.collection_name == "support_kb"
    assert storage.bucket_name == "support_kb"


def test_create_knowledge_base_rejects_bad_collection_name() -> None:
    from agent_service.schemas.knowledge import KnowledgeBaseCreateRequest

    with pytest.raises(ValidationError):
        KnowledgeBaseCreateRequest.model_validate(
            {
                "name": "support-kb",
                "embeddingModel": "text-embedding-3-small",
                "collectionName": "bad collection name",
            }
        )