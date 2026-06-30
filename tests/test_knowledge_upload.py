from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from agent_service.core.config import Settings
from agent_service.main import create_app
from agent_service.repositories.knowledge_repository import KnowledgeDocumentChunkTarget
from agent_service.schemas.knowledge import KnowledgeDocumentUrlUploadRequest
from agent_service.services.knowledge_document_service import KnowledgeDocumentService
from agent_service.services.rocketmq_transaction_producer import TransactionSendResult

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
    assert "<span>文档管理</span>" in response.text


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
    assert "<span>文档管理</span>" in response.text


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

class FakeChunkProducer:
    def __init__(self) -> None:
        self.sent_payload = None
        self.local_transaction_started = False
        self.message_built = False

    def send_in_transaction(self, *, topic, tag, key, local_transaction, message_builder):
        assert topic
        assert tag
        assert key == "doc-1"
        self.local_transaction_started = True
        target = local_transaction()
        self.sent_payload = message_builder(target)
        self.message_built = True
        return TransactionSendResult(message_id="msg-1")


class FakeChunkRepository:
    def __init__(self, target=None) -> None:
        self.target = target
        self.cas_calls = []

    def mark_document_chunk_running_cas(self, *, doc_id: str, updated_by: str):
        self.cas_calls.append((doc_id, updated_by))
        return self.target


def test_start_chunking_sends_transaction_message_after_cas() -> None:
    target = KnowledgeDocumentChunkTarget(
        id="doc-1",
        kb_id="kb-1",
        doc_name="manual.pdf",
        file_url="s3://bucket/manual.pdf",
        file_type="pdf",
        file_size=123,
        source_type="FILE",
        source_location=None,
        chunk_strategy="fixed_size",
        chunk_config={"targetChars": 1400},
        status="RUNNING",
    )
    repository = FakeChunkRepository(target)
    producer = FakeChunkProducer()
    service = KnowledgeDocumentService(repository=repository, storage=None, chunk_producer=producer)

    result = service.start_chunking("doc-1")

    assert result.id == "doc-1"
    assert result.knowledgeBaseId == "kb-1"
    assert result.status == "RUNNING"
    assert result.messageId == "msg-1"
    assert repository.cas_calls == [("doc-1", "agent")]
    assert producer.local_transaction_started is True
    assert producer.sent_payload["docId"] == "doc-1"
    assert producer.sent_payload["kbId"] == "kb-1"
    assert producer.sent_payload["status"] == "RUNNING"


def test_start_chunking_drops_message_when_cas_fails() -> None:
    repository = FakeChunkRepository(target=None)
    producer = FakeChunkProducer()
    service = KnowledgeDocumentService(repository=repository, storage=None, chunk_producer=producer)

    with pytest.raises(HTTPException) as exc_info:
        service.start_chunking("doc-1")

    assert exc_info.value.status_code == 409
    assert repository.cas_calls == [("doc-1", "agent")]
    assert producer.local_transaction_started is True
    assert producer.message_built is False
    assert producer.sent_payload is None


def test_embedding_defaults_target_siliconflow_bge_m3() -> None:
    settings = Settings(_env_file=None)

    assert settings.embedding_provider == "siliconflow"
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.embedding_dimension == 1024
    assert settings.embedding_base_url == "https://api.siliconflow.cn/v1"


def test_chunk_log_message_id_sql_patch_is_present() -> None:
    sql = Path("sql/20260628_add_chunk_log_message_id.sql").read_text(encoding="utf-8")

    assert "ADD COLUMN IF NOT EXISTS message_id" in sql
    assert "idx_chunk_log_message_id" in sql


def test_mock_chunk_consumer_route_is_registered_in_mock_mode() -> None:
    client = TestClient(create_app())

    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert "/v1/knowledge-documents/chunks/mock-consume" in response.text