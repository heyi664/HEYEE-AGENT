from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

ALLOWED_CHUNK_STRATEGIES = {"fixed_size", "structure_aware"}


class KnowledgeBaseSummary(BaseModel):
    id: str
    name: str
    collectionName: str | None = None



class KnowledgeBaseCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    embeddingModel: str = Field(min_length=1, max_length=64)
    collectionName: str = Field(min_length=1, max_length=64)

    @field_validator("name", "embeddingModel", "collectionName")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("field must not be blank")
        return value

    @field_validator("collectionName")
    @classmethod
    def collection_name_must_be_safe(cls, value: str) -> str:
        import re

        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
            raise ValueError(
                "collectionName only supports letters, numbers, underscores and hyphens"
            )
        return value


class KnowledgeBaseCreateResponse(BaseModel):
    id: str
    name: str
    embeddingModel: str
    collectionName: str
    createdBy: str

class KnowledgeDocumentUrlUploadRequest(BaseModel):
    knowledgeBaseName: str = Field(min_length=1)
    sourceType: Literal["URL"]
    url: str = Field(min_length=1)
    chunkStrategy: str = Field(min_length=1)
    chunkConfig: str = Field(min_length=2)
    scheduleEnabled: bool = False
    scheduleCron: str | None = None

    @field_validator("knowledgeBaseName", "url", "chunkStrategy", "chunkConfig")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("field must not be blank")
        return value

    @model_validator(mode="after")
    def reject_schedule_for_now(self) -> KnowledgeDocumentUrlUploadRequest:
        # TODO: implement URL schedule sync by writing t_knowledge_document_schedule.
        if self.scheduleEnabled or self.scheduleCron:
            raise ValueError("URL schedule sync is not supported yet")
        return self


class KnowledgeDocumentUploadResult(BaseModel):
    id: str
    knowledgeBaseId: str
    knowledgeBaseName: str
    docName: str
    fileUrl: str
    fileType: str
    fileSize: int
    status: str = "PENDING"
    sourceType: Literal["FILE", "URL"]
    sourceLocation: str | None = None
    processMode: str = "CHUNK"
    chunkStrategy: str
    chunkConfig: dict[str, Any]
    chunkCount: int = 0
