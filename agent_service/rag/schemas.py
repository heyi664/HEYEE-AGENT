from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class RetrievedChunk(BaseModel):
    id: str
    text: str
    score: float | None = None


class RetrievedSource(BaseModel):
    title: str
    content: str
    score: float | None = None
    source_type: str | None = None
    url: str | None = None


class RewriteResult(BaseModel):
    original_question: str
    rewritten_question: str
    sub_questions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_sub_questions(self) -> RewriteResult:
        self.rewritten_question = self.rewritten_question.strip()
        cleaned = [item.strip() for item in self.sub_questions if item.strip()]
        self.sub_questions = cleaned or [self.rewritten_question]
        return self
