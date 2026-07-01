from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ModelCapability(StrEnum):
    CHAT = "chat"
    EMBEDDING = "embedding"
    RERANK = "rerank"


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    url: str | None = None
    api_key: str | None = None
    endpoints: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelCandidate:
    provider: str
    model: str
    id: str | None = None
    priority: int | None = None
    enabled: bool = True
    url: str | None = None
    supports_tools: bool = False
    supports_thinking: bool = False
    dimension: int | None = None
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class ModelGroup:
    default_model: str | None = None
    deep_thinking_model: str | None = None
    candidates: list[ModelCandidate] = field(default_factory=list)


@dataclass(frozen=True)
class AIModelConfig:
    providers: dict[str, ProviderConfig]
    chat: ModelGroup
    embedding: ModelGroup
    rerank: ModelGroup = field(default_factory=ModelGroup)


@dataclass(frozen=True)
class ModelTarget:
    id: str
    capability: ModelCapability
    candidate: ModelCandidate
    provider: ProviderConfig
