from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

from agent_service.core.config import get_settings
from agent_service.infra_ai.models import (
    AIModelConfig,
    ModelCandidate,
    ModelGroup,
    ProviderConfig,
)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _candidate_from_dict(value: Any) -> ModelCandidate:
    data = _as_dict(value)
    return ModelCandidate(
        id=_optional_str(data.get("id")),
        provider=str(data.get("provider") or "").strip(),
        model=str(data.get("model") or "").strip(),
        priority=_optional_int(data.get("priority")),
        enabled=bool(data.get("enabled", True)),
        url=_optional_str(data.get("url")),
        supports_tools=bool(data.get("supports_tools", data.get("supportsTools", False))),
        supports_thinking=bool(
            data.get("supports_thinking", data.get("supportsThinking", False))
        ),
        dimension=_optional_int(data.get("dimension")),
        timeout_seconds=_optional_float(data.get("timeout_seconds", data.get("timeoutSeconds"))),
    )


def _group_from_dict(value: Any) -> ModelGroup:
    data = _as_dict(value)
    return ModelGroup(
        default_model=_optional_str(data.get("default_model", data.get("defaultModel"))),
        deep_thinking_model=_optional_str(
            data.get("deep_thinking_model", data.get("deepThinkingModel"))
        ),
        candidates=[
            _candidate_from_dict(candidate)
            for candidate in data.get("candidates", [])
            if isinstance(candidate, dict)
        ],
    )


def _provider_from_dict(name: str, value: Any) -> ProviderConfig:
    data = _as_dict(value)
    api_key = _optional_str(data.get("api_key", data.get("apiKey")))
    api_key_env = _optional_str(data.get("api_key_env", data.get("apiKeyEnv")))
    if not api_key and api_key_env:
        api_key = os.getenv(api_key_env)
    endpoints = data.get("endpoints")
    return ProviderConfig(
        name=name,
        url=_optional_str(data.get("url")),
        api_key=api_key,
        endpoints={str(key): str(path) for key, path in _as_dict(endpoints).items()},
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


@lru_cache
def get_ai_model_config() -> AIModelConfig:
    settings = get_settings()
    if settings.ai_models_json:
        raw = json.loads(settings.ai_models_json)
        data = _as_dict(raw)
        providers = {
            str(name): _provider_from_dict(str(name), provider)
            for name, provider in _as_dict(data.get("providers")).items()
        }
        return AIModelConfig(
            providers=providers,
            chat=_group_from_dict(data.get("chat")),
            embedding=_group_from_dict(data.get("embedding")),
            rerank=_group_from_dict(data.get("rerank")),
        )
    return _legacy_config(settings)


def _legacy_config(settings: Any) -> AIModelConfig:
    providers = {
        settings.ai_provider: ProviderConfig(
            name=settings.ai_provider,
            url=settings.ai_base_url,
            api_key=settings.ai_api_key,
            endpoints={"chat": "/chat/completions"},
        ),
        settings.embedding_provider: ProviderConfig(
            name=settings.embedding_provider,
            url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            endpoints={"embedding": "/embeddings"},
        ),
        "noop": ProviderConfig(name="noop"),
    }
    chat = ModelGroup(
        default_model=settings.ai_model,
        candidates=[
            ModelCandidate(
                id=settings.ai_model,
                provider=settings.ai_provider,
                model=settings.ai_model,
                priority=1,
                supports_tools=True,
                timeout_seconds=settings.ai_timeout_seconds,
            )
        ],
    )
    embedding = ModelGroup(
        default_model=settings.embedding_model,
        candidates=[
            ModelCandidate(
                id=settings.embedding_model,
                provider=settings.embedding_provider,
                model=settings.embedding_model,
                priority=1,
                dimension=settings.embedding_dimension,
                timeout_seconds=settings.embedding_timeout_seconds,
            )
        ],
    )
    rerank = ModelGroup(
        candidates=[
            ModelCandidate(
                id="rerank-noop",
                provider="noop",
                model="noop",
                priority=100,
            )
        ]
    )
    return AIModelConfig(providers=providers, chat=chat, embedding=embedding, rerank=rerank)

