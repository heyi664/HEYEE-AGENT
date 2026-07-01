from __future__ import annotations

from agent_service.infra_ai.models import ModelCapability, ModelTarget, ProviderConfig


def resolve_model_url(target: ModelTarget) -> str:
    candidate_url = target.candidate.url
    if candidate_url:
        return candidate_url
    provider = target.provider
    if not provider.url:
        raise ValueError("Provider base URL is missing")
    path = _endpoint_path(provider, target.capability)
    return _join_url(provider.url, path)


def _endpoint_path(provider: ProviderConfig, capability: ModelCapability) -> str:
    path = provider.endpoints.get(capability.value)
    if not path:
        raise ValueError(f"Provider endpoint is missing: {capability.value}")
    return path


def _join_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + "/" + path.lstrip("/")
