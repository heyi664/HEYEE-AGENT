from __future__ import annotations

import pytest

from agent_service.infra_ai.health_store import ModelHealthStore
from agent_service.infra_ai.models import (
    AIModelConfig,
    ModelCandidate,
    ModelCapability,
    ModelGroup,
    ProviderConfig,
)
from agent_service.infra_ai.routing_executor import ModelRoutingExecutor
from agent_service.infra_ai.selector import ModelSelector


def test_model_selector_promotes_first_choice_and_filters_disabled() -> None:
    health_store = ModelHealthStore(
        failure_threshold=3,
        open_seconds=60,
        half_open_max_in_flight=1,
    )
    config = AIModelConfig(
        providers={"openai": ProviderConfig(name="openai", url="https://example.com")},
        chat=ModelGroup(
            default_model="preferred",
            candidates=[
                ModelCandidate(
                    id="fallback",
                    provider="openai",
                    model="fallback-model",
                    priority=0,
                    supports_tools=True,
                ),
                ModelCandidate(
                    id="preferred",
                    provider="openai",
                    model="preferred-model",
                    priority=9,
                    supports_tools=True,
                ),
                ModelCandidate(
                    id="disabled",
                    provider="openai",
                    model="disabled-model",
                    enabled=False,
                    priority=1,
                    supports_tools=True,
                ),
            ],
        ),
        embedding=ModelGroup(),
    )

    targets = ModelSelector(config, health_store).select_chat_candidates(require_tools=True)

    assert [target.id for target in targets] == ["preferred", "fallback"]
    assert all(target.capability == ModelCapability.CHAT for target in targets)


@pytest.mark.asyncio
async def test_model_routing_executor_resolves_client_and_falls_back_after_failure() -> None:
    health_store = ModelHealthStore(
        failure_threshold=3,
        open_seconds=60,
        half_open_max_in_flight=1,
    )
    provider = ProviderConfig(name="openai", url="https://example.com")
    targets = ModelSelector(
        AIModelConfig(
            providers={"openai": provider},
            chat=ModelGroup(
                candidates=[
                    ModelCandidate(id="missing", provider="openai", model="missing"),
                    ModelCandidate(id="first", provider="openai", model="first"),
                    ModelCandidate(id="second", provider="openai", model="second"),
                ]
            ),
            embedding=ModelGroup(),
        ),
        health_store,
    ).select_chat_candidates()
    calls: list[tuple[str, str]] = []

    def resolve_client(target):
        if target.id == "missing":
            return None
        return f"client:{target.candidate.provider}"

    async def call(client, target):
        calls.append((client, target.id))
        if target.id == "first":
            raise RuntimeError("boom")
        return "ok"

    result = await ModelRoutingExecutor(health_store).execute_with_fallback(
        ModelCapability.CHAT,
        targets,
        resolve_client,
        call,
    )

    assert result == "ok"
    assert calls == [("client:openai", "first"), ("client:openai", "second")]


@pytest.mark.asyncio
async def test_model_routing_executor_keeps_direct_target_helper() -> None:
    health_store = ModelHealthStore(
        failure_threshold=3,
        open_seconds=60,
        half_open_max_in_flight=1,
    )
    targets = ModelSelector(
        AIModelConfig(
            providers={"openai": ProviderConfig(name="openai", url="https://example.com")},
            chat=ModelGroup(
                candidates=[ModelCandidate(id="first", provider="openai", model="first")]
            ),
            embedding=ModelGroup(),
        ),
        health_store,
    ).select_chat_candidates()

    async def call(target):
        return target.id

    result = await ModelRoutingExecutor(health_store).execute_targets(targets, call)

    assert result == "first"
