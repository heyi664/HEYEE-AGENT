from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any

import pytest

from agent_service.infra_ai.models import (
    ModelCandidate,
    ModelCapability,
    ModelTarget,
    ProviderConfig,
)
from agent_service.services import stream_llm_service as stream_module
from agent_service.services.stream_llm_service import StreamLLMService


class RecordingCallback:
    def __init__(self) -> None:
        self.contents: list[str] = []
        self.errors: list[Exception] = []
        self.completed = False

    async def on_content(self, content: str) -> None:
        self.contents.append(content)

    async def on_complete(self) -> None:
        self.completed = True

    async def on_error(self, error: Exception) -> None:
        self.errors.append(error)


class FakeHandle:
    def __init__(self) -> None:
        self.cancelled = False
        self.task: asyncio.Task[None] | None = None

    def cancel(self) -> None:
        self.cancelled = True
        if self.task is not None and not self.task.done():
            self.task.cancel()


Action = Callable[[Any], Awaitable[None]]


class ScriptedClient:
    def __init__(self, action: Action) -> None:
        self._action = action
        self.calls = 0
        self.handles: list[FakeHandle] = []

    async def do_stream_chat(self, *args: Any, **kwargs: Any) -> FakeHandle:
        callback = args[3]
        self.calls += 1
        handle = FakeHandle()
        handle.task = asyncio.create_task(self._action(callback))
        self.handles.append(handle)
        return handle


class StaticSelector:
    def __init__(self, targets: list[ModelTarget]) -> None:
        self._targets = targets

    def select_chat_candidates(self, **kwargs: Any) -> list[ModelTarget]:
        return self._targets


class RecordingHealthStore:
    def __init__(self) -> None:
        self.successes: list[str] = []
        self.failures: list[str] = []

    def record_success(self, model_id: str) -> None:
        self.successes.append(model_id)

    def record_failure(self, model_id: str) -> None:
        self.failures.append(model_id)


class ClientRegistry:
    def __init__(self, clients: dict[str, ScriptedClient]) -> None:
        self._clients = clients

    def resolve(self, target: ModelTarget) -> ScriptedClient | None:
        return self._clients.get(target.id)


def _target(model_id: str) -> ModelTarget:
    candidate = ModelCandidate(provider="test", model=model_id, id=model_id)
    return ModelTarget(
        id=model_id,
        capability=ModelCapability.CHAT,
        candidate=candidate,
        provider=ProviderConfig(name="test"),
    )


def _service(
    monkeypatch: pytest.MonkeyPatch,
    clients: dict[str, ScriptedClient],
    *,
    timeout_seconds: float = 0.05,
) -> tuple[StreamLLMService, RecordingHealthStore]:
    monkeypatch.setattr(
        stream_module,
        "get_settings",
        lambda: SimpleNamespace(
            agent_mock_mode=False,
            ai_stream_first_token_timeout_seconds=timeout_seconds,
        ),
    )
    service = StreamLLMService(client_registry=ClientRegistry(clients))
    health = RecordingHealthStore()
    service._selector = StaticSelector([_target(model_id) for model_id in clients])
    service._health_store = health
    return service, health


@pytest.mark.asyncio
async def test_pre_token_error_falls_back_to_next_model(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail_before_token(callback: Any) -> None:
        await callback.on_error(RuntimeError("primary down"))

    async def answer(callback: Any) -> None:
        await callback.on_content("fallback answer")
        await callback.on_complete()

    primary = ScriptedClient(fail_before_token)
    fallback = ScriptedClient(answer)
    service, health = _service(monkeypatch, {"primary": primary, "fallback": fallback})
    callback = RecordingCallback()

    await service.start([], callback, temperature=0.1, top_p=None)

    assert callback.contents == ["fallback answer"]
    assert callback.completed is True
    assert callback.errors == []
    assert primary.handles[0].cancelled is True
    assert health.failures == ["primary"]
    assert health.successes == ["fallback"]


@pytest.mark.asyncio
async def test_first_token_timeout_cancels_candidate_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def stall(callback: Any) -> None:
        await asyncio.Event().wait()

    async def answer(callback: Any) -> None:
        await callback.on_content("after timeout")
        await callback.on_complete()

    stalled = ScriptedClient(stall)
    fallback = ScriptedClient(answer)
    service, health = _service(
        monkeypatch,
        {"stalled": stalled, "fallback": fallback},
        # 10 ms is below the Windows event-loop scheduling quantum and can make the
        # fallback task appear stalled even though the timeout path is correct.
        timeout_seconds=0.1,
    )
    callback = RecordingCallback()

    await service.start([], callback, temperature=0.1, top_p=None)

    assert stalled.handles[0].cancelled is True
    assert callback.contents == ["after timeout"]
    assert health.failures == ["stalled"]
    assert health.successes == ["fallback"]


@pytest.mark.asyncio
async def test_error_after_first_token_is_not_fallback_eligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def partial_then_error(callback: Any) -> None:
        await callback.on_content("partial")
        await callback.on_error(RuntimeError("generation interrupted"))

    async def should_not_run(callback: Any) -> None:
        await callback.on_content("wrong model")

    primary = ScriptedClient(partial_then_error)
    fallback = ScriptedClient(should_not_run)
    service, health = _service(monkeypatch, {"primary": primary, "fallback": fallback})
    callback = RecordingCallback()

    await service.start([], callback, temperature=0.1, top_p=None)

    assert callback.contents == ["partial"]
    assert len(callback.errors) == 1
    assert fallback.calls == 0
    assert health.failures == ["primary"]
