from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from agent_service.core.errors import ModelUnavailableError
from agent_service.infra_ai.clients import (
    ChatTurn,
    OpenAICompatibleChatModelClient,
    StreamCancellationHandle,
    ToolCallingUnavailable,
)
from agent_service.infra_ai.health_store import ModelHealthStore
from agent_service.infra_ai.models import (
    AIModelConfig,
    ModelCandidate,
    ModelCapability,
    ModelGroup,
    ModelTarget,
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


@pytest.mark.asyncio
async def test_openai_compatible_chat_client_posts_openai_style_request() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["content_type"] = request.headers.get("content-type")
        captured["json"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "AirPods Pro 2 has a one-year warranty.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 42},
            },
        )

    target = ModelTarget(
        id="qwen3-max",
        capability=ModelCapability.CHAT,
        candidate=ModelCandidate(
            id="qwen3-max",
            provider="bailian",
            model="qwen3-max",
            timeout_seconds=3,
        ),
        provider=ProviderConfig(
            name="bailian",
            url="https://dashscope.example.com/compatible-mode",
            api_key="sk-test",
            endpoints={"chat": "/v1/chat/completions"},
        ),
    )

    turn = await OpenAICompatibleChatModelClient(
        transport=httpx.MockTransport(handler)
    ).complete_turn(
        target,
        [{"role": "user", "content": "Warranty?"}],
        [],
    )

    assert isinstance(turn, ChatTurn)
    assert turn.content == "AirPods Pro 2 has a one-year warranty."
    assert turn.tool_calls == []
    assert turn.assistant_message == {
        "role": "assistant",
        "content": "AirPods Pro 2 has a one-year warranty.",
    }
    assert captured["url"] == (
        "https://dashscope.example.com/compatible-mode/v1/chat/completions"
    )
    assert captured["authorization"] == "Bearer sk-test"
    assert captured["content_type"] == "application/json"
    assert captured["json"] == {
        "model": "qwen3-max",
        "messages": [{"role": "user", "content": "Warranty?"}],
        "temperature": 0.2,
    }


@pytest.mark.asyncio
async def test_openai_compatible_chat_client_retries_transient_transport_errors() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.RemoteProtocolError(
                "Server disconnected without sending a response.",
                request=request,
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Recovered"}}]},
        )

    target = ModelTarget(
        id="qwen3-flash",
        capability=ModelCapability.CHAT,
        candidate=ModelCandidate(
            id="qwen3-flash",
            provider="bailian",
            model="qwen3-flash",
        ),
        provider=ProviderConfig(
            name="bailian",
            url="https://dashscope.example.com",
            api_key="sk-test",
            endpoints={"chat": "/compatible-mode/v1/chat/completions"},
        ),
    )

    turn = await OpenAICompatibleChatModelClient(
        transport=httpx.MockTransport(handler)
    ).complete_turn(target, [{"role": "user", "content": "Hello"}], [])

    assert attempts == 3
    assert turn.content == "Recovered"


@pytest.mark.asyncio
async def test_openai_compatible_chat_client_sends_tool_schema_and_parses_tool_calls() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "search_shops",
                                        "arguments": "{\"keyword\":\"hotpot\"}",
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    target = ModelTarget(
        id="glm-4.7",
        capability=ModelCapability.CHAT,
        candidate=ModelCandidate(id="glm-4.7", provider="siliconflow", model="glm-4.7"),
        provider=ProviderConfig(
            name="siliconflow",
            url="https://api.siliconflow.example.com",
            api_key="sk-test",
            endpoints={"chat": "/v1/chat/completions"},
        ),
    )
    schemas = [
        {
            "type": "function",
            "function": {
                "name": "search_shops",
                "parameters": {"type": "object"},
            },
        }
    ]

    turn = await OpenAICompatibleChatModelClient(
        transport=httpx.MockTransport(handler)
    ).complete_turn(
        target,
        [{"role": "user", "content": "Find hotpot"}],
        schemas,
    )

    request_json = captured["json"]
    assert isinstance(request_json, dict)
    assert request_json["tools"] == schemas
    assert request_json["tool_choice"] == "auto"
    assert turn.content is None
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].call_id == "call_1"
    assert turn.tool_calls[0].name == "search_shops"
    assert turn.tool_calls[0].arguments == {"keyword": "hotpot"}


@pytest.mark.asyncio
async def test_openai_compatible_chat_client_maps_tool_http_errors() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"error": {"message": "tools unsupported"}})

    target = ModelTarget(
        id="basic-model",
        capability=ModelCapability.CHAT,
        candidate=ModelCandidate(id="basic-model", provider="openai", model="basic-model"),
        provider=ProviderConfig(
            name="openai",
            url="https://openai.example.com",
            api_key="sk-test",
            endpoints={"chat": "/v1/chat/completions"},
        ),
    )

    with pytest.raises(ToolCallingUnavailable):
        await OpenAICompatibleChatModelClient(
            transport=httpx.MockTransport(handler)
        ).complete_turn(
            target,
            [{"role": "user", "content": "Use a tool"}],
            [{"type": "function", "function": {"name": "search_shops"}}],
        )


@pytest.mark.asyncio
async def test_openai_compatible_chat_client_raises_model_unavailable_for_http_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "server failed"}})

    target = ModelTarget(
        id="qwen3-max",
        capability=ModelCapability.CHAT,
        candidate=ModelCandidate(id="qwen3-max", provider="bailian", model="qwen3-max"),
        provider=ProviderConfig(
            name="bailian",
            url="https://dashscope.example.com",
            api_key="sk-test",
            endpoints={"chat": "/compatible-mode/v1/chat/completions"},
        ),
    )

    with pytest.raises(ModelUnavailableError, match="HTTP 500"):
        await OpenAICompatibleChatModelClient(
            transport=httpx.MockTransport(handler)
        ).complete_turn(
            target,
            [{"role": "user", "content": "Warranty?"}],
            [],
        )

class RecordingStreamCallback:
    def __init__(self) -> None:
        self.contents: list[str] = []
        self.thinking: list[str] = []
        self.completed = False
        self.error: Exception | None = None
        self.done = asyncio.Event()

    def on_content(self, content: str) -> None:
        self.contents.append(content)

    def on_thinking(self, content: str) -> None:
        self.thinking.append(content)

    def on_complete(self) -> None:
        self.completed = True
        self.done.set()

    def on_error(self, error: Exception) -> None:
        self.error = error
        self.done.set()


@pytest.mark.asyncio
async def test_openai_compatible_chat_client_streams_sse_deltas() -> None:
    captured: dict[str, object] = {}
    stream_body = "\n".join(
        [
            'data: {"choices":[{"delta":{"reasoning_content":"thinking "}}]}',
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            'data: {"choices":[{"finish_reason":"stop"}]}',
            "",
        ]
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["accept"] = request.headers.get("accept")
        captured["authorization"] = request.headers.get("authorization")
        captured["json"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, content=stream_body.encode("utf-8"))

    target = ModelTarget(
        id="qwen-plus",
        capability=ModelCapability.CHAT,
        candidate=ModelCandidate(id="qwen-plus", provider="bailian", model="qwen-plus"),
        provider=ProviderConfig(
            name="bailian",
            url="https://dashscope.example.com/compatible-mode",
            api_key="sk-test",
            endpoints={"chat": "/v1/chat/completions"},
        ),
    )
    callback = RecordingStreamCallback()

    handle = await OpenAICompatibleChatModelClient(
        transport=httpx.MockTransport(handler)
    ).do_stream_chat(
        target,
        [{"role": "user", "content": "Say hello"}],
        [],
        callback,
        reasoning_enabled=True,
    )
    await asyncio.wait_for(callback.done.wait(), timeout=1)
    await handle.wait()

    assert isinstance(handle, StreamCancellationHandle)
    assert callback.error is None
    assert callback.completed is True
    assert callback.thinking == ["thinking "]
    assert callback.contents == ["Hello", " world"]
    assert captured["url"] == (
        "https://dashscope.example.com/compatible-mode/v1/chat/completions"
    )
    assert captured["accept"] == "text/event-stream"
    assert captured["authorization"] == "Bearer sk-test"
    assert captured["json"] == {
        "model": "qwen-plus",
        "messages": [{"role": "user", "content": "Say hello"}],
        "temperature": 0.2,
        "stream": True,
    }


@pytest.mark.asyncio
async def test_openai_compatible_chat_client_stream_reports_http_errors() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server failed")

    target = ModelTarget(
        id="qwen3-max",
        capability=ModelCapability.CHAT,
        candidate=ModelCandidate(id="qwen3-max", provider="bailian", model="qwen3-max"),
        provider=ProviderConfig(
            name="bailian",
            url="https://dashscope.example.com",
            api_key="sk-test",
            endpoints={"chat": "/compatible-mode/v1/chat/completions"},
        ),
    )
    callback = RecordingStreamCallback()

    handle = await OpenAICompatibleChatModelClient(
        transport=httpx.MockTransport(handler)
    ).do_stream_chat(
        target,
        [{"role": "user", "content": "Warranty?"}],
        [],
        callback,
    )
    await asyncio.wait_for(callback.done.wait(), timeout=1)
    await handle.wait()

    assert callback.completed is False
    assert callback.error is not None
    assert "HTTP 500" in str(callback.error)
    assert callback.contents == []
