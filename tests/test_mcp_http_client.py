from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from agent_service.mcp.http_client import (
    MCP_PROTOCOL_VERSION,
    StreamableHttpMcpClient,
)


@pytest.mark.asyncio
async def test_java_mcp_initialize_list_and_call() -> None:
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer internal-token"
        assert request.headers["accept"] == "application/json, text/event-stream"
        payload = json.loads(request.content)
        method = payload["method"]
        methods.append(method)

        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "initialize":
            return _rpc_response(
                payload["id"],
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {
                        "name": "heyee-comments-mcp",
                        "version": "1.0.0",
                    },
                },
            )
        if method == "tools/list":
            return _rpc_response(
                payload["id"],
                {
                    "tools": [
                        {
                            "name": "search_shops",
                            "description": "Search shops.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "keyword": {"type": "string"},
                                },
                                "required": [],
                                "additionalProperties": False,
                            },
                        }
                    ]
                },
            )
        if method == "tools/call":
            assert payload["params"] == {
                "name": "search_shops",
                "arguments": {"keyword": "hotpot"},
            }
            return _rpc_response(
                payload["id"],
                {
                    "structuredContent": {
                        "data": [{"id": 5, "name": "Hotpot Shop"}],
                    },
                    "content": [
                        {
                            "type": "text",
                            "text": '[{"id":5,"name":"Hotpot Shop"}]',
                        }
                    ],
                    "isError": False,
                },
            )
        raise AssertionError(f"Unexpected MCP method: {method}")

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = StreamableHttpMcpClient(
        "http://java.test/mcp",
        token="internal-token",
        http_client=http_client,
    )

    server = await client.initialize()
    tools = await client.list_tools()
    result = await client.call_tool("search_shops", {"keyword": "hotpot"})

    assert server["serverInfo"]["name"] == "heyee-comments-mcp"
    assert [tool.name for tool in tools] == ["search_shops"]
    assert tools[0].input_schema["properties"]["keyword"]["type"] == "string"
    assert result == '[{"id":5,"name":"Hotpot Shop"}]'
    assert methods == [
        "initialize",
        "notifications/initialized",
        "tools/list",
        "tools/call",
    ]
    await http_client.aclose()


@pytest.mark.asyncio
async def test_mcp_client_accepts_sse_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload: dict[str, Any] = json.loads(request.content)
        if payload["method"] == "notifications/initialized":
            return httpx.Response(202)
        body = (
            "event: message\n"
            "data: "
            + json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "protocolVersion": MCP_PROTOCOL_VERSION,
                        "capabilities": {},
                        "serverInfo": {"name": "sse-server", "version": "1"},
                    },
                }
            )
            + "\n\n"
        )
        return httpx.Response(
            200,
            text=body,
            headers={"Content-Type": "text/event-stream"},
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = StreamableHttpMcpClient(
        "http://java.test/mcp",
        http_client=http_client,
    )

    result = await client.initialize()

    assert result["serverInfo"]["name"] == "sse-server"
    await http_client.aclose()


def _rpc_response(request_id: int, result: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result,
        },
        headers={"Content-Type": "application/json"},
    )