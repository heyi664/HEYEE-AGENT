from __future__ import annotations

import json
from itertools import count
from typing import Any, cast

import httpx

from agent_service.mcp.contracts import McpClientProtocol, McpToolDefinition

MCP_PROTOCOL_VERSION = "2025-03-26"


class McpProtocolError(RuntimeError):
    """Raised when the MCP server returns an invalid or error response."""


class McpToolCallError(RuntimeError):
    """Raised when an MCP tool reports isError=true."""


class StreamableHttpMcpClient(McpClientProtocol):
    def __init__(
        self,
        server_url: str,
        token: str | None = None,
        timeout_seconds: float = 10.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._server_url = server_url
        self._token = token.strip() if token and token.strip() else None
        self._request_ids = count(1)
        self._session_id: str | None = None
        self._initialized = False
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            timeout=timeout_seconds,
            trust_env=False,
        )

    async def initialize(self) -> dict[str, Any]:
        result = await self._request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "heyee-agent",
                    "version": "0.1.0",
                },
            },
        )
        protocol_version = str(result.get("protocolVersion") or "")
        if protocol_version != MCP_PROTOCOL_VERSION:
            raise McpProtocolError(
                f"Unsupported MCP protocol version: {protocol_version or 'missing'}"
            )
        await self._notify("notifications/initialized", {})
        self._initialized = True
        return result

    async def list_tools(self) -> list[McpToolDefinition]:
        await self._ensure_initialized()
        result = await self._request("tools/list", {})
        raw_tools = result.get("tools")
        if not isinstance(raw_tools, list):
            raise McpProtocolError("tools/list result does not contain a tools array")

        tools: list[McpToolDefinition] = []
        for raw_tool in raw_tools:
            if not isinstance(raw_tool, dict):
                continue
            name = str(raw_tool.get("name") or "").strip()
            if not name:
                continue
            input_schema = raw_tool.get("inputSchema")
            if not isinstance(input_schema, dict):
                input_schema = {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                }
            tools.append(
                McpToolDefinition(
                    name=name,
                    description=str(raw_tool.get("description") or ""),
                    input_schema=cast(dict[str, Any], input_schema),
                )
            )
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        await self._ensure_initialized()
        result = await self._request(
            "tools/call",
            {
                "name": name,
                "arguments": arguments,
            },
        )
        text = self._tool_result_text(result)
        if bool(result.get("isError")):
            raise McpToolCallError(text or f"MCP tool failed: {name}")
        if not text:
            raise McpProtocolError(f"MCP tool returned no content: {name}")
        return text

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.initialize()
    async def _request(
        self,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        request_id = next(self._request_ids)
        response = await self._post(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
        )
        payload = self._decode_response(response, request_id)
        error = payload.get("error")
        if isinstance(error, dict):
            code = error.get("code")
            message = error.get("message")
            raise McpProtocolError(f"MCP error {code}: {message}")
        result = payload.get("result")
        if not isinstance(result, dict):
            raise McpProtocolError(f"MCP method {method} returned no result object")
        return cast(dict[str, Any], result)

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        response = await self._post(
            {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            }
        )
        if response.status_code not in {200, 202, 204}:
            response.raise_for_status()

    async def _post(self, payload: dict[str, Any]) -> httpx.Response:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        response = await self._client.post(
            self._server_url,
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        session_id = response.headers.get("Mcp-Session-Id")
        if session_id:
            self._session_id = session_id
        return response

    def _decode_response(
        self,
        response: httpx.Response,
        expected_id: int,
    ) -> dict[str, Any]:
        if response.status_code in {202, 204} or not response.content:
            raise McpProtocolError("MCP request returned no response body")

        content_type = response.headers.get("content-type", "").lower()
        if "text/event-stream" in content_type:
            payloads = self._parse_sse(response.text)
            for payload in payloads:
                if payload.get("id") == expected_id:
                    return payload
            raise McpProtocolError("MCP SSE response did not contain the request ID")

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise McpProtocolError("MCP response is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise McpProtocolError("MCP response must be a JSON object")
        if payload.get("id") != expected_id:
            raise McpProtocolError("MCP response ID does not match the request")
        return cast(dict[str, Any], payload)

    def _parse_sse(self, body: str) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        data_lines: list[str] = []
        for line in body.splitlines() + [""]:
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
            elif not line and data_lines:
                try:
                    payload = json.loads("\n".join(data_lines))
                except json.JSONDecodeError as exc:
                    raise McpProtocolError("MCP SSE data is not valid JSON") from exc
                if isinstance(payload, dict):
                    payloads.append(cast(dict[str, Any], payload))
                data_lines = []
        return payloads

    def _tool_result_text(self, result: dict[str, Any]) -> str:
        structured = result.get("structuredContent")
        if isinstance(structured, dict) and "data" in structured:
            return json.dumps(
                structured["data"],
                ensure_ascii=False,
                separators=(",", ":"),
            )

        content = result.get("content")
        if not isinstance(content, list):
            return ""
        texts = [
            str(item.get("text"))
            for item in content
            if isinstance(item, dict)
            and item.get("type") == "text"
            and item.get("text") is not None
        ]
        return "\n".join(texts)