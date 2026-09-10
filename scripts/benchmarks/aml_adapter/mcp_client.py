"""Small, protocol-safe Streamable HTTP MCP client used by the AML adapter."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict

import requests


class McpToolError(RuntimeError):
    """Raised when an MCP tool call cannot produce a usable result."""


@dataclass(frozen=True)
class McpClient:
    """Call AutoMem's ``store_memory`` and ``recall_memory`` MCP tools.

    The AutoMem streamable transport is stateless, so tool calls do not require
    an initialize exchange or an MCP session header.
    """

    endpoint: str
    token: str | None = None
    timeout_seconds: float = 30.0

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        request_id = str(uuid.uuid4())
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "X-Request-Id": request_id,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        try:
            response = requests.post(
                self.endpoint,
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": arguments},
                },
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise McpToolError("AutoMem MCP is unavailable") from exc

        if response.status_code >= 400:
            raise McpToolError(f"AutoMem MCP returned HTTP {response.status_code}")

        try:
            payload = response.json()
        except ValueError as exc:
            raise McpToolError("AutoMem MCP returned invalid JSON") from exc

        if payload.get("error"):
            raise McpToolError("AutoMem MCP rejected the tool call")

        result = payload.get("result")
        if not isinstance(result, dict) or result.get("isError"):
            raise McpToolError("AutoMem MCP tool call failed")
        return result

    def store_memory(self, arguments: Dict[str, Any]) -> None:
        self.call_tool("store_memory", arguments)

    def recall_memory(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        result = self.call_tool("recall_memory", {**arguments, "format": "json"})
        content = result.get("content")
        if not isinstance(content, list) or not content:
            raise McpToolError("AutoMem MCP recall response had no content")
        raw = content[0].get("text") if isinstance(content[0], dict) else None
        if not isinstance(raw, str):
            raise McpToolError("AutoMem MCP recall response was not text")
        try:
            parsed = json.loads(raw)
        except ValueError as exc:
            raise McpToolError("AutoMem MCP recall response was not JSON") from exc
        if not isinstance(parsed, dict):
            raise McpToolError("AutoMem MCP recall result was not an object")
        return parsed
