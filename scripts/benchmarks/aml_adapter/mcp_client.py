"""Small, protocol-safe Streamable HTTP MCP client used by the AML adapter."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable

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

    def _request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send one stateless Streamable HTTP JSON-RPC request."""
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
                    "method": method,
                    "params": params,
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
            raise McpToolError("AutoMem MCP rejected the request")

        result = payload.get("result")
        if not isinstance(result, dict):
            raise McpToolError("AutoMem MCP returned an invalid result")
        return result

    def refresh_submission_tools(self) -> None:
        """Discover the live MCP surface and validate AML's required tools.

        The AutoMem Streamable HTTP bridge intentionally uses stateless MCP
        requests, so discovery is repeated at the start of every AML request.
        This avoids pinning a stale tool list across a long-lived adapter
        process while keeping the adapter limited to its three declared tools.
        """
        result = self._request("tools/list", {})
        tools = result.get("tools")
        if not isinstance(tools, list):
            raise McpToolError("AutoMem MCP tools/list response had no tools array")

        names = {
            tool.get("name")
            for tool in tools
            if isinstance(tool, dict) and isinstance(tool.get("name"), str)
        }
        required = {"store_memory", "recall_memory", "check_database_health"}
        missing = required - names
        if missing:
            raise McpToolError(
                "AutoMem MCP is missing required tool(s): " + ", ".join(sorted(missing))
            )

        self._validate_submission_schemas(tools)

    @staticmethod
    def _validate_submission_schemas(tools: Iterable[Dict[str, Any]]) -> None:
        """Fail closed if an upstream tool update drops required arguments."""
        required_properties = {
            "store_memory": {
                "content",
                "confidence",
                "importance",
                "metadata",
                "tags",
                "timestamp",
                "type",
            },
            "recall_memory": {
                "format",
                "limit",
                "query",
                "scope_fallback",
                "sort",
                "tag_match",
                "tag_mode",
                "tags",
            },
            "check_database_health": set(),
        }
        for tool in tools:
            name = tool.get("name")
            if name not in required_properties:
                continue
            schema = tool.get("inputSchema")
            properties = schema.get("properties") if isinstance(schema, dict) else None
            if not isinstance(properties, dict):
                raise McpToolError(f"AutoMem MCP {name} tool has no input schema")
            missing = required_properties[name] - set(properties)
            if missing:
                raise McpToolError(
                    f"AutoMem MCP {name} tool is missing required argument(s): "
                    + ", ".join(sorted(missing))
                )

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        result = self._request("tools/call", {"name": name, "arguments": arguments})
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
