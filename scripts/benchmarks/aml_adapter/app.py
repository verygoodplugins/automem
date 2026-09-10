"""Synchronous AML Add/Search API backed exclusively by AutoMem MCP tools.

Run with:
    AML_ADAPTER_MCP_URL=http://localhost:8080/mcp python -m flask --app \
        scripts.benchmarks.aml_adapter.app run --host 0.0.0.0 --port 8090
"""

from __future__ import annotations

import hashlib
import hmac
import os
from datetime import UTC, datetime
from typing import Any, Dict, Iterable, List

from flask import Flask, jsonify, request

from .mcp_client import McpClient, McpToolError

MCP_RECALL_MAX_LIMIT = 50
SCOPE_TAG_PREFIX = "aml-user-"


def _scope_tag(user_id: str) -> str:
    """Turn an opaque AML user ID into a safe, exact MCP tag scope."""
    digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
    return f"{SCOPE_TAG_PREFIX}{digest}"


def _iso_timestamp(timestamp_ms: Any) -> str | None:
    if timestamp_ms is None:
        return None
    if not isinstance(timestamp_ms, (int, float)) or isinstance(timestamp_ms, bool):
        raise ValueError("message timestamp must be a Unix-millisecond number")
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).isoformat().replace("+00:00", "Z")


def _detail(reason: str, status: int):
    return jsonify({"detail": {"reason": reason}}), status


def _required_string(payload: Dict[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _message_rows(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty array")
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in ("user", "assistant"):
            raise ValueError("message role must be user or assistant")
        if not isinstance(content, str) or not content:
            raise ValueError("message content must be a non-empty string")
        yield {
            "role": role,
            "content": content,
            "timestamp": _iso_timestamp(message.get("timestamp")),
        }


def _extract_results(recall: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
    raw_results = recall.get("results") or recall.get("memories") or []
    if not isinstance(raw_results, list):
        raise McpToolError("AutoMem MCP recall results were not an array")

    data: List[Dict[str, Any]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        memory = item.get("memory") if isinstance(item.get("memory"), dict) else item
        memory_id = memory.get("id") or memory.get("memory_id") or item.get("id")
        content = memory.get("content") or memory.get("text")
        if (
            not isinstance(memory_id, str)
            or not memory_id
            or not isinstance(content, str)
            or not content
        ):
            continue
        entry: Dict[str, Any] = {"id": memory_id, "content": content}
        score = item.get("final_score", item.get("score"))
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            entry["score"] = score
        created_at = memory.get("timestamp") or memory.get("created_at")
        if isinstance(created_at, str) and created_at:
            entry["created_at"] = created_at
        data.append(entry)
        if len(data) >= limit:
            break
    return data


def create_app(client: McpClient | None = None) -> Flask:
    app = Flask(__name__)
    adapter_key = os.getenv("AML_ADAPTER_API_KEY")
    mcp = client or McpClient(
        endpoint=os.getenv("AML_ADAPTER_MCP_URL", "http://localhost:8080/mcp"),
        token=os.getenv("AML_ADAPTER_MCP_TOKEN"),
        timeout_seconds=float(os.getenv("AML_ADAPTER_MCP_TIMEOUT_SECONDS", "30")),
    )

    def authorize() -> bool:
        """Accept AML's configured Token/Bearer/X-Api-Key authentication forms."""
        if not adapter_key:
            return True
        presented = request.headers.get("X-Api-Key")
        if not presented:
            authorization = request.headers.get("Authorization", "")
            scheme, _, value = authorization.partition(" ")
            if scheme.lower() in {"bearer", "token"}:
                presented = value
        return bool(presented) and hmac.compare_digest(presented, adapter_key)

    @app.get("/health")
    def health():
        try:
            mcp.call_tool("check_database_health", {})
        except McpToolError:
            return _detail("AutoMem MCP is not ready", 503)
        return jsonify({"status": "ok"})

    @app.post("/add")
    def add():
        if not authorize():
            return _detail("invalid memory system key", 401)
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return _detail("JSON object body is required", 422)
        try:
            request_id = _required_string(payload, "request_id")
            user_id = _required_string(payload, "user_id")
            session_id = _required_string(payload, "session_id")
            messages = list(_message_rows(payload))
        except ValueError as exc:
            return _detail(str(exc), 422)

        scope = _scope_tag(user_id)
        try:
            for message in messages:
                arguments: Dict[str, Any] = {
                    "content": message["content"],
                    "type": "Context",
                    "confidence": 1.0,
                    "importance": 0.5,
                    "tags": ["aml", scope],
                    "metadata": {
                        "aml": {
                            "request_id": request_id,
                            "session_id": session_id,
                            "role": message["role"],
                        }
                    },
                }
                if message["timestamp"]:
                    arguments["timestamp"] = message["timestamp"]
                mcp.store_memory(arguments)
        except McpToolError:
            return _detail("AutoMem MCP could not persist the submitted messages", 503)

        return jsonify(
            {
                "success": True,
                "request_id": request_id,
                "user_id": user_id,
                "session_id": session_id,
            }
        )

    @app.post("/search")
    def search():
        if not authorize():
            return _detail("invalid memory system key", 401)
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return _detail("JSON object body is required", 422)
        try:
            query = _required_string(payload, "query")
            user_id = _required_string(payload, "user_id")
            top_k = payload.get("top_k")
            if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
                raise ValueError("top_k must be a positive integer")
            options = payload.get("options")
            if options is not None and (
                not isinstance(options, list)
                or not all(isinstance(option, str) for option in options)
            ):
                raise ValueError("options must be an array of strings when supplied")
        except ValueError as exc:
            return _detail(str(exc), 422)

        try:
            recall = mcp.recall_memory(
                {
                    "query": query,
                    "tags": [_scope_tag(user_id)],
                    "tag_mode": "all",
                    "tag_match": "exact",
                    "scope_fallback": False,
                    # AutoMem MCP currently caps one recall call at 50. Returning
                    # fewer than AML's requested top_k remains contract-valid.
                    "limit": min(top_k, MCP_RECALL_MAX_LIMIT),
                    "sort": "score",
                }
            )
            data = _extract_results(recall, top_k)
        except McpToolError:
            return _detail("AutoMem MCP could not retrieve memories", 503)
        return jsonify({"data": data})

    return app


app = create_app()
