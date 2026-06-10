from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from werkzeug.exceptions import BadRequest, Forbidden


@dataclass(frozen=True)
class IsolationContext:
    graph_name: str
    collection_name: str
    isolated: bool = False


DEFAULT_ISOLATION_CONTEXT = IsolationContext("memories", "memories", isolated=False)

_ISOLATION_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _header_value(request_obj: Any, name: str) -> str:
    headers = getattr(request_obj, "headers", {}) or {}
    return str(headers.get(name, "") or "").strip()


def _parse_allowlist(env_name: str) -> set[str]:
    raw = os.getenv(env_name, "")
    return {item.strip() for item in raw.split(",") if item.strip()}


def _validate_name(header_name: str, value: str) -> None:
    if not _ISOLATION_NAME_RE.fullmatch(value):
        raise BadRequest(f"Invalid {header_name}: {value}")


def _ensure_allowed(env_name: str, label: str, value: str) -> None:
    allowed = _parse_allowlist(env_name)
    if not allowed:
        raise Forbidden(f"{label} isolation requires {env_name}")
    if value not in allowed:
        raise Forbidden(f"{label} '{value}' not allowed")


def resolve_isolation_context(
    *,
    default_graph_name: str,
    default_collection_name: str,
    request_obj=None,
) -> IsolationContext:
    if request_obj is None:
        return IsolationContext(default_graph_name, default_collection_name, isolated=False)

    graph_name = _header_value(request_obj, "X-Graph-Name")
    collection_name = _header_value(request_obj, "X-Collection-Name")

    if not graph_name and not collection_name:
        return IsolationContext(default_graph_name, default_collection_name, isolated=False)

    if bool(graph_name) != bool(collection_name):
        raise BadRequest("X-Graph-Name and X-Collection-Name must be provided together")

    _validate_name("X-Graph-Name", graph_name)
    _validate_name("X-Collection-Name", collection_name)
    _ensure_allowed("ALLOWED_GRAPHS", "Graph", graph_name)
    _ensure_allowed("ALLOWED_COLLECTIONS", "Collection", collection_name)

    return IsolationContext(graph_name, collection_name, isolated=True)


def resolve_flask_isolation_context(
    *,
    default_graph_name: str,
    default_collection_name: str,
) -> IsolationContext:
    try:
        from flask import has_request_context, request
    except Exception:
        return IsolationContext(default_graph_name, default_collection_name, isolated=False)

    if not has_request_context():
        return IsolationContext(default_graph_name, default_collection_name, isolated=False)

    return resolve_isolation_context(
        default_graph_name=default_graph_name,
        default_collection_name=default_collection_name,
        request_obj=request,
    )


def default_context(
    *,
    default_graph_name: str,
    default_collection_name: str,
) -> IsolationContext:
    return IsolationContext(default_graph_name, default_collection_name, isolated=False)
