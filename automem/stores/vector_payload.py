"""Canonical vector payload used by every memory writer and repair path."""

from __future__ import annotations

import json
from typing import Any, Mapping

from automem.utils.tags import _compute_tag_prefixes, _normalize_tag_list


def build_qdrant_payload(properties: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve graph state when an upsert replaces an entire vector payload."""
    metadata = properties.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (ValueError, TypeError):
            metadata = {}
    tags = _normalize_tag_list(properties.get("tags"))
    timestamp = properties.get("timestamp")
    return {
        "content": properties.get("content", ""),
        "tags": tags,
        "tag_prefixes": _compute_tag_prefixes(tags),
        "importance": properties.get("importance", 0.5),
        "timestamp": timestamp,
        "type": properties.get("type", "Memory"),
        "confidence": properties.get("confidence", 0.5),
        "t_valid": properties.get("t_valid") or timestamp,
        "t_invalid": properties.get("t_invalid"),
        "archived": bool(properties.get("archived", False)),
        "updated_at": properties.get("updated_at"),
        "last_accessed": properties.get("last_accessed"),
        "metadata": metadata if isinstance(metadata, dict) else {},
        "relevance_score": properties.get("relevance_score"),
    }
