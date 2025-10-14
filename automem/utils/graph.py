from __future__ import annotations

from typing import Any, Dict
from automem.utils.scoring import _parse_metadata_field


def _serialize_node(node: Any) -> Dict[str, Any]:
    properties = getattr(node, "properties", None)
    if isinstance(properties, dict):
        data = dict(properties)
    elif isinstance(node, dict):
        data = dict(node)
    else:
        return {"value": node}

    if "metadata" in data:
        data["metadata"] = _parse_metadata_field(data["metadata"])

    return data


def _summarize_relation_node(data: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}

    for key in ("id", "type", "timestamp", "summary", "importance", "confidence"):
        if key in data:
            summary[key] = data[key]

    content = data.get("content")
    if "summary" not in summary and isinstance(content, str):
        snippet = content.strip()
        if len(snippet) > 160:
            snippet = snippet[:157].rsplit(" ", 1)[0] + "…"
        summary["content"] = snippet

    tags = data.get("tags")
    if isinstance(tags, list) and tags:
        summary["tags"] = tags[:5]

    return summary

