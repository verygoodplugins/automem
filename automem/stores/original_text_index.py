"""Lifecycle of the dedicated graph index for original memory text."""

from __future__ import annotations

from typing import Any

LABEL = "MemoryOriginalText"
PROPERTY = "original_search_text"
_INSPECT = """
CALL db.indexes() YIELD label, properties, types, status
WHERE label = 'MemoryOriginalText'
RETURN label, properties, types, status
"""
_CREATE = "CALL db.idx.fulltext.createNodeIndex('MemoryOriginalText', 'original_search_text')"


def _read_status(graph: Any) -> str | None:
    result = graph.query(_INSPECT)
    for row in getattr(result, "result_set", []) or []:
        if len(row) < 4:
            continue
        label, properties, types, status = row[:4]
        if label != LABEL or PROPERTY not in (properties or []):
            continue
        field_types = types.get(PROPERTY, []) if isinstance(types, dict) else []
        if "FULLTEXT" not in field_types:
            continue
        return str(status).upper()
    return None


def original_text_index_status(graph: Any) -> str:
    """Inspect the index without creating or changing it."""
    status = _read_status(graph)
    if status == "OPERATIONAL":
        return "ready"
    if status in {"PENDING", "BUILDING"} or (status and "UNDER CONSTRUCTION" in status):
        return "building"
    return "unavailable" if status is None else "failed"


def ensure_original_text_index(graph: Any) -> str:
    """Create an isolated full-text index once; return ready/building/failed.

    A concurrent process may create it between inspection and creation. In that
    case re-inspect rather than treating an arbitrary creation error as success.
    """
    status = _read_status(graph)
    if status is None:
        try:
            graph.query(_CREATE)
        except Exception:
            status = _read_status(graph)
            if status is None:
                raise
        else:
            status = _read_status(graph)
    if status == "OPERATIONAL":
        return "ready"
    if status in {"PENDING", "BUILDING"} or (status and "UNDER CONSTRUCTION" in status):
        return "building"
    return "failed"
