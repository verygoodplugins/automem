from __future__ import annotations

from typing import Any, Iterable, List, Set

from automem.stores.vector_store import _build_qdrant_tag_filter


def _normalized_excluded_types(excluded_types: Iterable[str] | None) -> List[str]:
    return [
        str(memory_type).strip()
        for memory_type in (excluded_types or [])
        if str(memory_type).strip()
    ]


def _excluded_type_params(excluded_types: Iterable[str] | None) -> dict[str, Any]:
    excluded = _normalized_excluded_types(excluded_types)
    return {"excluded_types": excluded} if excluded else {}


def _memory_type_where(excluded_types: Iterable[str] | None, alias: str = "m") -> str:
    excluded = _normalized_excluded_types(excluded_types)
    if not excluded:
        return ""
    return f"WHERE NOT coalesce({alias}.type, '') IN $excluded_types"


def count_falkor_memories(graph: Any, excluded_types: Iterable[str] | None) -> int:
    """Count vector-sync-eligible graph memories."""
    result = graph.query(
        f"""
        MATCH (m:Memory)
        {_memory_type_where(excluded_types)}
        RETURN COUNT(m) as count
        """,
        _excluded_type_params(excluded_types),
    )
    rows = getattr(result, "result_set", []) or []
    if not rows:
        return 0
    return int(rows[0][0] or 0)


def fetch_falkor_memory_ids(graph: Any, excluded_types: Iterable[str] | None) -> Set[str]:
    """Fetch vector-sync-eligible graph memory IDs."""
    result = graph.query(
        f"""
        MATCH (m:Memory)
        {_memory_type_where(excluded_types)}
        RETURN m.id AS id
        """,
        _excluded_type_params(excluded_types),
    )
    ids: Set[str] = set()
    for row in getattr(result, "result_set", []) or []:
        if row[0]:
            ids.add(str(row[0]))
    return ids


def _qdrant_exclusion_filter(excluded_types: Iterable[str] | None) -> Any:
    return _build_qdrant_tag_filter(None, excluded_types=excluded_types)


def count_qdrant_points(
    qdrant_client: Any,
    collection_name: str,
    excluded_types: Iterable[str] | None,
) -> int:
    """Count vector-sync-eligible Qdrant points."""
    count_filter = _qdrant_exclusion_filter(excluded_types)
    count_fn = getattr(qdrant_client, "count", None)
    if count_fn is not None:
        result = count_fn(
            collection_name=collection_name,
            count_filter=count_filter,
            exact=True,
        )
        return int(getattr(result, "count", result) or 0)

    return len(fetch_qdrant_point_ids(qdrant_client, collection_name, excluded_types))


def fetch_qdrant_point_ids(
    qdrant_client: Any,
    collection_name: str,
    excluded_types: Iterable[str] | None,
) -> Set[str]:
    """Fetch vector-sync-eligible Qdrant point IDs."""
    point_ids: Set[str] = set()
    scroll_filter = _qdrant_exclusion_filter(excluded_types)
    offset = None

    while True:
        result = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=1000,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        points, next_offset = result
        for point in points:
            point_ids.add(str(point.id))

        if next_offset is None:
            break
        offset = next_offset

    return point_ids
