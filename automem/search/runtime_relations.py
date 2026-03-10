from __future__ import annotations

import uuid
from typing import Any, Dict, List, Set

from automem.config import (
    FILTERABLE_RELATIONS,
    canonicalize_relation_type,
    expand_relation_query_types,
    normalize_relation_type,
)


def _validate_memory_id(memory_id: str, abort_fn: Any) -> None:
    try:
        uuid.UUID(memory_id)
    except ValueError:
        abort_fn(400, description="memory_id must be a valid UUID")


def fetch_relations(
    *,
    graph: Any,
    memory_id: str,
    relation_limit: int,
    serialize_node_fn: Any,
    summarize_relation_node_fn: Any,
    logger: Any,
) -> List[Dict[str, Any]]:
    try:
        records = graph.query(
            """
            MATCH (m:Memory {id: $id})-[r]->(related:Memory)
            RETURN type(r) as relation_type,
                   coalesce(
                       r.strength,
                       r.score,
                       r.confidence,
                       r.similarity,
                       toFloat(r.count),
                       0.0
                   ) as strength,
                   r.kind as relation_kind,
                   related
            ORDER BY coalesce(r.updated_at, related.timestamp) DESC
            LIMIT $limit
            """,
            {"id": memory_id, "limit": relation_limit},
        )
    except Exception:  # pragma: no cover - log full stack trace in production
        logger.exception("Failed to fetch relations for memory %s", memory_id)
        return []

    connections: List[Dict[str, Any]] = []
    for row in getattr(records, "result_set", []) or []:
        if len(row) >= 4:
            relation_type, strength, relation_kind, related = row[:4]
        elif len(row) >= 3:
            relation_type, strength, related = row[:3]
            relation_kind = None
        else:  # pragma: no cover - defensive for malformed graph rows
            continue
        normalized_type, normalized_props = normalize_relation_type(
            relation_type,
            {"kind": relation_kind} if relation_kind else {},
        )
        connection = {
            "type": normalized_type,
            "strength": strength,
            "memory": summarize_relation_node_fn(serialize_node_fn(related)),
        }
        kind = normalized_props.get("kind")
        if kind:
            connection["kind"] = kind
        connections.append(connection)
    return connections


def get_related_memories(
    *,
    memory_id: str,
    request_args: Any,
    get_memory_graph_fn: Any,
    allowed_relations: Set[str],
    relation_limit: int,
    serialize_node_fn: Any,
    logger: Any,
    abort_fn: Any,
    jsonify_fn: Any,
) -> Any:
    """Return related memories by traversing relationship edges."""
    _validate_memory_id(memory_id, abort_fn)

    graph = get_memory_graph_fn()
    if graph is None:
        abort_fn(503, description="FalkorDB is unavailable")

    allowed_relations = allowed_relations or set(FILTERABLE_RELATIONS)
    rel_types_param = (request_args.get("relationship_types") or "").strip()
    if rel_types_param:
        requested = [part.strip() for part in rel_types_param.split(",") if part.strip()]
        rel_types = []
        seen_types: Set[str] = set()
        for relation_type in requested:
            normalized_type = canonicalize_relation_type(relation_type)
            if normalized_type in allowed_relations and normalized_type not in seen_types:
                rel_types.append(normalized_type)
                seen_types.add(normalized_type)
        if not rel_types:
            rel_types = sorted(allowed_relations)
    else:
        rel_types = sorted(allowed_relations)

    try:
        max_depth = int(request_args.get("max_depth", 1))
    except (TypeError, ValueError):
        max_depth = 1
    max_depth = max(1, min(max_depth, 3))

    try:
        limit = int(request_args.get("limit", relation_limit))
    except (TypeError, ValueError):
        limit = relation_limit
    limit = max(1, min(limit, 200))

    query_rel_types = expand_relation_query_types(rel_types)
    if query_rel_types:
        rel_pattern = ":" + "|".join(query_rel_types)
    else:
        rel_pattern = ""

    query = f"""
        MATCH (m:Memory {{id: $id}}){'-[r' + rel_pattern + f'*1..{max_depth}]-' if rel_pattern else f'-[r*1..{max_depth}]-'}(related:Memory)
        WHERE m.id <> related.id
        RETURN DISTINCT related
        ORDER BY coalesce(related.importance, 0.0) DESC, coalesce(related.timestamp, '') DESC
        LIMIT $limit
    """

    params = {"id": memory_id, "max_depth": max_depth, "limit": limit}

    try:
        result = graph.query(query, params)
    except Exception:
        logger.exception("Failed to traverse related memories for %s", memory_id)
        abort_fn(500, description="Failed to fetch related memories")

    related: List[Dict[str, Any]] = []
    for row in getattr(result, "result_set", []) or []:
        node = row[0]
        data = serialize_node_fn(node)
        if data.get("id") != memory_id:
            related.append(data)

    return jsonify_fn(
        {
            "status": "success",
            "memory_id": memory_id,
            "count": len(related),
            "related_memories": related,
            "relationship_types": rel_types,
            "max_depth": max_depth,
            "limit": limit,
        }
    )
