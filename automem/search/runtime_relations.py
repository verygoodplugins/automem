from __future__ import annotations

import uuid
from typing import Any, Dict, List, Set


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
            RETURN type(r) as relation_type, r.strength as strength, related
            ORDER BY coalesce(r.updated_at, related.timestamp) DESC
            LIMIT $limit
            """,
            {"id": memory_id, "limit": relation_limit},
        )
    except Exception:  # pragma: no cover - log full stack trace in production
        logger.exception("Failed to fetch relations for memory %s", memory_id)
        return []

    connections: List[Dict[str, Any]] = []
    for relation_type, strength, related in getattr(records, "result_set", []) or []:
        connections.append(
            {
                "type": relation_type,
                "strength": strength,
                "memory": summarize_relation_node_fn(serialize_node_fn(related)),
            }
        )
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

    rel_types_param = (request_args.get("relationship_types") or "").strip()
    if rel_types_param:
        requested = [part.strip().upper() for part in rel_types_param.split(",") if part.strip()]
        rel_types = [t for t in requested if t in allowed_relations]
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

    if rel_types:
        rel_pattern = ":" + "|".join(rel_types)
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
