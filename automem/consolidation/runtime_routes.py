from __future__ import annotations

import uuid
from typing import Any, Callable, Dict, Set


def create_association(
    *,
    request_obj: Any,
    coerce_importance_fn: Callable[[Any], float],
    get_memory_graph_fn: Callable[[], Any],
    allowed_relations: Set[str],
    relationship_types: Dict[str, Dict[str, Any]],
    utc_now_fn: Callable[[], str],
    abort_fn: Any,
    jsonify_fn: Any,
    logger: Any,
) -> Any:
    payload = request_obj.get_json(silent=True)
    if not isinstance(payload, dict):
        abort_fn(400, description="JSON body is required")

    memory1_id = (payload.get("memory1_id") or "").strip()
    memory2_id = (payload.get("memory2_id") or "").strip()
    relation_type = (payload.get("type") or "RELATES_TO").upper()
    strength = coerce_importance_fn(payload.get("strength", 0.5))

    if not memory1_id or not memory2_id:
        abort_fn(400, description="'memory1_id' and 'memory2_id' are required")
    for field_name, value in (("memory1_id", memory1_id), ("memory2_id", memory2_id)):
        try:
            uuid.UUID(value)
        except ValueError:
            abort_fn(400, description=f"'{field_name}' must be a valid UUID")
    if memory1_id == memory2_id:
        abort_fn(400, description="Cannot associate a memory with itself")
    if relation_type not in allowed_relations:
        abort_fn(400, description=f"Relation type must be one of {sorted(allowed_relations)}")

    graph = get_memory_graph_fn()
    if graph is None:
        abort_fn(503, description="FalkorDB is unavailable")

    timestamp = utc_now_fn()

    relationship_props = {
        "strength": strength,
        "updated_at": timestamp,
    }

    relation_config = relationship_types.get(relation_type, {})
    if "properties" in relation_config:
        for prop in relation_config["properties"]:
            if prop in payload:
                relationship_props[prop] = payload[prop]

    set_clauses = [f"r.{key} = ${key}" for key in relationship_props]
    set_clause = ", ".join(set_clauses)

    try:
        result = graph.query(
            f"""
            MATCH (m1:Memory {{id: $id1}})
            MATCH (m2:Memory {{id: $id2}})
            MERGE (m1)-[r:{relation_type}]->(m2)
            SET {set_clause}
            RETURN r
            """,
            {
                "id1": memory1_id,
                "id2": memory2_id,
                **relationship_props,
            },
        )
    except Exception:  # pragma: no cover - log full stack trace in production
        logger.exception("Failed to create association")
        abort_fn(500, description="Failed to create association")

    if not result.result_set:
        abort_fn(404, description="One or both memories do not exist")

    response = {
        "status": "success",
        "message": f"Association created between {memory1_id} and {memory2_id}",
        "relation_type": relation_type,
        "strength": strength,
    }

    for prop in relation_config.get("properties", []):
        if prop in relationship_props:
            response[prop] = relationship_props[prop]

    return jsonify_fn(response), 201


def consolidate_memories(
    *,
    request_obj: Any,
    get_memory_graph_fn: Callable[[], Any],
    init_consolidation_scheduler_fn: Callable[[], None],
    get_qdrant_client_fn: Callable[[], Any],
    memory_consolidator_cls: Any,
    persist_consolidation_run_fn: Callable[[Any, Dict[str, Any]], None],
    abort_fn: Any,
    jsonify_fn: Any,
    logger: Any,
) -> Any:
    """Run memory consolidation."""
    data = request_obj.get_json() or {}
    mode = data.get("mode", "full")
    dry_run = data.get("dry_run", True)

    graph = get_memory_graph_fn()
    if graph is None:
        abort_fn(503, description="FalkorDB is unavailable")

    init_consolidation_scheduler_fn()

    try:
        vector_store = get_qdrant_client_fn()
        consolidator = memory_consolidator_cls(graph, vector_store)
        results = consolidator.consolidate(mode=mode, dry_run=dry_run)

        if not dry_run:
            persist_consolidation_run_fn(graph, results)

        return jsonify_fn({"status": "success", "consolidation": results}), 200
    except Exception as e:
        logger.exception("Consolidation failed")
        return jsonify_fn({"error": "Consolidation failed", "details": str(e)}), 500


def consolidation_status(
    *,
    get_memory_graph_fn: Callable[[], Any],
    init_consolidation_scheduler_fn: Callable[[], None],
    build_scheduler_from_graph_fn: Callable[[Any], Any],
    load_recent_runs_fn: Callable[[Any, int], Any],
    consolidation_history_limit: int,
    consolidation_tick_seconds: int,
    state: Any,
    abort_fn: Any,
    jsonify_fn: Any,
    logger: Any,
) -> Any:
    """Get consolidation scheduler status."""
    graph = get_memory_graph_fn()
    if graph is None:
        abort_fn(503, description="FalkorDB is unavailable")

    try:
        init_consolidation_scheduler_fn()
        scheduler = build_scheduler_from_graph_fn(graph)
        history = load_recent_runs_fn(graph, consolidation_history_limit)
        next_runs = scheduler.get_next_runs() if scheduler else {}

        return (
            jsonify_fn(
                {
                    "status": "success",
                    "next_runs": next_runs,
                    "history": history,
                    "thread_alive": bool(
                        state.consolidation_thread and state.consolidation_thread.is_alive()
                    ),
                    "tick_seconds": consolidation_tick_seconds,
                }
            ),
            200,
        )
    except Exception as e:
        logger.exception("Failed to get consolidation status")
        return jsonify_fn({"error": "Failed to get status", "details": str(e)}), 500
