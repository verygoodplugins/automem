from __future__ import annotations

import json
from typing import Any, Dict, List


def update_memory(
    *,
    request_obj: Any,
    memory_id: str,
    get_memory_graph_fn: Any,
    get_qdrant_client_fn: Any,
    normalize_tag_list_fn: Any,
    compute_tag_prefixes_fn: Any,
    parse_metadata_field_fn: Any,
    normalize_timestamp_fn: Any,
    generate_real_embedding_fn: Any,
    serialize_node_fn: Any,
    collection_name: str,
    point_struct_cls: Any,
    utc_now_fn: Any,
    logger: Any,
    abort_fn: Any,
    jsonify_fn: Any,
) -> Any:
    payload = request_obj.get_json(silent=True)
    if not isinstance(payload, dict):
        abort_fn(400, description="JSON body is required")

    graph = get_memory_graph_fn()
    if graph is None:
        abort_fn(503, description="FalkorDB is unavailable")

    result = graph.query("MATCH (m:Memory {id: $id}) RETURN m", {"id": memory_id})
    if not getattr(result, "result_set", None):
        abort_fn(404, description="Memory not found")

    current_node = result.result_set[0][0]
    current = serialize_node_fn(current_node)

    new_content = payload.get("content", current.get("content"))
    tags = normalize_tag_list_fn(payload.get("tags", current.get("tags")))
    tags_lower = [t.strip().lower() for t in tags if isinstance(t, str) and t.strip()]
    tag_prefixes = compute_tag_prefixes_fn(tags_lower)
    importance = payload.get("importance", current.get("importance"))
    memory_type = payload.get("type", current.get("type"))
    confidence = payload.get("confidence", current.get("confidence"))
    timestamp = payload.get("timestamp", current.get("timestamp"))
    metadata_raw = payload.get("metadata", parse_metadata_field_fn(current.get("metadata")))
    updated_at = payload.get("updated_at", current.get("updated_at", utc_now_fn()))
    last_accessed = payload.get("last_accessed", current.get("last_accessed"))

    if metadata_raw is None:
        metadata: Dict[str, Any] = {}
    elif isinstance(metadata_raw, dict):
        metadata = metadata_raw
    else:
        abort_fn(400, description="'metadata' must be an object")
    metadata_json = json.dumps(metadata, default=str)

    if timestamp:
        try:
            timestamp = normalize_timestamp_fn(timestamp)
        except ValueError as exc:
            abort_fn(400, description=f"Invalid timestamp: {exc}")

    if updated_at:
        try:
            updated_at = normalize_timestamp_fn(updated_at)
        except ValueError as exc:
            abort_fn(400, description=f"Invalid updated_at: {exc}")

    if last_accessed:
        try:
            last_accessed = normalize_timestamp_fn(last_accessed)
        except ValueError as exc:
            abort_fn(400, description=f"Invalid last_accessed: {exc}")

    update_query = """
        MATCH (m:Memory {id: $id})
        SET m.content = $content,
            m.tags = $tags,
            m.tag_prefixes = $tag_prefixes,
            m.importance = $importance,
            m.type = $type,
            m.confidence = $confidence,
            m.timestamp = $timestamp,
            m.metadata = $metadata,
            m.updated_at = $updated_at,
            m.last_accessed = $last_accessed
        RETURN m
    """

    graph.query(
        update_query,
        {
            "id": memory_id,
            "content": new_content,
            "tags": tags,
            "tag_prefixes": tag_prefixes,
            "importance": importance,
            "type": memory_type,
            "confidence": confidence,
            "timestamp": timestamp,
            "metadata": metadata_json,
            "updated_at": updated_at,
            "last_accessed": last_accessed,
        },
    )

    qdrant_client = get_qdrant_client_fn()
    vector = None
    if qdrant_client is not None:
        if new_content != current.get("content"):
            vector = generate_real_embedding_fn(new_content)
        else:
            try:
                existing = qdrant_client.retrieve(
                    collection_name=collection_name,
                    ids=[memory_id],
                    with_vectors=True,
                )
                if existing:
                    vector = existing[0].vector
            except Exception:
                logger.exception("Failed to retrieve existing vector; regenerating")
                vector = generate_real_embedding_fn(new_content)

        if vector is not None:
            payload = {
                "content": new_content,
                "tags": tags,
                "tag_prefixes": tag_prefixes,
                "importance": importance,
                "timestamp": timestamp,
                "type": memory_type,
                "confidence": confidence,
                "updated_at": updated_at,
                "last_accessed": last_accessed,
                "metadata": metadata,
            }
            qdrant_client.upsert(
                collection_name=collection_name,
                points=[point_struct_cls(id=memory_id, vector=vector, payload=payload)],
            )

    return jsonify_fn({"status": "success", "memory_id": memory_id})


def delete_memory(
    *,
    memory_id: str,
    get_memory_graph_fn: Any,
    get_qdrant_client_fn: Any,
    qdrant_models_obj: Any,
    collection_name: str,
    abort_fn: Any,
    jsonify_fn: Any,
    logger: Any,
) -> Any:
    graph = get_memory_graph_fn()
    if graph is None:
        abort_fn(503, description="FalkorDB is unavailable")

    result = graph.query("MATCH (m:Memory {id: $id}) RETURN m", {"id": memory_id})
    if not getattr(result, "result_set", None):
        abort_fn(404, description="Memory not found")

    graph.query("MATCH (m:Memory {id: $id}) DETACH DELETE m", {"id": memory_id})

    qdrant_client = get_qdrant_client_fn()
    if qdrant_client is not None:
        try:
            if qdrant_models_obj is not None:
                selector = qdrant_models_obj.PointIdsList(points=[memory_id])
            else:
                selector = {"points": [memory_id]}
            qdrant_client.delete(collection_name=collection_name, points_selector=selector)
        except Exception:
            logger.exception("Failed to delete vector for memory %s", memory_id)

    return jsonify_fn({"status": "success", "memory_id": memory_id})


def memories_by_tag(
    *,
    request_obj: Any,
    normalize_tag_list_fn: Any,
    get_memory_graph_fn: Any,
    serialize_node_fn: Any,
    parse_metadata_field_fn: Any,
    abort_fn: Any,
    jsonify_fn: Any,
    logger: Any,
) -> Any:
    raw_tags = request_obj.args.getlist("tags") or request_obj.args.get("tags")
    tags = normalize_tag_list_fn(raw_tags)
    if not tags:
        abort_fn(400, description="'tags' query parameter is required")

    limit = max(1, min(int(request_obj.args.get("limit", 20)), 200))

    graph = get_memory_graph_fn()
    if graph is None:
        abort_fn(503, description="FalkorDB is unavailable")

    params = {
        "tags": [tag.lower() for tag in tags],
        "limit": limit,
    }

    query = """
        MATCH (m:Memory)
        WHERE ANY(tag IN coalesce(m.tags, []) WHERE toLower(tag) IN $tags)
        RETURN m
        ORDER BY m.importance DESC, m.timestamp DESC
        LIMIT $limit
    """

    try:
        result = graph.query(query, params)
    except Exception:
        logger.exception("Tag search failed")
        abort_fn(500, description="Failed to search by tag")

    memories: List[Dict[str, Any]] = []
    for row in getattr(result, "result_set", []) or []:
        data = serialize_node_fn(row[0])
        data["metadata"] = parse_metadata_field_fn(data.get("metadata"))
        memories.append(data)

    return jsonify_fn(
        {"status": "success", "tags": tags, "count": len(memories), "memories": memories}
    )
