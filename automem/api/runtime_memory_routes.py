from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List

MEMORY_CONTENT_SOFT_LIMIT = 500
MEMORY_CONTENT_HARD_LIMIT = 2000


def _parse_by_tag_request(
    *,
    request_obj: Any,
    normalize_tag_list_fn: Any,
    abort_fn: Any,
) -> tuple[List[str], int, int]:
    raw_tags = request_obj.args.getlist("tags") or request_obj.args.get("tags")
    tags = normalize_tag_list_fn(raw_tags)
    if not tags:
        abort_fn(400, description="'tags' query parameter is required")

    try:
        limit = int(request_obj.args.get("limit", 20))
    except (TypeError, ValueError):
        limit = 20
    limit = max(1, min(limit, 200))

    try:
        offset = int(request_obj.args.get("offset", 0))
    except (TypeError, ValueError):
        offset = 0
    offset = max(0, offset)

    return tags, limit, offset


def _load_memories_by_tag_page(
    *,
    graph: Any,
    tags: List[str],
    limit: int,
    offset: int,
    serialize_node_fn: Any,
    parse_metadata_field_fn: Any,
    logger: Any,
    abort_fn: Any,
) -> tuple[List[Dict[str, Any]], bool]:
    params = {
        "tags": [tag.lower() for tag in tags],
        "offset": offset,
        "limit_plus_one": limit + 1,
    }
    query = """
        MATCH (m:Memory)
        WHERE ANY(tag IN coalesce(m.tags, []) WHERE toLower(tag) IN $tags)
        RETURN m
        ORDER BY m.importance DESC, m.timestamp DESC, m.id ASC
        SKIP $offset
        LIMIT $limit_plus_one
    """

    try:
        result = graph.query(query, params)
    except Exception:
        logger.exception("Tag search failed")
        abort_fn(500, description="Failed to search by tag")

    rows = list(getattr(result, "result_set", []) or [])
    has_more = len(rows) > limit
    memories: List[Dict[str, Any]] = []
    for row in rows[:limit]:
        data = serialize_node_fn(row[0])
        data["metadata"] = parse_metadata_field_fn(data.get("metadata"))
        memories.append(data)

    return memories, has_more


def _delete_qdrant_points(
    *,
    qdrant_client: Any,
    collection_name: str,
    memory_ids: List[str],
    logger: Any,
) -> None:
    if qdrant_client is None or not memory_ids:
        return

    selector: Any = {"points": memory_ids}
    try:
        from qdrant_client.http import models as http_models  # type: ignore

        selector = http_models.PointIdsList(points=memory_ids)
    except Exception:
        pass

    try:
        qdrant_client.delete(collection_name=collection_name, points_selector=selector)
    except Exception:
        logger.exception("Failed to delete vectors for %d memories", len(memory_ids))


def _delete_graph_memories(
    *,
    graph: Any,
    memory_ids: List[str],
    logger: Any,
    abort_fn: Any,
) -> None:
    if not memory_ids:
        return

    try:
        graph.query("MATCH (m:Memory) WHERE m.id IN $ids DETACH DELETE m", {"ids": memory_ids})
    except Exception:
        logger.exception("Bulk delete by tag failed for %d memories", len(memory_ids))
        abort_fn(500, description="Failed to delete memories by tag")


def store_memory(
    *,
    request_obj: Any,
    perf_counter_fn: Any,
    normalize_tags_fn: Any,
    compute_tag_prefixes_fn: Any,
    coerce_importance_fn: Any,
    normalize_memory_type_fn: Any,
    memory_types: Any,
    type_aliases: Dict[str, Any],
    classify_memory_fn: Any,
    normalize_timestamp_fn: Any,
    coerce_embedding_fn: Any,
    get_memory_graph_fn: Any,
    get_qdrant_client_fn: Any,
    enqueue_enrichment_fn: Any,
    enqueue_embedding_fn: Any,
    collection_name: str,
    point_struct_cls: Any,
    state: Any,
    logger: Any,
    emit_event_fn: Any,
    utc_now_fn: Any,
    uuid4_fn: Any,
    abort_fn: Any,
    jsonify_fn: Any,
) -> Any:
    query_start = perf_counter_fn()
    payload = request_obj.get_json(silent=True)
    if not isinstance(payload, dict):
        abort_fn(400, description="JSON body is required")

    content = (payload.get("content") or "").strip()
    if not content:
        abort_fn(400, description="'content' is required")
    if len(content) > MEMORY_CONTENT_HARD_LIMIT:
        abort_fn(
            400,
            description=(
                f"Content exceeds maximum length of {MEMORY_CONTENT_HARD_LIMIT} characters "
                f"({len(content)} provided)."
            ),
        )

    needs_auto_summarize = len(content) > MEMORY_CONTENT_SOFT_LIMIT

    raw_memory_id = payload.get("id")
    memory_id = str(raw_memory_id).strip() if raw_memory_id else str(uuid4_fn())
    try:
        uuid.UUID(memory_id)
    except (ValueError, TypeError):
        abort_fn(400, description="'id' must be a valid UUID")

    tags = normalize_tags_fn(payload.get("tags"))
    tags_lower = [t.strip().lower() for t in tags if isinstance(t, str) and t.strip()]
    tag_prefixes = compute_tag_prefixes_fn(tags_lower)
    importance = coerce_importance_fn(payload.get("importance"))

    metadata_raw = payload.get("metadata")
    if metadata_raw is None:
        metadata: Dict[str, Any] = {}
    elif isinstance(metadata_raw, dict):
        metadata = metadata_raw
    else:
        abort_fn(400, description="'metadata' must be an object")

    if needs_auto_summarize:
        metadata["needs_auto_summarize"] = True

    metadata_json = json.dumps(metadata, default=str)

    raw_type = payload.get("type")
    type_confidence = payload.get("confidence")

    if raw_type:
        memory_type, was_normalized = normalize_memory_type_fn(raw_type)

        if not memory_type:
            valid_types = sorted(memory_types)
            alias_examples = ", ".join(f"'{k}'" for k in list(type_aliases.keys())[:5])
            abort_fn(
                400,
                description=(
                    f"Invalid memory type '{raw_type}'. "
                    f"Must be one of: {', '.join(valid_types)}, "
                    f"or aliases like {alias_examples}..."
                ),
            )

        if was_normalized and memory_type != raw_type:
            logger.debug("Normalized type '%s' -> '%s'", raw_type, memory_type)

        if type_confidence is None:
            type_confidence = 0.9
        else:
            type_confidence = coerce_importance_fn(type_confidence)
    else:
        memory_type, type_confidence = classify_memory_fn(content)

    t_valid = payload.get("t_valid")
    t_invalid = payload.get("t_invalid")
    if t_valid:
        try:
            t_valid = normalize_timestamp_fn(t_valid)
        except ValueError as exc:
            abort_fn(400, description=f"Invalid t_valid: {exc}")
    if t_invalid:
        try:
            t_invalid = normalize_timestamp_fn(t_invalid)
        except ValueError as exc:
            abort_fn(400, description=f"Invalid t_invalid: {exc}")

    try:
        embedding = coerce_embedding_fn(payload.get("embedding"))
    except ValueError as exc:
        abort_fn(400, description=str(exc))

    graph = get_memory_graph_fn()
    if graph is None:
        abort_fn(503, description="FalkorDB is unavailable")

    created_at = payload.get("timestamp")
    if created_at:
        try:
            created_at = normalize_timestamp_fn(created_at)
        except ValueError as exc:
            abort_fn(400, description=str(exc))
    else:
        created_at = utc_now_fn()

    updated_at = payload.get("updated_at")
    if updated_at:
        try:
            updated_at = normalize_timestamp_fn(updated_at)
        except ValueError as exc:
            abort_fn(400, description=f"Invalid updated_at: {exc}")
    else:
        updated_at = created_at

    last_accessed = payload.get("last_accessed")
    if last_accessed:
        try:
            last_accessed = normalize_timestamp_fn(last_accessed)
        except ValueError as exc:
            abort_fn(400, description=f"Invalid last_accessed: {exc}")
    else:
        last_accessed = updated_at

    try:
        graph.query(
            """
            MERGE (m:Memory {id: $id})
            SET m.content = $content,
                m.timestamp = $timestamp,
                m.importance = $importance,
                m.tags = $tags,
                m.tag_prefixes = $tag_prefixes,
                m.type = $type,
                m.confidence = $confidence,
                m.t_valid = $t_valid,
                m.t_invalid = $t_invalid,
                m.updated_at = $updated_at,
                m.last_accessed = $last_accessed,
                m.metadata = $metadata,
                m.processed = false
            RETURN m
            """,
            {
                "id": memory_id,
                "content": content,
                "timestamp": created_at,
                "importance": importance,
                "tags": tags,
                "tag_prefixes": tag_prefixes,
                "type": memory_type,
                "confidence": type_confidence,
                "t_valid": t_valid or created_at,
                "t_invalid": t_invalid,
                "updated_at": updated_at,
                "last_accessed": last_accessed,
                "metadata": metadata_json,
            },
        )
    except Exception:  # pragma: no cover - log full stack trace in production
        logger.exception("Failed to persist memory in FalkorDB")
        abort_fn(500, description="Failed to store memory in FalkorDB")

    enqueue_enrichment_fn(memory_id)

    embedding_status = "skipped"
    qdrant_client = get_qdrant_client_fn()

    if embedding is not None:
        embedding_status = "provided"
        qdrant_result = None
        if qdrant_client is not None:
            try:
                qdrant_client.upsert(
                    collection_name=collection_name,
                    points=[
                        point_struct_cls(
                            id=memory_id,
                            vector=embedding,
                            payload={
                                "content": content,
                                "tags": tags,
                                "tag_prefixes": tag_prefixes,
                                "importance": importance,
                                "timestamp": created_at,
                                "type": memory_type,
                                "confidence": type_confidence,
                                "updated_at": updated_at,
                                "last_accessed": last_accessed,
                                "metadata": metadata,
                            },
                        )
                    ],
                )
                qdrant_result = "stored"
            except Exception:  # pragma: no cover - log full stack trace in production
                logger.exception(
                    "Qdrant upsert failed for memory %s in collection %s",
                    memory_id,
                    collection_name,
                )
                qdrant_result = "failed"
    elif qdrant_client is not None:
        enqueue_embedding_fn(memory_id, content)
        embedding_status = "queued"
        qdrant_result = "queued"
    else:
        qdrant_result = "unconfigured"

    response = {
        "status": "success",
        "memory_id": memory_id,
        "stored_at": created_at,
        "type": memory_type,
        "confidence": type_confidence,
        "qdrant": qdrant_result,
        "embedding_status": embedding_status,
        "enrichment": "queued" if state.enrichment_queue else "disabled",
        "metadata": metadata,
        "timestamp": created_at,
        "updated_at": updated_at,
        "last_accessed": last_accessed,
        "query_time_ms": round((perf_counter_fn() - query_start) * 1000, 2),
    }

    logger.info(
        "memory_stored",
        extra={
            "memory_id": memory_id,
            "type": memory_type,
            "importance": importance,
            "tags_count": len(tags),
            "content_length": len(content),
            "latency_ms": response["query_time_ms"],
            "embedding_status": embedding_status,
            "qdrant_status": qdrant_result,
            "enrichment_queued": bool(state.enrichment_queue),
        },
    )

    emit_event_fn(
        "memory.store",
        {
            "id": memory_id,
            "content_preview": content[:100] + "..." if len(content) > 100 else content,
            "type": memory_type,
            "importance": importance,
            "tags": tags[:5],
            "size_bytes": len(content),
            "elapsed_ms": int(response["query_time_ms"]),
        },
        utc_now_fn,
    )

    return jsonify_fn(response), 201


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
    try:
        uuid.UUID(memory_id)
    except (ValueError, TypeError):
        abort_fn(400, description="memory_id must be a valid UUID")

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
    if "content" in payload:
        if not new_content:
            abort_fn(400, description="'content' is required")
        if len(new_content) > MEMORY_CONTENT_HARD_LIMIT:
            abort_fn(
                400,
                description=(
                    f"Content exceeds maximum length of {MEMORY_CONTENT_HARD_LIMIT} characters "
                    f"({len(new_content)} provided)."
                ),
            )
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

    if "content" in payload and new_content and len(new_content) > MEMORY_CONTENT_SOFT_LIMIT:
        metadata["needs_auto_summarize"] = True

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
            qdrant_payload = {
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
            try:
                qdrant_client.upsert(
                    collection_name=collection_name,
                    points=[point_struct_cls(id=memory_id, vector=vector, payload=qdrant_payload)],
                )
            except Exception:
                logger.exception(
                    "Qdrant upsert failed for memory %s in collection %s",
                    memory_id,
                    collection_name,
                )

    return jsonify_fn({"status": "success", "memory_id": memory_id})


def delete_memory(
    *,
    memory_id: str,
    get_memory_graph_fn: Any,
    get_qdrant_client_fn: Any,
    collection_name: str,
    abort_fn: Any,
    jsonify_fn: Any,
    logger: Any,
) -> Any:
    try:
        uuid.UUID(memory_id)
    except (ValueError, TypeError):
        abort_fn(400, description="memory_id must be a valid UUID")

    graph = get_memory_graph_fn()
    if graph is None:
        abort_fn(503, description="FalkorDB is unavailable")

    result = graph.query("MATCH (m:Memory {id: $id}) RETURN m", {"id": memory_id})
    if not getattr(result, "result_set", None):
        abort_fn(404, description="Memory not found")

    graph.query("MATCH (m:Memory {id: $id}) DETACH DELETE m", {"id": memory_id})

    _delete_qdrant_points(
        qdrant_client=get_qdrant_client_fn(),
        collection_name=collection_name,
        memory_ids=[memory_id],
        logger=logger,
    )

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
    get_qdrant_client_fn: Any = None,
    collection_name: str = "",
    on_access_fn: Any = None,
) -> Any:
    graph = get_memory_graph_fn()
    if graph is None:
        abort_fn(503, description="FalkorDB is unavailable")

    tags, limit, offset = _parse_by_tag_request(
        request_obj=request_obj,
        normalize_tag_list_fn=normalize_tag_list_fn,
        abort_fn=abort_fn,
    )

    if request_obj.method == "DELETE":
        deleted_count = 0
        while True:
            memories, _ = _load_memories_by_tag_page(
                graph=graph,
                tags=tags,
                limit=200,
                offset=0,
                serialize_node_fn=serialize_node_fn,
                parse_metadata_field_fn=parse_metadata_field_fn,
                logger=logger,
                abort_fn=abort_fn,
            )
            memory_ids = [str(memory.get("id")) for memory in memories if memory.get("id")]
            if not memory_ids:
                break

            _delete_graph_memories(
                graph=graph,
                memory_ids=memory_ids,
                logger=logger,
                abort_fn=abort_fn,
            )

            if get_qdrant_client_fn and collection_name:
                _delete_qdrant_points(
                    qdrant_client=get_qdrant_client_fn(),
                    collection_name=collection_name,
                    memory_ids=memory_ids,
                    logger=logger,
                )
            deleted_count += len(memory_ids)

        return jsonify_fn({"status": "success", "tags": tags, "deleted_count": deleted_count})

    memories, has_more = _load_memories_by_tag_page(
        graph=graph,
        tags=tags,
        limit=limit,
        offset=offset,
        serialize_node_fn=serialize_node_fn,
        parse_metadata_field_fn=parse_metadata_field_fn,
        logger=logger,
        abort_fn=abort_fn,
    )

    if on_access_fn and memories:
        accessed_ids = [str(memory.get("id")) for memory in memories if memory.get("id")]
        if accessed_ids:
            on_access_fn(accessed_ids)

    return jsonify_fn(
        {
            "status": "success",
            "tags": tags,
            "count": len(memories),
            "limit": limit,
            "offset": offset,
            "has_more": has_more,
            "memories": memories,
        }
    )
