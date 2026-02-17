from __future__ import annotations

import json
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set

from flask import Blueprint, abort, jsonify, request
from flask.typing import ResponseReturnValue

from automem.config import (
    CLASSIFICATION_MODEL,
    MEMORY_AUTO_SUMMARIZE,
    MEMORY_CONTENT_HARD_LIMIT,
    MEMORY_CONTENT_SOFT_LIMIT,
    MEMORY_SUMMARY_TARGET_LENGTH,
)
from automem.utils.text import should_summarize_content, summarize_content


def create_memory_blueprint(
    store_memory: Callable[[], Any],
    update_memory: Callable[[str], Any],
    delete_memory: Callable[[str], Any],
    by_tag: Callable[[], Any],
    associate: Callable[[], Any],
) -> Blueprint:
    """Compatibility wrapper around the legacy handlers in app.py."""
    bp = Blueprint("memory", __name__)

    @bp.route("/memory", methods=["POST"])
    def _store() -> Any:
        return store_memory()

    @bp.route("/memory/<memory_id>", methods=["PATCH"])
    def _update(memory_id: str) -> Any:
        return update_memory(memory_id)

    @bp.route("/memory/<memory_id>", methods=["DELETE"])
    def _delete(memory_id: str) -> Any:
        return delete_memory(memory_id)

    @bp.route("/memory/by-tag", methods=["GET"])
    def _by_tag() -> Any:
        return by_tag()

    @bp.route("/associate", methods=["POST"])
    def _associate() -> Any:
        return associate()

    return bp


def create_memory_blueprint_full(
    get_memory_graph: Callable[[], Any],
    get_qdrant_client: Callable[[], Any],
    normalize_tags: Callable[[Any], List[str]],
    normalize_tag_list: Callable[[Any], List[str]],
    compute_tag_prefixes: Callable[[List[str]], List[str]],
    coerce_importance: Callable[[Any], float],
    coerce_embedding: Callable[[Any], Optional[List[float]]],
    normalize_timestamp: Callable[[str], str],
    utc_now: Callable[[], str],
    serialize_node: Callable[[Any], Dict[str, Any]],
    parse_metadata_field: Callable[[Any], Any],
    generate_real_embedding: Callable[[str], List[float]],
    enqueue_enrichment: Callable[[str], None],
    enqueue_embedding: Callable[[str, str], None],
    memory_classify: Callable[[str], tuple[str, float]],
    point_struct: Any,
    collection_name: str,
    allowed_relations: Set[str] | List[str],
    relation_types: Dict[str, Any],
    state: Any,
    logger: Any,
    on_access: Optional[Callable[[List[str]], None]] = None,
    get_openai_client: Optional[Callable[[], Any]] = None,
) -> Blueprint:
    bp = Blueprint("memory", __name__)

    @bp.route("/memory", methods=["POST"])
    def store() -> Any:
        query_start = time.perf_counter()
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            abort(400, description="JSON body is required")

        content = (payload.get("content") or "").strip()
        if not content:
            abort(400, description="'content' is required")

        # Content size governance: check if summarization or rejection is needed
        original_content: Optional[str] = None
        content_action = should_summarize_content(
            content, MEMORY_CONTENT_SOFT_LIMIT, MEMORY_CONTENT_HARD_LIMIT
        )

        if content_action == "reject":
            abort(
                400,
                description=f"Content exceeds maximum length of {MEMORY_CONTENT_HARD_LIMIT} characters "
                f"({len(content)} provided). Please split into smaller memories or summarize.",
            )

        if content_action == "summarize" and MEMORY_AUTO_SUMMARIZE:
            openai_client = get_openai_client() if get_openai_client else None
            if openai_client:
                summary = summarize_content(
                    content,
                    openai_client,
                    CLASSIFICATION_MODEL,
                    MEMORY_SUMMARY_TARGET_LENGTH,
                )
                if summary:
                    original_content = content
                    content = summary
                    logger.info(
                        "Auto-summarized oversized memory: %d -> %d chars",
                        len(original_content),
                        len(content),
                    )
                else:
                    logger.warning(
                        "Auto-summarization failed for %d char content, storing as-is",
                        len(content),
                    )
            else:
                logger.warning(
                    "Content exceeds soft limit (%d chars) but OpenAI client unavailable for summarization",
                    len(content),
                )

        tags = normalize_tags(payload.get("tags"))
        tags_lower = [t.strip().lower() for t in tags if isinstance(t, str) and t.strip()]
        tag_prefixes = compute_tag_prefixes(tags_lower)
        importance = coerce_importance(payload.get("importance"))
        # Always generate server-side UUID to prevent collision/overwrite attacks
        memory_id = str(uuid.uuid4())

        metadata_raw = payload.get("metadata")
        if metadata_raw is None:
            metadata: Dict[str, Any] = {}
        elif isinstance(metadata_raw, dict):
            metadata = metadata_raw
        else:
            abort(400, description="'metadata' must be an object")

        # If content was summarized, preserve original in metadata for audit trail
        if original_content:
            metadata["original_content"] = original_content
            metadata["was_summarized"] = True
            metadata["original_length"] = len(original_content)

        metadata_json = json.dumps(metadata, default=str)

        # Accept explicit type/confidence or classify automatically
        memory_type = payload.get("type")
        type_confidence = payload.get("confidence")
        if memory_type:
            # Validate explicit type
            # (Memory types are validated by the classifier caller; keep permissive here)
            if type_confidence is None:
                type_confidence = 0.9
            else:
                type_confidence = coerce_importance(type_confidence)
        else:
            memory_type, type_confidence = memory_classify(content)

        t_valid = payload.get("t_valid")
        t_invalid = payload.get("t_invalid")
        if t_valid:
            try:
                t_valid = normalize_timestamp(t_valid)
            except ValueError as exc:
                abort(400, description=f"Invalid t_valid: {exc}")
        if t_invalid:
            try:
                t_invalid = normalize_timestamp(t_invalid)
            except ValueError as exc:
                abort(400, description=f"Invalid t_invalid: {exc}")

        try:
            embedding = coerce_embedding(payload.get("embedding"))
        except ValueError as exc:
            abort(400, description=str(exc))

        graph = get_memory_graph()
        if graph is None:
            abort(503, description="FalkorDB is unavailable")

        created_at = payload.get("timestamp")
        if created_at:
            try:
                created_at = normalize_timestamp(created_at)
            except ValueError as exc:
                abort(400, description=str(exc))
        else:
            created_at = utc_now()

        updated_at = payload.get("updated_at")
        if updated_at:
            try:
                updated_at = normalize_timestamp(updated_at)
            except ValueError as exc:
                abort(400, description=f"Invalid updated_at: {exc}")
        else:
            updated_at = created_at

        last_accessed = payload.get("last_accessed")
        if last_accessed:
            try:
                last_accessed = normalize_timestamp(last_accessed)
            except ValueError as exc:
                abort(400, description=f"Invalid last_accessed: {exc}")
        else:
            last_accessed = updated_at

        try:
            graph.query(
                """
                MERGE (m:Memory {id: $id})
                ON CREATE SET
                    m.content = $content,
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
        except Exception:
            logger.exception("Failed to persist memory in FalkorDB")
            abort(500, description="Failed to store memory in FalkorDB")

        # Queue enrichment
        enqueue_enrichment(memory_id)

        # Handle embeddings
        embedding_status = "skipped"
        qdrant_client = get_qdrant_client()
        if embedding is not None:
            embedding_status = "provided"
            qdrant_result = None
            if qdrant_client is not None:
                try:
                    qdrant_client.upsert(
                        collection_name=collection_name,
                        points=[
                            point_struct(
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
                except Exception:
                    logger.exception("Qdrant upsert failed")
                    qdrant_result = "failed"
        elif qdrant_client is not None:
            enqueue_embedding(memory_id, content)
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
            "query_time_ms": round((time.perf_counter() - query_start) * 1000, 2),
        }

        # Include summarization info in response
        if original_content:
            response["summarized"] = True
            response["original_length"] = len(original_content)
            response["summarized_length"] = len(content)

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
        return jsonify(response), 201

    @bp.route("/memory/<memory_id>", methods=["GET"])
    def get(memory_id: str) -> ResponseReturnValue:
        try:
            uuid.UUID(memory_id)
        except ValueError:
            abort(400, description="memory_id must be a valid UUID")

        graph = get_memory_graph()
        if graph is None:
            abort(503, description="Graph database unavailable")

        try:
            result = graph.query(
                "MATCH (m:Memory {id: $id}) RETURN m",
                {"id": memory_id},
            )
        except Exception:
            logger.exception("Failed to fetch memory %s", memory_id)
            abort(500, description="Failed to fetch memory")

        if not getattr(result, "result_set", None):
            abort(404, description="Memory not found")

        node = serialize_node(result.result_set[0][0])
        # Parse metadata field for consistency with by_tag endpoint
        node["metadata"] = parse_metadata_field(node.get("metadata"))

        # Update last_accessed timestamp for consistency with by_tag endpoint
        if on_access:
            on_access([memory_id])

        return jsonify({"status": "success", "memory": node})

    @bp.route("/memory/<memory_id>", methods=["PATCH"])
    def update(memory_id: str) -> Any:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            abort(400, description="JSON body is required")

        graph = get_memory_graph()
        if graph is None:
            abort(503, description="FalkorDB is unavailable")

        result = graph.query("MATCH (m:Memory {id: $id}) RETURN m", {"id": memory_id})
        if not getattr(result, "result_set", None):
            abort(404, description="Memory not found")

        current_node = result.result_set[0][0]
        current = serialize_node(current_node)

        new_content = payload.get("content", current.get("content"))
        tags = normalize_tag_list(payload.get("tags", current.get("tags")))
        tags_lower = [t.strip().lower() for t in tags if isinstance(t, str) and t.strip()]
        tag_prefixes = compute_tag_prefixes(tags_lower)
        importance = payload.get("importance", current.get("importance"))
        memory_type = payload.get("type", current.get("type"))
        confidence = payload.get("confidence", current.get("confidence"))
        timestamp = payload.get("timestamp", current.get("timestamp"))
        metadata_raw = payload.get("metadata", parse_metadata_field(current.get("metadata")))
        updated_at = payload.get("updated_at", current.get("updated_at", utc_now()))
        last_accessed = payload.get("last_accessed", current.get("last_accessed"))

        if metadata_raw is None:
            metadata: Dict[str, Any] = {}
        elif isinstance(metadata_raw, dict):
            metadata = metadata_raw
        else:
            abort(400, description="'metadata' must be an object")
        metadata_json = json.dumps(metadata, default=str)

        if timestamp:
            try:
                timestamp = normalize_timestamp(timestamp)
            except ValueError as exc:
                abort(400, description=f"Invalid timestamp: {exc}")

        if updated_at:
            try:
                updated_at = normalize_timestamp(updated_at)
            except ValueError as exc:
                abort(400, description=f"Invalid updated_at: {exc}")

        if last_accessed:
            try:
                last_accessed = normalize_timestamp(last_accessed)
            except ValueError as exc:
                abort(400, description=f"Invalid last_accessed: {exc}")

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

        qdrant_client = get_qdrant_client()
        vector = None
        if qdrant_client is not None:
            if new_content != current.get("content"):
                vector = generate_real_embedding(new_content)
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
                    vector = generate_real_embedding(new_content)

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
                    points=[point_struct(id=memory_id, vector=vector, payload=payload)],
                )

        return jsonify({"status": "success", "memory_id": memory_id})

    @bp.route("/memory/<memory_id>", methods=["DELETE"])
    def delete(memory_id: str) -> Any:
        graph = get_memory_graph()
        if graph is None:
            abort(503, description="FalkorDB is unavailable")

        result = graph.query("MATCH (m:Memory {id: $id}) RETURN m", {"id": memory_id})
        if not getattr(result, "result_set", None):
            abort(404, description="Memory not found")

        graph.query("MATCH (m:Memory {id: $id}) DETACH DELETE m", {"id": memory_id})

        qdrant_client = get_qdrant_client()
        if qdrant_client is not None:
            try:
                selector = {"points": [memory_id]}
                if hasattr(qdrant_client, "http"):
                    # Try to use HTTP models selector if available
                    try:
                        from qdrant_client.http import models as http_models  # type: ignore

                        selector = http_models.PointIdsList(points=[memory_id])
                    except Exception as e:
                        logger.warning(f"Failed to import qdrant_client.http.models: {e}")
                qdrant_client.delete(collection_name=collection_name, points_selector=selector)
            except Exception:
                logger.exception("Failed to delete vector for memory %s", memory_id)

        return jsonify({"status": "success", "memory_id": memory_id})

    @bp.route("/memory/by-tag", methods=["GET"])
    def by_tag() -> Any:
        raw_tags = request.args.getlist("tags") or request.args.get("tags")
        tags = normalize_tag_list(raw_tags)
        if not tags:
            abort(400, description="'tags' query parameter is required")

        try:
            limit = int(request.args.get("limit", 20))
        except (TypeError, ValueError):
            limit = 20
        limit = max(1, min(limit, 200))

        graph = get_memory_graph()
        if graph is None:
            abort(503, description="FalkorDB is unavailable")

        params = {"tags": [tag.lower() for tag in tags], "limit": limit}
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
            abort(500, description="Failed to search by tag")

        memories: List[Dict[str, Any]] = []
        for row in getattr(result, "result_set", []) or []:
            data = serialize_node(row[0])
            data["metadata"] = parse_metadata_field(data.get("metadata"))
            memories.append(data)

        # Update last_accessed for retrieved memories
        if on_access and memories:
            accessed_ids = [str(m.get("id")) for m in memories if m.get("id")]
            if accessed_ids:
                on_access(accessed_ids)

        return jsonify(
            {
                "status": "success",
                "tags": tags,
                "count": len(memories),
                "memories": memories,
            }
        )

    @bp.route("/associate", methods=["POST"])
    def associate() -> Any:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            abort(400, description="JSON body is required")

        memory1_id = (payload.get("memory1_id") or "").strip()
        memory2_id = (payload.get("memory2_id") or "").strip()
        relation_type = (payload.get("type") or "RELATES_TO").upper()
        strength = coerce_importance(payload.get("strength", 0.5))

        if not memory1_id or not memory2_id:
            abort(400, description="'memory1_id' and 'memory2_id' are required")
        if memory1_id == memory2_id:
            abort(400, description="Cannot associate a memory with itself")
        if relation_type not in set(allowed_relations):
            abort(
                400,
                description=f"Relation type must be one of {sorted(allowed_relations)}",
            )

        graph = get_memory_graph()
        if graph is None:
            abort(503, description="FalkorDB is unavailable")

        timestamp = utc_now()

        relationship_props = {"strength": strength, "updated_at": timestamp}
        relation_config = relation_types.get(relation_type, {})
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
                {"id1": memory1_id, "id2": memory2_id, **relationship_props},
            )
        except Exception:
            logger.exception("Failed to create association")
            abort(500, description="Failed to create association")

        if not getattr(result, "result_set", None):
            abort(404, description="One or both memories do not exist")

        response = {
            "status": "success",
            "message": f"Association created between {memory1_id} and {memory2_id}",
            "relation_type": relation_type,
            "strength": strength,
        }
        for prop in relation_config.get("properties", []):
            if prop in relationship_props:
                response[prop] = relationship_props[prop]
        return jsonify(response), 201

    return bp
