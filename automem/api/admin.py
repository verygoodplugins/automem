from __future__ import annotations

import json
from typing import Any, Callable, Dict, List
from flask import Blueprint, request, abort, jsonify


def _parse_metadata(raw: Any) -> Dict[str, Any]:
    """Parse metadata from FalkorDB which may be a string or dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            decoded = json.loads(raw)
            if isinstance(decoded, dict):
                return decoded
        except json.JSONDecodeError:
            pass
    return {}


def _parse_tags(raw: Any) -> List[str]:
    """Parse tags which may be a list or JSON string."""
    if isinstance(raw, list):
        return [str(t) for t in raw if t]
    if isinstance(raw, str) and raw:
        try:
            decoded = json.loads(raw)
            if isinstance(decoded, list):
                return [str(t) for t in decoded if t]
        except json.JSONDecodeError:
            pass
    return []


def create_admin_blueprint_full(
    require_admin_token: Callable[[], None],
    init_openai: Callable[[], None],
    get_openai_client: Callable[[], Any],
    get_qdrant_client: Callable[[], Any],
    get_memory_graph: Callable[[], Any],
    point_struct: Any,
    collection_name: str,
    vector_size: int,
    embedding_model: str,
    utc_now: Callable[[], str],
    logger: Any,
) -> Blueprint:
    bp = Blueprint("admin", __name__)

    @bp.route("/admin/reembed", methods=["POST"])
    def reembed() -> Any:
        require_admin_token()

        # Ensure OpenAI and Qdrant are available
        openai_client = get_openai_client()
        if openai_client is None:
            abort(503, description="OpenAI API key not configured - cannot generate real embeddings")

        qdrant_client = get_qdrant_client()
        if qdrant_client is None:
            abort(503, description="Qdrant is not available - cannot store embeddings")

        payload = request.get_json(silent=True) or {}
        try:
            batch_size = min(int(payload.get("batch_size", 32)), 100)
        except (TypeError, ValueError):
            batch_size = 32
        limit = payload.get("limit")
        force_reembed = bool(payload.get("force", False))

        graph = get_memory_graph()
        if graph is None:
            abort(503, description="FalkorDB is unavailable")

        # Fetch full memory data from FalkorDB (not just id and content)
        base_query = """
            MATCH (m:Memory)
            RETURN m.id AS id,
                   m.content AS content,
                   m.tags AS tags,
                   m.importance AS importance,
                   m.timestamp AS timestamp,
                   m.type AS type,
                   m.confidence AS confidence,
                   m.metadata AS metadata,
                   m.updated_at AS updated_at,
                   m.last_accessed AS last_accessed
            ORDER BY m.timestamp DESC
        """
        if not force_reembed:
            base_query = """
                MATCH (m:Memory)
                WHERE m.content IS NOT NULL
                RETURN m.id AS id,
                       m.content AS content,
                       m.tags AS tags,
                       m.importance AS importance,
                       m.timestamp AS timestamp,
                       m.type AS type,
                       m.confidence AS confidence,
                       m.metadata AS metadata,
                       m.updated_at AS updated_at,
                       m.last_accessed AS last_accessed
                ORDER BY m.timestamp DESC
            """
        if limit:
            try:
                base_query += f" LIMIT {int(limit)}"
            except (TypeError, ValueError):
                pass

        result = graph.query(base_query)
        to_process: List[Dict[str, Any]] = []
        for row in getattr(result, "result_set", []) or []:
            memory_id = row[0]
            content = row[1]
            if content:
                to_process.append({
                    "id": memory_id,
                    "content": content,
                    "tags": _parse_tags(row[2]),
                    "importance": row[3] if row[3] is not None else 0.5,
                    "timestamp": row[4],
                    "type": row[5] or "Context",
                    "confidence": row[6] if row[6] is not None else 0.6,
                    "metadata": _parse_metadata(row[7]),
                    "updated_at": row[8],
                    "last_accessed": row[9],
                })

        if not to_process:
            return jsonify({
                "status": "complete",
                "message": "No memories found to reembed",
                "processed": 0,
                "total": 0
            })

        processed = 0
        failed = 0
        failed_ids: List[str] = []

        # Process in batches
        for i in range(0, len(to_process), batch_size):
            batch = to_process[i : i + batch_size]
            texts = [mem["content"] for mem in batch]
            
            try:
                # Batch embedding request
                resp = openai_client.embeddings.create(
                    input=texts,
                    model=embedding_model,
                    dimensions=vector_size,
                )
                
                points = []
                for mem, data in zip(batch, resp.data):
                    embedding = data.embedding
                    # Preserve full payload from FalkorDB
                    payload_data = {
                        "content": mem["content"],
                        "tags": mem["tags"],
                        "importance": mem["importance"],
                        "timestamp": mem["timestamp"],
                        "type": mem["type"],
                        "confidence": mem["confidence"],
                        "metadata": mem["metadata"],
                        "updated_at": mem["updated_at"],
                        "last_accessed": mem["last_accessed"],
                    }
                    points.append(point_struct(id=mem["id"], vector=embedding, payload=payload_data))
                
                if points:
                    qdrant_client.upsert(collection_name=collection_name, points=points)
                    processed += len(points)
                    logger.info(f"Successfully reembedded batch of {len(points)} memories (preserving metadata)")
                    
            except Exception as e:
                logger.error(f"Failed to process batch starting at index {i}: {e}")
                failed += len(batch)
                failed_ids.extend([mem["id"] for mem in batch])

        response = {
            "status": "complete",
            "processed": processed,
            "failed": failed,
            "total": len(to_process),
            "batch_size": batch_size,
            "metadata_preserved": True,
        }
        if failed_ids:
            response["failed_ids"] = failed_ids[:10]
            if len(failed_ids) > 10:
                response["failed_ids_truncated"] = True
        return jsonify(response)

    return bp
