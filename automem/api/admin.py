from __future__ import annotations

from typing import Any, Callable, List, Tuple
from flask import Blueprint, request, abort, jsonify


def create_admin_blueprint_full(
    require_admin_token: Callable[[], None],
    init_openai: Callable[[], None],
    get_openai_client: Callable[[], Any],
    get_qdrant_client: Callable[[], Any],
    get_memory_graph: Callable[[], Any],
    point_struct: Any,
    collection_name: str,
    vector_size: int,
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

        if force_reembed:
            query = "MATCH (m:Memory) RETURN m.id, m.content ORDER BY m.timestamp DESC"
        else:
            query = (
                "MATCH (m:Memory)\n"
                "WHERE m.content IS NOT NULL\n"
                "RETURN m.id, m.content\n"
                "ORDER BY m.timestamp DESC"
            )
        if limit:
            try:
                query += f" LIMIT {int(limit)}"
            except (TypeError, ValueError):
                pass

        result = graph.query(query)
        to_process: List[Tuple[str, str]] = []
        for row in getattr(result, "result_set", []) or []:
            memory_id = row[0]
            content = row[1]
            if content:
                to_process.append((memory_id, content))

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
            points = []
            for memory_id, content in batch:
                try:
                    resp = openai_client.embeddings.create(
                        input=content,
                        model="text-embedding-3-small",
                        dimensions=vector_size,
                    )
                    embedding = resp.data[0].embedding
                    payload_data = {
                        "content": content,
                        "tags": [],
                        "importance": 0.5,
                        "timestamp": utc_now(),
                        "type": "Context",
                        "confidence": 0.6,
                        "metadata": {},
                    }
                    points.append(point_struct(id=memory_id, vector=embedding, payload=payload_data))
                    processed += 1
                except Exception as e:
                    logger.error(f"Failed to generate embedding for memory {memory_id}: {e}")
                    failed += 1
                    failed_ids.append(memory_id)

            if points:
                try:
                    qdrant_client.upsert(collection_name=collection_name, points=points)
                    logger.info(f"Successfully reembedded batch of {len(points)} memories")
                except Exception as e:
                    logger.error(f"Failed to upsert batch to Qdrant: {e}")
                    failed += len(points)
                    failed_ids.extend([p.id for p in points if hasattr(p, 'id')])
                    processed -= len(points)

        response = {
            "status": "complete",
            "processed": processed,
            "failed": failed,
            "total": len(to_process),
            "batch_size": batch_size,
        }
        if failed_ids:
            response["failed_ids"] = failed_ids[:10]
            if len(failed_ids) > 10:
                response["failed_ids_truncated"] = True
        return jsonify(response)

    return bp
