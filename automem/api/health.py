from __future__ import annotations

from typing import Any, Callable, Optional
from flask import Blueprint, jsonify


def create_health_blueprint(
    get_memory_graph: Callable[[], Any],
    get_qdrant_client: Callable[[], Any],
    state: Any,
    graph_name: str,
    collection_name: str,
    utc_now: Callable[[], str],
) -> Blueprint:
    bp = Blueprint("health", __name__)

    @bp.route("/health", methods=["GET"])
    def health() -> Any:
        graph_available = get_memory_graph() is not None
        qdrant_available = get_qdrant_client() is not None

        enrichment_thread_alive = bool(state.enrichment_thread and state.enrichment_thread.is_alive())
        with state.enrichment_lock:
            enrichment_pending = len(state.enrichment_pending)
            enrichment_inflight = len(state.enrichment_inflight)

        # Get memory count from FalkorDB (gracefully fail if unavailable)
        memory_count: Optional[int] = None
        if graph_available:
            try:
                graph = get_memory_graph()
                if graph:
                    result = graph.query("MATCH (m:Memory) RETURN COUNT(m) as count")
                    if getattr(result, "result_set", None):
                        memory_count = result.result_set[0][0]
            except Exception:
                pass

        # Get vector count from Qdrant (gracefully fail if unavailable)
        vector_count: Optional[int] = None
        if qdrant_available:
            try:
                qdrant = get_qdrant_client()
                if qdrant:
                    info = qdrant.get_collection(collection_name)
                    vector_count = info.points_count
            except Exception:
                pass

        # Determine sync status
        sync_status = "unknown"
        if memory_count is not None and vector_count is not None:
            if memory_count == vector_count:
                sync_status = "synced"
            elif vector_count < memory_count:
                sync_status = "drift_detected"
            else:
                # More vectors than memories (orphaned vectors)
                sync_status = "orphaned_vectors"

        # Overall status considers sync
        if not graph_available or not qdrant_available:
            status = "degraded"
        elif sync_status == "drift_detected":
            status = "degraded"
        else:
            status = "healthy"

        health_data = {
            "status": status,
            "falkordb": "connected" if graph_available else "disconnected",
            "qdrant": "connected" if qdrant_available else "disconnected",
            "memory_count": memory_count,
            "vector_count": vector_count,
            "sync_status": sync_status,
            "enrichment": {
                "status": "running" if enrichment_thread_alive else "stopped",
                "queue_depth": state.enrichment_queue.qsize() if state.enrichment_queue else 0,
                "pending": enrichment_pending,
                "inflight": enrichment_inflight,
                "processed": state.enrichment_stats.successes,
                "failed": state.enrichment_stats.failures,
            },
            "timestamp": utc_now(),
            "graph": graph_name,
        }
        return jsonify(health_data)

    return bp

