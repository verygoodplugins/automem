from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from flask import Blueprint, abort, jsonify, request


def create_consolidation_blueprint_full(
    get_memory_graph: Callable[[], Any],
    get_qdrant_client: Callable[[], Any],
    memory_consolidator_cls: Any,
    persist_run: Callable[[Any, Dict[str, Any]], None],
    build_scheduler: Callable[[Any], Any],
    load_recent_runs: Callable[[Any, int], List[Dict[str, Any]]],
    state: Any,
    tick_seconds: int,
    history_limit: int,
    logger: Any,
) -> Blueprint:
    bp = Blueprint("consolidation", __name__)

    @bp.route("/consolidate", methods=["POST"])
    def consolidate() -> Any:
        data = request.get_json(silent=True) or {}
        mode = data.get("mode", "full")
        dry_run = bool(data.get("dry_run", True))

        graph = get_memory_graph()
        if graph is None:
            abort(503, description="FalkorDB is unavailable")

        try:
            vector_store = get_qdrant_client()
            consolidator = memory_consolidator_cls(graph, vector_store)
            results = consolidator.consolidate(mode=mode, dry_run=dry_run)
            if not dry_run:
                persist_run(graph, results)
            return jsonify({"status": "success", "consolidation": results}), 200
        except Exception as e:
            logger.error(f"Consolidation failed: {e}")
            return jsonify({"error": "Consolidation failed", "details": str(e)}), 500

    @bp.route("/consolidate/status", methods=["GET"])
    def status() -> Any:
        graph = get_memory_graph()
        if graph is None:
            abort(503, description="FalkorDB is unavailable")

        try:
            scheduler = build_scheduler(graph)
            history = load_recent_runs(graph, history_limit)
            next_runs = scheduler.get_next_runs() if scheduler else {}
            return (
                jsonify(
                    {
                        "status": "success",
                        "next_runs": next_runs,
                        "history": history,
                        "thread_alive": bool(
                            state.consolidation_thread and state.consolidation_thread.is_alive()
                        ),
                        "tick_seconds": tick_seconds,
                    }
                ),
                200,
            )
        except Exception as e:
            logger.error(f"Failed to get consolidation status: {e}")
            return jsonify({"error": "Failed to get status", "details": str(e)}), 500

    return bp
