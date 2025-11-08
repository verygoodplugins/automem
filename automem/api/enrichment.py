from __future__ import annotations

from typing import Any, Callable
from flask import Blueprint, jsonify, request, abort


def create_enrichment_blueprint(
    require_admin_token: Callable[[], None],
    state: Any,
    enqueue_enrichment: Callable[..., None],
    max_attempts: int,
) -> Blueprint:
    bp = Blueprint("enrichment", __name__)

    @bp.route("/enrichment/status", methods=["GET"])
    def enrichment_status() -> Any:
        queue_size = state.enrichment_queue.qsize() if state.enrichment_queue else 0
        thread_alive = bool(state.enrichment_thread and state.enrichment_thread.is_alive())

        with state.enrichment_lock:
            pending = len(state.enrichment_pending)
            inflight = len(state.enrichment_inflight)

        response = {
            "status": "running" if thread_alive else "stopped",
            "queue_size": queue_size,
            "pending": pending,
            "inflight": inflight,
            "max_attempts": max_attempts,
            "stats": state.enrichment_stats.to_dict(),
        }
        return jsonify(response)

    @bp.route("/enrichment/reprocess", methods=["POST"])
    def enrichment_reprocess() -> Any:
        require_admin_token()

        payload = request.get_json(silent=True) or {}
        ids: set[str] = set()

        raw_ids = payload.get("ids") or request.args.get("ids")
        if isinstance(raw_ids, str):
            ids.update(part.strip() for part in raw_ids.split(",") if part.strip())
        elif isinstance(raw_ids, list):
            for item in raw_ids:
                if isinstance(item, str) and item.strip():
                    ids.add(item.strip())

        if not ids:
            abort(400, description="No memory ids provided for reprocessing")

        for memory_id in ids:
            enqueue_enrichment(memory_id, forced=True)

        return jsonify({
            "status": "queued",
            "count": len(ids),
            "ids": sorted(ids),
        }), 202

    return bp

