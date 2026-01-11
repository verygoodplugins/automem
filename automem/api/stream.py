"""Server-Sent Events endpoint for real-time observability.

Provides a /stream endpoint that emits events for memory operations,
enrichment, and consolidation tasks. Uses an in-memory subscriber
pattern with bounded queues per client.

Optionally logs events to a JSONL file for persistence (env: AUTOMEM_EVENT_LOG).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Lock
from typing import Any, Callable, Dict, Generator, List

from flask import Blueprint, Response

# Subscriber management - thread-safe list of client queues
_subscribers: List[Queue] = []
_subscribers_lock = Lock()

# Event log configuration
_event_log_path = os.getenv("AUTOMEM_EVENT_LOG", "")
_event_log_max = int(os.getenv("AUTOMEM_EVENT_LOG_MAX", "500"))
_event_log_lock = Lock()


def _write_event_to_log(event: Dict[str, Any]) -> None:
    """Append event to JSONL log file, truncating if needed.

    Thread-safe. Only writes if AUTOMEM_EVENT_LOG is set.
    """
    if not _event_log_path:
        return

    with _event_log_lock:
        path = Path(_event_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Read existing events
        events = []
        if path.exists():
            try:
                with open(path, "r") as f:
                    events = [line.strip() for line in f if line.strip()]
            except Exception:
                events = []

        # Append new event
        events.append(json.dumps(event))

        # Truncate to max
        if len(events) > _event_log_max:
            events = events[-_event_log_max:]

        # Write back
        with open(path, "w") as f:
            f.write("\n".join(events) + "\n")


def emit_event(event_type: str, data: Dict[str, Any], utc_now: Callable[[], str]) -> None:
    """Emit an event to all SSE subscribers.

    Thread-safe. Drops events if a subscriber queue is full (slow client).
    Also writes to JSONL log file if AUTOMEM_EVENT_LOG is configured.

    Args:
        event_type: Event type (e.g., "memory.store", "consolidation.run")
        data: Event payload data
        utc_now: Function that returns current UTC timestamp as ISO string
    """
    event = {
        "type": event_type,
        "timestamp": utc_now(),
        "data": data,
    }

    # Write to log file if enabled
    _write_event_to_log(event)

    event_str = f"data: {json.dumps(event)}\n\n"

    with _subscribers_lock:
        for sub_queue in _subscribers:
            try:
                sub_queue.put_nowait(event_str)
            except Full:
                pass  # Drop if subscriber is slow


def get_subscriber_count() -> int:
    """Return the number of active SSE subscribers."""
    with _subscribers_lock:
        return len(_subscribers)


def get_event_history(limit: int = 100) -> List[Dict[str, Any]]:
    """Return recent events from the log file.

    Args:
        limit: Maximum number of events to return

    Returns:
        List of event dictionaries, oldest first
    """
    if not _event_log_path:
        return []

    path = Path(_event_log_path)
    if not path.exists():
        return []

    with _event_log_lock:
        try:
            with open(path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            # Return last N events
            return [json.loads(line) for line in lines[-limit:]]
        except Exception:
            return []


def get_log_status() -> Dict[str, Any]:
    """Return event log status for display.

    Returns:
        Dict with enabled, path, size_bytes, event_count, max_events
    """
    enabled = bool(_event_log_path)
    size = 0
    count = 0

    if enabled:
        path = Path(_event_log_path)
        if path.exists():
            try:
                size = path.stat().st_size
                with open(path, "r") as f:
                    count = sum(1 for line in f if line.strip())
            except Exception:
                pass

    return {
        "enabled": enabled,
        "path": _event_log_path or None,
        "size_bytes": size,
        "event_count": count,
        "max_events": _event_log_max,
    }


def create_stream_blueprint(
    require_api_token: Callable[[], None],
) -> Blueprint:
    """Create the /stream SSE endpoint blueprint.

    Args:
        require_api_token: Function to validate API token (raises on failure)

    Returns:
        Flask Blueprint with /stream endpoint
    """
    bp = Blueprint("stream", __name__)

    def generate_events() -> Generator[str, None, None]:
        """SSE generator for a single client connection."""
        client_queue: Queue = Queue(maxsize=100)

        with _subscribers_lock:
            _subscribers.append(client_queue)

        try:
            while True:
                try:
                    event_str = client_queue.get(timeout=30)
                    yield event_str
                except Empty:
                    # Send keepalive comment to prevent proxy timeout
                    yield ": keepalive\n\n"
        finally:
            with _subscribers_lock:
                if client_queue in _subscribers:
                    _subscribers.remove(client_queue)

    @bp.route("/stream", methods=["GET"])
    def stream() -> Response:
        """SSE endpoint for real-time event streaming.

        Requires API token authentication. Streams events until client
        disconnects. Sends keepalive comments every 30 seconds.
        """
        require_api_token()
        return Response(
            generate_events(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
                "Connection": "keep-alive",
            },
        )

    @bp.route("/stream/status", methods=["GET"])
    def stream_status() -> Any:
        """Return SSE stream status (subscriber count)."""
        from flask import jsonify

        return jsonify(
            {
                "subscribers": get_subscriber_count(),
            }
        )

    @bp.route("/stream/history", methods=["GET"])
    def stream_history() -> Any:
        """Return cached events from log file for monitor hydration."""
        from flask import jsonify, request

        require_api_token()
        limit = request.args.get("limit", 100, type=int)
        events = get_event_history(min(limit, 500))
        return jsonify({"events": events, "count": len(events)})

    @bp.route("/stream/log-status", methods=["GET"])
    def stream_log_status() -> Any:
        """Return event log status (enabled, path, size, count)."""
        from flask import jsonify

        require_api_token()
        return jsonify(get_log_status())

    return bp
