"""Server-Sent Events endpoint for real-time observability.

Provides a /stream endpoint that emits events for memory operations,
enrichment, and consolidation tasks. Uses an in-memory subscriber
pattern with bounded queues per client.
"""

from __future__ import annotations

import json
from queue import Empty, Full, Queue
from threading import Lock
from typing import Any, Callable, Dict, Generator, List

from flask import Blueprint, Response

# Subscriber management - thread-safe list of client queues
_subscribers: List[Queue] = []
_subscribers_lock = Lock()


def emit_event(event_type: str, data: Dict[str, Any], utc_now: Callable[[], str]) -> None:
    """Emit an event to all SSE subscribers.

    Thread-safe. Drops events if a subscriber queue is full (slow client).

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

    return bp
