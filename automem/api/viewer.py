"""Viewer blueprint - serves the Graph Viewer SPA.

The viewer is an optional feature that can be enabled by setting
ENABLE_GRAPH_VIEWER=true. When enabled, it serves the pre-built
React application at /viewer/.

Since the viewer runs on the same origin as the API, it doesn't need
CORS and can access the API directly. Authentication is handled via
a token passed in the URL hash (client-side only, never sent to server).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from flask import Blueprint, Response, send_from_directory


def create_viewer_blueprint() -> Blueprint:
    """Create the viewer blueprint for serving the Graph Viewer SPA."""

    # Find the static files directory
    static_dir = Path(__file__).parent.parent / "static" / "viewer"

    bp = Blueprint(
        "viewer",
        __name__,
        url_prefix="/viewer",
        static_folder=str(static_dir),
        static_url_path="/static",
    )

    @bp.route("/")
    @bp.route("/<path:path>")
    def serve_viewer(path: str = "index.html") -> Any:
        """Serve the viewer SPA.

        For any path under /viewer/, serve the corresponding file from
        the static directory. If the file doesn't exist, serve index.html
        to support client-side routing.
        """
        if not static_dir.exists():
            return Response(
                "Graph Viewer not installed. Run 'npm run build' in packages/graph-viewer/",
                status=404,
                mimetype="text/plain",
            )

        # Check if the requested file exists
        file_path = static_dir / path
        if file_path.is_file():
            return send_from_directory(static_dir, path)

        # For SPA routing, always serve index.html for unknown paths
        return send_from_directory(static_dir, "index.html")

    @bp.route("/assets/<path:filename>")
    def serve_assets(filename: str) -> Any:
        """Serve static assets (JS, CSS, images)."""
        assets_dir = static_dir / "assets"
        return send_from_directory(assets_dir, filename)

    return bp


def is_viewer_enabled() -> bool:
    """Check if the graph viewer is enabled."""
    return os.environ.get("ENABLE_GRAPH_VIEWER", "true").lower() in ("true", "1", "yes")
