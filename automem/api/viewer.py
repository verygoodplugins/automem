"""Viewer compatibility blueprint.

The visualizer is now a standalone service. AutoMem keeps `/viewer/*`
as a compatibility entrypoint and forwards users to `GRAPH_VIEWER_URL`
while preserving URL hash tokens and passing `server=<automem-origin>`.
"""

from __future__ import annotations

import os
from typing import Any

from flask import Blueprint, Response, redirect, request


def create_viewer_blueprint() -> Blueprint:
    """Create the viewer compatibility blueprint for /viewer routes."""

    bp = Blueprint(
        "viewer",
        __name__,
        url_prefix="/viewer",
    )

    def _viewer_url() -> str:
        return (os.environ.get("GRAPH_VIEWER_URL") or "").strip().rstrip("/")

    def _viewer_unavailable_response() -> Response:
        return Response(
            "Graph Viewer URL is not configured. Set GRAPH_VIEWER_URL.",
            status=503,
            mimetype="text/plain",
        )

    def _is_asset_path(path: str) -> bool:
        if not path:
            return False
        if path.startswith("assets/"):
            return True
        file_ext = os.path.splitext(path)[1].lower()
        return file_ext in {
            ".js",
            ".css",
            ".svg",
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".ico",
            ".json",
            ".txt",
            ".map",
            ".woff",
            ".woff2",
            ".ttf",
        }

    def _build_redirect_url(base_url: str, path: str) -> str:
        trimmed_path = path.lstrip("/")
        target = f"{base_url}/{trimmed_path}" if trimmed_path else base_url
        query = request.query_string.decode("utf-8", errors="ignore")
        return f"{target}?{query}" if query else target

    def _build_bootstrap_html(base_url: str, path: str) -> str:
        normalized_base = f"{base_url}/"
        normalized_path = path.lstrip("/")
        # Keep template tiny and explicit; hash token remains client-side.
        return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>Redirecting to Graph Viewer...</title>
  </head>
  <body>
    <p>Redirecting to Graph Viewer...</p>
    <script>
      (function () {{
        const base = "{normalized_base}";
        const relPath = "{normalized_path}";
        const target = new URL(relPath, base);
        const incoming = new URLSearchParams(window.location.search);
        incoming.forEach((value, key) => target.searchParams.set(key, value));
        if (!target.searchParams.has("server")) {{
          target.searchParams.set("server", window.location.origin);
        }}
        window.location.replace(target.toString() + window.location.hash);
      }})();
    </script>
  </body>
</html>
"""

    @bp.route("/", defaults={"path": ""})
    @bp.route("/<path:path>")
    def serve_viewer(path: str) -> Any:
        """Redirect or bootstrap to standalone graph viewer service."""
        viewer_url = _viewer_url()
        if not viewer_url:
            return _viewer_unavailable_response()

        if _is_asset_path(path):
            return redirect(_build_redirect_url(viewer_url, path), code=302)

        html = _build_bootstrap_html(viewer_url, path)
        return Response(html, status=200, mimetype="text/html")

    return bp


def is_viewer_enabled() -> bool:
    """Check if the graph viewer compatibility route is enabled."""
    return os.environ.get("ENABLE_GRAPH_VIEWER", "true").lower() in ("true", "1", "yes")
