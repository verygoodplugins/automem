from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable

from flask import Blueprint, Response, abort, request

from automem.backup import (
    InvalidBackupInclude,
    backup_timestamp,
    parse_backup_include,
    stream_backup_tar_gz,
)


def _admin_key_fingerprint() -> str | None:
    token = (
        request.headers.get("X-Admin-Token")
        or request.headers.get("X-Admin-Api-Key")
        or request.args.get("admin_token")
    )
    if not token:
        return None
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]


def create_backup_blueprint(
    require_admin_token: Callable[[], None],
    get_memory_graph: Callable[[], Any],
    get_qdrant_client: Callable[[], Any],
    graph_name: str,
    collection_name: str,
    logger: Any,
) -> Blueprint:
    bp = Blueprint("backup", __name__)

    @bp.route("/backup", methods=["GET"], strict_slashes=False)
    def backup() -> Response:
        require_admin_token()

        try:
            includes = parse_backup_include(request.args.get("include"))
        except InvalidBackupInclude as exc:
            abort(400, description=str(exc))

        graph = get_memory_graph() if "falkordb" in includes else None
        if "falkordb" in includes and graph is None:
            abort(503, description="FalkorDB is unavailable")

        qdrant_client = get_qdrant_client() if "qdrant" in includes else None
        if "qdrant" in includes and qdrant_client is None:
            abort(503, description="Qdrant is unavailable")

        started = time.perf_counter()
        timestamp = backup_timestamp()
        audit_base = {
            "event": "backup.request",
            "key_fingerprint": _admin_key_fingerprint(),
            "include": list(includes),
            "timestamp": timestamp,
        }

        def audit_complete(stats: dict[str, Any]) -> None:
            artifacts = stats.get("artifacts") or {}
            falkordb_stats = artifacts.get("falkordb") or {}
            qdrant_stats = artifacts.get("qdrant") or {}
            audit = {
                **audit_base,
                "status": stats.get("status"),
                "byte_count": stats.get("bytes", 0),
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "node_count": falkordb_stats.get("node_count"),
                "relationship_count": falkordb_stats.get("relationship_count"),
                "point_count": qdrant_stats.get("points_count"),
            }
            if stats.get("error"):
                audit["error"] = stats["error"]
            logger.info("backup.request %s", json.dumps(audit, sort_keys=True))

        stream = stream_backup_tar_gz(
            includes=includes,
            timestamp=timestamp,
            graph=graph,
            graph_name=graph_name,
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            logger=logger,
            on_complete=audit_complete,
        )

        return Response(
            stream,
            mimetype="application/gzip",
            headers={
                "Content-Disposition": f'attachment; filename="automem-backup-{timestamp}.tar.gz"',
                "Cache-Control": "no-store",
            },
        )

    return bp
