from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Set
from uuid import uuid4

from flask import Blueprint, abort, jsonify, request, send_file


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


def _get_all_qdrant_ids(qdrant_client: Any, collection_name: str) -> Set[str]:
    """Fetch all point IDs from Qdrant collection."""
    all_ids: Set[str] = set()
    offset = None

    while True:
        result = qdrant_client.scroll(
            collection_name=collection_name,
            limit=1000,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        points, next_offset = result
        for point in points:
            all_ids.add(str(point.id))

        if next_offset is None:
            break
        offset = next_offset

    return all_ids


def _export_dir() -> Path:
    directory = Path(os.getenv("AUTOMEM_EXPORT_DIR", "./exports")).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _export_bundle_path(export_id: str) -> Path:
    return _export_dir() / f"{export_id}.json.gz"


def _export_manifest_path(export_id: str) -> Path:
    return _export_dir() / f"{export_id}.manifest.json"


def _load_export_manifest(export_id: str) -> Dict[str, Any]:
    path = _export_manifest_path(export_id)
    if not path.exists():
        abort(404, description="Export not found")
    return json.loads(path.read_text(encoding="utf-8"))


def _export_graph_snapshot(graph: Any, graph_name: str) -> Dict[str, Any]:
    nodes_result = graph.query(
        """
        MATCH (n)
        RETURN labels(n) AS labels,
               properties(n) AS props
        """
    )
    relationships_result = graph.query(
        """
        MATCH (a)-[r]->(b)
        RETURN properties(a).id AS source_id,
               type(r) AS rel_type,
               properties(b).id AS target_id,
               properties(r) AS props
        """
    )

    nodes = []
    for row in getattr(nodes_result, "result_set", []) or []:
        labels = row[0] if len(row) > 0 else []
        props = row[1] if len(row) > 1 else {}
        nodes.append({"labels": labels or [], "properties": props or {}})

    relationships = []
    for row in getattr(relationships_result, "result_set", []) or []:
        relationships.append(
            {
                "source_id": row[0] if len(row) > 0 else None,
                "type": row[1] if len(row) > 1 else None,
                "target_id": row[2] if len(row) > 2 else None,
                "properties": row[3] if len(row) > 3 else {},
            }
        )

    return {
        "graph_name": graph_name,
        "nodes": nodes,
        "relationships": relationships,
        "stats": {
            "node_count": len(nodes),
            "relationship_count": len(relationships),
        },
    }


def _export_qdrant_snapshot(qdrant_client: Any, collection_name: str) -> Dict[str, Any] | None:
    if qdrant_client is None:
        return None

    points = []
    offset = None
    while True:
        batch, next_offset = qdrant_client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for point in batch:
            points.append(
                {
                    "id": point.id,
                    "vector": getattr(point, "vector", None),
                    "payload": getattr(point, "payload", None),
                }
            )
        if next_offset is None:
            break
        offset = next_offset

    return {
        "collection_name": collection_name,
        "points": points,
        "stats": {
            "points_count": len(points),
        },
    }


def create_admin_blueprint_full(
    require_admin_token: Callable[[], None],
    init_openai: Callable[[], None],
    get_openai_client: Callable[[], Any],
    get_qdrant_client: Callable[[], Any],
    get_memory_graph: Callable[[], Any],
    point_struct: Any,
    collection_name: str,
    get_vector_size: Callable[[], int],
    embedding_model: str,
    utc_now: Callable[[], str],
    logger: Any,
    graph_name: str,
    get_service_profile: Callable[[], Dict[str, Any]],
    get_service_mode: Callable[[], str],
    get_service_tier: Callable[[], str],
) -> Blueprint:
    bp = Blueprint("admin", __name__)

    @bp.route("/admin/reembed", methods=["POST"])
    def reembed() -> Any:
        require_admin_token()

        # Ensure OpenAI and Qdrant are available
        openai_client = get_openai_client()
        if openai_client is None:
            abort(
                503, description="OpenAI API key not configured - cannot generate real embeddings"
            )

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
                to_process.append(
                    {
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
                    }
                )

        if not to_process:
            return jsonify(
                {
                    "status": "complete",
                    "message": "No memories found to reembed",
                    "processed": 0,
                    "total": 0,
                }
            )

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
                    dimensions=get_vector_size(),
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
                    points.append(
                        point_struct(id=mem["id"], vector=embedding, payload=payload_data)
                    )

                if points:
                    qdrant_client.upsert(collection_name=collection_name, points=points)
                    processed += len(points)
                    logger.info(
                        f"Successfully reembedded batch of {len(points)} memories (preserving metadata)"
                    )

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

    @bp.route("/admin/sync", methods=["POST"])
    def sync_missing() -> Any:
        """Find memories in FalkorDB missing from Qdrant and embed them.

        This is a non-destructive operation that only adds missing embeddings.
        Existing Qdrant vectors are not modified.

        Request body (JSON, optional):
            batch_size: int (default: 32, max: 100)
            dry_run: bool (default: False) - if True, only report missing count

        Returns:
            JSON with sync results including counts and timing.
        """
        require_admin_token()

        qdrant_client = get_qdrant_client()
        if qdrant_client is None:
            abort(503, description="Qdrant is not available")

        graph = get_memory_graph()
        if graph is None:
            abort(503, description="FalkorDB is unavailable")

        payload = request.get_json(silent=True) or {}
        try:
            batch_size = min(int(payload.get("batch_size", 32)), 100)
        except (TypeError, ValueError):
            batch_size = 32
        dry_run = bool(payload.get("dry_run", False))

        # Get all memory IDs from FalkorDB
        falkor_query = "MATCH (m:Memory) RETURN m.id AS id"
        falkor_result = graph.query(falkor_query)
        falkor_ids: Set[str] = set()
        for row in getattr(falkor_result, "result_set", []) or []:
            if row[0]:
                falkor_ids.add(str(row[0]))

        # Get all point IDs from Qdrant
        qdrant_ids = _get_all_qdrant_ids(qdrant_client, collection_name)

        # Find missing (in FalkorDB but not in Qdrant)
        missing_ids = falkor_ids - qdrant_ids
        orphaned_ids = qdrant_ids - falkor_ids  # In Qdrant but not FalkorDB

        if dry_run:
            return jsonify(
                {
                    "status": "dry_run",
                    "falkordb_count": len(falkor_ids),
                    "qdrant_count": len(qdrant_ids),
                    "missing_count": len(missing_ids),
                    "orphaned_count": len(orphaned_ids),
                    "missing_sample": list(missing_ids)[:10],
                    "orphaned_sample": list(orphaned_ids)[:10],
                }
            )

        if not missing_ids:
            return jsonify(
                {
                    "status": "already_synced",
                    "falkordb_count": len(falkor_ids),
                    "qdrant_count": len(qdrant_ids),
                    "synced": 0,
                    "orphaned_count": len(orphaned_ids),
                }
            )

        # Ensure OpenAI is available for embedding
        openai_client = get_openai_client()
        if openai_client is None:
            abort(503, description="OpenAI API key not configured - cannot generate embeddings")

        # Fetch full memory data for missing IDs
        # Process in batches to avoid huge IN clauses
        missing_list = list(missing_ids)
        to_process: List[Dict[str, Any]] = []

        for i in range(0, len(missing_list), 500):
            batch_ids = missing_list[i : i + 500]
            fetch_query = """
                MATCH (m:Memory)
                WHERE m.id IN $ids
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
            """
            result = graph.query(fetch_query, {"ids": batch_ids})
            for row in getattr(result, "result_set", []) or []:
                memory_id = row[0]
                content = row[1]
                if content:
                    to_process.append(
                        {
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
                        }
                    )

        # Generate embeddings and upsert to Qdrant
        synced = 0
        failed = 0
        failed_ids_list: List[str] = []

        for i in range(0, len(to_process), batch_size):
            batch = to_process[i : i + batch_size]
            texts = [mem["content"] for mem in batch]

            try:
                resp = openai_client.embeddings.create(
                    input=texts,
                    model=embedding_model,
                    dimensions=get_vector_size(),
                )

                points = []
                for mem, data in zip(batch, resp.data):
                    embedding = data.embedding
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
                    points.append(
                        point_struct(id=mem["id"], vector=embedding, payload=payload_data)
                    )

                if points:
                    qdrant_client.upsert(collection_name=collection_name, points=points)
                    synced += len(points)
                    logger.info(f"Synced batch of {len(points)} missing memories to Qdrant")

            except Exception as e:
                logger.error(f"Failed to sync batch starting at index {i}: {e}")
                failed += len(batch)
                failed_ids_list.extend([mem["id"] for mem in batch])

        response = {
            "status": "complete",
            "falkordb_count": len(falkor_ids),
            "qdrant_count": len(qdrant_ids) + synced,
            "synced": synced,
            "failed": failed,
            "orphaned_count": len(orphaned_ids),
        }
        if failed_ids_list:
            response["failed_ids"] = failed_ids_list[:10]
            if len(failed_ids_list) > 10:
                response["failed_ids_truncated"] = True
        return jsonify(response)

    @bp.route("/admin/exports", methods=["POST"])
    def create_export() -> Any:
        require_admin_token()

        payload = request.get_json(silent=True) or {}
        include_vectors = bool(payload.get("include_vectors", True))
        reason = str(payload.get("reason") or "").strip() or None
        export_id = str(payload.get("export_id") or uuid4().hex)

        graph = get_memory_graph()
        if graph is None:
            abort(503, description="FalkorDB is unavailable")

        qdrant_client = get_qdrant_client() if include_vectors else None
        graph_snapshot = _export_graph_snapshot(graph, graph_name)
        qdrant_snapshot = _export_qdrant_snapshot(qdrant_client, collection_name)

        bundle = {
            "export_id": export_id,
            "created_at": utc_now(),
            "reason": reason,
            "service": {
                "tier": get_service_tier(),
                "mode": get_service_mode(),
                "profile": get_service_profile(),
            },
            "graph": graph_snapshot,
            "qdrant": qdrant_snapshot,
        }

        bundle_path = _export_bundle_path(export_id)
        with gzip.open(bundle_path, "wt", encoding="utf-8") as handle:
            json.dump(bundle, handle, indent=2, default=str)

        manifest = {
            "status": "complete",
            "export_id": export_id,
            "created_at": bundle["created_at"],
            "reason": reason,
            "service": bundle["service"],
            "graph": graph_snapshot["stats"],
            "qdrant": qdrant_snapshot["stats"] if qdrant_snapshot else None,
            "include_vectors": qdrant_snapshot is not None,
            "bundle": {
                "filename": bundle_path.name,
                "bytes": bundle_path.stat().st_size,
            },
            "download_url": f"{request.host_url.rstrip('/')}/admin/exports/{export_id}/download",
            "status_url": f"{request.host_url.rstrip('/')}/admin/exports/{export_id}",
        }

        manifest_path = _export_manifest_path(export_id)
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

        logger.info(
            "Created service export",
            extra={
                "export_id": export_id,
                "tier": get_service_tier(),
                "mode": get_service_mode(),
                "include_vectors": bool(qdrant_snapshot is not None),
            },
        )
        return jsonify(manifest), 201

    @bp.route("/admin/exports/<export_id>", methods=["GET"])
    def export_status(export_id: str) -> Any:
        require_admin_token()
        return jsonify(_load_export_manifest(export_id))

    @bp.route("/admin/exports/<export_id>/download", methods=["GET"])
    def download_export(export_id: str) -> Any:
        require_admin_token()

        bundle_path = _export_bundle_path(export_id)
        if not bundle_path.exists():
            abort(404, description="Export bundle not found")

        return send_file(
            bundle_path,
            mimetype="application/gzip",
            as_attachment=True,
            download_name=bundle_path.name,
        )

    return bp
