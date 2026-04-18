"""Flask blueprint for agent-driven document storage.

Design (Stage 1 — "lean"):
- The file stays in the bucket as an opaque original.
- AutoMem never parses it — no pdfplumber, no trafilatura, no OCR.
- The uploading agent is expected to have read the file already and provided
  a human-quality ``title`` + ``summary``; those become the Memory's content
  so existing vector/keyword recall surfaces the doc.
- When an agent later wants the raw bytes, it calls ``GET
  /documents/:id/download`` to get a short-lived presigned URL and fetches
  the file itself (Claude can read PDFs directly from such URLs).

The "gate" requested by the user is the hard 422 on missing title/summary at
the HTTP boundary — the MCP tool definitions enforce it client-side too via
required parameters.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from flask import Blueprint, abort, current_app, jsonify, request
from flask.typing import ResponseReturnValue

logger = logging.getLogger(__name__)


# Filename sanitization: keep alphanumerics, dot, dash, underscore. Everything
# else becomes "_" so we never build S3 keys with control characters or path
# traversal surprises. The memory_id prefix keeps keys globally unique.
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename(name: str, *, fallback: str = "file") -> str:
    cleaned = _SAFE_FILENAME_RE.sub("_", name or "").strip("._-")
    return cleaned or fallback


def _parse_json_field(value: Optional[str], field_name: str) -> Any:
    """Parse an optional JSON-encoded form field. Aborts 400 if malformed."""
    if value is None or value == "":
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError) as exc:
        abort(400, description=f"'{field_name}' must be valid JSON ({exc})")


def create_documents_blueprint(
    *,
    bucket_store: Any,
    get_memory_graph_fn: Callable[[], Any],
    get_qdrant_client_fn: Callable[[], Any],
    normalize_tags_fn: Callable[[Any], List[str]],
    compute_tag_prefixes_fn: Callable[[List[str]], List[str]],
    coerce_importance_fn: Callable[[Any], float],
    enqueue_enrichment_fn: Callable[[str], None],
    enqueue_embedding_fn: Callable[[str, str], None],
    collection_name: str,
    utc_now_fn: Callable[[], str],
    state: Any,
    qdrant_models_obj: Any,
    max_bytes: int,
    presigned_expires: int,
) -> Blueprint:
    """Create the ``/documents`` blueprint. ``bucket_store`` may be None;
    when None, all endpoints return 503 with a clear message."""

    bp = Blueprint("documents", __name__)

    def _require_bucket() -> Any:
        if bucket_store is None:
            abort(
                503,
                description=(
                    "Document storage is not configured. Set S3_ENDPOINT, "
                    "S3_BUCKET, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY "
                    "(Railway Buckets provide all four)."
                ),
            )
        return bucket_store

    def _validate_memory_id(memory_id: str) -> None:
        try:
            uuid.UUID(memory_id)
        except (ValueError, TypeError):
            abort(400, description="memory_id must be a valid UUID")

    def _find_document(memory_id: str) -> Dict[str, Any]:
        """Return the Memory node + parsed metadata, or 404."""
        graph = get_memory_graph_fn()
        if graph is None:
            abort(503, description="FalkorDB is unavailable")

        result = graph.query("MATCH (m:Memory {id: $id}) RETURN m", {"id": memory_id})
        if not getattr(result, "result_set", None):
            abort(404, description="Memory not found")

        node = result.result_set[0][0]
        props = dict(getattr(node, "properties", {}) or {})
        mem_type = props.get("type")
        if mem_type != "Document":
            abort(
                404,
                description=(
                    "Memory exists but is not of type=Document "
                    f"(got {mem_type!r}); use /memory/:id for other types."
                ),
            )
        raw_meta = props.get("metadata")
        if isinstance(raw_meta, str):
            try:
                meta = json.loads(raw_meta) if raw_meta else {}
            except (TypeError, ValueError):
                meta = {}
        elif isinstance(raw_meta, dict):
            meta = raw_meta
        else:
            meta = {}
        props["metadata"] = meta
        return props

    # ------------------------------------------------------------------ POST
    @bp.route("/documents", methods=["POST"])
    def upload_document() -> ResponseReturnValue:  # noqa: WPS430 - Flask view
        """Multipart upload: stores file in bucket + creates Document Memory.

        Required form fields:
            file: the binary file payload
            title: agent-generated short title
            summary: agent-generated 1–3 sentence summary

        Optional:
            tags: JSON array of strings  OR  comma-separated string
            importance: float 0-1 (default 0.5)
            metadata: JSON object merged into Memory metadata
            memory_id: UUID to use (otherwise server-generated)
        """
        store = _require_bucket()
        query_start = time.perf_counter()

        # --- multipart validation --------------------------------------------
        uploaded = request.files.get("file")
        if uploaded is None or not uploaded.filename:
            abort(
                400,
                description=(
                    "Missing 'file' in multipart form data. Upload with "
                    "'Content-Type: multipart/form-data' and a 'file' field."
                ),
            )

        title = (request.form.get("title") or "").strip()
        summary = (request.form.get("summary") or "").strip()
        if not title or not summary:
            # THE GATE: force agents to read the file and describe it before
            # upload. We never extract text server-side, so the agent's
            # description IS the searchable content.
            abort(
                422,
                description=(
                    "Both 'title' and 'summary' are required. The agent must "
                    "read the file and generate an accurate title (< 200 chars)"
                    " and a 1-3 sentence summary BEFORE calling this endpoint."
                    " AutoMem does not parse file content; the agent's summary"
                    " is the indexed, searchable representation."
                ),
            )
        if len(title) > 300:
            abort(400, description="'title' must be 300 characters or fewer")
        # The summary becomes Memory content, which has its own hard limit;
        # but we don't want to accept a 20KB "summary" either.
        if len(summary) > 4000:
            abort(400, description="'summary' must be 4000 characters or fewer")

        # --- memory_id --------------------------------------------------------
        raw_memory_id = (request.form.get("memory_id") or "").strip()
        memory_id = raw_memory_id or str(uuid.uuid4())
        try:
            uuid.UUID(memory_id)
        except (ValueError, TypeError):
            abort(400, description="'memory_id' must be a valid UUID")

        # --- tags / importance / metadata ------------------------------------
        tags_raw: Any = request.form.get("tags")
        parsed_tags = _parse_json_field(tags_raw, "tags")
        if parsed_tags is None and tags_raw:
            # Accept comma-separated shorthand as well
            parsed_tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
        tags = normalize_tags_fn(parsed_tags)
        # Always add the "document" tag so list/filter queries are trivial.
        if "document" not in {t.lower() for t in tags}:
            tags.append("document")
        tags_lower = [t.strip().lower() for t in tags if isinstance(t, str) and t.strip()]
        tag_prefixes = compute_tag_prefixes_fn(tags_lower)

        importance = coerce_importance_fn(request.form.get("importance"))

        user_metadata = _parse_json_field(request.form.get("metadata"), "metadata")
        if user_metadata is None:
            user_metadata = {}
        elif not isinstance(user_metadata, dict):
            abort(400, description="'metadata' must be a JSON object")

        # --- size guard (cheap pre-check via Content-Length) ------------------
        content_length = request.content_length or 0
        if content_length and content_length > max_bytes:
            abort(
                413,
                description=(
                    f"Upload exceeds DOCUMENT_MAX_BYTES={max_bytes}; got "
                    f"{content_length} bytes."
                ),
            )

        # --- bucket upload ----------------------------------------------------
        filename = _safe_filename(uploaded.filename)
        mime = uploaded.mimetype or "application/octet-stream"
        bucket_key = f"documents/{memory_id}/{filename}"

        try:
            upload_info = store.upload(
                bucket_key,
                uploaded.stream,
                mime=mime,
                metadata={
                    "memory_id": memory_id,
                    "original_filename": uploaded.filename[:200],
                },
            )
        except Exception:
            logger.exception(
                "Bucket upload failed for memory_id=%s key=%s",
                memory_id,
                bucket_key,
            )
            abort(502, description="Upload to object store failed")

        # Size-after-the-fact check (Content-Length may have been absent on
        # chunked uploads). If over the cap, roll the object back.
        if upload_info["size"] > max_bytes:
            try:
                store.delete(bucket_key)
            except Exception:
                logger.exception("Failed to clean up oversized upload at %s", bucket_key)
            abort(
                413,
                description=(
                    f"Uploaded file {upload_info['size']} bytes exceeds "
                    f"DOCUMENT_MAX_BYTES={max_bytes}."
                ),
            )

        # --- assemble Memory node --------------------------------------------
        content = f"{title}\n\n{summary}".strip()
        created_at = utc_now_fn()

        storage_metadata: Dict[str, Any] = {
            "document": {
                "title": title,
                "filename": uploaded.filename,
                "safe_filename": filename,
                "mime": upload_info["content_type"],
                "size": upload_info["size"],
                "sha256": upload_info["sha256"],
                "etag": upload_info["etag"],
                "bucket_key": bucket_key,
                "source": "upload",
                "uploaded_at": created_at,
            }
        }
        # User metadata wins over storage_metadata only if they collide
        # outside the "document" key. We keep storage_metadata.document as
        # authoritative to avoid clients corrupting our bookkeeping.
        merged_metadata: Dict[str, Any] = {**user_metadata, **storage_metadata}
        metadata_json = json.dumps(merged_metadata, default=str)

        # --- persist to FalkorDB ---------------------------------------------
        graph = get_memory_graph_fn()
        if graph is None:
            # We already uploaded to the bucket; roll back.
            try:
                store.delete(bucket_key)
            except Exception:
                logger.exception("Cleanup failed after FalkorDB unavailable")
            abort(503, description="FalkorDB is unavailable")

        try:
            graph.query(
                """
                MERGE (m:Memory {id: $id})
                SET m.content = $content,
                    m.timestamp = $timestamp,
                    m.importance = $importance,
                    m.tags = $tags,
                    m.tag_prefixes = $tag_prefixes,
                    m.type = $type,
                    m.confidence = $confidence,
                    m.updated_at = $updated_at,
                    m.last_accessed = $last_accessed,
                    m.metadata = $metadata,
                    m.processed = false
                RETURN m
                """,
                {
                    "id": memory_id,
                    "content": content,
                    "timestamp": created_at,
                    "importance": importance,
                    "tags": tags,
                    "tag_prefixes": tag_prefixes,
                    "type": "Document",
                    "confidence": 1.0,  # Agent-provided classification
                    "updated_at": created_at,
                    "last_accessed": created_at,
                    "metadata": metadata_json,
                },
            )
        except Exception:
            logger.exception(
                "Failed to persist Document memory in FalkorDB; rolling back "
                "bucket upload for memory_id=%s",
                memory_id,
            )
            try:
                store.delete(bucket_key)
            except Exception:
                logger.exception("Bucket rollback failed for %s", bucket_key)
            abort(500, description="Failed to store document memory")

        # --- queue embedding + enrichment ------------------------------------
        qdrant_client = get_qdrant_client_fn()
        if qdrant_client is not None:
            enqueue_embedding_fn(memory_id, content)
            embedding_status = "queued"
        else:
            embedding_status = "unconfigured"

        try:
            enqueue_enrichment_fn(memory_id)
            enrichment_status = "queued" if state.enrichment_queue else "disabled"
        except Exception:
            logger.exception("Failed to enqueue enrichment for %s", memory_id)
            enrichment_status = "failed"

        # --- presigned URL for immediate agent use ---------------------------
        try:
            download_url = store.presigned_url(bucket_key, expires_in=presigned_expires)
        except Exception:
            logger.exception("Failed to generate presigned URL for %s", bucket_key)
            download_url = None

        response = {
            "status": "success",
            "memory_id": memory_id,
            "type": "Document",
            "title": title,
            "summary": summary,
            "tags": tags,
            "importance": importance,
            "document": merged_metadata["document"],
            "download_url": download_url,
            "download_url_expires_in": presigned_expires if download_url else None,
            "embedding_status": embedding_status,
            "enrichment": enrichment_status,
            "stored_at": created_at,
            "query_time_ms": round((time.perf_counter() - query_start) * 1000, 2),
        }

        logger.info(
            "document_uploaded",
            extra={
                "memory_id": memory_id,
                "bucket_key": bucket_key,
                "size": upload_info["size"],
                "mime": upload_info["content_type"],
                "latency_ms": response["query_time_ms"],
            },
        )
        return jsonify(response), 201

    # --------------------------------------------------------- GET download
    @bp.route("/documents/<memory_id>/download", methods=["GET"])
    def document_download(memory_id: str) -> ResponseReturnValue:
        store = _require_bucket()
        _validate_memory_id(memory_id)
        doc = _find_document(memory_id)

        doc_meta = doc.get("metadata", {}).get("document", {}) or {}
        bucket_key = doc_meta.get("bucket_key")
        if not bucket_key:
            abort(
                500,
                description=(
                    "Document memory is missing document.bucket_key in "
                    "metadata; cannot generate a download URL."
                ),
            )

        try:
            expires_in = int(request.args.get("expires_in", presigned_expires))
        except (TypeError, ValueError):
            abort(400, description="'expires_in' must be an integer")
        expires_in = max(30, min(expires_in, 3600))  # 30s .. 1h

        disposition = request.args.get("disposition")
        filename = doc_meta.get("filename") or doc_meta.get("safe_filename")
        rcd = None
        if disposition == "attachment" and filename:
            # Quote filename per RFC 6266; boto3 will pass this through verbatim.
            rcd = f'attachment; filename="{_safe_filename(filename)}"'

        try:
            url = store.presigned_url(
                bucket_key,
                expires_in=expires_in,
                response_content_disposition=rcd,
            )
        except Exception:
            logger.exception("Presign failed for %s", bucket_key)
            abort(502, description="Failed to generate presigned URL")

        return jsonify(
            {
                "status": "success",
                "memory_id": memory_id,
                "download_url": url,
                "expires_in": expires_in,
                "filename": filename,
                "mime": doc_meta.get("mime"),
                "size": doc_meta.get("size"),
            }
        )

    # -------------------------------------------------------------- GET list
    @bp.route("/documents", methods=["GET"])
    def list_documents() -> ResponseReturnValue:
        """Paged list of Document memories, filter by tag (any/all)."""
        _require_bucket()  # returns 503 with a clear message if unconfigured
        graph = get_memory_graph_fn()
        if graph is None:
            abort(503, description="FalkorDB is unavailable")

        raw_tags = request.args.getlist("tags") or request.args.get("tags")
        tag_filter: List[str] = []
        if isinstance(raw_tags, list):
            tag_filter = [t.strip().lower() for t in raw_tags if t and t.strip()]
        elif isinstance(raw_tags, str):
            tag_filter = [t.strip().lower() for t in raw_tags.split(",") if t.strip()]

        try:
            limit = int(request.args.get("limit", 25))
        except (TypeError, ValueError):
            limit = 25
        limit = max(1, min(limit, 200))

        if tag_filter:
            query = """
                MATCH (m:Memory {type: 'Document'})
                WHERE ANY(tag IN coalesce(m.tags, []) WHERE toLower(tag) IN $tags)
                RETURN m
                ORDER BY m.importance DESC, m.timestamp DESC
                LIMIT $limit
            """
            params: Dict[str, Any] = {"tags": tag_filter, "limit": limit}
        else:
            query = """
                MATCH (m:Memory {type: 'Document'})
                RETURN m
                ORDER BY m.importance DESC, m.timestamp DESC
                LIMIT $limit
            """
            params = {"limit": limit}

        try:
            result = graph.query(query, params)
        except Exception:
            logger.exception("Document list query failed")
            abort(500, description="Failed to list documents")

        docs: List[Dict[str, Any]] = []
        for row in getattr(result, "result_set", []) or []:
            node = row[0]
            props = dict(getattr(node, "properties", {}) or {})
            raw_meta = props.get("metadata")
            if isinstance(raw_meta, str):
                try:
                    meta = json.loads(raw_meta) if raw_meta else {}
                except (TypeError, ValueError):
                    meta = {}
            elif isinstance(raw_meta, dict):
                meta = raw_meta
            else:
                meta = {}
            doc_meta = meta.get("document") if isinstance(meta, dict) else None
            docs.append(
                {
                    "memory_id": props.get("id"),
                    "title": (doc_meta or {}).get("title"),
                    "content": props.get("content"),
                    "tags": props.get("tags") or [],
                    "importance": props.get("importance"),
                    "timestamp": props.get("timestamp"),
                    "updated_at": props.get("updated_at"),
                    "document": doc_meta,
                }
            )

        return jsonify(
            {
                "status": "success",
                "tags": tag_filter,
                "count": len(docs),
                "documents": docs,
            }
        )

    # ---------------------------------------------------------------- DELETE
    @bp.route("/documents/<memory_id>", methods=["DELETE"])
    def delete_document(memory_id: str) -> ResponseReturnValue:
        store = _require_bucket()
        _validate_memory_id(memory_id)
        doc = _find_document(memory_id)

        bucket_key = doc.get("metadata", {}).get("document", {}).get("bucket_key")

        graph = get_memory_graph_fn()
        if graph is None:
            abort(503, description="FalkorDB is unavailable")

        # Drop the graph node first (so subsequent recall can't hand out stale
        # presigned URLs); then the bucket object.
        try:
            graph.query("MATCH (m:Memory {id: $id}) DETACH DELETE m", {"id": memory_id})
        except Exception:
            logger.exception("Graph delete failed for %s", memory_id)
            abort(500, description="Failed to delete document memory")

        # Qdrant vector cleanup
        qdrant_client = get_qdrant_client_fn()
        if qdrant_client is not None:
            try:
                if qdrant_models_obj is not None:
                    selector = qdrant_models_obj.PointIdsList(points=[memory_id])
                else:
                    selector = {"points": [memory_id]}
                qdrant_client.delete(collection_name=collection_name, points_selector=selector)
            except Exception:
                logger.exception("Qdrant vector delete failed for %s", memory_id)

        bucket_result = "skipped"
        if bucket_key:
            try:
                store.delete(bucket_key)
                bucket_result = "deleted"
            except Exception:
                logger.exception("Bucket delete failed for %s", bucket_key)
                bucket_result = "failed"

        return jsonify(
            {
                "status": "success",
                "memory_id": memory_id,
                "graph": "deleted",
                "bucket": bucket_result,
            }
        )

    return bp
