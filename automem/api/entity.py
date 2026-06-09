"""Entity API endpoints for AutoMem.

Provides CRUD and query operations for first-class Entity nodes.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional

from flask import Blueprint, abort, jsonify, request

from automem.utils.entity_quality import validate_entity_tag

logger = logging.getLogger(__name__)


def _parse_aliases(raw: Any) -> List[str]:
    """Safely parse aliases from a graph result (may be list or JSON string)."""
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def _serialize_entity_row(row: list) -> Dict[str, Any]:
    """Convert a result row into a serialisable entity dict.

    Expected column order (12 columns):
        0: e.id, 1: e.slug, 2: e.category, 3: e.name, 4: e.aliases,
        5: e.identity, 6: e.identity_version, 7: e.identity_updated_at,
        8: e.identity_source_count, 9: ref_count,
        10: e.created_at, 11: e.last_referenced_at
    """
    return {
        "id": row[0],
        "slug": row[1],
        "category": row[2],
        "name": row[3],
        "aliases": _parse_aliases(row[4]),
        "identity": row[5],
        "identity_version": int(row[6] or 0),
        "identity_updated_at": row[7],
        "identity_source_count": int(row[8] or 0),
        "reference_count": int(row[9] or 0),
        "created_at": row[10] if len(row) > 10 else None,
        "last_referenced_at": row[11] if len(row) > 11 else None,
    }


def _audit_memory_entity_tags(
    graph: Any,
    *,
    summary: bool = False,
    limit: int = 100,
    offset: int = 0,
    category: Optional[str] = None,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    result = graph.query("MATCH (m:Memory) WHERE m.tags IS NOT NULL RETURN m.id, m.tags")
    accepted: Dict[str, Dict[str, Any]] = {}
    rejected: List[Dict[str, Any]] = []
    counts_by_reason: Counter[str] = Counter()
    counts_by_category: Dict[str, Counter[str]] = defaultdict(Counter)

    for row in getattr(result, "result_set", []) or []:
        memory_id = row[0]
        tags = row[1]
        if not isinstance(tags, list):
            continue
        for tag in tags:
            if not isinstance(tag, str) or not tag.startswith("entity:"):
                continue
            validation = validate_entity_tag(tag)
            if not validation.accepted:
                counts_by_reason[validation.reason] += 1
                counts_by_category[validation.category]["rejected"] += 1
                rejected.append(
                    {
                        "tag": tag,
                        "memory_id": memory_id,
                        "category": validation.category,
                        "slug": validation.slug,
                        "reason": validation.reason,
                    }
                )
                continue
            item = accepted.setdefault(
                validation.canonical_tag,
                {
                    "id": validation.canonical_tag,
                    "category": validation.category,
                    "slug": validation.canonical_slug,
                    "name": validation.name,
                    "source_tags": set(),
                    "memory_ids": set(),
                    "confidence": validation.confidence,
                },
            )
            item["source_tags"].add(tag)
            item["memory_ids"].add(memory_id)

    accepted_list = []
    for item in accepted.values():
        accepted_list.append(
            {
                "id": item["id"],
                "category": item["category"],
                "slug": item["slug"],
                "name": item["name"],
                "source_tags": sorted(item["source_tags"]),
                "references": len(item["memory_ids"]),
                "confidence": item["confidence"],
            }
        )
    accepted_list.sort(key=lambda item: item["id"])
    rejected.sort(key=lambda item: (item["reason"], item["tag"], str(item["memory_id"])))

    accepted_counts_by_category = Counter(item["category"] for item in accepted_list)
    for accepted_category, accepted_count in accepted_counts_by_category.items():
        counts_by_category[accepted_category]["accepted"] = accepted_count

    if category:
        accepted_list = [item for item in accepted_list if item["category"] == category]
        rejected = [item for item in rejected if item["category"] == category]
    if reason:
        accepted_list = []
        rejected = [item for item in rejected if item["reason"] == reason]

    safe_limit = max(0, min(int(limit), 500))
    safe_offset = max(0, int(offset))
    accepted_page = accepted_list[safe_offset : safe_offset + safe_limit]
    rejected_page = rejected[safe_offset : safe_offset + safe_limit]
    counts_by_category_payload = {
        key: {
            "accepted": value.get("accepted", 0),
            "rejected": value.get("rejected", 0),
        }
        for key, value in sorted(counts_by_category.items())
    }
    counts = {
        "accepted_entities": len(accepted_list),
        "rejected_entities": len(rejected),
        "accepted_total_unfiltered": len(accepted),
        "rejected_total_unfiltered": sum(counts_by_reason.values()),
    }

    if summary:
        return {
            "accepted_sample": accepted_page,
            "rejected_sample": rejected_page,
            "counts": counts,
            "counts_by_reason": dict(sorted(counts_by_reason.items())),
            "counts_by_category": counts_by_category_payload,
            "limit": safe_limit,
            "offset": safe_offset,
            "summary": True,
        }

    return {
        "accepted": accepted_page,
        "rejected_entities": rejected_page,
        "counts": counts,
        "counts_by_reason": dict(sorted(counts_by_reason.items())),
        "counts_by_category": counts_by_category_payload,
        "limit": safe_limit,
        "offset": safe_offset,
        "summary": False,
    }


def create_entity_blueprint(
    get_memory_graph: Callable[[], Any],
    logger: Any,
    require_admin_token_fn: Optional[Callable[[], None]] = None,
) -> Blueprint:
    """Create the entity API blueprint.

    Args:
        get_memory_graph: Factory that returns the FalkorDB graph instance.
        logger: Logger instance.
        require_admin_token_fn: Optional callable that aborts if the request
            lacks a valid admin token.  Used to gate write operations (merge).
    """
    bp = Blueprint("entity", __name__)

    @bp.route("/entities", methods=["GET"])
    def list_entities() -> Any:
        graph = get_memory_graph()
        if graph is None:
            abort(503, description="FalkorDB is unavailable")

        category = request.args.get("category")
        try:
            limit = min(int(request.args.get("limit", 100)), 500)
        except (ValueError, TypeError):
            abort(400, description="Invalid limit parameter — must be an integer")

        if category:
            result = graph.query(
                """
                MATCH (e:Entity)
                WHERE e.merged_into IS NULL AND e.category = $category
                OPTIONAL MATCH (e)-[ref:REFERENCED_IN]->()
                WITH e, count(ref) as ref_count
                RETURN e.id, e.slug, e.category, e.name, e.aliases,
                       e.identity, e.identity_version, e.identity_updated_at,
                       e.identity_source_count, ref_count,
                       e.created_at, e.last_referenced_at
                ORDER BY ref_count DESC
                LIMIT $limit
                """,
                {"category": category, "limit": limit},
            )
        else:
            result = graph.query(
                """
                MATCH (e:Entity)
                WHERE e.merged_into IS NULL
                OPTIONAL MATCH (e)-[ref:REFERENCED_IN]->()
                WITH e, count(ref) as ref_count
                RETURN e.id, e.slug, e.category, e.name, e.aliases,
                       e.identity, e.identity_version, e.identity_updated_at,
                       e.identity_source_count, ref_count,
                       e.created_at, e.last_referenced_at
                ORDER BY ref_count DESC
                LIMIT $limit
                """,
                {"limit": limit},
            )

        entities = []
        for row in getattr(result, "result_set", []) or []:
            entities.append(_serialize_entity_row(row))

        return jsonify({"status": "success", "entities": entities, "count": len(entities)})

    @bp.route("/entity/<slug>", methods=["GET"])
    def get_entity(slug: str) -> Any:
        graph = get_memory_graph()
        if graph is None:
            abort(503, description="FalkorDB is unavailable")

        # Try direct slug match first, then alias lookup
        result = graph.query(
            """
            MATCH (e:Entity)
            WHERE e.merged_into IS NULL AND (e.slug = $slug OR $slug IN e.aliases)
            OPTIONAL MATCH (e)-[ref:REFERENCED_IN]->()
            WITH e, count(ref) as ref_count
            RETURN e.id, e.slug, e.category, e.name, e.aliases,
                   e.identity, e.identity_version, e.identity_updated_at,
                   e.identity_source_count, ref_count,
                   e.created_at, e.last_referenced_at
            LIMIT 1
            """,
            {"slug": slug},
        )

        rows = getattr(result, "result_set", []) or []
        if not rows:
            abort(404, description=f"Entity '{slug}' not found")

        entity = _serialize_entity_row(rows[0])
        return jsonify({"status": "success", "entity": entity})

    @bp.route("/entities/merge-candidates", methods=["GET"])
    def merge_candidates() -> Any:
        graph = get_memory_graph()
        if graph is None:
            abort(503, description="FalkorDB is unavailable")

        try:
            from automem.consolidation.entity_dedup import find_merge_candidates

            auto_merge, review = find_merge_candidates(graph)
            auto_merge_ids = {(c.entity_a_id, c.entity_b_id) for c in auto_merge}
            candidates = []
            for c in auto_merge + review:
                candidates.append(
                    {
                        "entity_a": c.entity_a_id,
                        "entity_b": c.entity_b_id,
                        "canonical": c.canonical_id,
                        "alias": c.alias_id,
                        "confidence": c.confidence,
                        "reason": c.reason,
                        "auto_merge": (c.entity_a_id, c.entity_b_id) in auto_merge_ids,
                    }
                )
            return jsonify(
                {
                    "status": "success",
                    "candidates": candidates,
                    "count": len(candidates),
                }
            )
        except Exception as exc:
            logger.exception("Failed to find merge candidates")
            return (
                jsonify({"error": "Failed to find merge candidates", "details": str(exc)}),
                500,
            )

    @bp.route("/entities/audit", methods=["GET"])
    def audit_entities() -> Any:
        """Dry-run entity cleanup audit (admin-only)."""
        if require_admin_token_fn is not None:
            require_admin_token_fn()

        graph = get_memory_graph()
        if graph is None:
            abort(503, description="FalkorDB is unavailable")

        try:
            from automem.consolidation.entity_dedup import find_merge_candidates

            summary = request.args.get("summary", "").lower() in {"1", "true", "yes"}
            include_merge_candidates = request.args.get(
                "include_merge_candidates", ""
            ).lower() in {"1", "true", "yes"}
            try:
                limit = min(int(request.args.get("limit", 100)), 500)
                offset = max(int(request.args.get("offset", 0)), 0)
            except (ValueError, TypeError):
                abort(400, description="Invalid limit/offset parameter")
            tag_audit = _audit_memory_entity_tags(
                graph,
                summary=summary,
                limit=limit,
                offset=offset,
                category=request.args.get("category"),
                reason=request.args.get("reason"),
            )
            if summary and not include_merge_candidates:
                auto_merge, review = [], []
            else:
                auto_merge, review = find_merge_candidates(graph)
            merge_candidates_payload = [
                {
                    "entity_a": candidate.entity_a_id,
                    "entity_b": candidate.entity_b_id,
                    "canonical": candidate.canonical_id,
                    "alias": candidate.alias_id,
                    "confidence": candidate.confidence,
                    "reason": candidate.reason,
                    "auto_merge": auto,
                }
                for auto, candidates in ((True, auto_merge), (False, review))
                for candidate in candidates
            ]

            return jsonify(
                {
                    "status": "success",
                    "dry_run": True,
                    **(
                        {
                            "accepted_sample": tag_audit["accepted_sample"],
                            "rejected_sample": tag_audit["rejected_sample"],
                        }
                        if tag_audit.get("summary")
                        else {
                            "accepted": tag_audit["accepted"],
                            "rejected_entities": tag_audit["rejected_entities"],
                        }
                    ),
                    "merge_candidates": merge_candidates_payload,
                    "counts": {
                        **tag_audit["counts"],
                        "merge_candidates": len(merge_candidates_payload),
                        "auto_merge_candidates": len(auto_merge),
                        "review_merge_candidates": len(review),
                    },
                    "counts_by_reason": tag_audit.get("counts_by_reason", {}),
                    "counts_by_category": tag_audit.get("counts_by_category", {}),
                    "limit": tag_audit.get("limit"),
                    "offset": tag_audit.get("offset"),
                    "summary": tag_audit.get("summary", False),
                    "merge_candidates_skipped": summary and not include_merge_candidates,
                }
            )
        except Exception as exc:
            logger.exception("Failed to audit entities")
            return jsonify({"error": "Failed to audit entities", "details": str(exc)}), 500

    @bp.route("/entity/<slug>/merge", methods=["POST"])
    def merge_entity(slug: str) -> Any:
        """Merge one entity into another (admin-only)."""
        if require_admin_token_fn is not None:
            require_admin_token_fn()

        graph = get_memory_graph()
        if graph is None:
            abort(503, description="FalkorDB is unavailable")

        data = request.get_json(silent=True) or {}
        target_slug = data.get("merge_into")
        if not target_slug:
            abort(400, description="'merge_into' slug is required")

        # Resolve both entities
        alias_result = graph.query(
            "MATCH (e:Entity) WHERE e.slug = $slug AND e.merged_into IS NULL RETURN e.id LIMIT 1",
            {"slug": slug},
        )
        canonical_result = graph.query(
            "MATCH (e:Entity) WHERE e.slug = $slug AND e.merged_into IS NULL RETURN e.id LIMIT 1",
            {"slug": target_slug},
        )

        alias_rows = getattr(alias_result, "result_set", []) or []
        canonical_rows = getattr(canonical_result, "result_set", []) or []

        if not alias_rows:
            abort(404, description=f"Entity '{slug}' not found")
        if not canonical_rows:
            abort(404, description=f"Target entity '{target_slug}' not found")

        alias_id = alias_rows[0][0]
        canonical_id = canonical_rows[0][0]

        if alias_id == canonical_id:
            abort(400, description="Source and target entity must be different")

        try:
            from automem.consolidation.entity_dedup import merge_entities

            merge_result = merge_entities(graph, canonical_id, alias_id)
            return jsonify(
                {
                    "status": "success",
                    "merge": {
                        "canonical": merge_result.canonical_id,
                        "alias": merge_result.alias_id,
                        "alias_slug": merge_result.alias_slug,
                        "edges_moved": merge_result.edges_moved,
                    },
                }
            )
        except Exception as exc:
            logger.exception("Merge failed")
            return jsonify({"error": "Merge failed", "details": str(exc)}), 500

    return bp
