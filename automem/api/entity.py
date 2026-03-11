"""Entity API endpoints for AutoMem.

Provides CRUD and query operations for first-class Entity nodes.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

from flask import Blueprint, abort, jsonify, request

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


def create_entity_blueprint(
    get_memory_graph: Callable[[], Any],
    logger: Any,
) -> Blueprint:
    """Create the entity API blueprint."""

    bp = Blueprint("entity", __name__)

    @bp.route("/entities", methods=["GET"])
    def list_entities() -> Any:
        graph = get_memory_graph()
        if graph is None:
            abort(503, description="FalkorDB is unavailable")

        category = request.args.get("category")
        limit = min(int(request.args.get("limit", 100)), 500)

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
                candidates.append({
                    "entity_a": c.entity_a_id,
                    "entity_b": c.entity_b_id,
                    "canonical": c.canonical_id,
                    "alias": c.alias_id,
                    "confidence": c.confidence,
                    "reason": c.reason,
                    "auto_merge": (c.entity_a_id, c.entity_b_id) in auto_merge_ids,
                })
            return jsonify({"status": "success", "candidates": candidates, "count": len(candidates)})
        except Exception as exc:
            logger.exception("Failed to find merge candidates")
            return jsonify({"error": "Failed to find merge candidates", "details": str(exc)}), 500

    @bp.route("/entity/<slug>/merge", methods=["POST"])
    def merge_entity(slug: str) -> Any:
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

        try:
            from automem.consolidation.entity_dedup import merge_entities

            merge_result = merge_entities(graph, canonical_id, alias_id)
            return jsonify({
                "status": "success",
                "merge": {
                    "canonical": merge_result.canonical_id,
                    "alias": merge_result.alias_id,
                    "alias_slug": merge_result.alias_slug,
                    "edges_moved": merge_result.edges_moved,
                },
            })
        except Exception as exc:
            logger.exception("Merge failed")
            return jsonify({"error": "Merge failed", "details": str(exc)}), 500

    return bp
