"""Graph visualization API for AutoMem Memory Viewer.

Exposes memory graph structure in a format optimized for 3D visualization.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np
from flask import Blueprint, abort, jsonify, request

from automem.config import ALLOWED_RELATIONS, MEMORY_TYPES
from automem.utils.graph import _serialize_node

# Lazy load UMAP to avoid slow import on every request
_umap_module = None


def _get_umap():
    """Lazy load UMAP to avoid slow import at startup."""
    global _umap_module
    if _umap_module is None:
        try:
            import umap

            _umap_module = umap
        except ImportError:
            _umap_module = False
    return _umap_module if _umap_module else None


logger = logging.getLogger("automem.api.graph")

# Color palette for memory types (vibrant, distinguishable in 3D)
TYPE_COLORS: Dict[str, str] = {
    "Decision": "#3B82F6",  # Blue
    "Pattern": "#10B981",  # Emerald
    "Preference": "#8B5CF6",  # Purple
    "Style": "#EC4899",  # Pink
    "Habit": "#F59E0B",  # Amber
    "Insight": "#F97316",  # Orange
    "Context": "#6B7280",  # Gray
    "Memory": "#94A3B8",  # Slate (default)
}

# Edge colors by relationship type
RELATION_COLORS: Dict[str, str] = {
    "RELATES_TO": "#94A3B8",
    "LEADS_TO": "#3B82F6",
    "OCCURRED_BEFORE": "#6B7280",
    "PREFERS_OVER": "#8B5CF6",
    "EXEMPLIFIES": "#10B981",
    "CONTRADICTS": "#EF4444",
    "REINFORCES": "#22C55E",
    "INVALIDATED_BY": "#F97316",
    "EVOLVED_INTO": "#06B6D4",
    "DERIVED_FROM": "#A855F7",
    "PART_OF": "#64748B",
}


def create_graph_blueprint(
    get_memory_graph: Callable[[], Any],
    get_qdrant_client: Callable[[], Any],
    serialize_node: Callable[[Any], Dict[str, Any]],
    collection_name: str,
    logger: Any,
) -> Blueprint:
    """Create the graph visualization blueprint."""
    bp = Blueprint("graph", __name__, url_prefix="/graph")

    @bp.route("/snapshot", methods=["GET"])
    def snapshot() -> Any:
        """Return full graph structure for visualization.

        Query params:
            limit: Max nodes to return (default 500)
            min_importance: Filter by minimum importance (0.0-1.0)
            types: Comma-separated list of memory types to include
            since: ISO timestamp - only include memories after this time
            include_positions: If true, include pre-computed layout positions
        """
        query_start = time.perf_counter()

        limit = min(int(request.args.get("limit", 500)), 2000)
        min_importance = float(request.args.get("min_importance", 0.0))
        types_filter = (
            request.args.get("types", "").split(",") if request.args.get("types") else None
        )
        since = request.args.get("since")

        graph = get_memory_graph()
        if graph is None:
            abort(503, description="Graph database unavailable")

        # Build Cypher query for nodes
        where_clauses = ["m.importance >= $min_importance"]
        params: Dict[str, Any] = {"min_importance": min_importance, "limit": limit}

        if types_filter and types_filter[0]:
            where_clauses.append("m.type IN $types")
            params["types"] = [t.strip() for t in types_filter if t.strip()]

        if since:
            where_clauses.append("m.timestamp >= $since")
            params["since"] = since

        where_clause = " AND ".join(where_clauses)

        # Fetch nodes
        node_query = f"""
            MATCH (m:Memory)
            WHERE {where_clause}
            RETURN m
            ORDER BY m.importance DESC, m.timestamp DESC
            LIMIT $limit
        """

        try:
            node_result = graph.query(node_query, params)
            node_ids: Set[str] = set()
            nodes: List[Dict[str, Any]] = []

            for row in node_result.result_set:
                node_data = serialize_node(row[0])
                node_id = node_data.get("id")
                if node_id:
                    node_ids.add(node_id)

                    # Compute visual properties
                    importance = float(node_data.get("importance", 0.5))
                    confidence = float(node_data.get("confidence", 0.8))
                    mem_type = node_data.get("type", "Memory")

                    nodes.append(
                        {
                            "id": node_id,
                            "content": node_data.get("content", ""),
                            "type": mem_type,
                            "importance": importance,
                            "confidence": confidence,
                            "tags": node_data.get("tags", []),
                            "timestamp": node_data.get("timestamp"),
                            "updated_at": node_data.get("updated_at"),
                            "metadata": node_data.get("metadata", {}),
                            # Visual properties for 3D rendering
                            "color": TYPE_COLORS.get(mem_type, TYPE_COLORS["Memory"]),
                            "radius": 0.5 + (importance * 1.5),  # 0.5 to 2.0
                            "opacity": 0.4 + (confidence * 0.6),  # 0.4 to 1.0
                        }
                    )

            # Fetch edges between visible nodes
            edges: List[Dict[str, Any]] = []
            if node_ids:
                edge_query = """
                    MATCH (m1:Memory)-[r]->(m2:Memory)
                    WHERE m1.id IN $node_ids AND m2.id IN $node_ids
                    RETURN m1.id as source, m2.id as target, type(r) as rel_type, r as rel
                """
                edge_result = graph.query(edge_query, {"node_ids": list(node_ids)})

                for row in edge_result.result_set:
                    source, target, rel_type, rel = row
                    rel_props = dict(rel.properties) if hasattr(rel, "properties") else {}
                    strength = float(rel_props.get("strength", 0.5))

                    edges.append(
                        {
                            "id": f"{source}-{rel_type}-{target}",
                            "source": source,
                            "target": target,
                            "type": rel_type,
                            "strength": strength,
                            "color": RELATION_COLORS.get(rel_type, "#94A3B8"),
                            "properties": rel_props,
                        }
                    )

            # Compute statistics
            total_query = "MATCH (m:Memory) RETURN count(m) as total"
            total_result = graph.query(total_query)
            total_nodes = total_result.result_set[0][0] if total_result.result_set else 0

            total_edges_query = "MATCH ()-[r]->() RETURN count(r) as total"
            total_edges_result = graph.query(total_edges_query)
            total_edges = (
                total_edges_result.result_set[0][0] if total_edges_result.result_set else 0
            )

            elapsed = time.perf_counter() - query_start
            logger.info(f"graph/snapshot: {len(nodes)} nodes, {len(edges)} edges in {elapsed:.3f}s")

            return jsonify(
                {
                    "nodes": nodes,
                    "edges": edges,
                    "stats": {
                        "total_nodes": total_nodes,
                        "total_edges": total_edges,
                        "returned_nodes": len(nodes),
                        "returned_edges": len(edges),
                        "sampled": len(nodes) < total_nodes,
                        "sample_ratio": len(nodes) / max(total_nodes, 1),
                    },
                    "meta": {
                        "type_colors": TYPE_COLORS,
                        "relation_colors": RELATION_COLORS,
                        "query_time_ms": round(elapsed * 1000, 2),
                    },
                }
            )

        except Exception as e:
            logger.error(f"graph/snapshot failed: {e}")
            abort(500, description=str(e))

    @bp.route("/projected", methods=["GET"])
    def projected() -> Any:
        """Return graph with UMAP-projected 3D positions based on embeddings.

        Uses dimensionality reduction to position nodes so semantically similar
        memories appear close together, regardless of explicit edges or tags.

        Query params:
            limit: Max nodes to return (default 500)
            min_importance: Filter by minimum importance (0.0-1.0)
            types: Comma-separated list of memory types to include
            n_neighbors: UMAP n_neighbors param (default 15)
            min_dist: UMAP min_dist param (default 0.1)
            spread: UMAP spread param (default 1.0)
        """
        query_start = time.perf_counter()

        umap_mod = _get_umap()
        if umap_mod is None:
            abort(501, description="UMAP not available - install umap-learn")

        limit = min(int(request.args.get("limit", 500)), 2000)
        min_importance = float(request.args.get("min_importance", 0.0))
        types_filter = (
            request.args.get("types", "").split(",") if request.args.get("types") else None
        )

        # UMAP parameters
        n_neighbors = min(int(request.args.get("n_neighbors", 15)), 50)
        min_dist = float(request.args.get("min_dist", 0.1))
        spread = float(request.args.get("spread", 1.0))

        qdrant = get_qdrant_client()
        if qdrant is None:
            abort(503, description="Vector database unavailable for projection")

        graph = get_memory_graph()
        if graph is None:
            abort(503, description="Graph database unavailable")

        try:
            # Fetch embeddings from Qdrant with scroll
            all_points = []
            offset = None

            while True:
                scroll_result = qdrant.scroll(
                    collection_name=collection_name,
                    limit=100,
                    offset=offset,
                    with_vectors=True,
                    with_payload=True,
                )
                points, next_offset = scroll_result
                if not points:
                    break
                all_points.extend(points)
                if next_offset is None or len(all_points) >= limit:
                    break
                offset = next_offset

            if len(all_points) < 3:
                # Not enough points for UMAP
                abort(400, description="Need at least 3 memories for projection")

            # Filter by importance and type if specified
            filtered_points = []
            for p in all_points:
                payload = p.payload or {}
                importance = float(payload.get("importance", 0.5))
                mem_type = payload.get("type", "Memory")

                if importance < min_importance:
                    continue
                if types_filter and types_filter[0] and mem_type not in types_filter:
                    continue
                if p.vector is None:
                    continue

                filtered_points.append(p)

            if len(filtered_points) < 3:
                abort(400, description="Not enough memories match filters for projection")

            # Limit to requested count
            if len(filtered_points) > limit:
                # Sort by importance and take top N
                filtered_points.sort(
                    key=lambda p: float((p.payload or {}).get("importance", 0.5)),
                    reverse=True,
                )
                filtered_points = filtered_points[:limit]

            # Extract embeddings matrix
            embeddings = np.array([p.vector for p in filtered_points], dtype=np.float32)
            memory_ids = [p.id for p in filtered_points]

            # Run UMAP projection to 3D
            logger.info(f"Running UMAP on {len(embeddings)} embeddings...")
            umap_start = time.perf_counter()

            reducer = umap_mod.UMAP(
                n_components=3,
                n_neighbors=min(n_neighbors, len(embeddings) - 1),
                min_dist=min_dist,
                spread=spread,
                metric="cosine",
                random_state=42,  # Reproducible projections
            )
            projected_coords = reducer.fit_transform(embeddings)

            umap_time = time.perf_counter() - umap_start
            logger.info(f"UMAP completed in {umap_time:.2f}s")

            # Scale coordinates to reasonable visualization range
            # Center at origin and scale to ~100 unit radius
            projected_coords -= projected_coords.mean(axis=0)
            max_range = np.abs(projected_coords).max()
            if max_range > 0:
                projected_coords = projected_coords * (100.0 / max_range)

            # Build node list with projected positions
            nodes: List[Dict[str, Any]] = []
            node_ids: Set[str] = set()

            for i, point in enumerate(filtered_points):
                payload = point.payload or {}
                node_id = str(point.id)
                node_ids.add(node_id)

                importance = float(payload.get("importance", 0.5))
                confidence = float(payload.get("confidence", 0.8))
                mem_type = payload.get("type", "Memory")

                nodes.append(
                    {
                        "id": node_id,
                        "content": payload.get("content", ""),
                        "type": mem_type,
                        "importance": importance,
                        "confidence": confidence,
                        "tags": payload.get("tags", []),
                        "timestamp": payload.get("timestamp"),
                        "updated_at": payload.get("updated_at"),
                        "metadata": payload.get("metadata", {}),
                        # Visual properties
                        "color": TYPE_COLORS.get(mem_type, TYPE_COLORS["Memory"]),
                        "radius": 0.5 + (importance * 1.5),
                        "opacity": 0.4 + (confidence * 0.6),
                        # UMAP-projected positions
                        "x": float(projected_coords[i, 0]),
                        "y": float(projected_coords[i, 1]),
                        "z": float(projected_coords[i, 2]),
                    }
                )

            # Fetch edges between visible nodes
            edges: List[Dict[str, Any]] = []
            if node_ids:
                edge_query = """
                    MATCH (m1:Memory)-[r]->(m2:Memory)
                    WHERE m1.id IN $node_ids AND m2.id IN $node_ids
                    RETURN m1.id as source, m2.id as target, type(r) as rel_type, r as rel
                """
                edge_result = graph.query(edge_query, {"node_ids": list(node_ids)})

                for row in edge_result.result_set:
                    source, target, rel_type, rel = row
                    rel_props = dict(rel.properties) if hasattr(rel, "properties") else {}
                    strength = float(rel_props.get("strength", 0.5))

                    edges.append(
                        {
                            "id": f"{source}-{rel_type}-{target}",
                            "source": source,
                            "target": target,
                            "type": rel_type,
                            "strength": strength,
                            "color": RELATION_COLORS.get(rel_type, "#94A3B8"),
                            "properties": rel_props,
                        }
                    )

            elapsed = time.perf_counter() - query_start
            logger.info(
                f"graph/projected: {len(nodes)} nodes, {len(edges)} edges in {elapsed:.3f}s"
            )

            return jsonify(
                {
                    "nodes": nodes,
                    "edges": edges,
                    "stats": {
                        "total_nodes": len(all_points),
                        "returned_nodes": len(nodes),
                        "returned_edges": len(edges),
                    },
                    "projection": {
                        "method": "umap",
                        "n_neighbors": n_neighbors,
                        "min_dist": min_dist,
                        "spread": spread,
                        "dimensions": 3,
                        "umap_time_ms": round(umap_time * 1000, 2),
                    },
                    "meta": {
                        "type_colors": TYPE_COLORS,
                        "relation_colors": RELATION_COLORS,
                        "query_time_ms": round(elapsed * 1000, 2),
                    },
                }
            )

        except Exception as e:
            logger.error(f"graph/projected failed: {e}")
            abort(500, description=str(e))

    @bp.route("/neighbors/<memory_id>", methods=["GET"])
    def neighbors(memory_id: str) -> Any:
        """Get neighbors of a specific memory node.

        Returns both graph neighbors (via relationships) and semantic neighbors (via Qdrant).

        Query params:
            depth: How many hops to traverse (1-3, default 1)
            include_semantic: Include embedding-based neighbors (default true)
            semantic_limit: Max semantic neighbors (default 5)
        """
        query_start = time.perf_counter()

        depth = min(int(request.args.get("depth", 1)), 3)
        include_semantic = request.args.get("include_semantic", "true").lower() == "true"
        semantic_limit = min(int(request.args.get("semantic_limit", 5)), 20)

        graph = get_memory_graph()
        if graph is None:
            abort(503, description="Graph database unavailable")

        # Get the center node
        center_query = "MATCH (m:Memory {id: $id}) RETURN m"
        center_result = graph.query(center_query, {"id": memory_id})

        if not center_result.result_set:
            abort(404, description=f"Memory {memory_id} not found")

        center_node = serialize_node(center_result.result_set[0][0])

        # Get graph neighbors up to depth
        neighbor_query = f"""
            MATCH path = (m:Memory {{id: $id}})-[*1..{depth}]-(n:Memory)
            WHERE n.id <> $id
            UNWIND relationships(path) as r
            WITH DISTINCT n, r, startNode(r) as src, endNode(r) as tgt
            RETURN n, type(r) as rel_type, r, src.id as source_id, tgt.id as target_id
            LIMIT 100
        """

        neighbor_result = graph.query(neighbor_query, {"id": memory_id})

        graph_neighbors: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        seen_nodes: Set[str] = {memory_id}
        seen_edges: Set[str] = set()

        for row in neighbor_result.result_set:
            node, rel_type, rel, source_id, target_id = row
            node_data = serialize_node(node)
            node_id = node_data.get("id")

            if node_id and node_id not in seen_nodes:
                seen_nodes.add(node_id)
                importance = float(node_data.get("importance", 0.5))
                confidence = float(node_data.get("confidence", 0.8))
                mem_type = node_data.get("type", "Memory")

                graph_neighbors.append(
                    {
                        "id": node_id,
                        "content": node_data.get("content", ""),
                        "type": mem_type,
                        "importance": importance,
                        "confidence": confidence,
                        "tags": node_data.get("tags", []),
                        "color": TYPE_COLORS.get(mem_type, TYPE_COLORS["Memory"]),
                        "radius": 0.5 + (importance * 1.5),
                        "opacity": 0.4 + (confidence * 0.6),
                    }
                )

            edge_id = f"{source_id}-{rel_type}-{target_id}"
            if edge_id not in seen_edges:
                seen_edges.add(edge_id)
                rel_props = dict(rel.properties) if hasattr(rel, "properties") else {}
                strength = float(rel_props.get("strength", 0.5))

                edges.append(
                    {
                        "id": edge_id,
                        "source": source_id,
                        "target": target_id,
                        "type": rel_type,
                        "strength": strength,
                        "color": RELATION_COLORS.get(rel_type, "#94A3B8"),
                    }
                )

        # Get semantic neighbors from Qdrant
        semantic_neighbors: List[Dict[str, Any]] = []
        if include_semantic:
            qdrant = get_qdrant_client()
            if qdrant:
                try:
                    # Get the embedding for the center node
                    points = qdrant.retrieve(
                        collection_name=collection_name,
                        ids=[memory_id],
                        with_vectors=True,
                    )

                    if points and points[0].vector:
                        # Search for similar vectors
                        search_result = qdrant.search(
                            collection_name=collection_name,
                            query_vector=points[0].vector,
                            limit=semantic_limit + 1,  # +1 to exclude self
                            with_payload=True,
                        )

                        for hit in search_result:
                            if hit.id != memory_id and hit.id not in seen_nodes:
                                payload = hit.payload or {}
                                mem_type = payload.get("type", "Memory")
                                importance = float(payload.get("importance", 0.5))
                                confidence = float(payload.get("confidence", 0.8))

                                semantic_neighbors.append(
                                    {
                                        "id": hit.id,
                                        "content": payload.get("content", ""),
                                        "type": mem_type,
                                        "importance": importance,
                                        "confidence": confidence,
                                        "tags": payload.get("tags", []),
                                        "similarity": round(hit.score, 4),
                                        "color": TYPE_COLORS.get(mem_type, TYPE_COLORS["Memory"]),
                                        "radius": 0.5 + (importance * 1.5),
                                        "opacity": 0.4 + (confidence * 0.6),
                                    }
                                )
                except Exception as e:
                    logger.warning(f"Semantic neighbor search failed: {e}")

        elapsed = time.perf_counter() - query_start

        return jsonify(
            {
                "center": {
                    **center_node,
                    "color": TYPE_COLORS.get(
                        center_node.get("type", "Memory"), TYPE_COLORS["Memory"]
                    ),
                },
                "graph_neighbors": graph_neighbors,
                "semantic_neighbors": semantic_neighbors,
                "edges": edges,
                "meta": {
                    "depth": depth,
                    "query_time_ms": round(elapsed * 1000, 2),
                },
            }
        )

    @bp.route("/stats", methods=["GET"])
    def stats() -> Any:
        """Return graph statistics for dashboard display."""
        query_start = time.perf_counter()

        graph = get_memory_graph()
        if graph is None:
            abort(503, description="Graph database unavailable")

        try:
            # Count by type
            type_query = """
                MATCH (m:Memory)
                RETURN m.type as type, count(*) as count
                ORDER BY count DESC
            """
            type_result = graph.query(type_query)
            type_counts = {row[0]: row[1] for row in type_result.result_set if row[0]}

            # Count relationships by type
            rel_query = """
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(*) as count
                ORDER BY count DESC
            """
            rel_result = graph.query(rel_query)
            rel_counts = {row[0]: row[1] for row in rel_result.result_set if row[0]}

            # Importance distribution
            importance_query = """
                MATCH (m:Memory)
                RETURN
                    sum(CASE WHEN m.importance >= 0.8 THEN 1 ELSE 0 END) as high,
                    sum(CASE WHEN m.importance >= 0.5 AND m.importance < 0.8 THEN 1 ELSE 0 END) as medium,
                    sum(CASE WHEN m.importance < 0.5 THEN 1 ELSE 0 END) as low
            """
            importance_result = graph.query(importance_query)
            importance_dist = {}
            if importance_result.result_set:
                row = importance_result.result_set[0]
                importance_dist = {"high": row[0], "medium": row[1], "low": row[2]}

            # Recent activity (last 7 days)
            activity_query = """
                MATCH (m:Memory)
                WHERE m.timestamp >= datetime() - duration('P7D')
                RETURN date(datetime(m.timestamp)) as day, count(*) as count
                ORDER BY day DESC
            """
            try:
                activity_result = graph.query(activity_query)
                recent_activity = [
                    {"date": str(row[0]), "count": row[1]} for row in activity_result.result_set
                ]
            except Exception:
                recent_activity = []

            # Total counts
            totals_query = """
                MATCH (m:Memory)
                RETURN count(m) as nodes
            """
            totals_result = graph.query(totals_query)
            total_nodes = totals_result.result_set[0][0] if totals_result.result_set else 0

            edges_query = "MATCH ()-[r]->() RETURN count(r)"
            edges_result = graph.query(edges_query)
            total_edges = edges_result.result_set[0][0] if edges_result.result_set else 0

            elapsed = time.perf_counter() - query_start

            return jsonify(
                {
                    "totals": {
                        "nodes": total_nodes,
                        "edges": total_edges,
                    },
                    "by_type": type_counts,
                    "by_relationship": rel_counts,
                    "importance_distribution": importance_dist,
                    "recent_activity": recent_activity,
                    "meta": {
                        "type_colors": TYPE_COLORS,
                        "relation_colors": RELATION_COLORS,
                        "query_time_ms": round(elapsed * 1000, 2),
                    },
                }
            )

        except Exception as e:
            logger.error(f"graph/stats failed: {e}")
            abort(500, description=str(e))

    @bp.route("/types", methods=["GET"])
    def types() -> Any:
        """Return available memory types and their colors."""
        return jsonify(
            {
                "types": list(MEMORY_TYPES),
                "colors": TYPE_COLORS,
            }
        )

    @bp.route("/relations", methods=["GET"])
    def relations() -> Any:
        """Return available relationship types and their colors."""
        return jsonify(
            {
                "relations": list(ALLOWED_RELATIONS),
                "colors": RELATION_COLORS,
            }
        )

    return bp
