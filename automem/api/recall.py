from __future__ import annotations

from typing import Any, Callable, List, Dict, Optional, Tuple
from flask import Blueprint, request, abort, jsonify
import json
import time


def handle_recall(
    get_memory_graph: Callable[[], Any],
    get_qdrant_client: Callable[[], Any],
    normalize_tag_list: Callable[[Any], List[str]],
    normalize_timestamp: Callable[[str], str],
    parse_time_expression: Callable[[Optional[str]], Tuple[Optional[str], Optional[str]]],
    extract_keywords: Callable[[str], List[str]],
    compute_metadata_score: Callable[[Dict[str, Any], str, List[str]], tuple[float, Dict[str, float]]],
    result_passes_filters: Callable[[Dict[str, Any], Optional[str], Optional[str], Optional[List[str]], str, str], bool],
    graph_keyword_search: Callable[..., List[Dict[str, Any]]],
    vector_search: Callable[..., List[Dict[str, Any]]],
    vector_filter_only_tag_search: Callable[..., List[Dict[str, Any]]],
    recall_max_limit: int,
    logger: Any,
):
    query_start = time.perf_counter()
    query_text = (request.args.get("query") or "").strip()
    try:
        requested_limit = int(request.args.get("limit", 5))
    except (TypeError, ValueError):
        requested_limit = 5
    limit = max(1, min(requested_limit, recall_max_limit))

    embedding_param = request.args.get("embedding")
    time_query = request.args.get("time_query") or request.args.get("time")
    start_param = request.args.get("start")
    end_param = request.args.get("end")
    tags_param = request.args.getlist("tags") or request.args.get("tags")

    tag_mode = (request.args.get("tag_mode") or "any").strip().lower()
    if tag_mode not in {"any", "all"}:
        tag_mode = "any"

    tag_match = (request.args.get("tag_match") or "prefix").strip().lower()
    if tag_match not in {"exact", "prefix"}:
        tag_match = "prefix"

    time_start, time_end = parse_time_expression(time_query)
    start_time = time_start
    end_time = time_end

    if start_param:
        try:
            start_time = normalize_timestamp(start_param)
        except ValueError as exc:
            abort(400, description=f"Invalid start time: {exc}")
    if end_param:
        try:
            end_time = normalize_timestamp(end_param)
        except ValueError as exc:
            abort(400, description=f"Invalid end time: {exc}")

    tag_filters = normalize_tag_list(tags_param)

    seen_ids: set[str] = set()
    graph = get_memory_graph()
    qdrant_client = get_qdrant_client()

    results: List[Dict[str, Any]] = []
    vector_matches: List[Dict[str, Any]] = []
    if qdrant_client is not None:
        vector_matches = vector_search(
            qdrant_client,
            graph,
            query_text,
            embedding_param,
            limit,
            seen_ids,
            tag_filters,
            tag_mode,
            tag_match,
        )
        if start_time or end_time or tag_filters:
            vector_matches = [
                res
                for res in vector_matches
                if result_passes_filters(res, start_time, end_time, tag_filters, tag_mode, tag_match)
            ]
    results.extend(vector_matches[:limit])

    remaining_slots = max(0, limit - len(results))
    if remaining_slots and graph is not None:
        graph_matches = graph_keyword_search(
            graph,
            query_text,
            remaining_slots,
            seen_ids,
            start_time=start_time,
            end_time=end_time,
            tag_filters=tag_filters,
            tag_mode=tag_mode,
            tag_match=tag_match,
        )
        results.extend(graph_matches[:remaining_slots])

    tags_only_request = (not query_text and not (embedding_param and embedding_param.strip()) and bool(tag_filters))
    if tags_only_request and qdrant_client is not None and len(results) < limit:
        tag_only_results = vector_filter_only_tag_search(
            qdrant_client,
            tag_filters,
            tag_mode,
            tag_match,
            limit - len(results),
            seen_ids,
        )
        results.extend(tag_only_results)

    query_tokens = extract_keywords(query_text.lower()) if query_text else []
    for result in results:
        final_score, components = compute_metadata_score(result, query_text or "", query_tokens)
        result.setdefault("score_components", components)
        result["score_components"].update(components)
        result["final_score"] = final_score
        result["original_score"] = result.get("score", 0.0)
        result["score"] = final_score

    results = [
        res
        for res in results
        if result_passes_filters(res, start_time, end_time, tag_filters, tag_mode, tag_match)
    ]

    results.sort(
        key=lambda r: (
            -float(r.get("final_score", 0.0)),
            r.get("source") != "qdrant",
            -float(r.get("original_score", 0.0)),
            -float((r.get("memory") or {}).get("importance", 0.0) or 0.0),
        )
    )

    response = {
        "status": "success",
        "query": query_text,
        "results": results,
        "count": len(results),
        "vector_search": {
            "enabled": qdrant_client is not None,
            "matched": bool(vector_matches),
        },
    }
    if query_text:
        response["keywords"] = query_tokens
    if start_time or end_time:
        response["time_window"] = {"start": start_time, "end": end_time}
    if tag_filters:
        response["tags"] = tag_filters
    response["tag_mode"] = tag_mode
    response["tag_match"] = tag_match
    response["query_time_ms"] = round((time.perf_counter() - query_start) * 1000, 2)

    logger.info(
        "recall_complete",
        extra={
            "query": query_text[:100] if query_text else "",
            "results": len(results),
            "latency_ms": response["query_time_ms"],
            "vector_enabled": qdrant_client is not None,
            "vector_matches": len(vector_matches),
            "has_time_filter": bool(start_time or end_time),
            "has_tag_filter": bool(tag_filters),
            "limit": limit,
        },
    )
    return jsonify(response)


def create_recall_blueprint(
    get_memory_graph: Callable[[], Any],
    get_qdrant_client: Callable[[], Any],
    normalize_tag_list: Callable[[Any], List[str]],
    normalize_timestamp: Callable[[str], str],
    parse_time_expression: Callable[[Optional[str]], Tuple[Optional[str], Optional[str]]],
    extract_keywords: Callable[[str], List[str]],
    compute_metadata_score: Callable[[Dict[str, Any], str, List[str]], tuple[float, Dict[str, float]]],
    result_passes_filters: Callable[[Dict[str, Any], Optional[str], Optional[str], Optional[List[str]], str, str], bool],
    graph_keyword_search: Callable[..., List[Dict[str, Any]]],
    vector_search: Callable[..., List[Dict[str, Any]]],
    vector_filter_only_tag_search: Callable[..., List[Dict[str, Any]]],
    recall_max_limit: int,
    logger: Any,
    allowed_relations: List[str] | set[str] | tuple[str, ...] | Any = (),
    relation_limit: int = 5,
    serialize_node: Callable[[Any], Dict[str, Any]] | None = None,
    summarize_relation_node: Callable[[Dict[str, Any]], Dict[str, Any]] | None = None,
) -> Blueprint:
    bp = Blueprint("recall", __name__)

    @bp.route("/recall", methods=["GET"])
    def recall_memories() -> Any:
        return handle_recall(
            get_memory_graph,
            get_qdrant_client,
            normalize_tag_list,
            normalize_timestamp,
            parse_time_expression,
            extract_keywords,
            compute_metadata_score,
            result_passes_filters,
            graph_keyword_search,
            vector_search,
            vector_filter_only_tag_search,
            recall_max_limit,
            logger,
        )

    @bp.route("/startup-recall", methods=["GET"])
    def startup_recall() -> Any:
        graph = get_memory_graph()
        if graph is None:
            abort(503, description="FalkorDB is unavailable")

        try:
            lesson_query = """
                MATCH (m:Memory)
                WHERE 'critical' IN m.tags OR 'lesson' IN m.tags OR 'ai-assistant' IN m.tags
                RETURN m.id as id, m.content as content, m.tags as tags,
                       m.importance as importance, m.type as type, m.metadata as metadata
                ORDER BY m.importance DESC
                LIMIT 10
            """

            lesson_results = graph.query(lesson_query)
            lessons = []
            if getattr(lesson_results, "result_set", None):
                for row in lesson_results.result_set:
                    lessons.append({
                        'id': row[0],
                        'content': row[1],
                        'tags': row[2] if row[2] else [],
                        'importance': row[3] if row[3] else 0.5,
                        'type': row[4] if row[4] else 'Context',
                        'metadata': json.loads(row[5]) if row[5] else {}
                    })

            system_query = """
                MATCH (m:Memory)
                WHERE 'system' IN m.tags OR 'memory-recall' IN m.tags
                RETURN m.id as id, m.content as content, m.tags as tags
                LIMIT 5
            """

            system_results = graph.query(system_query)
            system_rules = []
            if getattr(system_results, "result_set", None):
                for row in system_results.result_set:
                    system_rules.append({
                        'id': row[0],
                        'content': row[1],
                        'tags': row[2] if row[2] else []
                    })

            response = {
                'status': 'success',
                'critical_lessons': lessons,
                'system_rules': system_rules,
                'lesson_count': len(lessons),
                'has_critical': any(l.get('importance', 0) >= 0.9 for l in lessons),
                'summary': f"Recalled {len(lessons)} lesson(s) and {len(system_rules)} system rule(s)"
            }
            return jsonify(response), 200
        except Exception as e:
            logger.error(f"Startup recall failed: {e}")
            return jsonify({
                "error": "Startup recall failed",
                "details": str(e)
            }), 500

    @bp.route("/analyze", methods=["GET"])
    def analyze_memories() -> Any:
        graph = get_memory_graph()
        if graph is None:
            abort(503, description="FalkorDB is unavailable")
        analytics = {
            "memory_types": {},
            "patterns": [],
            "preferences": [],
            "temporal_insights": {},
            "entity_frequency": {},
            "confidence_distribution": {},
        }
        try:
            type_result = graph.query(
                """
                MATCH (m:Memory)
                WHERE m.type IS NOT NULL
                RETURN m.type, COUNT(m) as count, AVG(m.confidence) as avg_confidence
                ORDER BY count DESC
                """
            )
            for mem_type, count, avg_conf in getattr(type_result, "result_set", []) or []:
                analytics["memory_types"][mem_type] = {
                    "count": count,
                    "average_confidence": round(avg_conf, 3) if avg_conf else 0,
                }

            pattern_result = graph.query(
                """
                MATCH (p:Pattern)
                WHERE p.confidence > 0.6
                RETURN p.type, p.content, p.confidence, p.observations
                ORDER BY p.confidence DESC
                LIMIT 10
                """
            )
            for p_type, content, confidence, observations in getattr(pattern_result, "result_set", []) or []:
                analytics["patterns"].append({
                    "type": p_type,
                    "description": content,
                    "confidence": round(confidence, 3) if confidence else 0,
                    "observations": observations or 0,
                })

            pref_result = graph.query(
                """
                MATCH (m1:Memory)-[r:PREFERS_OVER]->(m2:Memory)
                RETURN m1.content, m2.content, r.context, r.strength
                ORDER BY r.strength DESC
                LIMIT 10
                """
            )
            for preferred, over, context, strength in getattr(pref_result, "result_set", []) or []:
                analytics["preferences"].append({
                    "prefers": preferred,
                    "over": over,
                    "context": context,
                    "strength": round(strength, 3) if strength else 0,
                })

            try:
                temporal_result = graph.query(
                    """
                    MATCH (m:Memory)
                    WHERE m.timestamp IS NOT NULL
                    RETURN m.timestamp, m.importance
                    LIMIT 100
                    """
                )
                from collections import defaultdict
                hour_data = defaultdict(lambda: {"count": 0, "total_importance": 0})
                for timestamp, importance in getattr(temporal_result, "result_set", []) or []:
                    if timestamp and len(timestamp) > 13:
                        hour_str = timestamp[11:13]
                        if hour_str.isdigit():
                            hour = int(hour_str)
                            hour_data[hour]["count"] += 1
                            hour_data[hour]["total_importance"] += importance or 0.5
                for hour, data in hour_data.items():
                    if data["count"] > 0:
                        analytics["temporal_insights"][f"hour_{hour:02d}"] = {
                            "count": data["count"],
                            "avg_importance": round(data["total_importance"] / data["count"], 3)
                        }
            except Exception:
                pass

            entity_result = graph.query(
                """
                MATCH (m:Memory)
                WHERE m.metadata IS NOT NULL
                RETURN m.metadata
                LIMIT 200
                """
            )
            from collections import Counter
            entities = Counter()
            for (metadata_json,) in getattr(entity_result, "result_set", []) or []:
                try:
                    metadata = json.loads(metadata_json) if isinstance(metadata_json, str) else (metadata_json or {})
                except Exception:
                    metadata = {}
                if not isinstance(metadata, dict):
                    # Skip unsupported metadata shapes (e.g., SimpleNamespace in dummy graphs)
                    continue
                for key in ("entities", "keywords", "topics"):
                    for item in (metadata.get(key) or []):
                        val = str(item).strip().lower()
                        if len(val) >= 3:
                            entities[val] += 1
            analytics["entity_frequency"] = dict(entities.most_common(50))

            conf_result = graph.query(
                """
                MATCH (m:Memory)
                WHERE m.confidence IS NOT NULL
                RETURN m.confidence
                LIMIT 500
                """
            )
            conf_dist: Dict[str, int] = {"low": 0, "medium": 0, "high": 0}
            for (conf,) in getattr(conf_result, "result_set", []) or []:
                try:
                    c = float(conf or 0)
                except Exception:
                    c = 0.0
                if c < 0.4:
                    conf_dist["low"] += 1
                elif c < 0.7:
                    conf_dist["medium"] += 1
                else:
                    conf_dist["high"] += 1
            analytics["confidence_distribution"] = conf_dist
            return jsonify({"status": "success", "analytics": analytics, "elapsed_ms": 0}), 200
        except Exception as e:
            logger.error(f"Analyze failed: {e}")
            return jsonify({"error": "Analyze failed", "details": str(e)}), 500

    @bp.route("/memories/<memory_id>/related", methods=["GET"])
    def get_related_memories(memory_id: str) -> Any:
        graph = get_memory_graph()
        if graph is None:
            abort(503, description="FalkorDB is unavailable")

        allowed = set(allowed_relations) if allowed_relations else set()

        rel_types_param = (request.args.get("relationship_types") or "").strip()
        if rel_types_param:
            requested = [part.strip().upper() for part in rel_types_param.split(",") if part.strip()]
            rel_types = [t for t in requested if (t in allowed if allowed else True)]
            if not rel_types and allowed:
                rel_types = sorted(allowed)
        else:
            rel_types = sorted(allowed) if allowed else []

        try:
            max_depth = int(request.args.get("max_depth", 1))
        except (TypeError, ValueError):
            max_depth = 1
        max_depth = max(1, min(max_depth, 3))

        try:
            limit = int(request.args.get("limit", relation_limit))
        except (TypeError, ValueError):
            limit = relation_limit
        limit = max(1, min(limit, 200))

        rel_pattern = ":" + "|".join(rel_types) if rel_types else ""
        query = f"""
            MATCH (m:Memory {{id: $id}}){'-[r' + rel_pattern + f']-' if rel_pattern else '-[r]-'}(related:Memory)
            WHERE m.id <> related.id
            CALL apoc.path.expandConfig(related, {{
                relationshipFilter: '{'|'.join(rel_types)}',
                minLevel: 0,
                maxLevel: $max_depth,
                bfs: true,
                filterStartNode: true
            }}) YIELD path
            WITH DISTINCT related
            RETURN related
            ORDER BY coalesce(related.importance, 0.0) DESC, coalesce(related.timestamp, '') DESC
            LIMIT $limit
        """
        fallback_query = f"""
            MATCH (m:Memory {{id: $id}}){'-[r' + rel_pattern + f'*1..$max_depth]-' if rel_pattern else '-[r*1..$max_depth]-'}(related:Memory)
            WHERE m.id <> related.id
            RETURN DISTINCT related
            ORDER BY coalesce(related.importance, 0.0) DESC, coalesce(related.timestamp, '') DESC
            LIMIT $limit
        """
        params = {"id": memory_id, "max_depth": max_depth, "limit": limit}
        try:
            result = graph.query(query, params)
        except Exception:
            try:
                result = graph.query(fallback_query, params)
            except Exception:
                logger.exception("Failed to traverse related memories for %s", memory_id)
                abort(500, description="Failed to fetch related memories")

        related: List[Dict[str, Any]] = []
        for row in getattr(result, "result_set", []) or []:
            node = row[0]
            data = serialize_node(node) if serialize_node else {"value": node}
            if data.get("id") != memory_id:
                if summarize_relation_node:
                    # Present summarized node info
                    related.append(summarize_relation_node(data))
                else:
                    related.append(data)
        return jsonify({
            "status": "success",
            "memory_id": memory_id,
            "count": len(related),
            "related_memories": related,
            "relationship_types": rel_types,
            "max_depth": max_depth,
            "limit": limit,
        })

    return bp
