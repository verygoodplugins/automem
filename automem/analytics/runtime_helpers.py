from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Callable, Dict


def startup_recall(
    *,
    get_memory_graph_fn: Callable[[], Any],
    jsonify_fn: Callable[[Any], Any],
    abort_fn: Callable[..., Any],
    logger: Any,
) -> Any:
    """Recall critical lessons at session startup."""
    graph = get_memory_graph_fn()
    if graph is None:
        abort_fn(503, description="FalkorDB is unavailable")

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

        if lesson_results.result_set:
            for row in lesson_results.result_set:
                lessons.append(
                    {
                        "id": row[0],
                        "content": row[1],
                        "tags": row[2] if row[2] else [],
                        "importance": row[3] if row[3] else 0.5,
                        "type": row[4] if row[4] else "Context",
                        "metadata": json.loads(row[5]) if row[5] else {},
                    }
                )

        system_query = """
            MATCH (m:Memory)
            WHERE 'system' IN m.tags OR 'memory-recall' IN m.tags
            RETURN m.id as id, m.content as content, m.tags as tags
            LIMIT 5
        """

        system_results = graph.query(system_query)
        system_rules = []

        if system_results.result_set:
            for row in system_results.result_set:
                system_rules.append(
                    {"id": row[0], "content": row[1], "tags": row[2] if row[2] else []}
                )

        response = {
            "status": "success",
            "critical_lessons": lessons,
            "system_rules": system_rules,
            "lesson_count": len(lessons),
            "has_critical": any(l.get("importance", 0) >= 0.9 for l in lessons),
            "summary": f"Recalled {len(lessons)} lesson(s) and {len(system_rules)} system rule(s)",
        }

        return jsonify_fn(response), 200

    except Exception as e:
        logger.error(f"Startup recall failed: {e}")
        return jsonify_fn({"error": "Startup recall failed", "details": str(e)}), 500


def analyze_memories(
    *,
    get_memory_graph_fn: Callable[[], Any],
    extract_entities_fn: Callable[[str], Dict[str, Any]],
    utc_now_fn: Callable[[], str],
    perf_counter_fn: Callable[[], float],
    jsonify_fn: Callable[[Any], Any],
    abort_fn: Callable[..., Any],
    logger: Any,
) -> Any:
    """Analyze memory patterns, preferences, and insights."""
    query_start = perf_counter_fn()
    graph = get_memory_graph_fn()
    if graph is None:
        abort_fn(503, description="FalkorDB is unavailable")

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

        for mem_type, count, avg_conf in type_result.result_set:
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

        for p_type, content, confidence, observations in pattern_result.result_set:
            analytics["patterns"].append(
                {
                    "type": p_type,
                    "description": content,
                    "confidence": round(confidence, 3) if confidence else 0,
                    "observations": observations or 0,
                }
            )

        pref_result = graph.query(
            """
            MATCH (m1:Memory)-[r:PREFERS_OVER]->(m2:Memory)
            RETURN m1.content, m2.content, r.context, r.strength
            ORDER BY r.strength DESC
            LIMIT 10
            """
        )

        for preferred, over, context, strength in pref_result.result_set:
            analytics["preferences"].append(
                {
                    "prefers": preferred,
                    "over": over,
                    "context": context,
                    "strength": round(strength, 3) if strength else 0,
                }
            )

        try:
            temporal_result = graph.query(
                """
                MATCH (m:Memory)
                WHERE m.timestamp IS NOT NULL
                RETURN m.timestamp, m.importance
                LIMIT 100
                """
            )

            hour_data = defaultdict(lambda: {"count": 0, "total_importance": 0})

            for timestamp, importance in temporal_result.result_set:
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
                        "avg_importance": round(data["total_importance"] / data["count"], 3),
                    }
        except Exception:
            pass

        conf_result = graph.query(
            """
            MATCH (m:Memory)
            WHERE m.confidence IS NOT NULL
            RETURN
                CASE
                    WHEN m.confidence < 0.3 THEN 'low'
                    WHEN m.confidence < 0.7 THEN 'medium'
                    ELSE 'high'
                END as level,
                COUNT(m) as count
            """
        )

        for level, count in conf_result.result_set:
            analytics["confidence_distribution"][level] = count

        entity_result = graph.query(
            """
            MATCH (m:Memory)
            WHERE m.content IS NOT NULL
            RETURN m.content
            LIMIT 100
            """
        )

        entity_counts: Dict[str, Dict[str, int]] = {
            "tools": {},
            "projects": {},
        }

        for (content,) in entity_result.result_set:
            entities = extract_entities_fn(content)
            for tool in entities.get("tools", []):
                entity_counts["tools"][tool] = entity_counts["tools"].get(tool, 0) + 1
            for project in entities.get("projects", []):
                entity_counts["projects"][project] = entity_counts["projects"].get(project, 0) + 1

        analytics["entity_frequency"]["tools"] = sorted(
            entity_counts["tools"].items(), key=lambda x: x[1], reverse=True
        )[:5]
        analytics["entity_frequency"]["projects"] = sorted(
            entity_counts["projects"].items(), key=lambda x: x[1], reverse=True
        )[:5]

    except Exception:
        logger.exception("Failed to generate analytics")
        abort_fn(500, description="Failed to generate analytics")

    return jsonify_fn(
        {
            "status": "success",
            "analytics": analytics,
            "generated_at": utc_now_fn(),
            "query_time_ms": round((perf_counter_fn() - query_start) * 1000, 2),
        }
    )
