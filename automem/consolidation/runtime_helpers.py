from __future__ import annotations

import json
import uuid
from datetime import timedelta
from typing import Any, Callable, Dict, List, Set


def load_control_record(
    graph: Any,
    *,
    logger: Any,
    control_label: str,
    control_node_id: str,
    task_fields: Dict[str, str],
    utc_now_fn: Callable[[], str],
) -> Dict[str, Any]:
    """Fetch or create the consolidation control node."""
    bootstrap_timestamp = utc_now_fn()
    bootstrap_fields = sorted(set(task_fields.values()))
    bootstrap_set_clause = ",\n                ".join(
        f"c.{field} = coalesce(c.{field}, $now)" for field in bootstrap_fields
    )
    try:
        result = graph.query(
            f"""
            MERGE (c:{control_label} {{id: $id}})
            SET {bootstrap_set_clause}
            RETURN c
            """,
            {"id": control_node_id, "now": bootstrap_timestamp},
        )
    except Exception:
        logger.exception("Failed to load consolidation control record")
        return {}

    if not getattr(result, "result_set", None):
        return {}

    node = result.result_set[0][0]
    properties = getattr(node, "properties", None)
    if isinstance(properties, dict):
        return dict(properties)
    if isinstance(node, dict):
        return dict(node)
    return {}


def load_recent_runs(
    graph: Any, limit: int, *, logger: Any, run_label: str
) -> List[Dict[str, Any]]:
    """Return recent consolidation run records."""
    try:
        result = graph.query(
            f"""
            MATCH (r:{run_label})
            RETURN r
            ORDER BY r.started_at DESC
            LIMIT $limit
            """,
            {"limit": limit},
        )
    except Exception:
        logger.exception("Failed to load consolidation history")
        return []

    runs: List[Dict[str, Any]] = []
    for row in getattr(result, "result_set", []) or []:
        node = row[0]
        properties = getattr(node, "properties", None)
        if isinstance(properties, dict):
            runs.append(dict(properties))
        elif isinstance(node, dict):
            runs.append(dict(node))
    return runs


def apply_scheduler_overrides(
    scheduler: Any,
    *,
    decay_interval_seconds: int,
    creative_interval_seconds: int,
    cluster_interval_seconds: int,
    forget_interval_seconds: int,
) -> None:
    """Override default scheduler intervals using configuration."""
    overrides = {
        "decay": timedelta(seconds=decay_interval_seconds),
        "creative": timedelta(seconds=creative_interval_seconds),
        "cluster": timedelta(seconds=cluster_interval_seconds),
        "forget": timedelta(seconds=forget_interval_seconds),
    }

    for task, interval in overrides.items():
        if task in scheduler.schedules:
            scheduler.schedules[task]["interval"] = interval


def tasks_for_mode(mode: str, task_fields: Dict[str, str]) -> List[str]:
    """Map a consolidation mode to its task identifiers."""
    if mode == "full":
        return ["decay", "creative", "cluster", "forget", "full"]
    if mode in task_fields:
        return [mode]
    return [mode]


def persist_consolidation_run(
    graph: Any,
    result: Dict[str, Any],
    *,
    logger: Any,
    run_label: str,
    control_label: str,
    control_node_id: str,
    task_fields: Dict[str, str],
    history_limit: int,
    utc_now_fn: Callable[[], str],
) -> None:
    """Record consolidation outcomes and update scheduler metadata."""
    mode = result.get("mode", "unknown")
    completed_at = result.get("completed_at") or utc_now_fn()
    started_at = result.get("started_at") or completed_at
    success = bool(result.get("success"))
    dry_run = bool(result.get("dry_run"))

    try:
        graph.query(
            f"""
            CREATE (r:{run_label} {{
                id: $id,
                mode: $mode,
                task: $task,
                success: $success,
                dry_run: $dry_run,
                started_at: $started_at,
                completed_at: $completed_at,
                result: $result
            }})
            """,
            {
                "id": uuid.uuid4().hex,
                "mode": mode,
                "task": mode,
                "success": success,
                "dry_run": dry_run,
                "started_at": started_at,
                "completed_at": completed_at,
                "result": json.dumps(result, default=str),
            },
        )
    except Exception:
        logger.exception("Failed to record consolidation run history")

    for task in tasks_for_mode(mode, task_fields):
        field = task_fields.get(task)
        if not field:
            continue
        try:
            graph.query(
                f"""
                MERGE (c:{control_label} {{id: $id}})
                SET c.{field} = $timestamp
                """,
                {
                    "id": control_node_id,
                    "timestamp": completed_at,
                },
            )
        except Exception:
            logger.exception("Failed to update consolidation control for task %s", task)

    try:
        graph.query(
            f"""
            MATCH (r:{run_label})
            WITH r ORDER BY r.started_at DESC
            SKIP $keep
            DELETE r
            """,
            {"keep": history_limit},
        )
    except Exception:
        logger.exception("Failed to prune consolidation history")


def build_consolidator_from_config(
    graph: Any,
    vector_store: Any,
    *,
    memory_consolidator_cls: Any,
    delete_threshold: float,
    archive_threshold: float,
    grace_period_days: int,
    importance_protection_threshold: float,
    protected_types: Set[str],
) -> Any:
    return memory_consolidator_cls(
        graph,
        vector_store,
        delete_threshold=delete_threshold,
        archive_threshold=archive_threshold,
        grace_period_days=grace_period_days,
        importance_protection_threshold=importance_protection_threshold,
        protected_types=protected_types,
    )
