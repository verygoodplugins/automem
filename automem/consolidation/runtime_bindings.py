from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Set

from automem.consolidation.runtime_helpers import (
    apply_scheduler_overrides as _apply_scheduler_overrides_runtime,
)
from automem.consolidation.runtime_helpers import (
    build_consolidator_from_config as _build_consolidator_from_config_runtime,
)
from automem.consolidation.runtime_helpers import (
    load_control_record as _load_control_record_runtime,
)
from automem.consolidation.runtime_helpers import load_recent_runs as _load_recent_runs_runtime
from automem.consolidation.runtime_helpers import (
    persist_consolidation_run as _persist_consolidation_run_runtime,
)
from automem.consolidation.runtime_scheduler import (
    consolidation_worker as _consolidation_worker_runtime,
)
from automem.consolidation.runtime_scheduler import (
    init_consolidation_scheduler as _init_consolidation_scheduler_runtime,
)
from automem.consolidation.runtime_scheduler import (
    run_consolidation_tick as _run_consolidation_tick_runtime,
)
from consolidation import ConsolidationScheduler, MemoryConsolidator


@dataclass(frozen=True)
class ConsolidationRuntimeBindings:
    load_recent_runs: Callable[[Any, int], List[Dict[str, Any]]]
    persist_consolidation_run: Callable[[Any, Dict[str, Any]], None]
    build_consolidator_from_config: Callable[[Any, Any], MemoryConsolidator]
    build_scheduler_from_graph: Callable[[Any], ConsolidationScheduler]
    run_consolidation_tick: Callable[[], None]
    consolidation_worker: Callable[[], None]
    init_consolidation_scheduler: Callable[[], None]


def create_consolidation_runtime(
    *,
    state: Any,
    logger: Any,
    get_memory_graph_fn: Callable[[], Any],
    get_qdrant_client_fn: Callable[[], Any],
    emit_event_fn: Callable[[str, Dict[str, Any], Callable[[], str]], None],
    utc_now_fn: Callable[[], str],
    perf_counter_fn: Callable[[], float],
    parse_iso_datetime_fn: Callable[[Any], Any],
    stop_event_cls: Any,
    thread_cls: Any,
    run_label: str,
    control_label: str,
    control_node_id: str,
    task_fields: Dict[str, str],
    history_limit: int,
    tick_seconds: int,
    decay_interval_seconds: int,
    creative_interval_seconds: int,
    cluster_interval_seconds: int,
    forget_interval_seconds: int,
    delete_threshold: float,
    archive_threshold: float,
    grace_period_days: int,
    importance_protection_threshold: float,
    protected_types: Set[str],
    decay_importance_threshold: float,
) -> ConsolidationRuntimeBindings:
    def _load_control_record(graph: Any) -> Dict[str, Any]:
        return _load_control_record_runtime(
            graph,
            logger=logger,
            control_label=control_label,
            control_node_id=control_node_id,
            task_fields=task_fields,
            utc_now_fn=utc_now_fn,
        )

    def load_recent_runs(graph: Any, limit: int) -> List[Dict[str, Any]]:
        return _load_recent_runs_runtime(
            graph,
            limit,
            logger=logger,
            run_label=run_label,
        )

    def _apply_scheduler_overrides(scheduler: ConsolidationScheduler) -> None:
        _apply_scheduler_overrides_runtime(
            scheduler,
            decay_interval_seconds=decay_interval_seconds,
            creative_interval_seconds=creative_interval_seconds,
            cluster_interval_seconds=cluster_interval_seconds,
            forget_interval_seconds=forget_interval_seconds,
        )

    def persist_consolidation_run(graph: Any, result: Dict[str, Any]) -> None:
        _persist_consolidation_run_runtime(
            graph,
            result,
            logger=logger,
            run_label=run_label,
            control_label=control_label,
            control_node_id=control_node_id,
            task_fields=task_fields,
            history_limit=history_limit,
            utc_now_fn=utc_now_fn,
        )

    def build_consolidator_from_config(graph: Any, vector_store: Any) -> MemoryConsolidator:
        return _build_consolidator_from_config_runtime(
            graph,
            vector_store,
            memory_consolidator_cls=MemoryConsolidator,
            delete_threshold=delete_threshold,
            archive_threshold=archive_threshold,
            grace_period_days=grace_period_days,
            importance_protection_threshold=importance_protection_threshold,
            protected_types=set(protected_types),
        )

    def build_scheduler_from_graph(graph: Any) -> ConsolidationScheduler:
        vector_store = get_qdrant_client_fn()
        consolidator = build_consolidator_from_config(graph, vector_store)
        scheduler = ConsolidationScheduler(consolidator)
        _apply_scheduler_overrides(scheduler)

        control = _load_control_record(graph)
        for task, field in task_fields.items():
            iso_value = control.get(field)
            last_run = parse_iso_datetime_fn(iso_value)
            if last_run and task in scheduler.schedules:
                scheduler.schedules[task]["last_run"] = last_run

        return scheduler

    def run_consolidation_tick() -> None:
        _run_consolidation_tick_runtime(
            get_memory_graph_fn=get_memory_graph_fn,
            build_scheduler_from_graph_fn=build_scheduler_from_graph,
            persist_consolidation_run_fn=persist_consolidation_run,
            decay_importance_threshold=decay_importance_threshold,
            emit_event_fn=emit_event_fn,
            utc_now_fn=utc_now_fn,
            perf_counter_fn=perf_counter_fn,
            logger=logger,
        )

    def consolidation_worker() -> None:
        _consolidation_worker_runtime(
            state=state,
            logger=logger,
            consolidation_tick_seconds=tick_seconds,
            run_consolidation_tick_fn=run_consolidation_tick,
        )

    def init_consolidation_scheduler() -> None:
        _init_consolidation_scheduler_runtime(
            state=state,
            logger=logger,
            stop_event_cls=stop_event_cls,
            thread_cls=thread_cls,
            worker_target=consolidation_worker,
            run_consolidation_tick_fn=run_consolidation_tick,
        )

    return ConsolidationRuntimeBindings(
        load_recent_runs=load_recent_runs,
        persist_consolidation_run=persist_consolidation_run,
        build_consolidator_from_config=build_consolidator_from_config,
        build_scheduler_from_graph=build_scheduler_from_graph,
        run_consolidation_tick=run_consolidation_tick,
        consolidation_worker=consolidation_worker,
        init_consolidation_scheduler=init_consolidation_scheduler,
    )
