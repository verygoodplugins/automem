from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List

from automem.enrichment.runtime_worker import enqueue_enrichment as _enqueue_enrichment_runtime
from automem.enrichment.runtime_worker import enrichment_worker as _enrichment_worker_runtime
from automem.enrichment.runtime_worker import (
    init_enrichment_pipeline as _init_enrichment_pipeline_runtime,
)
from automem.enrichment.runtime_worker import update_last_accessed as _update_last_accessed_runtime


@dataclass(frozen=True)
class EnrichmentQueueRuntimeBindings:
    init_enrichment_pipeline: Callable[[], None]
    enqueue_enrichment: Callable[[str], None]
    update_last_accessed: Callable[[List[str]], None]
    enrichment_worker: Callable[[], None]


def create_enrichment_queue_runtime(
    *,
    get_state_fn: Callable[[], Any],
    logger: Any,
    queue_cls: Any,
    thread_cls: Any,
    enrichment_job_cls: Any,
    get_memory_graph_fn: Callable[[], Any],
    utc_now_fn: Callable[[], str],
    enrichment_idle_sleep_seconds: float,
    enrichment_max_attempts: int,
    enrichment_failure_backoff_seconds: float,
    empty_exc: Any,
    enrich_memory_fn: Callable[..., bool],
    emit_event_fn: Callable[[str, dict[str, Any], Callable[[], str]], None],
    perf_counter_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
) -> EnrichmentQueueRuntimeBindings:
    def init_enrichment_pipeline() -> None:
        _init_enrichment_pipeline_runtime(
            state=get_state_fn(),
            logger=logger,
            queue_cls=queue_cls,
            thread_cls=thread_cls,
            worker_target=enrichment_worker,
        )

    def enqueue_enrichment(memory_id: str, *, forced: bool = False, attempt: int = 0) -> None:
        _enqueue_enrichment_runtime(
            state=get_state_fn(),
            memory_id=memory_id,
            forced=forced,
            attempt=attempt,
            enrichment_job_cls=enrichment_job_cls,
        )

    def update_last_accessed(memory_ids: List[str]) -> None:
        _update_last_accessed_runtime(
            memory_ids=memory_ids,
            get_memory_graph_fn=get_memory_graph_fn,
            utc_now_fn=utc_now_fn,
            logger=logger,
        )

    def enrichment_worker() -> None:
        _enrichment_worker_runtime(
            state=get_state_fn(),
            logger=logger,
            enrichment_idle_sleep_seconds=enrichment_idle_sleep_seconds,
            enrichment_max_attempts=enrichment_max_attempts,
            enrichment_failure_backoff_seconds=enrichment_failure_backoff_seconds,
            empty_exc=empty_exc,
            enrich_memory_fn=enrich_memory_fn,
            emit_event_fn=emit_event_fn,
            utc_now_fn=utc_now_fn,
            enqueue_enrichment_fn=enqueue_enrichment,
            perf_counter_fn=perf_counter_fn,
            sleep_fn=sleep_fn,
        )

    return EnrichmentQueueRuntimeBindings(
        init_enrichment_pipeline=init_enrichment_pipeline,
        enqueue_enrichment=enqueue_enrichment,
        update_last_accessed=update_last_accessed,
        enrichment_worker=enrichment_worker,
    )
