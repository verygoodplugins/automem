from __future__ import annotations

from typing import Any, Callable, Dict, List


def init_enrichment_pipeline(
    *,
    state: Any,
    logger: Any,
    queue_cls: Any,
    thread_cls: Any,
    worker_target: Callable[[], None],
) -> None:
    """Initialize the background enrichment pipeline."""
    if state.enrichment_queue is not None:
        return

    state.enrichment_queue = queue_cls()
    state.enrichment_thread = thread_cls(target=worker_target, daemon=True)
    state.enrichment_thread.start()
    logger.info("Enrichment pipeline initialized")


def enqueue_enrichment(
    *,
    state: Any,
    memory_id: str,
    forced: bool,
    attempt: int,
    enrichment_job_cls: Any,
) -> None:
    if not memory_id or state.enrichment_queue is None:
        return

    job = enrichment_job_cls(memory_id=memory_id, attempt=attempt, forced=forced)

    with state.enrichment_lock:
        if not forced and (
            memory_id in state.enrichment_pending or memory_id in state.enrichment_inflight
        ):
            return

        state.enrichment_pending.add(memory_id)
        state.enrichment_queue.put(job)


def update_last_accessed(
    *,
    memory_ids: List[str],
    get_memory_graph_fn: Callable[[], Any],
    utc_now_fn: Callable[[], str],
    logger: Any,
) -> None:
    """Update last_accessed timestamp for retrieved memories (direct, synchronous)."""
    if not memory_ids:
        return

    graph = get_memory_graph_fn()
    if graph is None:
        return

    timestamp = utc_now_fn()
    try:
        graph.query(
            """
            UNWIND $ids AS mid
            MATCH (m:Memory {id: mid})
            SET m.last_accessed = $ts
            """,
            {"ids": memory_ids, "ts": timestamp},
        )
        logger.debug("Updated last_accessed for %d memories", len(memory_ids))
    except Exception:
        logger.exception("Failed to update last_accessed for memories")


def enrichment_worker(
    *,
    state: Any,
    logger: Any,
    enrichment_idle_sleep_seconds: float,
    enrichment_max_attempts: int,
    enrichment_failure_backoff_seconds: float,
    empty_exc: Any,
    enrich_memory_fn: Callable[..., bool],
    emit_event_fn: Callable[[str, Dict[str, Any], Callable[[], str]], None],
    utc_now_fn: Callable[[], str],
    enqueue_enrichment_fn: Callable[..., None],
    perf_counter_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
) -> None:
    """Background worker that processes memories for enrichment."""
    while True:
        try:
            if state.enrichment_queue is None:
                sleep_fn(enrichment_idle_sleep_seconds)
                continue

            try:
                job = state.enrichment_queue.get(timeout=enrichment_idle_sleep_seconds)
            except empty_exc:
                continue

            with state.enrichment_lock:
                state.enrichment_pending.discard(job.memory_id)
                state.enrichment_inflight.add(job.memory_id)

            enrich_start = perf_counter_fn()
            emit_event_fn(
                "enrichment.start",
                {
                    "memory_id": job.memory_id,
                    "attempt": job.attempt + 1,
                },
                utc_now_fn,
            )

            try:
                processed = enrich_memory_fn(job.memory_id, forced=job.forced)
                state.enrichment_stats.record_success(job.memory_id)
                elapsed_ms = int((perf_counter_fn() - enrich_start) * 1000)
                emit_event_fn(
                    "enrichment.complete",
                    {
                        "memory_id": job.memory_id,
                        "success": True,
                        "elapsed_ms": elapsed_ms,
                        "skipped": not processed,
                    },
                    utc_now_fn,
                )
                if not processed:
                    logger.debug("Enrichment skipped for %s (already processed)", job.memory_id)
            except Exception as exc:  # pragma: no cover - background thread
                state.enrichment_stats.record_failure(str(exc))
                elapsed_ms = int((perf_counter_fn() - enrich_start) * 1000)
                emit_event_fn(
                    "enrichment.failed",
                    {
                        "memory_id": job.memory_id,
                        "error": str(exc)[:100],
                        "attempt": job.attempt + 1,
                        "elapsed_ms": elapsed_ms,
                        "will_retry": job.attempt + 1 < enrichment_max_attempts,
                    },
                    utc_now_fn,
                )
                logger.exception("Failed to enrich memory %s", job.memory_id)
                if job.attempt + 1 < enrichment_max_attempts:
                    sleep_fn(enrichment_failure_backoff_seconds)
                    enqueue_enrichment_fn(job.memory_id, forced=job.forced, attempt=job.attempt + 1)
                else:
                    logger.error(
                        "Giving up on enrichment for %s after %s attempts",
                        job.memory_id,
                        job.attempt + 1,
                    )
            finally:
                with state.enrichment_lock:
                    state.enrichment_inflight.discard(job.memory_id)
                state.enrichment_queue.task_done()
        except Exception:  # pragma: no cover - defensive catch-all
            logger.exception("Error in enrichment worker loop")
            sleep_fn(enrichment_failure_backoff_seconds)
