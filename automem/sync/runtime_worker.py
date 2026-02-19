from __future__ import annotations

from typing import Any, Callable, Set


def init_sync_worker(
    *,
    state: Any,
    logger: Any,
    sync_auto_repair: bool,
    sync_check_interval_seconds: int,
    stop_event_cls: Any,
    thread_cls: Any,
    worker_target: Callable[[], None],
) -> None:
    """Initialize the background sync worker if auto-repair is enabled."""
    if not sync_auto_repair:
        logger.info("Sync auto-repair disabled (SYNC_AUTO_REPAIR=false)")
        return

    if state.sync_thread is not None:
        return

    state.sync_stop_event = stop_event_cls()
    state.sync_thread = thread_cls(target=worker_target, daemon=True)
    state.sync_thread.start()
    logger.info("Sync worker initialized (interval: %ds)", sync_check_interval_seconds)


def sync_worker(
    *,
    state: Any,
    logger: Any,
    sync_check_interval_seconds: int,
    run_sync_check_fn: Callable[[], None],
    sleep_fn: Callable[[float], None],
) -> None:
    """Background worker that detects and repairs FalkorDB/Qdrant sync drift."""
    while not state.sync_stop_event.is_set():
        try:
            if state.sync_stop_event.wait(timeout=sync_check_interval_seconds):
                break

            run_sync_check_fn()
        except Exception:
            logger.exception("Error in sync worker")
            sleep_fn(60)


def run_sync_check(
    *,
    state: Any,
    logger: Any,
    get_memory_graph_fn: Callable[[], Any],
    get_qdrant_client_fn: Callable[[], Any],
    collection_name: str,
    utc_now_fn: Callable[[], str],
    enqueue_embedding_fn: Callable[[str, str], None],
) -> None:
    """Check for sync drift and repair if needed."""
    graph = get_memory_graph_fn()
    qdrant = get_qdrant_client_fn()

    if graph is None or qdrant is None:
        logger.debug("Sync check skipped: services unavailable")
        return

    try:
        falkor_result = graph.query("MATCH (m:Memory) RETURN m.id AS id")
        falkor_ids: Set[str] = set()
        for row in getattr(falkor_result, "result_set", []) or []:
            if row[0]:
                falkor_ids.add(str(row[0]))

        qdrant_ids: Set[str] = set()
        offset = None
        while True:
            result = qdrant.scroll(
                collection_name=collection_name,
                limit=1000,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            points, next_offset = result
            for point in points:
                qdrant_ids.add(str(point.id))
            if next_offset is None:
                break
            offset = next_offset

        missing_ids = falkor_ids - qdrant_ids

        state.sync_last_run = utc_now_fn()
        state.sync_last_result = {
            "falkordb_count": len(falkor_ids),
            "qdrant_count": len(qdrant_ids),
            "missing_count": len(missing_ids),
        }

        if not missing_ids:
            logger.debug("Sync check: no drift detected (%d memories)", len(falkor_ids))
            return

        logger.warning(
            "Sync drift detected: %d memories missing from Qdrant (will auto-repair)",
            len(missing_ids),
        )

        for memory_id in missing_ids:
            mem_result = graph.query(
                "MATCH (m:Memory {id: $id}) RETURN m.content", {"id": memory_id}
            )
            if getattr(mem_result, "result_set", None):
                content = mem_result.result_set[0][0]
                if content:
                    enqueue_embedding_fn(memory_id, content)

        logger.info("Queued %d memories for sync repair", len(missing_ids))
    except Exception:
        logger.exception("Sync check failed")
