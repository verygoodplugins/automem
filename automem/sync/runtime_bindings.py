from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from automem.sync.runtime_worker import init_sync_worker as _init_sync_worker_runtime
from automem.sync.runtime_worker import run_sync_check as _run_sync_check_runtime
from automem.sync.runtime_worker import sync_worker as _sync_worker_runtime


@dataclass(frozen=True)
class SyncRuntimeBindings:
    init_sync_worker: Callable[[], None]
    sync_worker: Callable[[], None]
    run_sync_check: Callable[[], None]


def create_sync_runtime(
    *,
    get_state_fn: Callable[[], Any],
    logger: Any,
    sync_auto_repair: bool,
    sync_check_interval_seconds: int,
    stop_event_cls: Any,
    thread_cls: Any,
    sleep_fn: Callable[[float], None],
    get_memory_graph_fn: Callable[[], Any],
    get_qdrant_client_fn: Callable[[], Any],
    collection_name: str,
    utc_now_fn: Callable[[], str],
    enqueue_embedding_fn: Callable[[str, str], None],
) -> SyncRuntimeBindings:
    def run_sync_check() -> None:
        _run_sync_check_runtime(
            state=get_state_fn(),
            logger=logger,
            get_memory_graph_fn=get_memory_graph_fn,
            get_qdrant_client_fn=get_qdrant_client_fn,
            collection_name=collection_name,
            utc_now_fn=utc_now_fn,
            enqueue_embedding_fn=enqueue_embedding_fn,
        )

    def sync_worker() -> None:
        _sync_worker_runtime(
            state=get_state_fn(),
            logger=logger,
            sync_check_interval_seconds=sync_check_interval_seconds,
            run_sync_check_fn=run_sync_check,
            sleep_fn=sleep_fn,
        )

    def init_sync_worker() -> None:
        _init_sync_worker_runtime(
            state=get_state_fn(),
            logger=logger,
            sync_auto_repair=sync_auto_repair,
            sync_check_interval_seconds=sync_check_interval_seconds,
            stop_event_cls=stop_event_cls,
            thread_cls=thread_cls,
            worker_target=sync_worker,
        )

    return SyncRuntimeBindings(
        init_sync_worker=init_sync_worker,
        sync_worker=sync_worker,
        run_sync_check=run_sync_check,
    )
