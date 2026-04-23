from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Tuple

from automem.embedding.runtime_pipeline import embedding_worker as _embedding_worker_runtime
from automem.embedding.runtime_pipeline import enqueue_embedding as _enqueue_embedding_runtime
from automem.embedding.runtime_pipeline import (
    generate_and_store_embedding as _generate_and_store_embedding_runtime,
)
from automem.embedding.runtime_pipeline import (
    init_embedding_pipeline as _init_embedding_pipeline_runtime,
)
from automem.embedding.runtime_pipeline import (
    process_embedding_batch as _process_embedding_batch_runtime,
)
from automem.embedding.runtime_pipeline import (
    store_embedding_in_qdrant as _store_embedding_in_qdrant_runtime,
)


@dataclass(frozen=True)
class EmbeddingRuntimeBindings:
    init_embedding_pipeline: Callable[[], None]
    enqueue_embedding: Callable[[str, str], None]
    embedding_worker: Callable[[], None]
    process_embedding_batch: Callable[[List[Tuple[str, str]]], None]
    store_embedding_in_qdrant: Callable[[str, str, List[float]], None]
    generate_and_store_embedding: Callable[[str, str], None]


def create_embedding_runtime(
    *,
    get_state_fn: Callable[[], Any],
    logger: Any,
    queue_cls: Any,
    thread_cls: Any,
    batch_size: int,
    batch_timeout_seconds: float,
    empty_exc: Any,
    sleep_fn: Callable[[float], None],
    time_fn: Callable[[], float],
    get_qdrant_client_fn: Callable[[], Any],
    get_memory_graph_fn: Callable[[], Any],
    collection_name: str,
    point_struct_cls: Any,
    utc_now_fn: Callable[[], str],
    generate_real_embedding_fn: Callable[[str], List[float]],
    generate_real_embeddings_batch_fn: Callable[[List[str]], List[List[float]]],
) -> EmbeddingRuntimeBindings:
    def init_embedding_pipeline() -> None:
        _init_embedding_pipeline_runtime(
            state=get_state_fn(),
            logger=logger,
            queue_cls=queue_cls,
            thread_cls=thread_cls,
            worker_target=embedding_worker,
        )

    def enqueue_embedding(memory_id: str, content: str) -> None:
        _enqueue_embedding_runtime(state=get_state_fn(), memory_id=memory_id, content=content)

    def store_embedding_in_qdrant(memory_id: str, content: str, embedding: List[float]) -> None:
        _store_embedding_in_qdrant_runtime(
            memory_id=memory_id,
            content=content,
            embedding=embedding,
            get_qdrant_client_fn=get_qdrant_client_fn,
            get_memory_graph_fn=get_memory_graph_fn,
            collection_name=collection_name,
            point_struct_cls=point_struct_cls,
            utc_now_fn=utc_now_fn,
            logger=logger,
        )

    def generate_and_store_embedding(memory_id: str, content: str) -> None:
        _generate_and_store_embedding_runtime(
            memory_id=memory_id,
            content=content,
            generate_real_embedding_fn=generate_real_embedding_fn,
            store_embedding_in_qdrant_fn=store_embedding_in_qdrant,
        )

    def process_embedding_batch(batch: List[Tuple[str, str]]) -> None:
        _process_embedding_batch_runtime(
            state=get_state_fn(),
            batch=batch,
            logger=logger,
            generate_real_embeddings_batch_fn=generate_real_embeddings_batch_fn,
            store_embedding_in_qdrant_fn=store_embedding_in_qdrant,
        )

    def embedding_worker() -> None:
        _embedding_worker_runtime(
            state=get_state_fn(),
            logger=logger,
            batch_size=batch_size,
            batch_timeout_seconds=batch_timeout_seconds,
            empty_exc=empty_exc,
            process_batch_fn=process_embedding_batch,
            sleep_fn=sleep_fn,
            time_fn=time_fn,
        )

    return EmbeddingRuntimeBindings(
        init_embedding_pipeline=init_embedding_pipeline,
        enqueue_embedding=enqueue_embedding,
        embedding_worker=embedding_worker,
        process_embedding_batch=process_embedding_batch,
        store_embedding_in_qdrant=store_embedding_in_qdrant,
        generate_and_store_embedding=generate_and_store_embedding,
    )
