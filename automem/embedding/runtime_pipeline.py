from __future__ import annotations

import time
from typing import Any, Callable, List, Tuple

from automem.stores.vector_payload import build_qdrant_payload


class EmbeddingContentChanged(RuntimeError):
    """The graph changed after this job's vector source text was captured."""

    def __init__(self, memory_id: str, current_content: str) -> None:
        super().__init__(f"Memory {memory_id} content changed before vector storage")
        self.current_content = current_content


def init_embedding_pipeline(
    *,
    state: Any,
    logger: Any,
    queue_cls: Any,
    thread_cls: Any,
    worker_target: Callable[[], None],
) -> None:
    """Initialize the background embedding generation pipeline."""
    if state.embedding_queue is not None:
        return

    state.embedding_queue = queue_cls()
    state.embedding_thread = thread_cls(target=worker_target, daemon=True)
    state.embedding_thread.start()
    logger.info("Embedding pipeline initialized")


def enqueue_embedding(*, state: Any, memory_id: str, content: str) -> None:
    """Queue a memory for async embedding generation."""
    if not memory_id or not content or state.embedding_queue is None:
        return

    with state.embedding_lock:
        if memory_id in state.embedding_pending or memory_id in state.embedding_inflight:
            return

        state.embedding_pending.add(memory_id)
        state.embedding_queue.put((memory_id, content))


def embedding_worker(
    *,
    state: Any,
    logger: Any,
    batch_size: int,
    batch_timeout_seconds: float,
    empty_exc: Any,
    process_batch_fn: Callable[[List[Tuple[str, str]]], None],
    sleep_fn: Callable[[float], None],
    time_fn: Callable[[], float],
) -> None:
    """Background worker that generates embeddings and stores them in Qdrant with batching."""
    batch: List[Tuple[str, str]] = []
    batch_deadline = time_fn() + batch_timeout_seconds

    while True:
        try:
            if state.embedding_queue is None:
                sleep_fn(1)
                continue

            timeout = max(0.1, batch_deadline - time_fn())

            try:
                memory_id, content = state.embedding_queue.get(timeout=timeout)
                batch.append((memory_id, content))

                if len(batch) >= batch_size:
                    process_batch_fn(batch)
                    batch = []
                    batch_deadline = time_fn() + batch_timeout_seconds
            except empty_exc:
                if batch:
                    process_batch_fn(batch)
                    batch = []
                batch_deadline = time_fn() + batch_timeout_seconds
                continue
        except Exception:
            logger.exception("Error in embedding worker loop")
            if batch:
                try:
                    process_batch_fn(batch)
                except Exception:
                    logger.exception("Failed to process batch during error recovery")
                batch = []
            sleep_fn(1)
            batch_deadline = time_fn() + batch_timeout_seconds


def process_embedding_batch(
    *,
    state: Any,
    batch: List[Tuple[str, str]],
    logger: Any,
    generate_real_embeddings_batch_fn: Callable[[List[str]], List[List[float]]],
    store_embedding_in_qdrant_fn: Callable[[str, str, List[float]], Any],
    max_attempts: int = 3,
    retry_backoff_seconds: float = 1.0,
    sleep_fn: Callable[[float], None] | None = None,
) -> None:
    """Retry failed items only, retaining ownership until success or exhaustion.

    Generated vectors are cached during storage retries. Queue task accounting and
    duplicate suppression cover all attempts, including backoff.
    """
    if not batch:
        return
    sleep = sleep_fn or time.sleep
    remaining = list(batch)
    vectors: dict[str, List[float]] = {}
    with state.embedding_lock:
        for memory_id, _ in batch:
            state.embedding_pending.discard(memory_id)
            state.embedding_inflight.add(memory_id)

    def count(key: str, amount: int = 1, error: Exception | None = None) -> None:
        with state.embedding_lock:
            state.embedding_stats[key] += amount
            if error is not None:
                state.embedding_stats["last_error"] = str(error)

    try:
        for attempt in range(max(1, max_attempts)):
            if attempt:
                count("retries", len(remaining))
                sleep(max(0, retry_backoff_seconds) * (2 ** (attempt - 1)))
            need_vectors = [(mid, content) for mid, content in remaining if mid not in vectors]
            if need_vectors:
                try:
                    generated = generate_real_embeddings_batch_fn([c for _, c in need_vectors])
                    if len(generated) != len(need_vectors):
                        raise ValueError("Embedding provider returned an unexpected vector count")
                    vectors.update(
                        (mid, vector) for (mid, _), vector in zip(need_vectors, generated)
                    )
                except Exception as exc:
                    count("failures", len(need_vectors), exc)
                    logger.exception(
                        "Failed to generate batch embeddings (attempt %d)", attempt + 1
                    )
                    continue
            failed = []
            for memory_id, content in remaining:
                try:
                    stored = store_embedding_in_qdrant_fn(memory_id, content, vectors[memory_id])
                    count("skipped" if stored is False else "successes")
                except EmbeddingContentChanged as exc:
                    # The cached vector belongs to an older text version. Do not
                    # overwrite a completed PATCH with that vector and new payload.
                    count("failures", error=exc)
                    vectors.pop(memory_id, None)
                    failed.append((memory_id, exc.current_content))
                    logger.info("Refreshing stale embedding job for %s", memory_id)
                except Exception as exc:
                    count("failures", error=exc)
                    failed.append((memory_id, content))
                    logger.exception(
                        "Failed to store embedding for %s (attempt %d)", memory_id, attempt + 1
                    )
            remaining = failed
            if not remaining:
                break
        if remaining:
            count("exhausted", len(remaining))
            logger.error("Embedding retries exhausted for %s", [mid for mid, _ in remaining])
    finally:
        with state.embedding_lock:
            for memory_id, _ in batch:
                state.embedding_inflight.discard(memory_id)
        for _ in batch:
            state.embedding_queue.task_done()


def store_embedding_in_qdrant(
    *,
    memory_id: str,
    content: str,
    embedding: List[float],
    get_qdrant_client_fn: Callable[[], Any],
    get_memory_graph_fn: Callable[[], Any],
    collection_name: str,
    point_struct_cls: Any,
    utc_now_fn: Callable[[], str],
    logger: Any,
) -> bool:
    """Store a pre-generated embedding in Qdrant with memory metadata."""
    qdrant_client = get_qdrant_client_fn()
    if qdrant_client is None:
        raise RuntimeError("Qdrant is unavailable")

    graph = get_memory_graph_fn()
    if graph is None:
        raise RuntimeError("FalkorDB is unavailable")

    result = graph.query("MATCH (m:Memory {id: $id}) RETURN m", {"id": memory_id})
    if not getattr(result, "result_set", None):
        logger.warning("Memory %s not found in FalkorDB, skipping Qdrant update", memory_id)
        return False

    node = result.result_set[0][0]
    properties = getattr(node, "properties", {})
    current_content = properties.get("content", content)
    if current_content != content:
        if not isinstance(current_content, str):
            raise RuntimeError(f"Memory {memory_id} has invalid content")
        raise EmbeddingContentChanged(memory_id, current_content)
    payload = build_qdrant_payload({"content": content, **properties})
    qdrant_client.upsert(
        collection_name=collection_name,
        points=[point_struct_cls(id=memory_id, vector=embedding, payload=payload)],
        wait=True,
    )
    # Wait for application before re-reading the source. A PATCH can commit after
    # the first read while Qdrant is applying this request, even if it was issued
    # earlier. Revalidate after our write so that race triggers a fresh retry.
    latest = graph.query("MATCH (m:Memory {id: $id}) RETURN m", {"id": memory_id})
    if getattr(latest, "result_set", None):
        latest_properties = getattr(latest.result_set[0][0], "properties", {})
        latest_content = latest_properties.get("content", content)
        if latest_content != content:
            if not isinstance(latest_content, str):
                raise RuntimeError(f"Memory {memory_id} has invalid content")
            raise EmbeddingContentChanged(memory_id, latest_content)
    logger.info("Stored embedding for %s in Qdrant", memory_id)
    return True


def generate_and_store_embedding(
    *,
    memory_id: str,
    content: str,
    generate_real_embedding_fn: Callable[[str], List[float]],
    store_embedding_in_qdrant_fn: Callable[[str, str, List[float]], None],
) -> None:
    """Generate embedding for content and store in Qdrant (legacy single-item API)."""
    embedding = generate_real_embedding_fn(content)
    store_embedding_in_qdrant_fn(memory_id, content, embedding)
