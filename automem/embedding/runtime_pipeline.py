from __future__ import annotations

import json
from typing import Any, Callable, List, Tuple


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
    store_embedding_in_qdrant_fn: Callable[[str, str, List[float]], None],
) -> None:
    """Process a batch of embeddings efficiently."""
    if not batch:
        return

    memory_ids = [item[0] for item in batch]
    contents = [item[1] for item in batch]

    with state.embedding_lock:
        for memory_id in memory_ids:
            state.embedding_pending.discard(memory_id)
            state.embedding_inflight.add(memory_id)

    try:
        embeddings = generate_real_embeddings_batch_fn(contents)

        for memory_id, content, embedding in zip(memory_ids, contents, embeddings, strict=True):
            try:
                store_embedding_in_qdrant_fn(memory_id, content, embedding)
                logger.debug("Generated and stored embedding for %s", memory_id)
            except Exception:  # pragma: no cover
                logger.exception("Failed to store embedding for %s", memory_id)
    except Exception:  # pragma: no cover
        logger.exception("Failed to generate batch embeddings")
    finally:
        with state.embedding_lock:
            for memory_id in memory_ids:
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
) -> None:
    """Store a pre-generated embedding in Qdrant with memory metadata."""
    qdrant_client = get_qdrant_client_fn()
    if qdrant_client is None:
        return

    graph = get_memory_graph_fn()
    if graph is None:
        return

    result = graph.query("MATCH (m:Memory {id: $id}) RETURN m", {"id": memory_id})
    if not getattr(result, "result_set", None):
        logger.warning("Memory %s not found in FalkorDB, skipping Qdrant update", memory_id)
        return

    node = result.result_set[0][0]
    properties = getattr(node, "properties", {})
    try:
        metadata_payload = json.loads(properties.get("metadata", "{}"))
    except (json.JSONDecodeError, TypeError):
        logger.warning("Malformed metadata JSON for %s; defaulting to empty object", memory_id)
        metadata_payload = {}

    try:
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                point_struct_cls(
                    id=memory_id,
                    vector=embedding,
                    payload={
                        "content": properties.get("content", content),
                        "tags": properties.get("tags", []),
                        "tag_prefixes": properties.get("tag_prefixes", []),
                        "importance": properties.get("importance", 0.5),
                        "timestamp": properties.get("timestamp", utc_now_fn()),
                        "type": properties.get("type", "Context"),
                        "confidence": properties.get("confidence", 0.5),
                        "updated_at": properties.get("updated_at", utc_now_fn()),
                        "last_accessed": properties.get("last_accessed", utc_now_fn()),
                        "metadata": metadata_payload,
                        "relevance_score": properties.get("relevance_score"),
                    },
                )
            ],
        )
        logger.info("Stored embedding for %s in Qdrant", memory_id)
    except Exception:  # pragma: no cover
        logger.exception("Qdrant upsert failed for %s", memory_id)


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
