"""Behavioral regressions for background retries and complete vector payloads."""

import logging
from functools import partial
from queue import Queue
from threading import Lock
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from automem.embedding.runtime_pipeline import process_embedding_batch, store_embedding_in_qdrant
from automem.service_state import ServiceState

LOG = logging.getLogger(__name__)


def store(qdrant, properties):
    impl = partial(
        store_embedding_in_qdrant,
        get_qdrant_client_fn=lambda: qdrant,
        get_memory_graph_fn=lambda: SimpleNamespace(
            query=lambda *args: SimpleNamespace(
                result_set=[[SimpleNamespace(properties=properties)]]
            )
        ),
        collection_name="custom",
        point_struct_cls=lambda **kw: SimpleNamespace(**kw),
        utc_now_fn=lambda: "2026-10-20T00:00:00Z",
        logger=LOG,
    )
    return lambda memory_id, content, embedding: impl(
        memory_id=memory_id, content=content, embedding=embedding
    )


def run_batch(qdrant, properties, generate=None, **kwargs):
    state = ServiceState()
    state.embedding_queue = Queue()
    batch = [("a", "alpha"), ("b", "beta")]
    for job in batch:
        state.embedding_queue.put(job)
        state.embedding_pending.add(job[0])
        state.embedding_queue.get()
    generate = generate or Mock(return_value=[[1.0], [2.0]])
    process_embedding_batch(
        state=state,
        batch=batch,
        logger=LOG,
        generate_real_embeddings_batch_fn=generate,
        store_embedding_in_qdrant_fn=store(qdrant, properties),
        **kwargs
    )
    assert not state.embedding_inflight
    assert not state.embedding_pending
    assert state.embedding_queue.unfinished_tasks == 0
    return state, generate


def test_async_payload_preserves_state_and_hierarchical_tags():
    qdrant = Mock()
    properties = {
        "content": "alpha",
        "tags": ["auth/login"],
        "metadata": '{"source":"x"}',
        "t_valid": "2026-10-21T00:00:00Z",
        "t_invalid": "2026-10-22T00:00:00Z",
        "archived": True,
        "relevance_score": 0.4,
    }
    store(qdrant, properties)("a", "alpha", [1.0])
    payload = qdrant.upsert.call_args.kwargs["points"][0].payload
    assert payload["t_valid"] == properties["t_valid"]
    assert payload["t_invalid"] == properties["t_invalid"]
    assert payload["archived"] is True
    assert payload["tag_prefixes"] == ["auth", "auth:login"]
    assert payload["metadata"] == {"source": "x"}


def test_real_upsert_error_is_visible_to_worker():
    qdrant = Mock()
    qdrant.upsert.side_effect = RuntimeError("unavailable")
    with pytest.raises(RuntimeError, match="unavailable"):
        store(qdrant, {})("a", "alpha", [1.0])


def test_upsert_failure_retries_only_failed_item(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    qdrant = Mock()
    qdrant.upsert.side_effect = [RuntimeError("transient"), None, None]
    state, generate = run_batch(qdrant, {})
    assert [c.kwargs["points"][0].id for c in qdrant.upsert.call_args_list] == ["a", "b", "a"]
    assert generate.call_count == 1
    assert state.embedding_stats["successes"] == 2
    assert state.embedding_stats["failures"] == 1
    assert state.embedding_stats["retries"] == 1


def test_provider_failure_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    qdrant = Mock()
    generate = Mock(side_effect=[RuntimeError("rate limit"), [[1.0], [2.0]]])
    state, _ = run_batch(qdrant, {}, generate)
    assert generate.call_count == 2
    assert qdrant.upsert.call_count == 2
    assert state.embedding_stats["successes"] == 2


def test_permanent_failure_is_bounded_and_counted(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    qdrant = Mock()
    qdrant.upsert.side_effect = RuntimeError("unavailable")
    state, _ = run_batch(qdrant, {})
    assert qdrant.upsert.call_count == 6
    assert state.embedding_stats["exhausted"] == 2
    assert state.embedding_stats["failures"] == 6


@pytest.mark.parametrize("endpoint", ["/admin/reembed", "/admin/sync"])
def test_admin_writers_preserve_full_graph_state(endpoint, monkeypatch):
    from flask import Flask

    from automem.api import admin

    row = [
        "00000000-0000-0000-0000-000000000001",
        "alpha",
        ["auth/login"],
        0.8,
        "2026-10-20T00:00:00Z",
        "Insight",
        0.9,
        '{"source":"x"}',
        None,
        None,
        "2026-10-21T00:00:00Z",
        "2026-10-22T00:00:00Z",
        True,
        0.4,
    ]
    graph = Mock()
    graph.query.return_value = SimpleNamespace(result_set=[row])
    qdrant = Mock()
    monkeypatch.setattr(admin, "fetch_falkor_memory_ids", lambda *a, **kw: {row[0]})
    monkeypatch.setattr(admin, "_get_all_qdrant_ids", lambda *a, **kw: set())
    app = Flask(__name__)
    app.register_blueprint(
        admin.create_admin_blueprint_full(
            require_admin_token=lambda: None,
            get_qdrant_client=lambda: qdrant,
            get_memory_graph=lambda: graph,
            point_struct=lambda **kw: SimpleNamespace(**kw),
            collection_name="custom",
            generate_real_embeddings_batch=lambda *a, **kw: [[1.0]],
            utc_now=lambda: row[4],
            logger=LOG,
        )
    )
    response = app.test_client().post(endpoint, json={"force": True})
    assert response.status_code == 200, response.json
    payload = qdrant.upsert.call_args.kwargs["points"][0].payload
    assert payload["t_valid"] == row[10]
    assert payload["t_invalid"] == row[11]
    assert payload["archived"] is True
    assert payload["relevance_score"] == 0.4
    assert payload["tag_prefixes"] == ["auth", "auth:login"]
    assert payload["metadata"] == {"source": "x"}
    query = graph.query.call_args.args[0]
    assert "m.archived AS archived" in query
    assert "m.t_invalid AS t_invalid" in query


def test_reembedding_script_preserves_graph_state(monkeypatch):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/reembed_embeddings.py"
    spec = importlib.util.spec_from_file_location("reembed_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    row = [
        "00000000-0000-0000-0000-000000000001",
        "alpha",
        ["auth/login"],
        0.8,
        "2026-10-20T00:00:00Z",
        "Insight",
        0.9,
        '{"source":"x"}',
        None,
        None,
        "2026-10-21T00:00:00Z",
        "2026-10-22T00:00:00Z",
        True,
        0.4,
    ]
    graph = Mock()
    graph.query.return_value = SimpleNamespace(result_set=[row])
    memories = module.fetch_memories(graph)
    qdrant = Mock()
    monkeypatch.setattr(module, "get_qdrant_client", lambda: qdrant)
    provider = Mock()
    provider.generate_embeddings_batch.return_value = [[1.0]]
    module.reembed_memories(memories, 32, provider)
    payload = qdrant.upsert.call_args.kwargs["points"][0].payload
    assert payload["t_valid"] == row[10]
    assert payload["t_invalid"] == row[11]
    assert payload["archived"] is True
    assert payload["relevance_score"] == 0.4
    assert payload["tag_prefixes"] == ["auth", "auth:login"]


def test_deleted_memory_is_skipped_without_retry():
    impl = partial(
        store_embedding_in_qdrant,
        get_qdrant_client_fn=lambda: Mock(),
        get_memory_graph_fn=lambda: SimpleNamespace(
            query=lambda *a: SimpleNamespace(result_set=[])
        ),
        collection_name="custom",
        point_struct_cls=Mock(),
        utc_now_fn=lambda: "now",
        logger=LOG,
    )
    assert impl(memory_id="gone", content="deleted", embedding=[1.0]) is False


def test_payloads_respect_current_state_filter():
    from datetime import datetime, timezone

    from automem.api.recall import _state_reason_for_memory
    from automem.stores.vector_payload import build_qdrant_payload

    now = datetime(2026, 10, 20, tzinfo=timezone.utc)
    for properties, reason in [
        ({"archived": True}, "archived"),
        ({"t_valid": "2026-10-21T00:00:00Z"}, "not_yet_valid"),
        ({"t_invalid": "2026-10-19T00:00:00Z"}, "expired"),
    ]:
        assert _state_reason_for_memory(build_qdrant_payload(properties), now) == reason


def test_background_generation_accepts_configured_placeholder_but_not_failure_fallback():
    from automem.embedding.placeholder import PlaceholderEmbeddingProvider
    from automem.embedding.runtime_helpers import generate_real_embeddings_batch

    state = ServiceState()
    state.effective_vector_size = 4
    state.embedding_provider = PlaceholderEmbeddingProvider(dimension=4)
    kwargs = dict(
        init_embedding_provider=lambda: None,
        state=state,
        logger=LOG,
        placeholder_embedding=lambda _: pytest.fail("automatic fallback must not run"),
        allow_placeholder_fallback=False,
        allow_placeholder_provider=True,
    )
    assert len(generate_real_embeddings_batch(["alpha"], **kwargs)[0]) == 4
    provider = Mock()
    provider.provider_name.return_value = "real-provider"
    provider.generate_embeddings_batch.side_effect = RuntimeError("rate limited")
    provider.generate_embedding.side_effect = RuntimeError("rate limited")
    state.embedding_provider = provider
    with pytest.raises(RuntimeError):
        generate_real_embeddings_batch(["alpha"], **kwargs)


def test_retry_backoff_and_configured_attempt_limit():
    qdrant = Mock()
    qdrant.upsert.side_effect = RuntimeError("unavailable")
    delays = []
    state, generate = run_batch(
        qdrant, {}, max_attempts=4, retry_backoff_seconds=0.5, sleep_fn=delays.append
    )
    assert delays == [0.5, 1, 2]
    assert qdrant.upsert.call_count == 8
    assert generate.call_count == 1
    assert state.embedding_stats["exhausted"] == 2


def test_inflight_duplicate_enqueue_is_suppressed_during_backoff():
    from automem.embedding.runtime_pipeline import enqueue_embedding

    state = ServiceState()
    state.embedding_queue = Queue()
    enqueue_embedding(state=state, memory_id="a", content="alpha")
    batch = [state.embedding_queue.get()]
    attempts = Mock(side_effect=[RuntimeError("transient"), True])

    def backoff(_):
        assert state.embedding_inflight == {"a"}
        enqueue_embedding(state=state, memory_id="a", content="alpha")
        assert state.embedding_queue.empty()

    process_embedding_batch(
        state=state,
        batch=batch,
        logger=LOG,
        generate_real_embeddings_batch_fn=lambda _: [[1.0]],
        store_embedding_in_qdrant_fn=attempts,
        sleep_fn=backoff,
    )
    assert attempts.call_count == 2
    assert state.embedding_queue.unfinished_tasks == 0


def test_content_edit_during_backoff_never_overwrites_correct_patch_vector():
    """Use the real store helper: PATCH finishes while the old job is retrying."""
    properties = {}
    qdrant = Mock()
    point_store = {}
    successful_writes = []
    failures = 0

    def upsert(*, collection_name, points, wait):
        nonlocal failures
        point = points[0]
        if point.id == "a" and failures == 0:
            failures += 1
            raise RuntimeError("temporary Qdrant failure")
        successful_writes.append((point.id, point.payload["content"], point.vector))
        point_store[point.id] = {"content": point.payload["content"], "vector": point.vector}

    qdrant.upsert.side_effect = upsert
    patched = False

    def backoff(_):
        nonlocal patched
        if not patched:
            # Simulate PATCH's graph update and completed synchronous vector upsert.
            properties["content"] = "updated alpha"
            point_store["a"] = {"content": "updated alpha", "vector": [3.0]}
            patched = True
        assert point_store["a"] == {"content": "updated alpha", "vector": [3.0]}

    generate = Mock(side_effect=[[[1.0], [2.0]], [[3.0]]])
    state, _ = run_batch(qdrant, properties, generate, sleep_fn=backoff)
    assert successful_writes == [("b", "beta", [2.0]), ("a", "updated alpha", [3.0])]
    assert generate.call_args_list[1].args == (["updated alpha"],)
    assert state.embedding_stats["successes"] == 2
    assert point_store["a"] == {"content": "updated alpha", "vector": [3.0]}


def test_stale_queued_text_regenerates_from_current_graph_content():
    state = ServiceState()
    state.embedding_queue = Queue()
    batch = [("a", "old queued text")]
    state.embedding_queue.put(batch[0])
    state.embedding_queue.get()
    qdrant = Mock()
    generate = Mock(side_effect=[[[1.0]], [[2.0]]])
    process_embedding_batch(
        state=state,
        batch=batch,
        logger=LOG,
        generate_real_embeddings_batch_fn=generate,
        store_embedding_in_qdrant_fn=store(qdrant, {"content": "current graph text"}),
        sleep_fn=lambda _: None,
    )
    assert qdrant.upsert.call_count == 1
    point = qdrant.upsert.call_args.kwargs["points"][0]
    assert point.vector == [2.0]
    assert point.payload["content"] == "current graph text"
    assert generate.call_args_list[1].args == (["current graph text"],)
    assert state.embedding_stats["successes"] == 1
    assert state.embedding_queue.unfinished_tasks == 0


@pytest.mark.parametrize("max_attempts", [1, 3])
def test_patch_between_source_read_and_upsert_is_detected_and_repaired(max_attempts):
    """PATCH wins the graph race while the old Qdrant write is in progress."""
    state = ServiceState()
    state.embedding_queue = Queue()
    job = ("a", "alpha")
    state.embedding_queue.put(job)
    state.embedding_queue.get()
    current = {"content": "alpha"}
    stored = {}
    upserts = 0

    def read_graph(*args):
        # Return a snapshot, as FalkorDB does, rather than a mutable shared dict.
        return SimpleNamespace(result_set=[[SimpleNamespace(properties=dict(current))]])

    def upsert(*, collection_name, points, **kwargs):
        nonlocal upserts
        upserts += 1
        point = points[0]
        if upserts == 1:
            # PATCH commits both stores after the old read, then the delayed old
            # background upsert lands over the PATCH point.
            current["content"] = "new alpha"
            stored.update(content="new alpha", vector=[2.0])
        stored.update(content=point.payload["content"], vector=point.vector)

    qdrant = Mock()
    qdrant.upsert.side_effect = upsert
    graph = SimpleNamespace(query=read_graph)
    impl = partial(
        store_embedding_in_qdrant,
        get_qdrant_client_fn=lambda: qdrant,
        get_memory_graph_fn=lambda: graph,
        collection_name="custom",
        point_struct_cls=lambda **kw: SimpleNamespace(**kw),
        utc_now_fn=lambda: "now",
        logger=LOG,
    )
    process_embedding_batch(
        state=state,
        batch=[job],
        logger=LOG,
        generate_real_embeddings_batch_fn=Mock(side_effect=[[[1.0]], [[2.0]]]),
        store_embedding_in_qdrant_fn=lambda mid, text, vector: impl(
            memory_id=mid, content=text, embedding=vector
        ),
        sleep_fn=lambda _: None,
        max_attempts=max_attempts,
    )
    if max_attempts == 1:
        assert state.embedding_stats["successes"] == 0
        assert state.embedding_stats["exhausted"] == 1
    else:
        assert stored == {"content": "new alpha", "vector": [2.0]}
        assert upserts == 2
        assert state.embedding_stats["successes"] == 1
    assert state.embedding_queue.unfinished_tasks == 0
