from queue import Queue
from threading import Lock
from types import SimpleNamespace

import pytest
from werkzeug.exceptions import BadRequest, Forbidden

import app
from automem.embedding.runtime_pipeline import (
    enqueue_embedding,
    process_embedding_batch,
    store_embedding_in_qdrant,
)
from automem.enrichment.runtime_orchestration import enrich_memory, jit_enrich_lightweight
from automem.enrichment.runtime_worker import enqueue_enrichment
from automem.isolation import DEFAULT_ISOLATION_CONTEXT, IsolationContext, resolve_isolation_context
from tests.support.fake_graph import FakeGraph


class _TenantFalkor:
    def __init__(self, default_graph: FakeGraph) -> None:
        self.graphs = {"memories": default_graph}
        self.selected: list[str] = []

    def select_graph(self, name: str) -> FakeGraph:
        self.selected.append(name)
        if name not in self.graphs:
            self.graphs[name] = FakeGraph()
        return self.graphs[name]


class _Qdrant:
    def __init__(self) -> None:
        self.points: dict[str, dict[str, object]] = {}
        self.collections = {"memories"}
        self.created_collections: list[str] = []
        self.upsert_calls: list[tuple[str, list[object]]] = []
        self.delete_calls: list[tuple[str, object]] = []
        self.scroll_calls: list[dict[str, object]] = []
        self.set_payload_calls: list[dict[str, object]] = []

    def upsert(self, collection_name: str, points: list[object]) -> None:
        self.upsert_calls.append((collection_name, points))
        for point in points:
            self.points[str(point.id)] = {"payload": point.payload, "vector": point.vector}

    def delete(self, collection_name: str, points_selector: object) -> None:
        self.delete_calls.append((collection_name, points_selector))

    def get_collections(self) -> SimpleNamespace:
        return SimpleNamespace(
            collections=[SimpleNamespace(name=name) for name in sorted(self.collections)]
        )

    def create_collection(self, collection_name: str, **_kwargs) -> None:
        self.collections.add(collection_name)
        self.created_collections.append(collection_name)

    def create_payload_index(self, **_kwargs) -> None:
        return None

    def scroll(self, collection_name: str, **kwargs) -> tuple[list[SimpleNamespace], None]:
        self.scroll_calls.append({"collection": collection_name, **kwargs})
        return [
            SimpleNamespace(id=point_id, payload=point["payload"])
            for point_id, point in self.points.items()
        ], None

    def set_payload(
        self, collection_name: str, points: list[str], payload: dict[str, object]
    ) -> None:
        self.set_payload_calls.append(
            {"collection": collection_name, "points": points, "payload": payload}
        )


@pytest.fixture
def mock_state(monkeypatch):
    state = app.ServiceState()
    state.memory_graph = FakeGraph()
    state.falkordb = _TenantFalkor(state.memory_graph)
    state.qdrant = _Qdrant()
    state.openai_client = object()

    monkeypatch.setattr(app, "state", state)
    monkeypatch.setattr(app, "init_falkordb", lambda: None)
    monkeypatch.setattr(app, "init_qdrant", lambda: None)
    monkeypatch.setattr(app, "_generate_real_embedding", lambda _content: [0.1] * 768)
    monkeypatch.setattr(app, "API_TOKEN", "test-token")
    monkeypatch.setattr(app, "ADMIN_TOKEN", "test-admin-token")

    return state


@pytest.fixture
def client():
    app.app.config["TESTING"] = True
    with app.app.test_client() as test_client:
        yield test_client


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-token"}


@pytest.fixture
def admin_headers(auth_headers: dict[str, str]) -> dict[str, str]:
    return {**auth_headers, "X-Admin-Token": "test-admin-token"}


def _closure_value(function, name: str):
    for freevar, cell in zip(function.__code__.co_freevars, function.__closure__ or []):
        if freevar == name:
            return cell.cell_contents
    raise AssertionError(f"{name} not found in closure")


def test_resolve_isolation_context_defaults_without_request_context() -> None:
    assert (
        resolve_isolation_context(
            default_graph_name="memories",
            default_collection_name="memories",
        )
        == DEFAULT_ISOLATION_CONTEXT
    )


def test_resolve_isolation_context_rejects_single_header() -> None:
    request_obj = SimpleNamespace(headers={"X-Graph-Name": "tenant_a"})

    with pytest.raises(BadRequest):
        resolve_isolation_context(
            default_graph_name="memories",
            default_collection_name="memories",
            request_obj=request_obj,
        )


def test_resolve_isolation_context_requires_allowlists(monkeypatch) -> None:
    monkeypatch.delenv("ALLOWED_GRAPHS", raising=False)
    monkeypatch.delenv("ALLOWED_COLLECTIONS", raising=False)
    request_obj = SimpleNamespace(
        headers={"X-Graph-Name": "tenant_a", "X-Collection-Name": "tenant_a_vectors"}
    )

    with pytest.raises(Forbidden):
        resolve_isolation_context(
            default_graph_name="memories",
            default_collection_name="memories",
            request_obj=request_obj,
        )


def test_resolve_isolation_context_accepts_allowlisted_pair(monkeypatch) -> None:
    monkeypatch.setenv("ALLOWED_GRAPHS", "tenant_a, tenant_b")
    monkeypatch.setenv("ALLOWED_COLLECTIONS", "tenant_a_vectors, tenant_b_vectors")
    request_obj = SimpleNamespace(
        headers={"X-Graph-Name": "tenant_a", "X-Collection-Name": "tenant_a_vectors"}
    )

    assert resolve_isolation_context(
        default_graph_name="memories",
        default_collection_name="memories",
        request_obj=request_obj,
    ) == IsolationContext("tenant_a", "tenant_a_vectors", isolated=True)


def test_resolve_isolation_context_rejects_invalid_name(monkeypatch) -> None:
    monkeypatch.setenv("ALLOWED_GRAPHS", "tenant_a")
    monkeypatch.setenv("ALLOWED_COLLECTIONS", "tenant_a_vectors")
    request_obj = SimpleNamespace(
        headers={"X-Graph-Name": "tenant/a", "X-Collection-Name": "tenant_a_vectors"}
    )

    with pytest.raises(BadRequest):
        resolve_isolation_context(
            default_graph_name="memories",
            default_collection_name="memories",
            request_obj=request_obj,
        )


def test_store_with_single_isolation_header_returns_400(client, mock_state, auth_headers) -> None:
    response = client.post(
        "/memory",
        headers={**auth_headers, "X-Graph-Name": "tenant_a"},
        json={"content": "Tenant memory", "embedding": [0.1] * 1024},
    )

    assert response.status_code == 400


def test_store_with_allowlisted_headers_uses_tenant_graph_and_collection(
    client, mock_state, auth_headers, monkeypatch
) -> None:
    monkeypatch.setenv("ALLOWED_GRAPHS", "tenant_a")
    monkeypatch.setenv("ALLOWED_COLLECTIONS", "tenant_a_vectors")
    tenant_graph = mock_state.falkordb.select_graph("tenant_a")

    response = client.post(
        "/memory",
        headers={
            **auth_headers,
            "X-Graph-Name": "tenant_a",
            "X-Collection-Name": "tenant_a_vectors",
        },
        json={"content": "Tenant memory", "embedding": [0.1] * 1024},
    )

    assert response.status_code == 201
    assert len(tenant_graph.memories) == 1
    assert len(mock_state.memory_graph.memories) == 0
    assert "tenant_a_vectors" in mock_state.qdrant.created_collections
    assert mock_state.qdrant.upsert_calls[-1][0] == "tenant_a_vectors"


def test_get_memory_with_allowlisted_headers_reads_tenant_graph(
    client, mock_state, auth_headers, monkeypatch
) -> None:
    monkeypatch.setenv("ALLOWED_GRAPHS", "tenant_a")
    monkeypatch.setenv("ALLOWED_COLLECTIONS", "tenant_a_vectors")
    memory_id = "00000000-0000-4000-8000-000000000001"
    tenant_graph = mock_state.falkordb.select_graph("tenant_a")
    tenant_graph.memories[memory_id] = {
        "id": memory_id,
        "content": "Tenant-only memory",
        "tags": ["tenant"],
        "metadata": "{}",
        "timestamp": "2026-01-01T00:00:00+00:00",
    }

    response = client.get(
        f"/memory/{memory_id}",
        headers={
            **auth_headers,
            "X-Graph-Name": "tenant_a",
            "X-Collection-Name": "tenant_a_vectors",
        },
    )

    assert response.status_code == 200
    assert response.get_json()["memory"]["content"] == "Tenant-only memory"


def test_delete_by_tag_with_allowlisted_headers_deletes_tenant_collection(
    client, mock_state, auth_headers, monkeypatch
) -> None:
    monkeypatch.setenv("ALLOWED_GRAPHS", "tenant_a")
    monkeypatch.setenv("ALLOWED_COLLECTIONS", "tenant_a_vectors")
    memory_id = "00000000-0000-4000-8000-000000000002"
    tenant_graph = mock_state.falkordb.select_graph("tenant_a")
    tenant_graph.memories[memory_id] = {
        "id": memory_id,
        "content": "Tenant memory",
        "tags": ["tenant"],
        "metadata": "{}",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "importance": 0.7,
    }
    mock_state.memory_graph.memories["00000000-0000-4000-8000-000000000003"] = {
        "id": "00000000-0000-4000-8000-000000000003",
        "content": "Default memory",
        "tags": ["tenant"],
        "metadata": "{}",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "importance": 0.7,
    }

    response = client.delete(
        "/memory/by-tag?tags=tenant",
        headers={
            **auth_headers,
            "X-Graph-Name": "tenant_a",
            "X-Collection-Name": "tenant_a_vectors",
        },
    )

    assert response.status_code == 200
    assert memory_id not in tenant_graph.memories
    assert "00000000-0000-4000-8000-000000000003" in mock_state.memory_graph.memories
    assert mock_state.qdrant.delete_calls[-1][0] == "tenant_a_vectors"


def test_recall_tag_only_with_allowlisted_headers_uses_tenant_collection(
    client, mock_state, auth_headers, monkeypatch
) -> None:
    monkeypatch.setenv("ALLOWED_GRAPHS", "tenant_a")
    monkeypatch.setenv("ALLOWED_COLLECTIONS", "tenant_a_vectors")
    mock_state.qdrant.points["00000000-0000-4000-8000-000000000004"] = {
        "vector": [0.1] * 1024,
        "payload": {
            "content": "Tenant vector memory",
            "tags": ["tenant"],
            "tag_prefixes": ["tenant"],
            "importance": 0.9,
            "timestamp": "2026-01-01T00:00:00+00:00",
        },
    }

    response = client.get(
        "/recall?tags=tenant&limit=1",
        headers={
            **auth_headers,
            "X-Graph-Name": "tenant_a",
            "X-Collection-Name": "tenant_a_vectors",
        },
    )

    assert response.status_code == 200
    assert mock_state.qdrant.scroll_calls[-1]["collection"] == "tenant_a_vectors"


def test_embedding_queue_preserves_isolation_context() -> None:
    state = SimpleNamespace(
        embedding_queue=Queue(),
        embedding_lock=Lock(),
        embedding_pending=set(),
        embedding_inflight=set(),
    )
    context = IsolationContext("tenant_a", "tenant_a_vectors", isolated=True)
    memory_id = "00000000-0000-4000-8000-000000000005"

    enqueue_embedding(
        state=state,
        memory_id=memory_id,
        content="Tenant queued memory",
        isolation_context=context,
    )
    queued = state.embedding_queue.get_nowait()
    seen_contexts = []

    process_embedding_batch(
        state=state,
        batch=[queued],
        logger=SimpleNamespace(debug=lambda *_args, **_kwargs: None, exception=print),
        generate_real_embeddings_batch_fn=lambda _contents: [[0.1] * 1024],
        store_embedding_in_qdrant_fn=lambda *_args, **kwargs: seen_contexts.append(
            kwargs.get("isolation_context")
        ),
    )

    assert seen_contexts == [context]


def test_async_embedding_store_ensures_tenant_collection() -> None:
    qdrant = _Qdrant()
    graph = FakeGraph()
    context = IsolationContext("tenant_a", "tenant_a_vectors", isolated=True)
    memory_id = "00000000-0000-4000-8000-000000000007"
    graph.memories[memory_id] = {
        "id": memory_id,
        "content": "Tenant async memory",
        "tags": ["tenant"],
        "metadata": "{}",
        "timestamp": "2026-01-01T00:00:00+00:00",
    }
    ensured = []

    store_embedding_in_qdrant(
        memory_id=memory_id,
        content="Tenant async memory",
        embedding=[0.1] * 1024,
        get_qdrant_client_fn=lambda: qdrant,
        get_memory_graph_fn=lambda isolation_context=None: graph,
        collection_name="memories",
        point_struct_cls=lambda id, vector, payload: SimpleNamespace(
            id=id,
            vector=vector,
            payload=payload,
        ),
        utc_now_fn=lambda: "2026-01-01T00:00:00+00:00",
        logger=SimpleNamespace(warning=lambda *_args, **_kwargs: None, exception=print, info=print),
        isolation_context=context,
        ensure_qdrant_collection_fn=ensured.append,
    )

    assert ensured == ["tenant_a_vectors"]
    assert qdrant.upsert_calls[-1][0] == "tenant_a_vectors"


def test_enrichment_queue_preserves_isolation_context() -> None:
    state = SimpleNamespace(
        enrichment_queue=Queue(),
        enrichment_lock=Lock(),
        enrichment_pending=set(),
        enrichment_inflight=set(),
    )
    context = IsolationContext("tenant_a", "tenant_a_vectors", isolated=True)
    memory_id = "00000000-0000-4000-8000-000000000006"

    enqueue_enrichment(
        state=state,
        memory_id=memory_id,
        forced=True,
        attempt=1,
        enrichment_job_cls=app.EnrichmentJob,
        isolation_context=context,
    )

    job = state.enrichment_queue.get_nowait()
    assert job.memory_id == memory_id
    assert job.isolation_context == context


def test_app_enrichment_worker_callback_forwards_isolation_context(monkeypatch) -> None:
    context = IsolationContext("tenant_a", "tenant_a_vectors", isolated=True)
    memory_id = "00000000-0000-4000-8000-000000000009"
    seen = {}

    def fake_enrich(memory_id_arg, *, forced=False, isolation_context=None):
        seen["memory_id"] = memory_id_arg
        seen["forced"] = forced
        seen["isolation_context"] = isolation_context
        return True

    monkeypatch.setattr(app, "enrich_memory", fake_enrich)
    callback = _closure_value(app.enrichment_worker, "enrich_memory_fn")

    assert callback(memory_id, forced=True, isolation_context=context) is True
    assert seen == {
        "memory_id": memory_id,
        "forced": True,
        "isolation_context": context,
    }


def test_jit_enrichment_uses_tenant_collection_for_payload_sync() -> None:
    qdrant = _Qdrant()
    graph = FakeGraph()
    context = IsolationContext("tenant_a", "tenant_a_vectors", isolated=True)
    memory_id = "00000000-0000-4000-8000-000000000008"
    graph.memories[memory_id] = {
        "id": memory_id,
        "content": "Met Alice at Launchpad.",
        "tags": ["meeting"],
        "metadata": "{}",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "processed": False,
        "enriched": False,
    }

    updated = jit_enrich_lightweight(
        memory_id=memory_id,
        properties=graph.memories[memory_id],
        get_memory_graph_fn=lambda isolation_context=None: graph,
        get_qdrant_client_fn=lambda: qdrant,
        parse_metadata_field_fn=lambda raw: {},
        normalize_tag_list_fn=lambda tags: list(tags or []),
        extract_entities_fn=lambda _content: {"people": {"Alice"}},
        slugify_fn=lambda value: str(value).lower(),
        compute_tag_prefixes_fn=lambda tags: tags,
        enrichment_enable_summaries=False,
        generate_summary_fn=lambda content, _existing=None: content,
        utc_now_fn=lambda: "2026-01-01T00:00:00+00:00",
        collection_name="memories",
        logger=SimpleNamespace(
            debug=lambda *_args, **_kwargs: None,
            exception=lambda *_args, **_kwargs: None,
        ),
        isolation_context=context,
    )

    assert updated is not None
    assert qdrant.set_payload_calls[-1]["collection"] == "tenant_a_vectors"


def test_full_enrichment_uses_tenant_collection_for_payload_sync() -> None:
    qdrant = _Qdrant()
    graph = FakeGraph()
    context = IsolationContext("tenant_a", "tenant_a_vectors", isolated=True)
    memory_id = "00000000-0000-4000-8000-000000000010"
    graph.memories[memory_id] = {
        "id": memory_id,
        "content": "Met Alice at Launchpad.",
        "tags": ["meeting"],
        "metadata": "{}",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "processed": False,
        "enriched": False,
    }

    processed = enrich_memory(
        memory_id=memory_id,
        forced=True,
        get_memory_graph_fn=lambda isolation_context=None: graph,
        get_qdrant_client_fn=lambda: qdrant,
        parse_metadata_field_fn=lambda raw: {},
        normalize_tag_list_fn=lambda tags: list(tags or []),
        extract_entities_fn=lambda _content: {"people": ["Alice"]},
        slugify_fn=lambda value: str(value).lower(),
        compute_tag_prefixes_fn=lambda tags: tags,
        find_temporal_relationships_fn=lambda _graph, _memory_id: 0,
        detect_patterns_fn=lambda _graph, _memory_id, _content: [],
        link_semantic_neighbors_fn=lambda _graph, _memory_id, _collection_name: [],
        enrichment_enable_summaries=False,
        generate_summary_fn=lambda content, _existing=None: content,
        utc_now_fn=lambda: "2026-01-01T00:00:00+00:00",
        collection_name="memories",
        unexpected_response_exc=Exception,
        logger=SimpleNamespace(debug=lambda *_args, **_kwargs: None, exception=print),
        isolation_context=context,
    )

    assert processed is True
    assert qdrant.set_payload_calls[-1]["collection"] == "tenant_a_vectors"


def test_admin_reembed_with_allowlisted_headers_uses_tenant_collection(
    client, mock_state, admin_headers, monkeypatch
) -> None:
    monkeypatch.setenv("ALLOWED_GRAPHS", "tenant_a")
    monkeypatch.setenv("ALLOWED_COLLECTIONS", "tenant_a_vectors")
    memory_id = "00000000-0000-4000-8000-000000000011"
    tenant_graph = mock_state.falkordb.select_graph("tenant_a")
    tenant_graph.memories[memory_id] = {
        "id": memory_id,
        "content": "Tenant reembed memory",
        "tags": ["tenant"],
        "metadata": "{}",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "importance": 0.7,
    }
    mock_state.openai_client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=lambda **_kwargs: SimpleNamespace(data=[SimpleNamespace(embedding=[0.2] * 1024)])
        )
    )

    response = client.post(
        "/admin/reembed",
        headers={
            **admin_headers,
            "X-Graph-Name": "tenant_a",
            "X-Collection-Name": "tenant_a_vectors",
        },
        json={"batch_size": 10, "limit": 1},
    )

    assert response.status_code == 200
    assert mock_state.qdrant.upsert_calls[-1][0] == "tenant_a_vectors"
