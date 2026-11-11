from types import SimpleNamespace

import pytest

from automem.stores.original_text_index import (
    ensure_original_text_index,
    original_text_index_status,
)


class Graph:
    def __init__(self, statuses, create_error=None):
        self.statuses = iter(statuses)
        self.create_error = create_error
        self.queries = []

    def query(self, text):
        self.queries.append(text)
        if "db.indexes" in text:
            status = next(self.statuses)
            rows = (
                []
                if status is None
                else [
                    [
                        "MemoryOriginalText",
                        ["original_search_text"],
                        {"original_search_text": ["FULLTEXT"]},
                        status,
                    ]
                ]
            )
            return SimpleNamespace(result_set=rows)
        if self.create_error:
            raise self.create_error
        return SimpleNamespace(result_set=[])


def test_existing_ready_index_is_not_recreated():
    graph = Graph(["OPERATIONAL"])
    assert ensure_original_text_index(graph) == "ready"
    assert len(graph.queries) == 1


def test_absent_index_is_created_and_rechecked():
    graph = Graph([None, "OPERATIONAL"])
    assert ensure_original_text_index(graph) == "ready"
    assert len(graph.queries) == 3
    assert "createNodeIndex" in graph.queries[1]


def test_race_is_accepted_only_after_verifying_other_creator():
    graph = Graph([None, "OPERATIONAL"], RuntimeError("duplicate"))
    assert ensure_original_text_index(graph) == "ready"
    graph = Graph([None, None], RuntimeError("permission denied"))
    with pytest.raises(RuntimeError, match="permission denied"):
        ensure_original_text_index(graph)


def test_pending_and_failed_states_are_visible():
    assert ensure_original_text_index(Graph(["PENDING"])) == "building"
    assert (
        ensure_original_text_index(Graph(["[Indexing] 100/1000: UNDER CONSTRUCTION"])) == "building"
    )
    assert (
        original_text_index_status(Graph(["[Indexing] 100/1000: UNDER CONSTRUCTION"])) == "building"
    )
    assert ensure_original_text_index(Graph(["FAILED"])) == "failed"
    assert ensure_original_text_index(Graph([None, None])) == "failed"


def test_index_status_is_read_only_and_recovers_from_building():
    graph = Graph(["BUILDING", "OPERATIONAL", None])
    assert original_text_index_status(graph) == "building"
    assert original_text_index_status(graph) == "ready"
    assert original_text_index_status(graph) == "unavailable"
    assert all("db.indexes" in query for query in graph.queries)


def test_falkordb_initialization_keeps_graph_available_when_index_creation_fails(monkeypatch):
    from unittest.mock import Mock
    from automem.stores.runtime_clients import init_falkordb

    class UnavailableGraph:
        def query(self, _text):
            raise RuntimeError("index unavailable")

    graph = UnavailableGraph()
    client = SimpleNamespace(select_graph=lambda _name: graph)
    state = SimpleNamespace(memory_graph=None, falkordb=None)
    monkeypatch.delenv("FALKORDB_PASSWORD", raising=False)
    init_falkordb(
        state=state,
        logger=Mock(),
        falkordb_cls=lambda **_kwargs: client,
        graph_name="memories",
        falkordb_port=6379,
    )
    assert state.memory_graph is graph
    assert state.original_text_index_status == "unavailable"


def test_health_refreshes_index_readiness_and_marks_failure_degraded(monkeypatch):
    from threading import Lock
    from flask import Flask
    from automem.api.health import create_health_blueprint

    graph = Graph(["BUILDING", "OPERATIONAL", None])
    state = SimpleNamespace(
        original_text_index_status="building",
        enrichment_thread=None,
        enrichment_lock=Lock(),
        enrichment_pending=set(),
        enrichment_inflight=set(),
        enrichment_queue=None,
        enrichment_stats=SimpleNamespace(successes=0, failures=0),
        embedding_stats={},
        embedding_queue=None,
        embedding_pending=set(),
        embedding_inflight=set(),
        effective_vector_size=None,
    )
    qdrant = SimpleNamespace(
        get_collection=lambda _name: SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=1024)))
        )
    )
    monkeypatch.setattr("automem.api.health.count_falkor_memories", lambda *_args: 0)
    monkeypatch.setattr("automem.api.health.count_qdrant_points", lambda *_args: 0)
    app = Flask(__name__)
    app.register_blueprint(
        create_health_blueprint(
            lambda: graph, lambda: qdrant, state, "memories", "memories", lambda: "now"
        )
    )
    with app.test_client() as client:
        first = client.get("/health").get_json()
        second = client.get("/health").get_json()
        third = client.get("/health").get_json()
    assert first["original_text_index"] == "building"
    assert second["original_text_index"] == "ready"
    assert third["original_text_index"] == "unavailable"
    assert third["status"] == "degraded"
