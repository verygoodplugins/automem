"""Consolidation must never read or mutate another deployment's collection."""

import copy
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

from automem.consolidation.runtime_bindings import create_consolidation_runtime
from consolidation import MemoryConsolidator
from tests.support.fake_graph import FakeGraph, FakeResult

CUSTOM_COLLECTION = "tenant-memories"
MEMORY_IDS = ["delete", "archive", "protected", "preserve"]


class Graph(FakeGraph):
    def query(self, query, params=None, **kwargs):
        if "WHERE m.relevance_score > 0.3" in query and "ORDER BY rand()" not in query:
            return FakeResult([[mid, "related content", "Memory"] for mid in MEMORY_IDS])
        return super().query(query, params, **kwargs)


class Collections:
    def __init__(self, fail_writes=False):
        points = {mid: {"untouched": True} for mid in MEMORY_IDS}
        self.collections = {
            "memories": copy.deepcopy(points),
            CUSTOM_COLLECTION: copy.deepcopy(points),
        }
        self.calls = []
        self.fail_writes = fail_writes

    def retrieve(self, *, collection_name, ids, **kwargs):
        self.calls.append(("retrieve", collection_name))
        return [SimpleNamespace(id=mid, vector=[1.0, 0.0]) for mid in ids]

    def scroll(self, *, collection_name, **kwargs):
        self.calls.append(("scroll", collection_name))
        return [SimpleNamespace(id=mid, vector=[1.0, 0.0]) for mid in MEMORY_IDS], None

    def delete(self, *, collection_name, points_selector):
        self.calls.append(("delete", collection_name))
        if self.fail_writes:
            raise ConnectionError("vector store unavailable")
        ids = (
            points_selector["points"]
            if isinstance(points_selector, dict)
            else points_selector.points
        )
        for mid in ids:
            self.collections[collection_name].pop(mid, None)

    def set_payload(self, *, collection_name, points, payload):
        self.calls.append(("set_payload", collection_name))
        if self.fail_writes:
            raise ConnectionError("vector store unavailable")
        for mid in points:
            self.collections[collection_name][mid].update(payload)


def exercise_consolidation(monkeypatch, *, fail_writes=False):
    graph = Graph()
    store = Collections(fail_writes=fail_writes)
    consolidator = MemoryConsolidator(graph, vector_store=store, grace_period_days=0)
    # Set the configured value directly as well as testing constructor/runtime wiring
    # below, so this regression reaches every real vector operation on old code.
    consolidator.collection_name = CUSTOM_COLLECTION
    graph.sample_rows = [[mid, "related content", "Memory", None, None] for mid in MEMORY_IDS]
    assert consolidator.discover_creative_associations() == []
    assert consolidator.cluster_similar_memories()[0]["size"] == 4

    scores = {"delete": 0.01, "archive": 0.1, "protected": 0.1, "preserve": 0.5}
    graph.forgetting_rows = [
        [mid, "content", scores[mid], None, "Memory", 0.1, None, mid == "protected"]
        for mid in MEMORY_IDS
    ]
    monkeypatch.setattr(
        consolidator, "calculate_relevance_score", lambda memory, *args: scores[memory["id"]]
    )
    result = consolidator.apply_controlled_forgetting(dry_run=False)
    assert [memory["id"] for memory in result["deleted"]] == ["delete"]
    assert [memory["id"] for memory in result["archived"]] == ["archive"]
    assert [memory["id"] for memory in result["protected"]] == ["protected"]
    graph.decay_rows = [["preserve", "content", None, 0.1, None, 0.5]]
    assert consolidator._apply_decay()["processed"] == 1
    return store


def test_all_seven_vector_operations_use_configured_collection(monkeypatch):
    store = exercise_consolidation(monkeypatch)
    assert store.calls == [
        ("retrieve", CUSTOM_COLLECTION),
        ("scroll", CUSTOM_COLLECTION),
        ("delete", CUSTOM_COLLECTION),
        ("set_payload", CUSTOM_COLLECTION),
        ("set_payload", CUSTOM_COLLECTION),
        ("set_payload", CUSTOM_COLLECTION),
        ("set_payload", CUSTOM_COLLECTION),
    ]
    assert store.collections["memories"] == {mid: {"untouched": True} for mid in MEMORY_IDS}
    assert "delete" not in store.collections[CUSTOM_COLLECTION]
    assert store.collections[CUSTOM_COLLECTION]["archive"]["archived"] is True
    assert store.collections[CUSTOM_COLLECTION]["protected"]["relevance_score"] == 0.1
    assert store.collections[CUSTOM_COLLECTION]["preserve"]["relevance_score"] == 0.5


def test_failed_vector_mutations_are_visible_at_warning_level(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger="consolidation"):
        exercise_consolidation(monkeypatch, fail_writes=True)
    assert len(caplog.records) == 5  # delete, archive, protected, preserve, decay
    assert all(record.levelno >= logging.WARNING for record in caplog.records)


def test_constructor_retains_default_and_accepts_custom_collection():
    assert MemoryConsolidator(Graph()).collection_name == "memories"
    assert MemoryConsolidator(Graph(), collection_name=CUSTOM_COLLECTION).collection_name == (
        CUSTOM_COLLECTION
    )


def test_runtime_factory_passes_configured_collection():
    runtime = create_consolidation_runtime(
        state=SimpleNamespace(),
        logger=MagicMock(),
        get_memory_graph_fn=lambda: None,
        get_qdrant_client_fn=lambda: None,
        collection_name=CUSTOM_COLLECTION,
        emit_event_fn=MagicMock(),
        utc_now_fn=lambda: "2026-10-20T00:00:00Z",
        perf_counter_fn=lambda: 0,
        parse_iso_datetime_fn=lambda value: None,
        stop_event_cls=MagicMock(),
        thread_cls=MagicMock(),
        run_label="Run",
        control_label="Control",
        control_node_id="singleton",
        task_fields={},
        history_limit=10,
        tick_seconds=60,
        decay_interval_seconds=60,
        creative_interval_seconds=60,
        cluster_interval_seconds=60,
        forget_interval_seconds=0,
        delete_threshold=0,
        archive_threshold=0,
        grace_period_days=90,
        importance_protection_threshold=0.7,
        protected_types={"Decision", "Insight"},
        decay_importance_threshold=0.3,
    )
    assert runtime.build_consolidator_from_config(Graph(), Collections()).collection_name == (
        CUSTOM_COLLECTION
    )
