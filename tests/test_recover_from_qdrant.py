"""Recovery must preserve existing graph data and report incomplete imports."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "recover_from_qdrant.py"


def load_recovery_module():
    spec = importlib.util.spec_from_file_location("recover_from_qdrant", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeGraph:
    def __init__(self):
        self.queries = []
        self.nodes = {"existing": {"id": "existing", "content": "Newer graph content"}}
        self.relationships = [("existing", "RELATED_TO", "entity")]

    def query(self, statement, params=None):
        self.queries.append((statement, params))
        assert "DELETE" not in statement
        assert "MERGE (m:Memory {id: $id})" in statement
        assert "ON CREATE SET" in statement
        assert params is not None
        self.nodes.setdefault(params["id"], params["properties"])
        return SimpleNamespace(result_set=[[params["id"]]])


class FakeFalkorDB:
    def __init__(self):
        self.graph_names = []
        self.graph = FakeGraph()

    def select_graph(self, name):
        self.graph_names.append(name)
        return self.graph


def memory(memory_id="new", **payload):
    return {
        "id": memory_id,
        "payload": {"content": "Recovered memory", "timestamp": "2026-10-20T12:00:00Z", **payload},
    }


def test_recovery_preserves_ids_existing_memories_and_relationships(monkeypatch):
    monkeypatch.setenv("FALKORDB_GRAPH", "custom_memories")
    recovery = load_recovery_module()
    client = FakeFalkorDB()

    for item in [memory("existing"), memory(), memory()]:
        assert recovery.restore_memory_to_graph_only(item, client)

    assert client.graph_names == ["custom_memories"] * 3
    assert set(client.graph.nodes) == {"existing", "new"}
    assert client.graph.nodes["existing"]["content"] == "Newer graph content"
    assert client.graph.relationships == [("existing", "RELATED_TO", "entity")]


def test_recovery_uses_api_node_shape_and_parameterizes_payload():
    recovery = load_recovery_module()
    client = FakeFalkorDB()
    item = memory(
        "quote'id",
        content="O'Reilly \\ books",
        tags=["Auth/Login", "bugfix", "a'b"],
        metadata={"source": "x'", "nested": {"count": 2}, "enrichment": {"version": 1}},
        t_valid="2026-10-19T12:00:00Z",
        t_invalid="2026-10-22T12:00:00Z",
        archived=True,
        updated_at="2026-10-21T12:00:00Z",
        last_accessed="2026-10-21T13:00:00Z",
        relevance_score=0.25,
    )
    assert recovery.restore_memory_to_graph_only(item, client)
    query, params = client.graph.queries[0]
    node = client.graph.nodes[item["id"]]
    assert item["id"] not in query
    assert item["payload"]["content"] not in query
    assert node["tag_prefixes"] == ["auth", "auth:login", "bugfix", "a'b"]
    assert json.loads(node["metadata"]) == item["payload"]["metadata"]
    assert "source" not in node
    assert node["t_valid"] == "2026-10-19T12:00:00+00:00"
    assert node["t_invalid"] == "2026-10-22T12:00:00+00:00"
    assert node["updated_at"] == "2026-10-21T12:00:00+00:00"
    assert node["last_accessed"] == "2026-10-21T13:00:00+00:00"
    assert node["archived"] is True
    assert node["processed"] is False
    assert node["relevance_score"] == 0.25
    assert params["id"] == item["id"]


def test_recovery_fills_legacy_payload_defaults():
    recovery = load_recovery_module()
    client = FakeFalkorDB()
    assert recovery.restore_memory_to_graph_only(memory(tags=["Auth/Login"]), client)
    node = client.graph.nodes["new"]
    assert node["t_valid"] == node["timestamp"]
    assert node["updated_at"] == node["last_accessed"] == node["timestamp"]
    assert node["type"] == "Context"
    assert node["archived"] is False
    assert json.loads(node["metadata"]) == {}


@pytest.mark.parametrize("items", [[], [memory(), memory("bad", timestamp="invalid")]])
def test_recovery_validates_entire_source_before_connecting_to_graph(monkeypatch, items):
    recovery = load_recovery_module()
    monkeypatch.setattr(recovery, "get_all_memories", lambda: items)
    connect = Mock()
    monkeypatch.setattr(recovery, "FalkorDB", connect)
    assert recovery.main() == 1
    connect.assert_not_called()


def test_recovery_source_failure_never_mutates_graph(monkeypatch):
    recovery = load_recovery_module()
    monkeypatch.setattr(recovery, "get_all_memories", Mock(side_effect=RuntimeError("source down")))
    connect = Mock()
    monkeypatch.setattr(recovery, "FalkorDB", connect)
    assert recovery.main() == 1
    connect.assert_not_called()


def test_recovery_partial_write_failure_returns_failure(monkeypatch):
    recovery = load_recovery_module()
    client = FakeFalkorDB()
    monkeypatch.setattr(recovery, "get_all_memories", lambda: [memory("first"), memory("second")])
    monkeypatch.setattr(recovery, "FalkorDB", lambda **kwargs: client)
    monkeypatch.setattr(recovery, "restore_memory_to_graph_only", Mock(side_effect=[True, False]))
    assert recovery.main() == 1


def test_recovery_success_and_unconfirmed_write(monkeypatch):
    recovery = load_recovery_module()
    client = FakeFalkorDB()
    monkeypatch.setattr(recovery, "get_all_memories", lambda: [memory()])
    monkeypatch.setattr(recovery, "FalkorDB", lambda **kwargs: client)
    assert recovery.main() == 0
    client.graph.query = Mock(return_value=SimpleNamespace(result_set=[]))
    assert not recovery.restore_memory_to_graph_only(memory("another"), client)


@pytest.mark.parametrize(
    "item",
    [
        {"id": None, "payload": {"content": "x"}},
        {"id": "x", "payload": None},
        memory(metadata=[]),
        memory(metadata="not json"),
        memory(importance=float("nan")),
        memory(relevance_score="0.2"),
        memory(relevance_score=float("nan")),
        memory(archived="false"),
    ],
)
def test_recovery_rejects_invalid_source_before_any_graph_write(item):
    recovery = load_recovery_module()
    client = FakeFalkorDB()
    assert not recovery.restore_memory_to_graph_only(item, client)
    assert client.graph.queries == []


def test_recovery_fetches_all_payload_pages_without_vectors_and_closes_client(monkeypatch):
    recovery = load_recovery_module()
    point = SimpleNamespace(id="new", payload=memory()["payload"])
    client = Mock()
    client.scroll.side_effect = [([point], "next"), ([], None)]
    monkeypatch.setattr(recovery, "QdrantClient", lambda **kwargs: client)
    monkeypatch.setattr(recovery.time, "sleep", lambda seconds: None)
    assert recovery.get_all_memories() == [memory()]
    calls = client.scroll.call_args_list
    assert len(calls) == 2
    assert calls[1].kwargs["offset"] == "next"
    assert all(call.kwargs["with_vectors"] is False for call in calls)
    client.close.assert_called_once()


def test_recovery_closes_source_client_after_fetch_failure(monkeypatch):
    recovery = load_recovery_module()
    client = Mock()
    client.scroll.side_effect = RuntimeError("source down")
    monkeypatch.setattr(recovery, "QdrantClient", lambda **kwargs: client)
    with pytest.raises(RuntimeError, match="source down"):
        recovery.get_all_memories()
    client.close.assert_called_once()


def test_recovery_accepts_canonical_vector_payload_with_unset_relevance():
    """Normal memories have a null relevance score until consolidation computes it."""
    recovery = load_recovery_module()
    client = FakeFalkorDB()
    # Shape emitted by the canonical vector payload builder (including JSON nulls).
    payload = {
        "content": "Normal memory before first decay",
        "tags": ["auth/login"],
        "tag_prefixes": ["auth", "auth:login"],
        "importance": 0.5,
        "timestamp": "2026-10-20T12:00:00Z",
        "type": "Memory",
        "confidence": 0.5,
        "t_valid": "2026-10-20T12:00:00Z",
        "t_invalid": None,
        "archived": False,
        "updated_at": None,
        "last_accessed": None,
        "metadata": {},
        "relevance_score": None,
    }
    assert recovery.restore_memory_to_graph_only({"id": "canonical", "payload": payload}, client)
    properties = client.graph.nodes["canonical"]
    assert properties["relevance_score"] == 1.0
    assert properties["t_valid"] == "2026-10-20T12:00:00+00:00"
    assert properties["t_invalid"] is None
    assert properties["tag_prefixes"] == ["auth", "auth:login"]
    payload["relevance_score"] = 0.0
    assert (
        recovery.memory_properties({"id": "decayed", "payload": payload})["relevance_score"] == 0.0
    )
