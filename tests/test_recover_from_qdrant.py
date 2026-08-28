"""Regression coverage for the Qdrant-to-FalkorDB recovery helper."""

import importlib.util
from pathlib import Path


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

    def query(self, statement, params=None):
        self.queries.append((statement, params))


class FakeFalkorDB:
    def __init__(self):
        self.graph_names = []
        self.graph = FakeGraph()

    def select_graph(self, name):
        self.graph_names.append(name)
        return self.graph


def test_recovery_uses_configured_graph_for_clear_and_restore(monkeypatch):
    monkeypatch.setenv("FALKORDB_GRAPH", "custom_memories")
    recovery = load_recovery_module()
    client = FakeFalkorDB()

    assert recovery.clear_graph(client)
    assert recovery.restore_memory_to_graph_only(
        {
            "id": "memory-1",
            "payload": {
                "content": "Recovered memory",
                "tags": [],
            },
        },
        client,
    )

    assert client.graph_names == ["custom_memories", "custom_memories"]
