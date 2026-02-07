from __future__ import annotations

import json

import pytest

import app


class FakeResult:
    def __init__(self, rows):
        self.result_set = rows


class FakeNode:
    def __init__(self, properties):
        self.properties = properties


class FakeGraph:
    def __init__(self):
        self.temporal_calls = []
        self.pattern_calls = []
        self.exemplifies_calls = []
        self.update_calls = []

    def query(self, query: str, params: dict | None = None, **kwargs) -> FakeResult:
        params = params or {}

        if "MATCH (m:Memory {id: $id}) RETURN m" in query and "RETURN m2.id" not in query:
            node = FakeNode(
                {
                    "id": "mem-1",
                    "content": 'Met with Alice about SuperWhisper deployment on project "Launchpad".',
                    "tags": ["meeting"],
                    "metadata": {},
                    "processed": False,
                    "summary": None,
                }
            )
            return FakeResult([[node]])

        if "RETURN m2.id" in query and "PRECEDED_BY" not in query:
            return FakeResult([["mem-older"]])

        if "MERGE (m1)-[r:PRECEDED_BY]" in query:
            self.temporal_calls.append(params)
            return FakeResult([])

        if "MATCH (m:Memory)" in query and "m.type = $type" in query:
            return FakeResult(
                [
                    ["mem-a", "Pattern insight about automation"],
                    ["mem-b", "Another automation pattern emerges"],
                    ["mem-c", "Automation habit noted"],
                ]
            )

        if "MERGE (p:Pattern" in query:
            self.pattern_calls.append(params)
            return FakeResult([])

        if "MERGE (m)-[r:EXEMPLIFIES]" in query:
            self.exemplifies_calls.append(params)
            return FakeResult([])

        if "SET m.metadata" in query:
            self.update_calls.append(params)
            return FakeResult([])

        return FakeResult([])


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    monkeypatch.setattr(app, "init_falkordb", lambda: None)
    monkeypatch.setattr(app, "init_qdrant", lambda: None)
    monkeypatch.setattr(app, "get_qdrant_client", lambda: None)

    original_graph = app.state.memory_graph
    original_stats = app.state.enrichment_stats
    original_pending = set(app.state.enrichment_pending)
    original_inflight = set(app.state.enrichment_inflight)

    app.state.memory_graph = None
    app.state.enrichment_stats = app.EnrichmentStats()
    app.state.enrichment_pending.clear()
    app.state.enrichment_inflight.clear()

    yield

    app.state.memory_graph = original_graph
    app.state.enrichment_stats = original_stats
    app.state.enrichment_pending.clear()
    app.state.enrichment_pending.update(original_pending)
    app.state.enrichment_inflight.clear()
    app.state.enrichment_inflight.update(original_inflight)


def test_extract_entities_basic():
    content = "Deployed SuperWhisper with Alice during Project Launchpad review"
    entities = app.extract_entities(content)
    assert "SuperWhisper" in entities["tools"]
    assert "Launchpad" in entities["projects"]


def test_enrich_memory_updates_metadata(monkeypatch):
    fake_graph = FakeGraph()
    app.state.memory_graph = fake_graph

    processed = app.enrich_memory("mem-1", forced=True)
    assert processed is True

    assert fake_graph.temporal_calls, "Should create temporal relationships"
    assert fake_graph.pattern_calls, "Should update pattern nodes"
    assert fake_graph.exemplifies_calls, "Should create EXEMPLIFIES relationship"
    assert fake_graph.update_calls, "Should update memory metadata"

    update_payload = fake_graph.update_calls[-1]
    metadata = json.loads(update_payload["metadata"])
    assert metadata["entities"]["projects"] == ["Launchpad"]
    assert metadata["enrichment"]["temporal_links"] == 1
    assert metadata["enrichment"]["patterns_detected"]
    assert update_payload["summary"].startswith("Met with Alice")
    assert "entity:projects:launchpad" in update_payload["tags"]
