from __future__ import annotations

import json

import pytest

import app
from tests.support.fake_graph import FakeGraph


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
    fake_graph = FakeGraph(seed_enrichment_fixture=True)
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
