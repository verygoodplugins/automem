from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

import consolidation as consolidation_module
from consolidation import MemoryConsolidator
from tests.support.fake_graph import FakeGraph  # noqa: F401 - used via dummy_graph fixture


class FakeVectorStore:
    def __init__(self) -> None:
        self.deletions: List[tuple[str, Dict[str, Any]]] = []

    def delete(self, collection_name: str, points_selector: Dict[str, Any]) -> None:
        self.deletions.append((collection_name, points_selector))


@pytest.fixture(autouse=True)
def freeze_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use a fixed timestamp to keep decay calculations deterministic."""

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz: timezone | None = None) -> datetime:
            base = datetime(2024, 1, 1, tzinfo=timezone.utc)
            return base if tz is None else base.astimezone(tz)

    monkeypatch.setattr(consolidation_module, "datetime", FixedDatetime)
    yield
    monkeypatch.setattr(consolidation_module, "datetime", datetime)


def iso_days_ago(days: int) -> str:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return (base - timedelta(days=days)).isoformat()


def test_calculate_relevance_score_accounts_for_relationships(dummy_graph) -> None:
    graph = dummy_graph
    graph.relationship_counts["m1"] = 0
    consolidator = MemoryConsolidator(graph)

    common_memory = {
        "id": "m1",
        "timestamp": iso_days_ago(1),
        "importance": 0.6,
        "confidence": 0.6,
    }

    baseline = consolidator.calculate_relevance_score(common_memory.copy())

    # Clear the LRU cache to ensure the updated relationship count is fetched
    consolidator._get_relationship_count_cached_impl.cache_clear()

    graph.relationship_counts["m1"] = 6
    boosted = consolidator.calculate_relevance_score(common_memory.copy())

    assert boosted > baseline
    assert 0 < boosted <= 1


def test_discover_creative_associations_builds_connections(dummy_graph) -> None:
    graph = dummy_graph
    graph.sample_rows = [
        ["decision-a", "Chose approach A", "Decision", [1.0, 0.0, 0.0], iso_days_ago(3)],
        ["decision-b", "Chose approach B", "Decision", [0.0, 1.0, 0.0], iso_days_ago(4)],
        ["insight", "Insight about A", "Insight", [0.9, 0.1, 0.0], iso_days_ago(5)],
    ]

    consolidator = MemoryConsolidator(graph)
    associations = consolidator.discover_creative_associations(sample_size=3)

    assert any(item["type"] == "CONTRASTS_WITH" for item in associations)


def test_cluster_similar_memories_groups_items(dummy_graph) -> None:
    graph = dummy_graph
    graph.cluster_rows = [
        ["m1", "Alpha", [1.0, 0.0], "Insight"],
        ["m2", "Alpha follow-up", [0.95, 0.05], "Insight"],
        ["m3", "Alpha summary", [1.02, -0.02], "Pattern"],
    ]

    consolidator = MemoryConsolidator(graph)
    clusters = consolidator.cluster_similar_memories()

    assert clusters
    assert clusters[0]["size"] == 3
    assert clusters[0]["dominant_type"] in {"Insight", "Pattern"}


def build_forgetting_rows() -> List[List[Any]]:
    return [
        [
            "recent-keep",
            "Fresh important memory",
            0.8,
            iso_days_ago(2),
            "Insight",
            0.9,
            iso_days_ago(1),
        ],
        [
            "archive-candidate",
            "Memory to archive",
            0.2,
            iso_days_ago(15),
            "Memory",
            0.4,
            iso_days_ago(15),
        ],
        [
            "old-delete",
            "Superseded note",
            0.05,
            iso_days_ago(90),
            "Memory",
            0.2,
            iso_days_ago(90),
        ],
    ]


def test_apply_controlled_forgetting_dry_run(dummy_graph) -> None:
    graph = dummy_graph
    graph.relationship_counts["recent-keep"] = 5
    graph.forgetting_rows = build_forgetting_rows()

    consolidator = MemoryConsolidator(graph)
    stats = consolidator.apply_controlled_forgetting(dry_run=True)

    assert stats["examined"] == 3
    assert stats["preserved"] == 2
    assert len(stats["archived"]) == 0
    assert len(stats["deleted"]) == 1
    assert len(stats["protected"]) == 1
    assert graph.deleted == []


def test_apply_controlled_forgetting_updates_graph_and_vector_store(dummy_graph) -> None:
    graph = dummy_graph
    graph.relationship_counts["recent-keep"] = 5
    graph.forgetting_rows = build_forgetting_rows()

    vector_store = FakeVectorStore()
    consolidator = MemoryConsolidator(graph, vector_store=vector_store)

    stats = consolidator.apply_controlled_forgetting(dry_run=False)

    assert stats["preserved"] == 2
    assert graph.updated_scores  # recent memory updated in graph
    assert graph.archived == []
    assert graph.deleted == ["old-delete"]
    assert vector_store.deletions
    collection, selector = vector_store.deletions[0]
    assert collection == "memories"
    points = selector.get("point_ids") or selector.get("points")
    assert points == ["old-delete"]


def test_apply_decay_updates_scores(dummy_graph) -> None:
    graph = dummy_graph
    graph.relationship_counts = {"a": 0, "b": 2}
    graph.decay_rows = [
        ["a", "Early note", iso_days_ago(10), 0.5, iso_days_ago(10), 0.5],
        ["b", "Recent insight", iso_days_ago(1), 0.7, iso_days_ago(1), 0.9],
    ]

    consolidator = MemoryConsolidator(graph)
    stats = consolidator._apply_decay()

    assert stats["processed"] == 2
    assert len(graph.updated_scores) == 2
    assert stats["avg_relevance_after"] <= 1
