"""Regression tests for the FalkorDB backup export pagination.

Covers the two ways the previous ``SKIP <offset> LIMIT <batch_size>`` paging
lost data: batches that grew slower with depth until they exceeded the server's
``TIMEOUT``, and FalkorDB's ``RESULTSET_SIZE`` cap silently truncating a result
set that the loop then read as "the last page".
"""

from __future__ import annotations

import gzip
import io
import json
import re
from typing import Any

import pytest

from automem.backup import BackupError, export_falkordb_artifact
from tests.support.fake_graph import FakeGraph


def build_graph(memory_count: int, degrees: dict[int, int] | None = None) -> FakeGraph:
    """Seed a graph with `memory_count` memories and per-node out-degrees."""
    graph = FakeGraph()
    for index in range(memory_count):
        memory_id = f"mem-{index:05d}"
        graph.memories[memory_id] = {"id": memory_id, "content": f"memory {index}"}
        graph.nodes.add(memory_id)

    for source, degree in (degrees or {}).items():
        for step in range(degree):
            graph.relationships.append(
                {
                    "id1": f"mem-{source:05d}",
                    "id2": f"mem-{(source + step + 1) % memory_count:05d}",
                    "type": "RELATES_TO",
                    "strength": 0.5,
                }
            )
    return graph


def export(graph: FakeGraph, **kwargs: Any) -> dict[str, Any]:
    artifact = export_falkordb_artifact(
        graph=graph, graph_name="memories", timestamp="20260101_000000", **kwargs
    )
    with gzip.open(io.BytesIO(artifact.data), "rt", encoding="utf-8") as handle:
        return json.load(handle)


def assert_exported_everything(graph: FakeGraph, payload: dict[str, Any]) -> None:
    assert payload["stats"]["node_count"] == len(graph.memories)
    assert payload["stats"]["relationship_count"] == len(graph.relationships)
    assert len(payload["nodes"]) == len(graph.memories)

    exported_ids = [node["id"] for node in payload["nodes"]]
    assert len(set(exported_ids)) == len(exported_ids), "a node was exported twice"

    exported_edges = sorted(
        (rel["source_id"], rel["type"], rel["target_id"]) for rel in payload["relationships"]
    )
    assert len(exported_edges) == len(graph.relationships)
    node_ids = set(exported_ids)
    assert all(edge[0] in node_ids and edge[2] in node_ids for edge in exported_edges)


def test_export_round_trips_a_small_graph() -> None:
    graph = build_graph(50, {index: 2 for index in range(50)})
    assert_exported_everything(graph, export(graph))


def test_export_is_complete_when_resultset_cap_is_below_batch_size() -> None:
    """RESULTSET_SIZE under batch_size used to end the export after one batch."""
    graph = build_graph(2500, {index: 3 for index in range(2500)})
    graph.resultset_cap = 1000

    payload = export(graph)

    assert_exported_everything(graph, payload)
    assert payload["stats"] == {"node_count": 2500, "relationship_count": 7500}


def test_export_subdivides_a_dense_relationship_range() -> None:
    """A node id window holding more relationships than the cap must be split."""
    degrees = {index: 1 for index in range(300)}
    degrees.update({index: 60 for index in range(40, 45)})
    graph = build_graph(300, degrees)
    graph.resultset_cap = 100

    assert_exported_everything(graph, export(graph))


def test_export_pages_a_single_high_degree_node() -> None:
    """One node with more out-edges than the cap falls back to bounded SKIP."""
    degrees = {index: 2 for index in range(200)}
    degrees[7] = 250
    graph = build_graph(200, degrees)
    graph.resultset_cap = 100

    payload = export(graph)

    assert_exported_everything(graph, payload)
    hub_edges = [rel for rel in payload["relationships"] if rel["source_id"] == 7]
    assert len(hub_edges) == 250


def test_export_handles_out_degree_exactly_at_the_cap() -> None:
    """Rows == cap is ambiguous between complete and truncated; treat it as truncated."""
    degrees = {index: 1 for index in range(50)}
    degrees[3] = 100
    graph = build_graph(50, degrees)
    graph.resultset_cap = 100

    assert_exported_everything(graph, export(graph))


def test_export_rejects_a_silently_truncated_result() -> None:
    """A backup that lost rows must fail rather than look complete."""

    class LossyGraph(FakeGraph):
        def _apply_result_limits(self, rows: list[Any], query: str) -> list[Any]:
            kept = super()._apply_result_limits(rows, query)
            return [row for index, row in enumerate(kept) if index % 5]

    graph = LossyGraph()
    for index in range(300):
        memory_id = f"mem-{index:05d}"
        graph.memories[memory_id] = {"id": memory_id}

    with pytest.raises(BackupError, match="short by"):
        export(graph)


def test_export_handles_an_empty_graph() -> None:
    payload = export(FakeGraph())

    assert payload["nodes"] == []
    assert payload["relationships"] == []
    assert payload["stats"] == {"node_count": 0, "relationship_count": 0}


def test_export_avoids_deep_skip_and_always_sets_a_timeout() -> None:
    """Deep SKIP is what exceeded the server TIMEOUT on a large graph."""
    graph = build_graph(400, {index: 3 for index in range(400)})

    export(graph)

    assert graph.query_kwargs, "no queries were issued"
    assert all(kwargs.get("timeout") for kwargs in graph.query_kwargs)

    offsets = [
        int(match.group(1))
        for query, _params in graph.queries
        if (match := re.search(r"\bSKIP\s+(\d+)", query))
    ]
    assert all(offset == 0 for offset in offsets), f"deep SKIP still in use: {offsets}"
