"""Recall must exclude internal artifact memory types (e.g. ``MetaPattern``).

Consolidation creates ``type='MetaPattern'`` cluster-summary nodes with
importance 0.0 ("cluster with N memories over 0 days"). They carry no user
value but leak into ``/recall`` via bare-token keyword matches (e.g. "berlin"),
padding results and misleading the consumer. These tests pin the exclusion at
the universal chokepoint (``_result_passes_filters``) and at the graph-keyword
Cypher layer (so artifacts don't even consume candidate slots).
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

import automem.search.runtime_recall_helpers as recall_helpers
from automem.search.runtime_recall_helpers import (
    _graph_keyword_search,
    _result_passes_filters,
    configure_recall_helpers,
)
from automem.utils.text import _extract_keywords
from tests.support.fake_graph import FakeResult

_HELPER_STATE_ATTRS = (
    "_parse_iso_datetime",
    "_prepare_tag_filters",
    "_build_graph_tag_predicate",
    "_build_qdrant_tag_filter",
    "_serialize_node",
    "_fetch_relations",
    "_extract_keywords",
    "_coerce_embedding",
    "_generate_real_embedding",
    "_logger",
    "_collection_name",
)


@pytest.fixture(autouse=True)
def _restore_recall_helper_state():
    previous = {name: getattr(recall_helpers, name) for name in _HELPER_STATE_ATTRS}
    yield
    for name, value in previous.items():
        setattr(recall_helpers, name, value)


class _ScriptedGraph:
    """Graph stub returning preset rows and recording issued queries."""

    def __init__(self, rows):
        self._rows = rows
        self.queries = []

    def query(self, query, params=None):
        self.queries.append((query, params))
        return FakeResult(self._rows)


def _configure_helpers() -> None:
    configure_recall_helpers(
        parse_iso_datetime=lambda value: (
            datetime.fromisoformat(str(value).replace("Z", "+00:00")) if value else None
        ),
        prepare_tag_filters=lambda tags: [
            str(tag).strip().lower() for tag in (tags or []) if isinstance(tag, str) and tag.strip()
        ],
        build_graph_tag_predicate=lambda _mode, _match: "true",
        build_qdrant_tag_filter=lambda *_a, **_k: None,
        serialize_node=lambda node: dict(getattr(node, "properties", node)),
        fetch_relations=lambda *_a, **_k: [],
        extract_keywords=_extract_keywords,
        coerce_embedding=lambda _v: None,
        generate_real_embedding=lambda _t: [0.1, 0.2, 0.3],
        logger=SimpleNamespace(exception=lambda *_a, **_k: None, debug=lambda *_a, **_k: None),
        collection_name="memories",
    )


def _result(memory_type: str) -> dict:
    return {
        "id": f"mem-{memory_type}",
        "memory": {
            "id": f"mem-{memory_type}",
            "type": memory_type,
            "content": "cluster with 12 memories over 0 days in berlin",
            "tags": ["entity:concepts:berlin"],
            "timestamp": "2026-06-11T00:00:00+00:00",
        },
    }


def test_result_passes_filters_excludes_metapattern() -> None:
    _configure_helpers()
    assert _result_passes_filters(_result("MetaPattern"), None, None) is False


def test_result_passes_filters_keeps_normal_types() -> None:
    _configure_helpers()
    for kept in ("Memory", "Context", "Decision", "Insight", "Preference"):
        assert _result_passes_filters(_result(kept), None, None) is True


def test_graph_keyword_search_emits_artifact_exclusion() -> None:
    _configure_helpers()
    graph = _ScriptedGraph([])  # rows irrelevant; we inspect the emitted Cypher
    _graph_keyword_search(graph, "berlin benefits", limit=10, seen_ids=set())

    assert graph.queries, "expected a Cypher query to be issued"
    issued_query, params = graph.queries[-1]
    params = params or {}
    assert "$excluded_types" in issued_query
    assert "MetaPattern" in (params.get("excluded_types") or [])
