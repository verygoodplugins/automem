"""Regression tests for issue #190: keyword scores must stay within 0-1.

The graph keyword search Cypher returns a raw additive score (up to
3 * len(keywords) + 3 with a phrase bonus). That raw value must be
normalized before it reaches score blending, where every other channel
(vector cosine, metadata, trending importance) lives in 0-1.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

import automem.search.runtime_recall_helpers as recall_helpers
from automem.search.runtime_recall_helpers import _graph_keyword_search, configure_recall_helpers
from automem.utils.scoring import _compute_metadata_score
from automem.utils.text import _extract_keywords
from tests.support.fake_graph import FakeNode, FakeResult

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
    """Graph stub returning preset [node, raw_score] rows for any query."""

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
        build_qdrant_tag_filter=lambda *_args, **_kwargs: None,
        serialize_node=lambda node: dict(getattr(node, "properties", node)),
        fetch_relations=lambda *_args, **_kwargs: [],
        extract_keywords=_extract_keywords,
        coerce_embedding=lambda _value: None,
        generate_real_embedding=lambda _text: [0.1, 0.2, 0.3],
        logger=SimpleNamespace(exception=lambda *_args, **_kwargs: None),
        collection_name="memories",
    )


def _node(memory_id: str) -> FakeNode:
    return FakeNode(
        {
            "id": memory_id,
            "content": f"content for {memory_id}",
            "tags": ["automem"],
            "importance": 0.5,
            "timestamp": "2026-06-11T00:00:00+00:00",
        }
    )


def test_keyword_scores_are_normalized_to_unit_range() -> None:
    _configure_helpers()
    query_text = "automem keyword scoring"
    keywords = _extract_keywords(query_text.strip().lower())
    assert keywords, "fixture query must extract keywords"
    # Raw Cypher maximum: content (+2) and tag (+1) per keyword, plus the
    # whole-phrase bonus (+2 content, +1 tag).
    max_raw = 3 * len(keywords) + 3

    graph = _ScriptedGraph(
        [
            [_node("mem-max"), max_raw],
            [_node("mem-partial"), 2],
        ]
    )
    results = _graph_keyword_search(graph, query_text, limit=10, seen_ids=set())

    assert [r["id"] for r in results] == ["mem-max", "mem-partial"]
    by_id = {r["id"]: r for r in results}
    assert by_id["mem-max"]["match_score"] == pytest.approx(1.0)
    assert by_id["mem-partial"]["match_score"] == pytest.approx(2 / max_raw)
    for record in results:
        assert 0.0 <= record["match_score"] <= 1.0
        assert record["score"] == record["match_score"]


def test_keyword_score_observed_repro_stays_below_one() -> None:
    """The production repro: 3 keywords fully matched + phrase hit = raw 11."""
    _configure_helpers()
    query_text = "automem keyword scoring"
    keywords = _extract_keywords(query_text.strip().lower())
    assert len(keywords) == 3
    graph = _ScriptedGraph([[_node("mem-repro"), 11]])

    results = _graph_keyword_search(graph, query_text, limit=10, seen_ids=set())

    assert results[0]["match_score"] == pytest.approx(11 / 12)


def test_phrase_only_query_normalizes_against_phrase_maximum() -> None:
    _configure_helpers()
    # Stop-word-only query: no keywords extracted, phrase branch used.
    query_text = "from the with"
    assert _extract_keywords(query_text) == []
    graph = _ScriptedGraph([[_node("mem-phrase"), 2]])

    results = _graph_keyword_search(graph, query_text, limit=10, seen_ids=set())

    assert results[0]["match_score"] == pytest.approx(2 / 3)


def test_consumer_clamps_keyword_component_to_one() -> None:
    """Even if a producer misbehaves, the blend must never see keyword > 1."""
    result = {
        "match_type": "keyword",
        "match_score": 11.0,
        "memory": {
            "id": "mem-1",
            "content": "automem keyword scoring memory",
            "tags": ["automem"],
            "importance": 0.9,
            "timestamp": "2026-06-11T00:00:00+00:00",
        },
    }
    final, components = _compute_metadata_score(
        result, "automem keyword scoring", ["automem", "keyword", "scoring"]
    )

    assert components["keyword"] == pytest.approx(1.0)
    assert final < 2.0
