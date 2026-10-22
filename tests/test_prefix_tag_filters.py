"""Prefix tag filters use hierarchy delimiters; exact tags retain their spelling."""

from types import SimpleNamespace

import pytest
from flask import Flask

import automem.search.runtime_recall_helpers as helpers
from automem.stores.graph_store import _build_graph_tag_predicate
from automem.stores.vector_store import _build_qdrant_tag_filter
from automem.utils.tags import _compute_tag_prefixes, _prepare_tag_filters
from tests.test_api_endpoints import MockQdrantClient


@pytest.fixture(autouse=True)
def configured_helpers(monkeypatch):
    monkeypatch.setattr(helpers, "_prepare_tag_filters", _prepare_tag_filters)
    monkeypatch.setattr(helpers, "_parse_iso_datetime", lambda value: value)
    monkeypatch.setattr(helpers, "_build_graph_tag_predicate", _build_graph_tag_predicate)
    monkeypatch.setattr(helpers, "_logger", SimpleNamespace(exception=lambda *args: None))
    monkeypatch.setattr(helpers, "_extract_keywords", lambda query: query.split())
    monkeypatch.setattr(helpers, "_serialize_node", lambda node: node)
    monkeypatch.setattr(helpers, "_fetch_relations", lambda *args: [])


@pytest.mark.parametrize("mode", ["any", "all"])
@pytest.mark.parametrize("with_prefixes", [True, False])
def test_result_filters_match_slash_hierarchy_and_reject_hyphen(mode, with_prefixes):
    for tag, expected in [("auth/login:password", True), ("auth-login", False)]:
        memory = {"tags": [tag]}
        if with_prefixes:
            memory["tag_prefixes"] = _compute_tag_prefixes(memory["tags"])
        result = {"memory": memory}
        assert (
            helpers._result_passes_filters(result, None, None, [" Auth/Login "], mode, "prefix")
            is expected
        )
        assert (
            helpers._result_passes_filters(result, None, None, ["auth"], mode, "prefix") is expected
        )


def test_exact_filter_keeps_delimiter_distinction():
    result = {"memory": {"tags": ["auth/login"], "tag_prefixes": ["auth", "auth:login"]}}
    assert helpers._result_passes_filters(result, None, None, ["auth/login"], "any", "exact")
    assert not helpers._result_passes_filters(result, None, None, ["auth:login"], "any", "exact")


@pytest.mark.parametrize("mode", ["any", "all"])
@pytest.mark.parametrize("match, expected", [("prefix", "auth:login"), ("exact", "auth/login")])
@pytest.mark.parametrize("path", ["trending", "keyword", "metadata"])
def test_graph_query_receives_filters_for_selected_mode(mode, match, expected, path):
    queries = []
    graph = SimpleNamespace(
        query=lambda query, params: queries.append((query, params))
        or SimpleNamespace(result_set=[])
    )
    with Flask(__name__).test_request_context("/recall"):
        kwargs = {"tag_filters": [" Auth/Login "], "tag_mode": mode, "tag_match": match}
        if path == "trending":
            helpers._graph_trending_results(graph, 5, set(), **kwargs)
        elif path == "keyword":
            helpers._graph_keyword_search(graph, "password", 5, set(), **kwargs)
        else:
            helpers._metadata_keyword_search(graph, "password", 5, set(), **kwargs)
    assert queries[0][1]["tag_filters"] == [expected]


@pytest.mark.parametrize("mode", ["any", "all"])
def test_qdrant_executes_slash_filter_as_hierarchical_prefix(mode):
    client = MockQdrantClient()
    tags = ["auth/login:password", "auth-login", "auth:login"]
    client.points = {
        i: {"payload": {"tags": [tag], "tag_prefixes": _compute_tag_prefixes([tag])}}
        for i, tag in enumerate(tags)
    }
    points, _ = client.scroll(
        "tags", scroll_filter=_build_qdrant_tag_filter([" Auth/Login "], mode, "prefix")
    )
    assert {point.id for point in points} == {0, 2}
    points, _ = client.scroll(
        "tags", scroll_filter=_build_qdrant_tag_filter(["auth/login:password"], mode, "exact")
    )
    assert {point.id for point in points} == {0}
