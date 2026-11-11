"""Original-text fulltext recall is scoped, ranked, state-aware, and hidden."""

from types import SimpleNamespace
from unittest.mock import Mock

import app
import pytest
import automem.api.recall as recall_api
import automem.search.runtime_recall_helpers as helper
import automem.utils.scoring as scoring
from automem.api.recall import handle_recall
from automem.utils.scoring import _compute_metadata_score
from automem.utils.text import _extract_keywords


class Node:
    def __init__(self, memory_id, text, tags=None):
        self.properties = {
            "id": memory_id,
            "content": "short summary",
            "tags": tags or ["scope"],
            "original_search_text": text,
        }


class Graph:
    def __init__(self, nodes):
        self.nodes = nodes
        self.calls = []

    def query(self, query, params):
        self.calls.append((query, dict(params)))
        offset, batch = params["offset"], params["batch"]
        return SimpleNamespace(
            result_set=[[node, 9.0] for node in self.nodes[offset : offset + batch]]
        )


def configure(monkeypatch):
    monkeypatch.setattr(helper, "_serialize_node", lambda node: dict(node.properties))
    monkeypatch.setattr(helper, "_fetch_relations", lambda _graph, _id: [])
    monkeypatch.setattr(helper, "_prepare_tag_filters", lambda tags: tags or [])
    monkeypatch.setattr(helper, "_build_graph_tag_predicate", lambda *_: "m.tags IS NOT NULL")
    monkeypatch.setattr(helper, "_parse_iso_datetime", lambda value: None)
    monkeypatch.setattr(helper, "_logger", Mock())


def test_query_terms_do_not_allow_redisearch_operators():
    terms = helper._original_text_query_tokens('abc-987 | * @secret:李四 "(value)"')
    assert terms == ["abc", "987", "secret", "李四", "value"]
    assert not any(char in term for term in terms for char in '|*@:()"-')


def test_original_search_pages_past_excluded_and_state_filtered_candidates(monkeypatch):
    configure(monkeypatch)
    nodes = [Node(f"stale-{i}", "needle 123", ["scope"]) for i in range(40)]
    nodes += [Node("wrong", "needle 123", ["different"]), Node("hit", "needle 123", ["scope"])]
    graph = Graph(nodes)
    result = helper._original_text_search(
        graph,
        "needle 123",
        1,
        set(),
        tag_filters=["scope"],
        tag_match="exact",
        candidate_filter=lambda page: [r for r in page if not r["id"].startswith("stale-")],
    )
    assert [r["id"] for r in result] == ["hit"]
    assert len(graph.calls) == 2
    assert graph.calls[0][1]["query"] == '"needle" | "123"'
    assert "m.tags IS NOT NULL" in graph.calls[0][0]
    assert "original_search_text" not in result[0]["memory"]
    assert result[0]["score_components"]["original_text"] == 1


def test_original_search_can_upgrade_vector_seen_id(monkeypatch):
    configure(monkeypatch)
    graph = Graph([Node("seen", "needle 123")])
    results = helper._original_text_search(
        graph, "needle 123", 5, {"seen"}, include_seen_ids={"seen"}
    )
    assert [item["id"] for item in results] == ["seen"]
    assert results[0]["match_score"] == 1.0


def test_upgrades_do_not_starve_novel_candidates_across_pages(monkeypatch):
    configure(monkeypatch)
    nodes = [Node(f"seen-{i}", "needle") for i in range(320)] + [Node("novel", "needle")]
    seen = {node.properties["id"] for node in nodes[:-1]}
    graph = Graph(nodes)
    results = helper._original_text_search(graph, "needle", 1, seen, include_seen_ids=seen)
    assert {r["id"] for r in results} == seen | {"novel"}
    assert len(graph.calls) == 11


def test_scan_cap_logs_incomplete_results(monkeypatch):
    configure(monkeypatch)
    monkeypatch.setattr(helper, "_ORIGINAL_TEXT_MAX_SCAN", 40)
    graph = Graph([Node(f"stale-{i}", "needle") for i in range(50)])
    results = helper._original_text_search(
        graph, "needle", 1, set(), candidate_filter=lambda _page: []
    )
    assert results == []
    assert sum(call[1]["batch"] for call in graph.calls) == 40
    assert helper._logger.warning.call_count == 1
    assert "truncated" in helper._logger.warning.call_args[0][0]


def test_coverage_reranks_index_candidates_before_limit(monkeypatch):
    configure(monkeypatch)
    graph = Graph([Node("partial", "needle"), Node("full", "needle 123")])
    result = helper._original_text_search(graph, "needle 123", 1, set())
    assert [r["id"] for r in result] == ["full"]
    assert result[0]["match_score"] == 1.0


def test_current_state_replacement_filter_runs_before_cap_across_pages(monkeypatch):
    configure(monkeypatch)
    graph = Graph(
        [Node(f"superseded-{i}", "needle") for i in range(40)] + [Node("current", "needle")]
    )
    monkeypatch.setattr(
        recall_api,
        "_active_replacements_for_memories",
        lambda _g, ids, _now, _logger: {
            memory_id: {"id": "other-scope", "tags": ["different"]}
            for memory_id in ids
            if memory_id.startswith("superseded-")
        },
    )

    def current_only(page):
        return recall_api._apply_current_state_filter(
            results=page,
            graph=graph,
            result_passes_filters=helper._result_passes_filters,
            start_time=None,
            end_time=None,
            tag_filters=["scope"],
            tag_mode="any",
            tag_match="exact",
            exclude_tags=None,
            logger=Mock(),
        )[0]

    results = helper._original_text_search(
        graph,
        "needle",
        1,
        set(),
        tag_filters=["scope"],
        tag_match="exact",
        candidate_filter=current_only,
    )
    assert [r["id"] for r in results] == ["current"]
    assert len(graph.calls) == 2


def test_missing_index_fails_safe(monkeypatch):
    configure(monkeypatch)
    graph = SimpleNamespace(query=Mock(side_effect=RuntimeError("index unavailable")))
    assert helper._original_text_search(graph, "needle", 5, set()) == []
    helper._logger.exception.assert_called_once()


def test_original_channel_runs_with_full_vector_pool_and_history_mode(monkeypatch):
    vector = {
        "id": "same",
        "memory": {"id": "same", "content": "short"},
        "match_type": "vector",
        "match_score": 0.2,
        "score": 0.2,
    }
    lexical = {
        "id": "same",
        "memory": {"id": "same", "content": "short"},
        "match_type": "keyword",
        "match_score": 0.9,
        "score": 0.9,
    }
    calls = []

    def original(_graph, _query, _limit, _seen, **kwargs):
        calls.append(kwargs)
        return kwargs["candidate_filter"]([lexical])

    monkeypatch.setattr(
        recall_api, "_apply_current_state_filter", lambda **kwargs: (kwargs["results"], {})
    )
    with app.app.test_request_context(
        "/recall?query=needle&limit=1&state_mode=history&adaptive_floor=false"
    ):
        response = handle_recall(
            get_memory_graph=lambda: object(),
            get_qdrant_client=lambda: object(),
            normalize_tag_list=lambda value: value if isinstance(value, list) else [],
            normalize_timestamp=lambda value: value,
            parse_time_expression=lambda _value: (None, None),
            extract_keywords=_extract_keywords,
            compute_metadata_score=lambda result, *_args: (float(result.get("match_score", 0)), {}),
            result_passes_filters=lambda *_args: True,
            graph_keyword_search=lambda *_a, **_k: [],
            vector_search=lambda *_a, **_k: [vector],
            vector_filter_only_tag_search=lambda *_a, **_k: [],
            recall_max_limit=50,
            logger=Mock(),
            original_text_search=original,
        )
    assert calls and calls[0]["include_seen_ids"] == {"same"}
    assert response.get_json()["results"][0]["match_type"] == "keyword"


def test_current_pool_filters_before_limit_and_reports_original_suppressions(monkeypatch):
    monkeypatch.setattr(recall_api, "_active_replacements_for_memories", lambda *_args: {})
    stale = {
        "id": "stale",
        "memory": {"id": "stale", "content": "stale", "t_invalid": "2020-01-01T00:00:00+00:00"},
        "match_type": "vector",
        "match_score": 0.99,
        "score": 0.99,
    }
    original_stale = {
        "id": "original-stale",
        "memory": {
            "id": "original-stale",
            "content": "stale original",
            "t_invalid": "2020-01-01T00:00:00+00:00",
        },
        "match_type": "keyword",
        "match_score": 0.95,
        "score": 0.95,
    }
    current = {
        "id": "current",
        "memory": {"id": "current", "content": "current"},
        "match_type": "keyword",
        "match_score": 0.8,
        "score": 0.8,
    }

    def original(_graph, _query, _limit, _seen, **kwargs):
        return kwargs["candidate_filter"]([original_stale, current])

    with app.app.test_request_context(
        "/recall?query=needle&limit=1&state_debug=true&adaptive_floor=false"
    ):
        response = handle_recall(
            get_memory_graph=lambda: object(),
            get_qdrant_client=lambda: object(),
            normalize_tag_list=lambda value: value if isinstance(value, list) else [],
            normalize_timestamp=lambda value: value,
            parse_time_expression=lambda _value: (None, None),
            extract_keywords=_extract_keywords,
            compute_metadata_score=lambda result, *_args: (float(result.get("match_score", 0)), {}),
            result_passes_filters=lambda *_args: True,
            graph_keyword_search=lambda *_a, **_k: [],
            vector_search=lambda *_a, **_k: [stale],
            vector_filter_only_tag_search=lambda *_a, **_k: [],
            recall_max_limit=50,
            logger=Mock(),
            original_text_search=original,
        )
    body = response.get_json()
    assert [r["id"] for r in body["results"]] == ["current"]
    assert body["state_filter"]["suppressed_count"] == 2
    assert {item["id"] for item in body["state_filter"]["suppressed"]} == {
        "stale",
        "original-stale",
    }


def test_expanded_stale_memory_does_not_duplicate_checked_replacement(monkeypatch):
    current = {
        "id": "current",
        "memory": {"id": "current", "content": "current fact"},
        "match_type": "vector",
        "match_score": 0.9,
        "score": 0.9,
    }
    stale = {
        "id": "stale",
        "memory": {"id": "stale", "content": "old fact"},
        "match_type": "relation",
        "match_score": 0.8,
        "score": 0.8,
        "final_score": 0.8,
    }
    monkeypatch.setattr(recall_api, "_expand_related_memories", lambda **_kwargs: [stale])
    monkeypatch.setattr(
        recall_api,
        "_active_replacements_for_memories",
        lambda _graph, ids, _now, _logger: (
            {
                "stale": {
                    "id": "current",
                    "content": "current fact",
                    "_state_relation": {"type": "INVALIDATED_BY"},
                }
            }
            if "stale" in ids
            else {}
        ),
    )
    with app.app.test_request_context(
        "/recall?query=needle&limit=2&expand_relations=true&state_debug=true&adaptive_floor=false"
    ):
        response = handle_recall(
            get_memory_graph=lambda: object(),
            get_qdrant_client=lambda: object(),
            normalize_tag_list=lambda value: value if isinstance(value, list) else [],
            normalize_timestamp=lambda value: value,
            parse_time_expression=lambda _value: (None, None),
            extract_keywords=_extract_keywords,
            compute_metadata_score=lambda result, *_args: (float(result.get("match_score", 0)), {}),
            result_passes_filters=lambda *_args: True,
            graph_keyword_search=lambda *_a, **_k: [],
            vector_search=lambda *_a, **_k: [current],
            vector_filter_only_tag_search=lambda *_a, **_k: [],
            recall_max_limit=50,
            logger=Mock(),
        )
    body = response.get_json()
    assert [result["id"] for result in body["results"]] == ["current"]
    assert body["state_filter"]["suppressed_count"] == 1
    assert body["state_filter"]["replacement_count"] == 0


def test_only_selected_original_candidate_hydrates_relations(monkeypatch):
    configure(monkeypatch)
    nodes = [Node(f"seen-{i}", "needle") for i in range(40)] + [Node("novel", "needle")]
    for node in nodes:
        node.properties["content"] = node.properties["id"]

    class RecallGraph(Graph):
        def query(self, query, params):
            if "db.idx.fulltext.queryNodes" in query:
                return super().query(query, params)
            return SimpleNamespace(result_set=[])

    graph = RecallGraph(nodes)
    fetch_relations = Mock(
        return_value=[{"type": "RELATED_TO", "memory": {"id": "obsolete", "content": "old fact"}}]
    )
    monkeypatch.setattr(helper, "_fetch_relations", fetch_relations)
    monkeypatch.setattr(recall_api, "_active_replacements_for_memories", lambda *_args: {})
    expansion_seeds = []

    def expand(**kwargs):
        expansion_seeds.extend(kwargs["seed_results"])
        return []

    monkeypatch.setattr(recall_api, "_expand_related_memories", expand)
    vector = [
        {
            "id": n.properties["id"],
            "memory": {"id": n.properties["id"], "content": n.properties["id"]},
            "source": "qdrant",
            "match_type": "vector",
            "match_score": 0.2,
            "score": 0.2,
        }
        for n in nodes[:-1]
    ]
    with app.app.test_request_context(
        "/recall?query=needle&limit=1&expand_relations=true&adaptive_floor=false"
    ):
        response = handle_recall(
            get_memory_graph=lambda: graph,
            get_qdrant_client=lambda: object(),
            normalize_tag_list=lambda value: value if isinstance(value, list) else [],
            normalize_timestamp=lambda value: value,
            parse_time_expression=lambda _value: (None, None),
            extract_keywords=_extract_keywords,
            compute_metadata_score=lambda result, *_args: (float(result.get("match_score", 0)), {}),
            result_passes_filters=lambda *_args: True,
            graph_keyword_search=lambda *_a, **_k: [],
            vector_search=lambda *_a, **_k: vector,
            vector_filter_only_tag_search=lambda *_a, **_k: [],
            recall_max_limit=50,
            logger=Mock(),
            original_text_search=helper._original_text_search,
        )
    body = response.get_json()
    assert len(body["results"]) == 1
    selected = body["results"][0]
    assert len(expansion_seeds) == 1 and expansion_seeds[0]["id"] == selected["id"]
    fetch_relations.assert_called_once_with(graph, selected["id"])
    assert selected["relations"] == [{"type": "RELATED_TO", "memory": {"id": "obsolete"}}]
    assert "_hydrate_original_text_relations" not in selected


def test_state_replacement_retains_existing_provenance(monkeypatch):
    configure(monkeypatch)
    graph = Graph([Node("old", "needle")])
    provenance = [{"type": "INVALIDATED_BY", "from": "old"}]
    fetch = Mock(return_value=[])
    monkeypatch.setattr(helper, "_fetch_relations", fetch)

    def replace(_page):
        return [
            {
                "id": "new",
                "memory": {"id": "new"},
                "match_score": 1.0,
                "match_type": "state_replacement",
                "relations": provenance,
            }
        ]

    results = helper._original_text_search(graph, "needle", 1, set(), candidate_filter=replace)
    helper._hydrate_vector_relations(graph, results)
    assert results[0]["relations"] == provenance
    fetch.assert_not_called()


def test_original_text_evidence_survives_supersession_and_min_score(monkeypatch):
    configure(monkeypatch)
    monkeypatch.setattr(scoring, "SEARCH_WEIGHT_KEYWORD", 0.35)
    graph = Graph([Node("stale", "needle obsolete-private-fact")])
    monkeypatch.setattr(
        recall_api,
        "_active_replacements_for_memories",
        lambda _graph, ids, _now, _logger: (
            {
                "stale": {
                    "id": "current",
                    "content": "corrected fact",
                    "tags": ["scope"],
                    "_state_relation": {"type": "INVALIDATED_BY"},
                }
            }
            if "stale" in ids
            else {}
        ),
    )
    with app.app.test_request_context(
        "/recall?query=needle&limit=1&min_score=0.3&state_mode=current&adaptive_floor=false"
    ):
        response = handle_recall(
            get_memory_graph=lambda: graph,
            get_qdrant_client=lambda: None,
            normalize_tag_list=lambda value: value if isinstance(value, list) else [],
            normalize_timestamp=lambda value: value,
            parse_time_expression=lambda _value: (None, None),
            extract_keywords=_extract_keywords,
            compute_metadata_score=_compute_metadata_score,
            result_passes_filters=lambda *_args: True,
            graph_keyword_search=lambda *_a, **_k: [],
            vector_search=lambda *_a, **_k: [],
            vector_filter_only_tag_search=lambda *_a, **_k: [],
            recall_max_limit=50,
            logger=Mock(),
            original_text_search=helper._original_text_search,
        )
    body = response.get_json()
    assert [r["id"] for r in body["results"]] == ["current"]
    result = body["results"][0]
    assert result["match_type"] == "state_replacement"
    assert result["state_replaces"] == "stale"
    assert result["score_components"]["keyword"] == 1.0
    assert result["score_components"]["vector"] == 0.0
    assert "obsolete-private-fact" not in response.get_data(as_text=True)


def test_state_replacement_inherited_evidence_is_bounded_and_channel_specific():
    replacement = {
        "memory": {"id": "current", "content": "corrected fact"},
        "match_type": "state_replacement",
        "state_replaces": "stale",
        "match_score": 1.0,
        "score_components": {"original_text": 7.0},
    }
    _, components = _compute_metadata_score(replacement, "needle", ["needle"])
    assert components["keyword"] == 1.0
    assert components["vector"] == 0.0
    replacement["score_components"] = {}
    _, components = _compute_metadata_score(replacement, "needle", ["needle"])
    assert components["keyword"] == 0.0
    assert components["vector"] == 0.0


@pytest.mark.parametrize(
    ("weak_count", "direct_current"),
    [(1, False), (32, False), (1, True)],
    ids=["same-page", "cross-page", "replacement-also-matched"],
)
def test_superseded_hits_keep_strongest_original_evidence(monkeypatch, weak_count, direct_current):
    configure(monkeypatch)
    monkeypatch.setattr(scoring, "SEARCH_WEIGHT_KEYWORD", 0.35)
    nodes = [Node(f"weak-{i}", "needle") for i in range(weak_count)]
    if direct_current:
        nodes[0].properties["id"] = "current"
    nodes.append(Node("strong", "needle 123 private-obsolete-fact"))
    graph = Graph(nodes)
    monkeypatch.setattr(
        recall_api,
        "_active_replacements_for_memories",
        lambda _graph, ids, _now, _logger: {
            memory_id: {
                "id": "current",
                "content": "corrected fact",
                "tags": ["scope"],
                "_state_relation": {"type": "INVALIDATED_BY"},
            }
            for memory_id in ids
            if memory_id != "current"
        },
    )
    with app.app.test_request_context(
        "/recall?query=needle%20123&limit=1&min_score=0.3&state_mode=current&adaptive_floor=false"
    ):
        response = handle_recall(
            get_memory_graph=lambda: graph,
            get_qdrant_client=lambda: None,
            normalize_tag_list=lambda value: value if isinstance(value, list) else [],
            normalize_timestamp=lambda value: value,
            parse_time_expression=lambda _value: (None, None),
            extract_keywords=_extract_keywords,
            compute_metadata_score=_compute_metadata_score,
            result_passes_filters=lambda *_args: True,
            graph_keyword_search=lambda *_a, **_k: [],
            vector_search=lambda *_a, **_k: [],
            vector_filter_only_tag_search=lambda *_a, **_k: [],
            recall_max_limit=50,
            logger=Mock(),
            original_text_search=helper._original_text_search,
        )
    body = response.get_json()
    assert [r["id"] for r in body["results"]] == ["current"]
    assert body["results"][0]["state_replaces"] == "strong"
    assert body["results"][0]["score_components"]["original_text"] == 1.0
    assert "private-obsolete-fact" not in response.get_data(as_text=True)


def test_vector_replacement_keeps_evidence_after_context_injection_rerank(monkeypatch):
    monkeypatch.setattr(scoring, "SEARCH_WEIGHT_VECTOR", 0.35)
    stale = {
        "id": "stale",
        "memory": {"id": "stale", "content": "private-obsolete-fact"},
        "source": "qdrant",
        "match_type": "vector",
        "match_score": 1.0,
        "score": 1.0,
    }
    monkeypatch.setattr(
        recall_api,
        "_active_replacements_for_memories",
        lambda _graph, ids, _now, _logger: (
            {
                "stale": {
                    "id": "current",
                    "content": "corrected fact",
                    "tags": [],
                    "_state_relation": {"type": "INVALIDATED_BY"},
                }
            }
            if "stale" in ids
            else {}
        ),
    )
    injected = []

    def inject(results, *_args):
        injected.append(True)
        results.append(
            {
                "id": "context",
                "memory": {"id": "context", "content": "unrelated", "tags": ["priority"]},
                "match_type": "keyword",
                "match_score": 0.05,
                "score": 0.05,
            }
        )
        return True

    monkeypatch.setattr(recall_api, "_inject_priority_memories", inject)
    with app.app.test_request_context(
        "/recall?query=needle&limit=2&min_score=0.3&context_tags=priority&state_mode=current&adaptive_floor=false"
    ):
        response = handle_recall(
            get_memory_graph=lambda: object(),
            get_qdrant_client=lambda: object(),
            normalize_tag_list=lambda value: value if isinstance(value, list) else [],
            normalize_timestamp=lambda value: value,
            parse_time_expression=lambda _value: (None, None),
            extract_keywords=_extract_keywords,
            compute_metadata_score=_compute_metadata_score,
            result_passes_filters=lambda *_args: True,
            graph_keyword_search=lambda *_a, **_k: [],
            vector_search=lambda *_a, **_k: [stale],
            vector_filter_only_tag_search=lambda *_a, **_k: [],
            recall_max_limit=50,
            logger=Mock(),
        )
    body = response.get_json()
    assert injected
    current = next(item for item in body["results"] if item["id"] == "current")
    assert current["match_type"] == "state_replacement"
    assert current["state_replaces"] == "stale"
    assert current["score_components"]["vector"] == 1.0
    assert current["score_components"]["inherited_vector"] == 1.0
    assert "private-obsolete-fact" not in response.get_data(as_text=True)


@pytest.mark.parametrize("channel", ["keyword", "metadata", "exact"])
def test_replacement_retains_other_topical_evidence_after_rerank(monkeypatch, channel):
    monkeypatch.setattr(scoring, "SEARCH_WEIGHT_KEYWORD", 0.35)
    monkeypatch.setattr(scoring, "SEARCH_WEIGHT_METADATA", 0.35)
    monkeypatch.setattr(scoring, "SEARCH_WEIGHT_EXACT", 0.2)
    monkeypatch.setattr(recall_api, "RECALL_METADATA_SEARCH_ENABLED", True)
    stale = {
        "id": "stale",
        "memory": {"id": "stale", "content": "private-obsolete-fact"},
        "source": "graph",
        "match_type": "metadata" if channel == "metadata" else "keyword",
        "match_score": 0.0 if channel == "exact" else 1.0,
        "score": 0.0 if channel == "exact" else 1.0,
    }
    if channel == "exact":
        stale["memory"]["metadata"] = {"topic": "needle"}
    monkeypatch.setattr(
        recall_api,
        "_active_replacements_for_memories",
        lambda _graph, ids, _now, _logger: (
            {
                "stale": {
                    "id": "current",
                    "content": "corrected fact",
                    "tags": [],
                    "_state_relation": {"type": "INVALIDATED_BY"},
                }
            }
            if "stale" in ids
            else {}
        ),
    )
    injected = []

    def inject(results, *_args):
        injected.append(True)
        results.append(
            {
                "id": "context",
                "memory": {"id": "context", "content": "unrelated", "tags": ["priority"]},
                "match_type": "keyword",
                "match_score": 0.05,
                "score": 0.05,
            }
        )
        return True

    monkeypatch.setattr(recall_api, "_inject_priority_memories", inject)
    min_score = 0.15 if channel == "exact" else 0.3
    with app.app.test_request_context(
        f"/recall?query=needle&limit=2&min_score={min_score}&context_tags=priority&state_mode=current&adaptive_floor=false"
    ):
        response = handle_recall(
            get_memory_graph=lambda: object(),
            get_qdrant_client=lambda: None,
            normalize_tag_list=lambda value: value if isinstance(value, list) else [],
            normalize_timestamp=lambda value: value,
            parse_time_expression=lambda _value: (None, None),
            extract_keywords=_extract_keywords,
            compute_metadata_score=_compute_metadata_score,
            result_passes_filters=lambda *_args: True,
            graph_keyword_search=lambda *_a, **_k: [] if channel == "metadata" else [stale],
            metadata_keyword_search=lambda *_a, **_k: [stale] if channel == "metadata" else [],
            vector_search=lambda *_a, **_k: [],
            vector_filter_only_tag_search=lambda *_a, **_k: [],
            recall_max_limit=50,
            logger=Mock(),
        )
    assert injected
    result = next(item for item in response.get_json()["results"] if item["id"] == "current")
    assert result["score_components"][channel] == 1.0
    assert result["score_components"][f"inherited_{channel}"] == 1.0
    assert result["score_components"]["vector"] == 0.0
    for other in {"keyword", "metadata", "exact"} - {channel}:
        assert result["score_components"][other] == 0.0
        assert f"inherited_{other}" not in result["score_components"]
    assert "private-obsolete-fact" not in response.get_data(as_text=True)
