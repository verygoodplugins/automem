import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import app
import automem.api.recall as recall_api
from automem.api.recall import handle_recall
from automem.search.runtime_recall_helpers import _metadata_keyword_search, configure_recall_helpers
from automem.utils.scoring import _compute_metadata_score
from automem.utils.text import _extract_keywords
from tests.support.fake_graph import FakeGraph


def _serialize_node(node):
    return dict(getattr(node, "properties", node))


def _configure_metadata_helpers(fetch_relations=None) -> None:
    configure_recall_helpers(
        parse_iso_datetime=lambda value: (
            datetime.fromisoformat(str(value).replace("Z", "+00:00")) if value else None
        ),
        prepare_tag_filters=lambda tags: [
            str(tag).strip().lower() for tag in (tags or []) if isinstance(tag, str) and tag.strip()
        ],
        build_graph_tag_predicate=lambda _mode, _match: "true",
        build_qdrant_tag_filter=lambda *_args, **_kwargs: None,
        serialize_node=_serialize_node,
        fetch_relations=fetch_relations or (lambda *_args, **_kwargs: []),
        extract_keywords=_extract_keywords,
        coerce_embedding=lambda _value: None,
        generate_real_embedding=lambda _text: [0.1, 0.2, 0.3],
        logger=SimpleNamespace(exception=lambda *_args, **_kwargs: None),
        collection_name="memories",
    )


def test_handle_recall_runs_metadata_sidecar_for_strong_non_field_queries() -> None:
    calls = []

    def metadata_keyword_search(_graph, query_text, *_args, **_kwargs):
        calls.append(query_text)
        return [
            {
                "id": "meta-1",
                "score": 0.8,
                "match_score": 0.8,
                "match_type": "metadata",
                "source": "graph",
                "memory": {
                    "id": "meta-1",
                    "content": "Recall planning note.",
                    "tags": [],
                    "metadata": {"source_agent": "hub-developer"},
                },
                "relations": [],
            }
        ]

    with app.app.test_request_context("/recall?query=hub-developer%20run%20notes&limit=5"):
        response = handle_recall(
            get_memory_graph=lambda: object(),
            get_qdrant_client=lambda: None,
            normalize_tag_list=lambda value: value if isinstance(value, list) else [],
            normalize_timestamp=lambda value: value,
            parse_time_expression=lambda _value: (None, None),
            extract_keywords=_extract_keywords,
            compute_metadata_score=lambda result, _query, _tokens, _context: (
                float(result["match_score"]),
                {"metadata": float(result["match_score"])},
            ),
            result_passes_filters=lambda *_args, **_kwargs: True,
            graph_keyword_search=lambda *_args, **_kwargs: [],
            vector_search=lambda *_args, **_kwargs: [],
            vector_filter_only_tag_search=lambda *_args, **_kwargs: [],
            metadata_keyword_search=metadata_keyword_search,
            recall_max_limit=50,
            logger=Mock(),
        )

    data = response.get_json()
    assert calls == ["hub-developer run notes"]
    assert data["results"][0]["id"] == "meta-1"
    assert data["results"][0]["match_type"] == "metadata"


def test_handle_recall_skips_metadata_sidecar_when_disabled() -> None:
    metadata_keyword_search = Mock(return_value=[])
    previous = recall_api.RECALL_METADATA_SEARCH_ENABLED
    recall_api.RECALL_METADATA_SEARCH_ENABLED = False

    try:
        with app.app.test_request_context("/recall?query=hub-developer%20run%20notes"):
            handle_recall(
                get_memory_graph=lambda: object(),
                get_qdrant_client=lambda: None,
                normalize_tag_list=lambda value: value if isinstance(value, list) else [],
                normalize_timestamp=lambda value: value,
                parse_time_expression=lambda _value: (None, None),
                extract_keywords=_extract_keywords,
                compute_metadata_score=lambda result, _query, _tokens, _context: (
                    float(result.get("match_score", 0.0)),
                    {},
                ),
                result_passes_filters=lambda *_args, **_kwargs: True,
                graph_keyword_search=lambda *_args, **_kwargs: [],
                vector_search=lambda *_args, **_kwargs: [],
                vector_filter_only_tag_search=lambda *_args, **_kwargs: [],
                metadata_keyword_search=metadata_keyword_search,
                recall_max_limit=50,
                logger=Mock(),
            )
    finally:
        recall_api.RECALL_METADATA_SEARCH_ENABLED = previous

    metadata_keyword_search.assert_not_called()


def test_metadata_match_type_uses_metadata_score_component() -> None:
    score, components = _compute_metadata_score(
        {
            "match_type": "metadata",
            "match_score": 0.8,
            "memory": {
                "content": "Recall planning note.",
                "metadata": {"source_agent": "hub-developer"},
            },
        },
        "memories with source agent hub-developer",
        ["memories", "source", "agent", "hub-developer"],
    )

    assert components["metadata"] == 0.8
    assert components["keyword"] == 0.0
    assert score > 0


def test_metadata_keyword_search_matches_whitelisted_hidden_metadata() -> None:
    _configure_metadata_helpers()
    graph = FakeGraph()
    graph.memories["meta-1"] = {
        "id": "meta-1",
        "content": "Recall planning note.",
        "tags": ["automem"],
        "metadata": json.dumps(
            {
                "source_agent": "hub-developer",
                "original_content": "source agent forbidden-noise",
                "entities": {"people": ["Hub Developer"], "organizations": ["AutoMem"]},
            }
        ),
        "importance": 0.4,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    graph.memories["people-only"] = {
        "id": "people-only",
        "content": "Unrelated note.",
        "tags": ["automem"],
        "metadata": {"entities": {"people": ["Hub Developer"]}},
        "importance": 0.9,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    results = _metadata_keyword_search(
        graph,
        "memories with source agent hub-developer",
        10,
        set(),
        tag_filters=["automem"],
    )

    assert [result["id"] for result in results] == ["meta-1"]
    assert results[0]["match_type"] == "metadata"
    assert results[0]["score_components"]["metadata"] > 0


def test_metadata_keyword_search_does_not_require_field_words() -> None:
    _configure_metadata_helpers()
    graph = FakeGraph()
    graph.memories["meta-1"] = {
        "id": "meta-1",
        "content": "Recall planning note.",
        "tags": ["automem"],
        "metadata": {"source_agent": "hub-developer"},
        "importance": 0.4,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    results = _metadata_keyword_search(
        graph,
        "hub developer",
        10,
        set(),
        tag_filters=["automem"],
    )

    assert [result["id"] for result in results] == ["meta-1"]


def test_metadata_keyword_search_fetches_relations_only_for_returned_results() -> None:
    relation_calls: list[str] = []

    def counting_fetch_relations(_graph, memory_id, *_args, **_kwargs):
        relation_calls.append(str(memory_id))
        return []

    _configure_metadata_helpers(fetch_relations=counting_fetch_relations)
    graph = FakeGraph()
    for index in range(8):
        graph.memories[f"meta-{index}"] = {
            "id": f"meta-{index}",
            "content": "Recall planning note.",
            "tags": ["automem"],
            "metadata": {"source_agent": "hub-developer"},
            "importance": 0.4,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    results = _metadata_keyword_search(
        graph,
        "hub developer",
        3,
        set(),
        tag_filters=["automem"],
    )

    assert len(results) == 3
    assert sorted(relation_calls) == sorted(result["id"] for result in results)


def test_metadata_keyword_search_uses_field_words_as_scoring_context() -> None:
    _configure_metadata_helpers()
    graph = FakeGraph()
    graph.memories["generic-entity"] = {
        "id": "generic-entity",
        "content": "General MCP note.",
        "tags": ["automem"],
        "metadata": {"entities": {"organizations": ["MCP"]}},
        "importance": 0.95,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    graph.memories["repo-target"] = {
        "id": "repo-target",
        "content": "Unrelated implementation note.",
        "tags": ["automem"],
        "metadata": {"repo": "verygoodplugins/streamdeck-mcp"},
        "importance": 0.2,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    results = _metadata_keyword_search(
        graph,
        "repo verygoodplugins streamdeck mcp",
        10,
        set(),
        tag_filters=["automem"],
    )

    assert [result["id"] for result in results] == ["repo-target"]


def test_metadata_keyword_search_rejects_partial_repo_field_matches() -> None:
    _configure_metadata_helpers()
    graph = FakeGraph()
    graph.memories["other-repo"] = {
        "id": "other-repo",
        "content": "WhatsApp implementation note.",
        "tags": ["automem"],
        "metadata": {"repo": "verygoodplugins/whatsapp-mcp"},
        "importance": 0.95,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    graph.memories["repo-target"] = {
        "id": "repo-target",
        "content": "Stream Deck implementation note.",
        "tags": ["automem"],
        "metadata": {"repo": "verygoodplugins/streamdeck-mcp"},
        "importance": 0.2,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    results = _metadata_keyword_search(
        graph,
        "repo verygoodplugins streamdeck mcp",
        10,
        set(),
        tag_filters=["automem"],
    )

    assert [result["id"] for result in results] == ["repo-target"]


def test_metadata_keyword_search_allows_short_values_when_field_is_explicit() -> None:
    _configure_metadata_helpers()
    graph = FakeGraph()
    graph.memories["repo-target"] = {
        "id": "repo-target",
        "content": "Unrelated implementation note.",
        "tags": ["automem"],
        "metadata": {"repo": "mcp"},
        "importance": 0.2,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    graph.memories["generic-entity"] = {
        "id": "generic-entity",
        "content": "General MCP note.",
        "tags": ["automem"],
        "metadata": {"entities": {"organizations": ["MCP"]}},
        "importance": 0.95,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    results = _metadata_keyword_search(
        graph,
        "repo mcp",
        10,
        set(),
        tag_filters=["automem"],
    )

    assert [result["id"] for result in results] == ["repo-target"]


def test_metadata_keyword_search_skips_entities_without_entity_field_context() -> None:
    _configure_metadata_helpers()
    graph = FakeGraph()
    graph.memories["entity-only"] = {
        "id": "entity-only",
        "content": "Unrelated note.",
        "tags": ["automem"],
        "metadata": {"entities": {"organizations": ["Root Cause"]}},
        "importance": 0.95,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    results = _metadata_keyword_search(
        graph,
        "root cause",
        10,
        set(),
        tag_filters=["automem"],
    )

    assert results == []


def test_metadata_keyword_search_allows_entities_with_entity_field_context() -> None:
    _configure_metadata_helpers()
    graph = FakeGraph()
    graph.memories["entity-only"] = {
        "id": "entity-only",
        "content": "Unrelated note.",
        "tags": ["automem"],
        "metadata": {"entities": {"organizations": ["Root Cause"]}},
        "importance": 0.95,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    results = _metadata_keyword_search(
        graph,
        "entities organizations root cause",
        10,
        set(),
        tag_filters=["automem"],
    )

    assert [result["id"] for result in results] == ["entity-only"]


def test_metadata_keyword_search_rejects_weak_single_token_noise() -> None:
    _configure_metadata_helpers()
    graph = FakeGraph()
    graph.memories["weak"] = {
        "id": "weak",
        "content": "Unrelated operational note.",
        "tags": ["automem"],
        "metadata": {"source": "blog"},
        "importance": 0.9,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    results = _metadata_keyword_search(
        graph,
        "Which blog posts were recently published?",
        10,
        set(),
        tag_filters=["automem"],
    )

    assert results == []


def test_metadata_keyword_search_respects_tag_filters() -> None:
    _configure_metadata_helpers()
    graph = FakeGraph()
    for memory_id, tags in (("allowed", ["automem"]), ("blocked", ["other"])):
        graph.memories[memory_id] = {
            "id": memory_id,
            "content": "Recall planning note.",
            "tags": tags,
            "metadata": {"repo": "verygoodplugins/automem"},
            "importance": 0.5,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    results = _metadata_keyword_search(
        graph,
        "memories from repo verygoodplugins automem",
        10,
        set(),
        tag_filters=["automem"],
    )

    assert [result["id"] for result in results] == ["allowed"]
