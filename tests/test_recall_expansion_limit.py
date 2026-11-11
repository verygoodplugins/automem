"""The ranked recall limit applies to seeds, expansions and fallback together."""

from copy import deepcopy
from unittest.mock import Mock

import pytest

import app
import automem.api.recall as recall
from tests.support.fake_graph import FakeGraph


def record(memory_id, score, *, kind="vector", timestamp="2022-01-01T00:00:00Z", **memory):
    return {
        "id": memory_id,
        "memory": {"id": memory_id, "content": memory_id, "timestamp": timestamp, **memory},
        "source": "graph",
        "match_type": kind,
        "match_score": score,
        "score": score,
        "final_score": score,
        "relations": [],
    }


@pytest.fixture
def run_recall(monkeypatch):
    graph = FakeGraph()
    monkeypatch.setattr(recall, "_hydrate_vector_relations", lambda *_args: None)
    monkeypatch.setattr(recall, "_hydrate_missing_summaries_from_graph", lambda *_args: None)
    monkeypatch.setattr(recall, "_filter_inline_relation_state", lambda *_args, **_kwargs: None)

    def run(
        query,
        seeds,
        relation=(),
        entity=(),
        fallback=(),
        on_access=None,
        passes_filters=None,
        original=(),
    ):
        fallback_calls = []
        if relation is not None:
            monkeypatch.setattr(
                recall, "_expand_related_memories", lambda **_kwargs: deepcopy(list(relation))
            )
        monkeypatch.setattr(
            recall, "_expand_entity_memories", lambda **_kwargs: deepcopy(list(entity))
        )

        def scope_fallback(**kwargs):
            fallback_calls.append(kwargs)
            return deepcopy(list(fallback))

        monkeypatch.setattr(recall, "_apply_scope_fallback", scope_fallback)
        with app.app.test_request_context(query):
            response = recall.handle_recall(
                get_memory_graph=lambda: graph,
                get_qdrant_client=lambda: object(),
                normalize_tag_list=lambda value: value if isinstance(value, list) else [],
                normalize_timestamp=lambda value: value,
                parse_time_expression=lambda _value: (None, None),
                extract_keywords=lambda _query: [],
                compute_metadata_score=lambda result, *_args: (result["match_score"], {}),
                result_passes_filters=passes_filters or (lambda *_args: True),
                graph_keyword_search=lambda *_args, **_kwargs: [],
                vector_search=lambda *_args, **_kwargs: deepcopy(list(seeds)),
                vector_filter_only_tag_search=lambda *_args, **_kwargs: [],
                recall_max_limit=50,
                logger=Mock(),
                on_access=on_access,
                original_text_search=(
                    (lambda *_args, **kwargs: kwargs["candidate_filter"](deepcopy(list(original))))
                    if original
                    else None
                ),
            )
        return response.get_json(), fallback_calls

    run.graph = graph
    return run


def test_relation_and_entity_expansion_compete_with_seeds_within_limit(run_recall):
    body, _ = run_recall(
        "/recall?query=needle&limit=2&expand_relations=true&expand_entities=true&adaptive_floor=false",
        [record("seed-a", 0.4), record("seed-b", 0.3)],
        [record("relation", 0.9, kind="relation")],
        [record("entity", 0.7, kind="vector")],
    )
    assert body["count"] == 2
    assert [result["id"] for result in body["results"]] == ["relation", "entity"]
    assert body["expansion"]["expanded_count"] == 1
    assert body["expansion"]["candidate_count"] == 1
    assert body["entity_expansion"]["expanded_count"] == 1
    assert body["entity_expansion"]["candidate_count"] == 1


def test_priority_id_remains_guaranteed_when_expansion_is_stronger(run_recall):
    body, _ = run_recall(
        "/recall?query=needle&limit=2&priority_ids=seed-b&expand_relations=true&adaptive_floor=false",
        [record("seed-a", 0.4), record("seed-b", 0.3)],
        [record("relation", 0.9, kind="relation")],
    )
    assert body["count"] == 2
    assert {result["id"] for result in body["results"]} == {"seed-b", "relation"}


def test_superseded_priority_id_guarantees_current_replacement(run_recall):
    run_recall.graph.memories.update(
        {
            "old": {"id": "old", "content": "old"},
            "new": {"id": "new", "content": "new"},
        }
    )
    run_recall.graph.relationships.append(
        {"id1": "old", "id2": "new", "type": "INVALIDATED_BY", "strength": 0.9}
    )
    body, _ = run_recall(
        "/recall?query=needle&limit=1&priority_ids=old&expand_relations=true&adaptive_floor=false",
        [record("old", 0.4)],
        [record("relation", 0.9, kind="relation")],
    )
    assert [result["id"] for result in body["results"]] == ["new"]
    assert body["results"][0]["state_replaces"] == "old"
    assert body["expansion"]["expanded_count"] == 0


def test_priority_alias_and_current_id_do_not_duplicate_the_replacement():
    replacement = record("new", 0.4)
    replacement["state_replaces"] = "old"
    other = record("other", 0.3)
    selected = recall._guarantee_priority_results(
        [replacement, other], {"priority_id_order": ["old", "new"]}, 2
    )
    assert [result["id"] for result in selected] == ["new", "other"]


def test_superseded_priority_does_not_restore_filtered_replacement(run_recall):
    run_recall.graph.memories.update(
        {"old": {"id": "old", "content": "old"}, "new": {"id": "new", "content": "new"}}
    )
    run_recall.graph.relationships.append(
        {"id1": "old", "id2": "new", "type": "INVALIDATED_BY", "strength": 0.9}
    )
    body, _ = run_recall(
        "/recall?query=needle&limit=1&priority_ids=old&expand_relations=true&adaptive_floor=false",
        [record("old", 0.4)],
        [record("relation", 0.9, kind="relation")],
        passes_filters=lambda result, *_args: result["id"] != "new",
    )
    assert [result["id"] for result in body["results"]] == ["relation"]


@pytest.mark.parametrize("allow_replacement,expected", [(True, "new"), (False, "relation")])
def test_priority_current_replacement_already_in_pool(run_recall, allow_replacement, expected):
    run_recall.graph.memories.update(
        {"old": {"id": "old", "content": "old"}, "new": {"id": "new", "content": "new"}}
    )
    run_recall.graph.relationships.append(
        {"id1": "old", "id2": "new", "type": "INVALIDATED_BY", "strength": 0.9}
    )
    replacement = record("new", 0.2, kind="relation")
    body, _ = run_recall(
        "/recall?query=needle&limit=1&priority_ids=old&expand_relations=true&adaptive_floor=false",
        [record("old", 0.4)],
        [replacement, record("relation", 0.9, kind="relation")],
        passes_filters=lambda result, *_args: allow_replacement or result["id"] != "new",
    )
    assert [result["id"] for result in body["results"]] == [expected]
    assert body["count"] == 1
    assert not any(key.startswith("_priority") for key in body["state_filter"])
    if allow_replacement:
        # This integration base resolves the stale seed before expansion.
        assert body["results"][0]["final_score"] == 0.4
        assert body["results"][0]["state_replaces"] == "old"
        assert body["results"][0]["match_type"] == "state_replacement"
        assert body["results"][0]["score_components"]["inherited_vector"] == 0.4


def test_existing_head_in_seed_pool_keeps_priority_alias(run_recall):
    run_recall.graph.memories.update(
        {"old": {"id": "old", "content": "old"}, "new": {"id": "new", "content": "new"}}
    )
    run_recall.graph.relationships.append(
        {"id1": "old", "id2": "new", "type": "INVALIDATED_BY", "strength": 0.9}
    )
    body, _ = run_recall(
        "/recall?query=needle&limit=1&priority_ids=old&expand_relations=true&adaptive_floor=false",
        [record("old", 0.1), record("new", 0.2)],
        [record("relation", 0.9, kind="relation")],
    )
    assert [result["id"] for result in body["results"]] == ["new"]
    assert body["results"][0]["final_score"] == 0.2
    assert "state_replaces" not in body["results"][0]


def test_original_text_early_filter_preserves_priority_for_existing_head(run_recall):
    run_recall.graph.memories.update(
        {"old": {"id": "old", "content": "old"}, "new": {"id": "new", "content": "new"}}
    )
    run_recall.graph.relationships.append(
        {"id1": "old", "id2": "new", "type": "INVALIDATED_BY", "strength": 0.9}
    )
    body, _ = run_recall(
        "/recall?query=needle&limit=1&priority_ids=old&expand_relations=true&adaptive_floor=false",
        [record("new", 0.2)],
        [record("relation", 0.9, kind="relation")],
        original=[record("old", 0.1, kind="keyword")],
    )
    assert [result["id"] for result in body["results"]] == ["new"]
    assert body["results"][0]["final_score"] == 0.2
    assert "state_replaces" not in body["results"][0]


@pytest.mark.parametrize(
    "sort,expected", [("time_desc", ["relation", "seed-b"]), ("time_asc", ["seed-a", "seed-b"])]
)
def test_expansion_selection_respects_requested_time_order(run_recall, sort, expected):
    body, _ = run_recall(
        f"/recall?query=needle&limit=2&sort={sort}&expand_relations=true&adaptive_floor=false",
        [record("seed-a", 0.4, timestamp="2020"), record("seed-b", 0.3, timestamp="2022")],
        [record("relation", 0.9, kind="relation", timestamp="2024")],
    )
    assert body["count"] == 2
    assert [result["id"] for result in body["results"]] == expected


@pytest.mark.parametrize("history,expected", [(False, "seed"), (True, "expired")])
def test_expired_expansion_is_only_selected_in_history_mode(run_recall, history, expected):
    body, _ = run_recall(
        f"/recall?query=needle&limit=1&state_mode={'history' if history else 'current'}"
        "&expand_relations=true&adaptive_floor=false",
        [record("seed", 0.4)],
        [record("expired", 0.9, kind="relation", t_invalid="2020-01-01T00:00:00Z")],
    )
    assert body["count"] == 1
    assert body["results"][0]["id"] == expected
    assert body["expansion"]["expanded_count"] == int(history)
    assert body["expansion"]["candidate_count"] == 1


def test_scope_fallback_only_fills_after_scoped_results_are_selected(run_recall):
    outside = record("outside", 0.9)
    outside["outside_tag_scope"] = True
    body, calls = run_recall(
        "/recall?query=needle&tags=scope&limit=2&scope_fallback=true"
        "&expand_relations=true&min_score=0.2&adaptive_floor=false",
        [record("seed", 0.4)],
        [record("low-expansion", 0.1, kind="relation")],
        fallback=[outside],
    )
    assert body["count"] == 2
    assert [result["id"] for result in body["results"]] == ["seed", "outside"]
    assert len(calls) == 1
    assert body["expansion"]["expanded_count"] == 0
    assert body["expansion"]["candidate_count"] == 1


def test_only_returned_direct_seeds_are_marked_accessed(run_recall):
    on_access = Mock()
    body, _ = run_recall(
        "/recall?query=needle&limit=2&expand_relations=true&adaptive_floor=false",
        [record("seed-a", 0.4), record("seed-b", 0.3)],
        [record("relation", 0.9, kind="relation")],
        on_access=on_access,
    )
    assert [result["id"] for result in body["results"]] == ["relation", "seed-a"]
    on_access.assert_called_once_with(["seed-a"])


def test_real_relation_expansion_is_ranked_and_reported_as_returned(run_recall):
    run_recall.graph.memories.update(
        {
            "seed": {"id": "seed", "content": "seed"},
            "neighbor": {"id": "neighbor", "content": "neighbor"},
        }
    )
    run_recall.graph.relationships.append(
        {"id1": "seed", "id2": "neighbor", "type": "RELATES_TO", "strength": 0.9}
    )
    body, _ = run_recall(
        "/recall?query=needle&limit=1&expand_relations=true&adaptive_floor=false",
        [record("seed", 0.4)],
        relation=None,
    )
    assert [result["id"] for result in body["results"]] == ["neighbor"]
    assert body["count"] == 1
    assert body["expansion"]["expanded_count"] == 1
    assert body["expansion"]["candidate_count"] == 1
