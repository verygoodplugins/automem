"""Recall must over-fetch vector candidates so the richer final scoring can
re-rank a wide enough pool.

Reproduces the production miss: a high-importance, exact-topic memory that
ranked just past the requested limit in *pure vector* similarity was discarded
before its importance/keyword signal could ever lift it, while the pool filled
with high-frequency-token ("berlin") noise. Over-fetching the candidate pool
lets the existing re-rank surface it.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

import app
import automem.api.recall as recall_module
import automem.search.runtime_recall_helpers as recall_helpers
from tests.support.fake_graph import FakeGraph


class _FakeQdrant:
    """Records the requested limit and returns hits[:limit] (Qdrant top-K)."""

    def __init__(self, hits):
        self._hits = hits
        self.last_limit = None

    def search(self, *, collection_name, query_vector, limit, with_payload=True, query_filter=None):
        self.last_limit = limit
        return list(self._hits[:limit])


def _hit(memory_id, score, *, content, importance, tags):
    payload = {
        "id": memory_id,
        "content": content,
        "tags": tags,
        "importance": importance,
        "confidence": 0.9,
        "type": "Context",
        "enriched": True,  # skip JIT enrichment during recall
        "timestamp": "2026-05-18T00:00:00+00:00",
    }
    return SimpleNamespace(id=memory_id, payload=payload, score=score)


@pytest.fixture
def overfetch_env(monkeypatch):
    state = app.ServiceState()
    state.memory_graph = FakeGraph()

    # 8 high-similarity but low-value "berlin" decoys, then the real target at
    # vector-rank index 8 — just beyond the default per_query_limit of 5.
    decoys = [
        _hit(
            f"decoy-{i}",
            0.60 - i * 0.001,
            content="berlin weather morning note",
            importance=0.0,
            tags=["entity:concepts:berlin"],
        )
        for i in range(8)
    ]
    target = _hit(
        "lennard-bafoeg",
        0.50,
        content="lennard bafoeg benefits situation resolved german bureaucracy",
        importance=0.9,
        tags=["lennard", "bafoeg", "benefits"],
    )
    fake_qdrant = _FakeQdrant(decoys + [target])
    state.qdrant = fake_qdrant

    monkeypatch.setattr(app, "state", state)
    monkeypatch.setattr(app, "init_falkordb", lambda: None)
    monkeypatch.setattr(app, "init_qdrant", lambda: None)
    monkeypatch.setattr(app, "API_TOKEN", "test-token")
    monkeypatch.setattr(app, "ADMIN_TOKEN", "test-admin-token")

    # Deterministic embedding + no graph relation lookups for vector hits.
    # Pin a real logger so the vector path is independent of any helper-state
    # pollution left by sibling tests that reconfigure the recall helpers.
    monkeypatch.setattr(recall_helpers, "_generate_real_embedding", lambda _t: [0.1] * 8)
    monkeypatch.setattr(recall_helpers, "_fetch_relations", lambda *_a, **_k: [])
    monkeypatch.setattr(recall_helpers, "_logger", logging.getLogger("automem.test.recall"))

    return fake_qdrant


def _recall(fake_qdrant):
    client = app.app.test_client()
    resp = client.get(
        "/recall?query=lennard%20benefits%20situation&limit=5&current_only=false",
        headers={"Authorization": "Bearer test-token"},
    )
    assert resp.status_code == 200, resp.get_data(as_text=True)
    body = resp.get_json()
    ids = [r.get("id") or (r.get("memory") or {}).get("id") for r in body["results"]]
    return ids, fake_qdrant.last_limit


def test_vector_overfetch_requests_widened_limit(overfetch_env):
    ids, last_limit = _recall(overfetch_env)
    # default RECALL_VECTOR_OVERFETCH=4 -> 5 * 4 = 20 candidates fetched
    assert last_limit == 20
    # the relevant memory ranked past 1x is now recalled and re-ranked to the top
    assert "lennard-bafoeg" in ids
    assert len(ids) <= 5  # response size unchanged by over-fetch


def test_vector_overfetch_off_switch_uses_1x(overfetch_env, monkeypatch):
    monkeypatch.setattr(recall_module, "RECALL_VECTOR_OVERFETCH", 1)
    ids, last_limit = _recall(overfetch_env)
    assert last_limit == 5
    # without over-fetch the target never enters the candidate pool
    assert "lennard-bafoeg" not in ids


def test_vector_fetch_cap_never_reduces_below_requested_limit(overfetch_env, monkeypatch):
    monkeypatch.setattr(recall_module, "RECALL_VECTOR_FETCH_CAP", 3)
    _ids, last_limit = _recall(overfetch_env)
    assert last_limit == 5


def test_tag_scoped_vector_fetch_respects_cap(overfetch_env, monkeypatch):
    monkeypatch.setattr(recall_module, "RECALL_VECTOR_FETCH_CAP", 20)

    client = app.app.test_client()
    resp = client.get(
        "/recall?query=lennard%20benefits%20situation&tags=benefits&limit=5&current_only=false",
        headers={"Authorization": "Bearer test-token"},
    )

    assert resp.status_code == 200, resp.get_data(as_text=True)
    assert overfetch_env.last_limit == 20


def test_vector_overfetch_hydrates_relations_after_trim(overfetch_env, monkeypatch):
    relation_calls = []

    def _counting_fetch_relations(_graph, memory_id):
        relation_calls.append(memory_id)
        return []

    monkeypatch.setattr(recall_helpers, "_fetch_relations", _counting_fetch_relations)

    ids, last_limit = _recall(overfetch_env)

    assert last_limit == 20
    assert relation_calls
    assert set(relation_calls).issubset(set(ids))
    assert len(relation_calls) <= len(ids) <= 5


def test_vector_overfetch_does_not_hide_metadata_upgrade() -> None:
    metadata_seen_sets = []
    metadata_include_sets = []

    def _result(memory_id, score, match_type="vector"):
        return {
            "id": memory_id,
            "score": score,
            "match_score": score,
            "match_type": match_type,
            "source": "qdrant" if match_type == "vector" else "graph",
            "memory": {
                "id": memory_id,
                "content": f"content for {memory_id}",
                "tags": [],
                "target_score": score,
                "type": "Context",
                "enriched": True,
                "timestamp": "2026-05-18T00:00:00+00:00",
            },
            "relations": [],
        }

    vector_results = [
        _result("decoy-0", 0.95),
        _result("decoy-1", 0.94),
        _result("decoy-2", 0.93),
        _result("decoy-3", 0.92),
        _result("decoy-4", 0.91),
        _result("metadata-target", 0.10),
    ]

    def _vector_search(_qdrant, _graph, _query, _embedding, _limit, seen_ids, *_args):
        for result in vector_results:
            seen_ids.add(result["id"])
        return [dict(result) for result in vector_results]

    def _metadata_keyword_search(_graph, _query, _limit, seen_ids, **kwargs):
        include_seen_ids = set(kwargs.get("include_seen_ids") or [])
        metadata_seen_sets.append(set(seen_ids))
        metadata_include_sets.append(include_seen_ids)
        if "metadata-target" in seen_ids and "metadata-target" not in include_seen_ids:
            return []
        return [_result("metadata-target", 1.25, match_type="metadata")]

    graph = FakeGraph()

    with app.app.test_request_context("/recall?query=metadata%20target&limit=5&current_only=false"):
        response = recall_module.handle_recall(
            get_memory_graph=lambda: graph,
            get_qdrant_client=lambda: object(),
            normalize_tag_list=lambda value: value if isinstance(value, list) else [],
            normalize_timestamp=lambda value: value,
            parse_time_expression=lambda _value: (None, None),
            extract_keywords=lambda _query: ["metadata", "target"],
            compute_metadata_score=lambda result, _query, _tokens, _context: (
                float((result.get("memory") or {}).get("target_score") or 0.0),
                {},
            ),
            result_passes_filters=lambda *_args, **_kwargs: True,
            graph_keyword_search=lambda *_args, **_kwargs: [],
            vector_search=_vector_search,
            vector_filter_only_tag_search=lambda *_args, **_kwargs: [],
            metadata_keyword_search=_metadata_keyword_search,
            recall_max_limit=50,
            logger=SimpleNamespace(
                debug=lambda *_args, **_kwargs: None,
                info=lambda *_args, **_kwargs: None,
                exception=lambda *_args, **_kwargs: None,
            ),
        )

    results = response.get_json()["results"]
    vector_ids = {result["id"] for result in vector_results}
    assert metadata_seen_sets == [vector_ids]
    assert metadata_include_sets == [vector_ids]
    assert results[0]["id"] == "metadata-target"
    assert results[0]["match_type"] == "metadata"


def test_metadata_vector_duplicates_do_not_consume_trimmed_slots() -> None:
    def _result(memory_id, score, match_type="vector"):
        return {
            "id": memory_id,
            "score": score,
            "match_score": score,
            "match_type": match_type,
            "source": "qdrant" if match_type == "vector" else "graph",
            "memory": {
                "id": memory_id,
                "content": f"content for {memory_id}",
                "tags": [],
                "target_score": score,
                "type": "Context",
                "enriched": True,
                "timestamp": "2026-05-18T00:00:00+00:00",
            },
            "relations": [],
        }

    vector_results = [
        _result("metadata-target", 1.10),
        _result("unique-1", 1.00),
        _result("unique-2", 0.90),
        _result("unique-3", 0.80),
        _result("unique-4", 0.70),
    ]

    def _vector_search(_qdrant, _graph, _query, _embedding, _limit, seen_ids, *_args):
        for result in vector_results:
            seen_ids.add(result["id"])
        return [dict(result) for result in vector_results]

    def _metadata_keyword_search(_graph, _query, _limit, _seen_ids, **_kwargs):
        return [_result("metadata-target", 1.25, match_type="metadata")]

    with app.app.test_request_context("/recall?query=metadata%20target&limit=5&current_only=false"):
        response = recall_module.handle_recall(
            get_memory_graph=lambda: FakeGraph(),
            get_qdrant_client=lambda: object(),
            normalize_tag_list=lambda value: value if isinstance(value, list) else [],
            normalize_timestamp=lambda value: value,
            parse_time_expression=lambda _value: (None, None),
            extract_keywords=lambda _query: ["metadata", "target"],
            compute_metadata_score=lambda result, _query, _tokens, _context: (
                float((result.get("memory") or {}).get("target_score") or 0.0),
                {},
            ),
            result_passes_filters=lambda *_args, **_kwargs: True,
            graph_keyword_search=lambda *_args, **_kwargs: [],
            vector_search=_vector_search,
            vector_filter_only_tag_search=lambda *_args, **_kwargs: [],
            metadata_keyword_search=_metadata_keyword_search,
            recall_max_limit=50,
            logger=SimpleNamespace(
                debug=lambda *_args, **_kwargs: None,
                info=lambda *_args, **_kwargs: None,
                exception=lambda *_args, **_kwargs: None,
            ),
        )

    ids = [result["id"] for result in response.get_json()["results"]]
    assert ids == ["metadata-target", "unique-1", "unique-2", "unique-3", "unique-4"]


def test_metadata_overlap_does_not_spend_metadata_only_slots() -> None:
    metadata_seen_sets = []
    metadata_include_sets = []

    def _result(memory_id, score, match_type="vector"):
        return {
            "id": memory_id,
            "score": score,
            "match_score": score,
            "match_type": match_type,
            "source": "qdrant" if match_type == "vector" else "graph",
            "memory": {
                "id": memory_id,
                "content": f"content for {memory_id}",
                "tags": [],
                "target_score": score,
                "type": "Context",
                "enriched": True,
                "timestamp": "2026-05-18T00:00:00+00:00",
            },
            "relations": [],
        }

    vector_results = [_result(f"vector-{i}", 0.95 - i * 0.01) for i in range(5)]
    metadata_rows = [
        *[_result(f"vector-{i}", 1.0 - i * 0.01, match_type="metadata") for i in range(5)],
        _result("metadata-only", 1.30, match_type="metadata"),
    ]

    def _vector_search(_qdrant, _graph, _query, _embedding, _limit, seen_ids, *_args):
        for result in vector_results:
            seen_ids.add(result["id"])
        return [dict(result) for result in vector_results]

    def _metadata_keyword_search(_graph, _query, limit, seen_ids, **kwargs):
        include_seen_ids = set(kwargs.get("include_seen_ids") or [])
        metadata_seen_sets.append(set(seen_ids))
        metadata_include_sets.append(include_seen_ids)

        matches = []
        metadata_only_count = 0
        for result in metadata_rows:
            memory_id = result["id"]
            already_seen = memory_id in seen_ids
            if already_seen and memory_id not in include_seen_ids:
                continue
            matches.append(dict(result))
            if not already_seen:
                metadata_only_count += 1
            if metadata_only_count >= limit:
                break
        return matches

    with app.app.test_request_context("/recall?query=metadata%20target&limit=5&current_only=false"):
        response = recall_module.handle_recall(
            get_memory_graph=lambda: FakeGraph(),
            get_qdrant_client=lambda: object(),
            normalize_tag_list=lambda value: value if isinstance(value, list) else [],
            normalize_timestamp=lambda value: value,
            parse_time_expression=lambda _value: (None, None),
            extract_keywords=lambda _query: ["metadata", "target"],
            compute_metadata_score=lambda result, _query, _tokens, _context: (
                float((result.get("memory") or {}).get("target_score") or 0.0),
                {},
            ),
            result_passes_filters=lambda *_args, **_kwargs: True,
            graph_keyword_search=lambda *_args, **_kwargs: [],
            vector_search=_vector_search,
            vector_filter_only_tag_search=lambda *_args, **_kwargs: [],
            metadata_keyword_search=_metadata_keyword_search,
            recall_max_limit=50,
            logger=SimpleNamespace(
                debug=lambda *_args, **_kwargs: None,
                info=lambda *_args, **_kwargs: None,
                exception=lambda *_args, **_kwargs: None,
            ),
        )

    vector_ids = {result["id"] for result in vector_results}
    ids = [result["id"] for result in response.get_json()["results"]]
    assert metadata_seen_sets == [vector_ids]
    assert metadata_include_sets == [vector_ids]
    assert "metadata-only" in ids


def test_priority_tail_does_not_block_context_injection() -> None:
    def _result(memory_id, score, *, tags=None, mem_type="Context", match_type="vector"):
        return {
            "id": memory_id,
            "score": score,
            "match_score": score,
            "match_type": match_type,
            "source": "qdrant",
            "memory": {
                "id": memory_id,
                "content": f"content for {memory_id}",
                "tags": tags or [],
                "target_score": score,
                "type": mem_type,
                "enriched": True,
                "timestamp": "2026-05-18T00:00:00+00:00",
            },
            "relations": [],
        }

    vector_results = [
        _result("normal-1", 1.00),
        _result("normal-2", 0.95),
        _result("normal-3", 0.90),
        _result("normal-4", 0.85),
        _result("normal-5", 0.80),
        _result("style-tail", 0.05, tags=["coding-style"], mem_type="Style"),
    ]
    injected_results = [
        _result(
            "style-injected",
            1.20,
            tags=["coding-style", "python"],
            mem_type="Style",
            match_type="tag",
        )
    ]
    injection_seen_sets = []

    def _vector_search(_qdrant, _graph, _query, _embedding, _limit, seen_ids, *_args):
        for result in vector_results:
            seen_ids.add(result["id"])
        return [dict(result) for result in vector_results]

    def _vector_filter_only_tag_search(_qdrant, _tags, _mode, _match, _limit, seen_ids):
        injection_seen_sets.append(set(seen_ids))
        return [dict(result) for result in injected_results if result["id"] not in seen_ids]

    with app.app.test_request_context(
        "/recall?query=context&limit=5&active_path=/workspace/tool.py&current_only=false"
    ):
        response = recall_module.handle_recall(
            get_memory_graph=lambda: None,
            get_qdrant_client=lambda: object(),
            normalize_tag_list=lambda value: value if isinstance(value, list) else [],
            normalize_timestamp=lambda value: value,
            parse_time_expression=lambda _value: (None, None),
            extract_keywords=lambda _query: ["context"],
            compute_metadata_score=lambda result, _query, _tokens, _context: (
                float((result.get("memory") or {}).get("target_score") or 0.0),
                {},
            ),
            result_passes_filters=lambda *_args, **_kwargs: True,
            graph_keyword_search=lambda *_args, **_kwargs: [],
            vector_search=_vector_search,
            vector_filter_only_tag_search=_vector_filter_only_tag_search,
            metadata_keyword_search=lambda *_args, **_kwargs: [],
            recall_max_limit=50,
            logger=SimpleNamespace(
                debug=lambda *_args, **_kwargs: None,
                info=lambda *_args, **_kwargs: None,
                exception=lambda *_args, **_kwargs: None,
            ),
        )

    data = response.get_json()
    ids = [result["id"] for result in data["results"]]
    assert injection_seen_sets
    assert "style-tail" not in injection_seen_sets[0]
    assert ids[0] == "style-injected"
    assert data["context_priority"]["injected"] is True
