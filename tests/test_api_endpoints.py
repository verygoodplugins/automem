"""Comprehensive test suite for AutoMem Flask API endpoints."""

import json
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
from flask import Flask
from flask.testing import FlaskClient
from qdrant_client import models as qdrant_models

import app
from app import _normalize_timestamp, utc_now
from automem import config
from automem.api import recall as recall_module
from automem.api.recall import _expand_related_memories, handle_recall
from automem.utils import scoring
from automem.utils.scoring import _compute_metadata_score, _compute_recency_score
from automem.utils.text import _extract_keywords
from automem.utils.time import query_has_temporal_intent
from tests.support.fake_graph import FakeGraph


class MockQdrantClient:
    """Mock Qdrant client for testing."""

    def __init__(self):
        self.points = {}
        self.upsert_calls = []
        self.search_calls = []
        self.delete_calls = []

    def upsert(self, collection_name, points):
        """Mock upsert operation."""
        self.upsert_calls.append((collection_name, points))
        for point in points:
            self.points[point.id] = {"vector": point.vector, "payload": point.payload}

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        *,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> list[Any]:
        """Mock search operation."""
        _ = with_payload, with_vectors  # Used by real client, not needed in mock
        self.search_calls.append(
            {"collection": collection_name, "vector": query_vector, "limit": limit}
        )
        # Return mock search results
        return []

    def retrieve(self, collection_name, ids, with_payload=True, with_vectors=False):
        """Mock retrieve operation."""
        results = []
        for id in ids:
            if id in self.points:
                mock_point = Mock()
                mock_point.payload = self.points[id]["payload"]
                mock_point.vector = self.points[id]["vector"] if with_vectors else None
                results.append(mock_point)
        return results

    def delete(self, collection_name, points_selector):
        """Mock delete operation."""
        self.delete_calls.append((collection_name, points_selector))
        if hasattr(points_selector, "points"):
            for point_id in points_selector.points:
                if point_id in self.points:
                    del self.points[point_id]
        elif isinstance(points_selector, dict):
            for point_id in points_selector.get("points", []):
                if point_id in self.points:
                    del self.points[point_id]

    def scroll(self, collection_name, scroll_filter=None, limit=10, with_payload=True):
        """Mock scroll to support tag-only queries."""
        matches = []
        for point_id, point in self.points.items():
            payload = point["payload"]
            if self._filter_matches(payload, scroll_filter):
                mock_point = SimpleNamespace(id=point_id, payload=payload)
                matches.append(mock_point)
            if len(matches) >= limit:
                break
        return matches, None

    def _filter_matches(self, payload, scroll_filter):
        if scroll_filter is None:
            return True
        must_conditions = getattr(scroll_filter, "must", []) or []
        for condition in must_conditions:
            field_values = payload.get(condition.key) or []
            normalized = [
                str(value).strip().lower()
                for value in field_values
                if isinstance(value, str) and value.strip()
            ]
            match = condition.match
            if isinstance(match, qdrant_models.MatchAny):
                targets = {
                    str(value).strip().lower()
                    for value in (match.any or [])
                    if isinstance(value, str)
                }
                if not targets or not any(val in targets for val in normalized):
                    return False
            elif isinstance(match, qdrant_models.MatchValue):
                target = str(match.value).strip().lower()
                if target not in normalized:
                    return False
        return True


@pytest.fixture
def mock_state(monkeypatch):
    """Create mock service state with graph and Qdrant."""
    state = app.ServiceState()
    state.memory_graph = FakeGraph()
    state.qdrant = MockQdrantClient()  # Changed from qdrant_client to qdrant
    state.openai_client = Mock()  # Mock OpenAI client

    monkeypatch.setattr(app, "state", state)
    monkeypatch.setattr(app, "init_falkordb", lambda: None)
    monkeypatch.setattr(app, "init_qdrant", lambda: None)
    monkeypatch.setattr(app, "init_openai", lambda: None)

    # Mock embedding generation
    monkeypatch.setattr(app, "_generate_real_embedding", lambda content: [0.1] * 768)
    monkeypatch.setattr(app, "_generate_placeholder_embedding", lambda content: [0.0] * 768)

    # Mock API tokens for auth
    monkeypatch.setattr(app, "API_TOKEN", "test-token")
    monkeypatch.setattr(app, "ADMIN_TOKEN", "test-admin-token")

    return state


@pytest.fixture
def client():
    """Create Flask test client."""
    app.app.config["TESTING"] = True
    with app.app.test_client() as client:
        yield client


@pytest.fixture
def auth_headers():
    """Provide authorization headers for testing."""
    return {"Authorization": "Bearer test-token"}


@pytest.fixture
def admin_headers():
    """Provide admin authorization headers for testing."""
    return {"Authorization": "Bearer test-token", "X-Admin-Token": "test-admin-token"}


# ==================== Test Health Endpoint ====================


def test_viewer_bootstrap_response(client, mock_state, monkeypatch):
    """Test /viewer bootstrap response for standalone visualizer redirect."""
    _ = mock_state
    monkeypatch.setenv("GRAPH_VIEWER_URL", "https://viewer.example.com")

    response = client.get("/viewer/?foo=bar")

    assert response.status_code == 200
    assert response.mimetype == "text/html"
    body = response.get_data(as_text=True)
    assert "viewer.example.com" in body
    assert "server" in body


def test_viewer_asset_redirect(client, mock_state, monkeypatch):
    """Test static asset compatibility route redirects to standalone visualizer."""
    _ = mock_state
    monkeypatch.setenv("GRAPH_VIEWER_URL", "https://viewer.example.com")

    response = client.get("/viewer/assets/index.js?v=1", follow_redirects=False)

    assert response.status_code == 302
    assert response.headers["Location"] == "https://viewer.example.com/assets/index.js?v=1"


def test_viewer_unavailable_without_target_url(client, mock_state, monkeypatch):
    """Test /viewer returns 503 when GRAPH_VIEWER_URL is not configured."""
    _ = mock_state
    monkeypatch.delenv("GRAPH_VIEWER_URL", raising=False)

    response = client.get("/viewer/")

    assert response.status_code == 503
    assert "GRAPH_VIEWER_URL" in response.get_data(as_text=True)


def test_cors_preflight_for_graph_endpoint(client, mock_state):
    """Test OPTIONS preflight is not blocked by auth middleware."""
    _ = mock_state
    response = client.options(
        "/graph/stats",
        headers={
            "Origin": "https://viewer.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code in (200, 204)
    assert "Access-Control-Allow-Origin" in response.headers


def test_health_endpoint_all_services_up(client, mock_state):
    """Test health endpoint when all services are available."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["falkordb"] == "connected"
    assert data["qdrant"] == "connected"
    assert "timestamp" in data
    assert "graph" in data


def test_health_endpoint_qdrant_down(client, mock_state):
    """Test health endpoint when Qdrant is unavailable."""
    mock_state.qdrant = None
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "degraded"
    assert data["falkordb"] == "connected"
    assert data["qdrant"] == "disconnected"


def test_health_endpoint_falkordb_down(client, mock_state):
    """Test health endpoint when FalkorDB is unavailable."""
    mock_state.memory_graph = None
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "degraded"
    assert data["falkordb"] == "disconnected"
    assert data["qdrant"] == "connected"


# ==================== Test Memory Recall ====================


def _make_floor_result(idx: int, score: float, query: str = "autojack") -> dict[str, Any]:
    return {
        "id": f"floor-{idx}",
        "score": score,
        "match_score": score,
        "match_type": "vector",
        "source": "qdrant",
        "memory": {
            "id": f"floor-{idx}",
            "content": f"{query} memory {idx}",
            "target_score": score,
            "importance": 0.1,
        },
        "relations": [],
    }


def _call_handle_recall_for_scores(query: str, scores: list[float]):
    vector_results = [
        _make_floor_result(idx, score, query=query) for idx, score in enumerate(scores)
    ]

    with app.app.test_request_context(f"/recall?query={query}&limit={len(scores)}"):
        response = handle_recall(
            get_memory_graph=lambda: None,
            get_qdrant_client=lambda: object(),
            normalize_tag_list=lambda value: value if isinstance(value, list) else [],
            normalize_timestamp=_normalize_timestamp,
            parse_time_expression=lambda _value: (None, None),
            extract_keywords=_extract_keywords,
            compute_metadata_score=lambda result, _query, _tokens, _context: (
                float(result["memory"]["target_score"]),
                {"keyword": 1.0},
            ),
            result_passes_filters=lambda *_args, **_kwargs: True,
            graph_keyword_search=lambda *_args, **_kwargs: [],
            vector_search=lambda *_args, **_kwargs: list(vector_results),
            vector_filter_only_tag_search=lambda *_args, **_kwargs: [],
            recall_max_limit=50,
            logger=Mock(),
        )

    return response.get_json()


def test_recall_with_text_and_tags_overfetches_vector_candidates():
    seen_limits: list[int] = []

    def _vector_search(*args, **kwargs):
        limit = args[4]
        seen_limits.append(limit)
        return [_make_floor_result(idx, 1.0 - (idx * 0.01), query="locomo") for idx in range(limit)]

    with app.app.test_request_context(
        "/recall?query=locomo&tags=automem&tags=locomo&tag_mode=all&limit=10"
    ):
        response = handle_recall(
            get_memory_graph=lambda: None,
            get_qdrant_client=lambda: object(),
            normalize_tag_list=lambda value: value if isinstance(value, list) else [],
            normalize_timestamp=_normalize_timestamp,
            parse_time_expression=lambda _value: (None, None),
            extract_keywords=_extract_keywords,
            compute_metadata_score=lambda result, _query, _tokens, _context: (
                float(result["memory"]["target_score"]),
                {"keyword": 1.0},
            ),
            result_passes_filters=lambda *_args, **_kwargs: True,
            graph_keyword_search=lambda *_args, **_kwargs: [],
            vector_search=_vector_search,
            vector_filter_only_tag_search=lambda *_args, **_kwargs: [],
            recall_max_limit=50,
            logger=Mock(),
        )

    data = response.get_json()
    assert seen_limits == [50]
    assert len(data["results"]) == 10


def test_compute_metadata_score_uses_content_keyword_fallback_for_vector_results():
    score, components = _compute_metadata_score(
        {
            "match_type": "vector",
            "match_score": 0.7,
            "memory": {"content": "AutoJack recall debugging notes"},
        },
        "AutoJack recall",
        ["autojack", "recall"],
    )

    assert components["keyword"] == 1.0
    assert score > config.SEARCH_WEIGHT_VECTOR * 0.7


def test_compute_metadata_score_uses_partial_content_keyword_fallback():
    _score, components = _compute_metadata_score(
        {
            "match_type": "vector",
            "match_score": 0.7,
            "memory": {"content": "AutoJack debugging notes"},
        },
        "AutoJack recall",
        ["autojack", "recall"],
    )

    assert components["keyword"] == 0.5


def test_compute_metadata_score_leaves_keyword_zero_without_content_hits():
    _score, components = _compute_metadata_score(
        {
            "match_type": "vector",
            "match_score": 0.7,
            "memory": {"content": "Unrelated debugging notes"},
        },
        "AutoJack recall",
        ["autojack", "recall"],
    )

    assert components["keyword"] == 0.0


def test_compute_metadata_score_preserves_keyword_match_score_for_keyword_results():
    _score, components = _compute_metadata_score(
        {
            "match_type": "keyword",
            "match_score": 0.42,
            "memory": {"content": "AutoJack recall debugging notes"},
        },
        "AutoJack recall",
        ["autojack", "recall"],
    )

    assert components["keyword"] == 0.42


def test_compute_metadata_score_preserves_keyword_match_score_for_trending_results():
    _score, components = _compute_metadata_score(
        {
            "match_type": "trending",
            "match_score": 0.33,
            "memory": {"content": "AutoJack recall debugging notes"},
        },
        "AutoJack recall",
        ["autojack", "recall"],
    )

    assert components["keyword"] == 0.33


def test_compute_metadata_score_ignores_generated_entities_for_generic_tag_score(monkeypatch):
    # Pin the cap (config.py runs load_dotenv() at import, so a tuned .env
    # could otherwise leak in); 1 hit / min(3 tokens, cap 3) == 1/3 either way.
    monkeypatch.setattr(scoring, "SEARCH_TAG_SCORE_TOKEN_CAP", 3)

    _score, components = _compute_metadata_score(
        {
            "match_type": "vector",
            "match_score": 0.7,
            "memory": {
                "content": "Benchmark notes",
                "tags": ["automem"],
                "metadata": {
                    "entities": {
                        "people": ["Result Json"],
                        "organizations": ["Root Cause"],
                    },
                    "source": "locomo",
                },
            },
        },
        "result json locomo",
        ["result", "json", "locomo"],
    )

    assert components["tag"] == 1 / 3


def _tag_score_result(tags: list) -> dict:
    return {
        "match_type": "vector",
        "match_score": 0.7,
        "memory": {"content": "Benchmark notes", "tags": tags},
    }


def test_compute_metadata_score_tag_score_single_token_full_credit(monkeypatch):
    monkeypatch.setattr(scoring, "SEARCH_TAG_SCORE_TOKEN_CAP", 3)

    _score, components = _compute_metadata_score(
        _tag_score_result(["automem"]),
        "automem",
        ["automem"],
    )

    assert components["tag"] == 1.0


def test_compute_metadata_score_tag_score_caps_denominator_for_long_queries(monkeypatch):
    monkeypatch.setattr(scoring, "SEARCH_TAG_SCORE_TOKEN_CAP", 3)

    _score, components = _compute_metadata_score(
        _tag_score_result(["alpha", "bravo"]),
        "alpha bravo charlie delta echo",
        ["alpha", "bravo", "charlie", "delta", "echo"],
    )

    # 2 hits over min(5, cap=3) instead of the legacy 2/5
    assert components["tag"] == pytest.approx(2 / 3, abs=1e-9)


def test_compute_metadata_score_tag_score_clips_at_one_when_hits_exceed_cap(monkeypatch):
    monkeypatch.setattr(scoring, "SEARCH_TAG_SCORE_TOKEN_CAP", 3)

    _score, components = _compute_metadata_score(
        _tag_score_result(["alpha", "bravo", "charlie", "delta"]),
        "alpha bravo charlie delta echo",
        ["alpha", "bravo", "charlie", "delta", "echo"],
    )

    # 4 hits over a capped denominator of 3 would exceed 1.0; clip it.
    assert components["tag"] == 1.0


def test_compute_metadata_score_tag_score_cap_zero_restores_legacy_denominator(monkeypatch):
    monkeypatch.setattr(scoring, "SEARCH_TAG_SCORE_TOKEN_CAP", 0)

    _score, components = _compute_metadata_score(
        _tag_score_result(["alpha", "bravo"]),
        "alpha bravo charlie delta echo",
        ["alpha", "bravo", "charlie", "delta", "echo"],
    )

    # Legacy behavior: denominator is the full query length (2/5)
    assert components["tag"] == pytest.approx(0.4, abs=1e-9)


def test_compute_metadata_score_tag_score_short_query_below_cap_uses_query_length(monkeypatch):
    monkeypatch.setattr(scoring, "SEARCH_TAG_SCORE_TOKEN_CAP", 3)

    _score, components = _compute_metadata_score(
        _tag_score_result(["alpha"]),
        "alpha zulu",
        ["alpha", "zulu"],
    )

    # Below the cap, the denominator stays min(len(tokens), cap) == 2
    assert components["tag"] == pytest.approx(0.5, abs=1e-9)


def test_tag_score_token_cap_config_falls_back_on_negative_values():
    # 0 is a valid sentinel (legacy full-length denominator), so only negative
    # values fall back to the default; unparseable values raise like the
    # neighboring int()/float() env parses.
    assert config._non_negative_int_or_default("-1", 3) == 3
    assert config._non_negative_int_or_default("0", 3) == 0
    assert config._non_negative_int_or_default("5", 3) == 5
    with pytest.raises(ValueError):
        config._non_negative_int_or_default("not-an-int", 3)


# ==================== Relevance Gate (issue #130) ====================


def _pin_default_scoring(monkeypatch, gate: float = 0.0) -> None:
    """Pin scoring config to documented defaults.

    config.py runs load_dotenv() at import, so a tuned .env could otherwise
    leak into these tests.
    """
    monkeypatch.setattr(scoring, "SEARCH_WEIGHT_VECTOR", 0.35)
    monkeypatch.setattr(scoring, "SEARCH_WEIGHT_KEYWORD", 0.35)
    monkeypatch.setattr(scoring, "SEARCH_WEIGHT_METADATA", 0.35)
    monkeypatch.setattr(scoring, "SEARCH_WEIGHT_RELATION", 0.25)
    monkeypatch.setattr(scoring, "SEARCH_WEIGHT_TAG", 0.2)
    monkeypatch.setattr(scoring, "SEARCH_WEIGHT_IMPORTANCE", 0.1)
    monkeypatch.setattr(scoring, "SEARCH_WEIGHT_CONFIDENCE", 0.05)
    monkeypatch.setattr(scoring, "SEARCH_WEIGHT_RECENCY", 0.1)
    monkeypatch.setattr(scoring, "SEARCH_WEIGHT_EXACT", 0.2)
    monkeypatch.setattr(scoring, "SEARCH_WEIGHT_RELEVANCE", 0.0)
    monkeypatch.setattr(scoring, "SEARCH_RECENCY_WINDOW_DAYS", 180.0)
    monkeypatch.setattr(scoring, "SEARCH_RECENCY_CURVE", "linear")
    monkeypatch.setattr(scoring, "SEARCH_TAG_SCORE_TOKEN_CAP", 3)
    monkeypatch.setattr(scoring, "RECALL_RELEVANCE_GATE", gate)


def test_compute_metadata_score_gate_zero_default_matches_legacy(monkeypatch):
    """Gate disabled (default 0.0): scores must match the pre-gate formula exactly."""
    _pin_default_scoring(monkeypatch, gate=0.0)

    cases = [
        # Importance-heavy, zero topical evidence (the issue #130 shape)
        {
            "match_type": "vector",
            "match_score": 0.0,
            "memory": {
                "content": "Entirely unrelated lessons learned",
                "tags": ["automem"],
                "importance": 0.9,
                "confidence": 0.8,
                "timestamp": _timestamp_days_ago(90),
            },
        },
        # Vector-heavy, low importance
        {
            "match_type": "vector",
            "match_score": 0.8,
            "memory": {
                "content": "Entirely unrelated lessons learned",
                "tags": [],
                "importance": 0.1,
                "timestamp": _timestamp_days_ago(90),
            },
        },
        # Keyword result with tag crumbs
        {
            "match_type": "keyword",
            "match_score": 0.4,
            "memory": {
                "content": "automem recall notes",
                "tags": ["automem", "recall"],
                "importance": 0.5,
                "timestamp": _timestamp_days_ago(90),
            },
        },
    ]

    for result in cases:
        score, components = _compute_metadata_score(result, "automem recall", ["automem", "recall"])
        expected = (
            0.35 * components["vector"]
            + 0.35 * components["keyword"]
            + 0.35 * components["metadata"]
            + 0.25 * components["relation"]
            + 0.2 * components["tag"]
            + 0.1 * components["importance"]
            + 0.05 * components["confidence"]
            + 0.1 * components["recency"]
            + 0.2 * components["exact"]
        )
        assert score == pytest.approx(expected, abs=1e-12)
        # Gate-off must never scale the query-independent components
        memory = result["memory"]
        assert components["importance"] == memory.get("importance", 0.0)
        assert components["confidence"] == memory.get("confidence", 0.0)
        assert components["recency"] == pytest.approx(0.5, abs=1e-6)
        assert components["relevance_gated"] is False
        assert "evidence" in components


def test_compute_metadata_score_gate_demotes_zero_evidence_high_importance(monkeypatch):
    """With the gate on, off-topic high-importance loses to on-topic low-importance."""
    _pin_default_scoring(monkeypatch, gate=0.2)

    off_topic_high_importance = {
        "match_type": "vector",
        "match_score": 0.0,
        "memory": {
            "content": "Entirely unrelated critical lessons",
            "tags": [],
            "importance": 1.0,
            "confidence": 1.0,
            "timestamp": _timestamp_days_ago(0),
        },
    }
    on_topic_low_importance = {
        "match_type": "vector",
        "match_score": 0.5,
        "memory": {
            "content": "Different words about other things",
            "tags": [],
            "importance": 0.0,
            "timestamp": _timestamp_days_ago(400),
        },
    }

    off_score, off_components = _compute_metadata_score(
        off_topic_high_importance,
        "quarterly metrics dashboard",
        ["quarterly", "metrics", "dashboard"],
    )
    on_score, on_components = _compute_metadata_score(
        on_topic_low_importance,
        "quarterly metrics dashboard",
        ["quarterly", "metrics", "dashboard"],
    )

    assert off_score < on_score
    assert off_components["relevance_gated"] is True
    assert off_components["evidence"] == 0.0
    # evidence 0 -> scale 0 -> query-independent components zeroed
    assert off_components["importance"] == 0.0
    assert off_components["confidence"] == 0.0
    assert off_components["recency"] == 0.0
    assert on_components["relevance_gated"] is False
    assert on_components["evidence"] == 0.5


def test_compute_metadata_score_gate_ramp_is_linear(monkeypatch):
    """Evidence at half the gate scales query-independent components by half."""
    _pin_default_scoring(monkeypatch, gate=0.2)

    result = {
        "match_type": "vector",
        "match_score": 0.1,  # evidence = half of the 0.2 gate
        "memory": {
            "content": "Different words about other things",
            "tags": ["quarterly"],  # one tag crumb out of three tokens
            "importance": 0.8,
            "confidence": 0.4,
            "timestamp": _timestamp_days_ago(90),  # recency 0.5 at 180d window
        },
    }

    _score, components = _compute_metadata_score(
        result, "quarterly metrics dashboard", ["quarterly", "metrics", "dashboard"]
    )

    assert components["relevance_gated"] is True
    assert components["evidence"] == pytest.approx(0.1, abs=1e-9)
    assert components["importance"] == pytest.approx(0.4, abs=1e-9)  # 0.8 * 0.5
    assert components["confidence"] == pytest.approx(0.2, abs=1e-9)  # 0.4 * 0.5
    assert components["recency"] == pytest.approx(0.25, abs=1e-6)  # 0.5 * 0.5
    assert components["tag"] == pytest.approx((1 / 3) * 0.5, abs=1e-9)


def test_compute_metadata_score_gate_leaves_evidence_at_threshold_untouched(monkeypatch):
    _pin_default_scoring(monkeypatch, gate=0.2)

    result = {
        "match_type": "vector",
        "match_score": 0.2,  # exactly at the gate
        "memory": {
            "content": "Different words about other things",
            "tags": [],
            "importance": 0.8,
            "confidence": 0.4,
            "timestamp": _timestamp_days_ago(90),
        },
    }

    _score, components = _compute_metadata_score(
        result, "quarterly metrics dashboard", ["quarterly", "metrics", "dashboard"]
    )

    assert components["relevance_gated"] is False
    assert components["importance"] == 0.8
    assert components["confidence"] == 0.4
    assert components["recency"] == pytest.approx(0.5, abs=1e-6)


def test_compute_metadata_score_gate_inactive_without_query_tokens(monkeypatch):
    """No query tokens (tag-only / time-only recall): gate must not apply."""
    _pin_default_scoring(monkeypatch, gate=0.2)

    result = {
        "match_type": "tag",
        "match_score": 0.9,
        "memory": {
            "content": "Tag-only recall result",
            "tags": ["automem"],
            "importance": 0.9,
            "confidence": 0.7,
            "timestamp": _timestamp_days_ago(0),
        },
    }

    _score, components = _compute_metadata_score(result, "", [])

    assert components["relevance_gated"] is False
    assert components["importance"] == 0.9
    assert components["confidence"] == 0.7


def test_compute_metadata_score_gate_does_not_touch_context_bonus(monkeypatch):
    """The context bonus is the explicit soft channel and must never be gated."""
    _pin_default_scoring(monkeypatch, gate=0.2)

    profile = {
        "weights": {"tag": 0.45},
        "priority_tags": {"automem"},
        "priority_types": set(),
        "priority_ids": set(),
        "priority_keywords": set(),
    }
    result = {
        "match_type": "vector",
        "match_score": 0.0,
        "memory": {
            "content": "Entirely unrelated note",
            "tags": ["automem"],
            "importance": 1.0,
            "timestamp": _timestamp_days_ago(0),
        },
    }

    _score, components = _compute_metadata_score(
        result, "quarterly metrics dashboard", ["quarterly", "metrics", "dashboard"], profile
    )

    assert components["relevance_gated"] is True
    assert components["context"] == pytest.approx(0.45, abs=1e-9)


def test_compute_metadata_score_gate_scales_relevance_score(monkeypatch):
    """relevance_score (consolidation decay) is query-independent and gated
    with the other crumbs. With the default SEARCH_WEIGHT_RELEVANCE=0.0 this
    is a no-op today; pin a non-zero weight to observe the scaling."""
    _pin_default_scoring(monkeypatch, gate=0.2)
    monkeypatch.setattr(scoring, "SEARCH_WEIGHT_RELEVANCE", 0.3)

    result = {
        "match_type": "vector",
        "match_score": 0.1,  # evidence = half of the 0.2 gate
        "memory": {
            "content": "Different words about other things",
            "tags": [],
            "importance": 0.0,
            "relevance_score": 0.8,
            "timestamp": _timestamp_days_ago(90),  # recency 0.5 at 180d window
        },
    }

    score, components = _compute_metadata_score(
        result, "quarterly metrics dashboard", ["quarterly", "metrics", "dashboard"]
    )

    assert components["relevance_gated"] is True
    assert components["relevance"] == pytest.approx(0.4, abs=1e-9)  # 0.8 * 0.5
    # The component is scaled before weighting, so the final score reflects it.
    expected = (
        0.35 * components["vector"] + 0.1 * components["recency"] + 0.3 * components["relevance"]
    )
    assert score == pytest.approx(expected, abs=1e-9)


def test_relevance_gate_config_clamps_to_unit_interval():
    # Negatives clamp to 0.0 (gate disabled); values above 1.0 clamp to 1.0
    # because evidence components are bounded at ~1.0, so a larger gate would
    # only dampen every result uniformly. Unparseable values raise like the
    # neighboring float() env parses.
    assert config._clamped_unit_interval("-0.5") == 0.0
    assert config._clamped_unit_interval("0.0") == 0.0
    assert config._clamped_unit_interval("0.2") == 0.2
    assert config._clamped_unit_interval("1.0") == 1.0
    assert config._clamped_unit_interval("1.5") == 1.0
    with pytest.raises(ValueError):
        config._clamped_unit_interval("not-a-float")


def _timestamp_days_ago(days: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def test_compute_recency_score_age_zero_scores_one():
    assert _compute_recency_score(_timestamp_days_ago(0)) == pytest.approx(1.0, abs=1e-6)


def test_compute_recency_score_future_timestamp_scores_one():
    assert _compute_recency_score(_timestamp_days_ago(-5)) == 1.0


def test_compute_recency_score_linear_half_window_scores_half():
    assert _compute_recency_score(_timestamp_days_ago(90)) == pytest.approx(0.5, abs=1e-6)


def test_compute_recency_score_linear_beyond_window_scores_zero():
    assert _compute_recency_score(_timestamp_days_ago(180)) == pytest.approx(0.0, abs=1e-6)
    assert _compute_recency_score(_timestamp_days_ago(400)) == 0.0


def test_compute_recency_score_exp_window_is_half_life(monkeypatch):
    monkeypatch.setattr(scoring, "SEARCH_RECENCY_CURVE", "exp")

    assert _compute_recency_score(_timestamp_days_ago(180)) == pytest.approx(0.5, abs=1e-6)
    assert _compute_recency_score(_timestamp_days_ago(360)) == pytest.approx(0.25, abs=1e-6)


def test_compute_recency_score_missing_timestamp_scores_zero():
    assert _compute_recency_score(None) == 0.0
    assert _compute_recency_score("") == 0.0


def test_compute_recency_score_unparseable_timestamp_scores_zero():
    assert _compute_recency_score("not-a-timestamp") == 0.0


def test_compute_recency_score_respects_configured_window(monkeypatch):
    monkeypatch.setattr(scoring, "SEARCH_RECENCY_WINDOW_DAYS", 90.0)

    assert _compute_recency_score(_timestamp_days_ago(45)) == pytest.approx(0.5, abs=1e-6)
    assert _compute_recency_score(_timestamp_days_ago(90)) == pytest.approx(0.0, abs=1e-6)
    assert _compute_recency_score(_timestamp_days_ago(120)) == 0.0


def test_compute_recency_score_defaults_match_legacy_behavior(monkeypatch):
    # Pin the default window/curve explicitly (config.py runs load_dotenv() at
    # import, so a tuned .env would otherwise leak into this test) and verify
    # the historical linear-decay-over-180-days behavior.
    monkeypatch.setattr(scoring, "SEARCH_RECENCY_WINDOW_DAYS", 180.0)
    monkeypatch.setattr(scoring, "SEARCH_RECENCY_CURVE", "linear")

    assert _compute_recency_score(_timestamp_days_ago(90)) == pytest.approx(0.5, abs=1e-6)


def test_recency_window_config_rejects_non_positive_values():
    # A window <= 0 would cause a request-time ZeroDivisionError (window == 0)
    # or unbounded scores (window < 0); the config guard falls back to 180.
    assert config._positive_or_default("0", 180.0) == 180.0
    assert config._positive_or_default("-30", 180.0) == 180.0
    assert config._positive_or_default("90", 180.0) == 90.0
    with pytest.raises(ValueError):
        config._positive_or_default("not-a-number", 180.0)


def test_recall_with_query(client, mock_state, auth_headers):
    """Test memory recall with text query."""
    # Store a memory first
    memory_data = {
        "content": "Test memory about Python programming",
        "tags": ["python", "programming"],
        "importance": 0.8,
    }
    mock_state.memory_graph.memories["dddddddd-dddd-dddd-dddd-dddddddddddd"] = {
        "id": "dddddddd-dddd-dddd-dddd-dddddddddddd",
        "content": memory_data["content"],
        "tags": memory_data["tags"],
        "importance": memory_data["importance"],
        "timestamp": utc_now(),
    }

    response = client.get("/recall?query=Python&limit=10", headers=auth_headers)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "results" in data


def test_recall_with_tags(client, mock_state, auth_headers):
    """Test memory recall filtered by tags."""
    response = client.get("/recall?tags=python&tags=ai&limit=5", headers=auth_headers)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "tags" in data
    assert "python" in data["tags"]
    assert "ai" in data["tags"]


def test_recall_with_time_query(client, mock_state, auth_headers):
    """Test memory recall with natural language time query."""
    response = client.get("/recall?time_query=last week", headers=auth_headers)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "time_window" in data


def test_recall_time_sorting(client, mock_state, auth_headers):
    """Test that sort=time_desc returns most recent memories first within a time window."""
    mock_state.memory_graph.memories.clear()

    now = datetime.now(timezone.utc)
    older_ts = (now - timedelta(days=2)).isoformat()
    newer_ts = (now - timedelta(days=1)).isoformat()

    mock_state.memory_graph.memories["10000000-0000-0000-0000-000000000001"] = {
        "id": "10000000-0000-0000-0000-000000000001",
        "content": "Older memory",
        "tags": ["cursor"],
        "importance": 0.1,
        "timestamp": older_ts,
        "updated_at": older_ts,
        "last_accessed": older_ts,
    }
    mock_state.memory_graph.memories["10000000-0000-0000-0000-000000000002"] = {
        "id": "10000000-0000-0000-0000-000000000002",
        "content": "Newer memory",
        "tags": ["cursor"],
        "importance": 0.1,
        "timestamp": newer_ts,
        "updated_at": newer_ts,
        "last_accessed": newer_ts,
    }

    response = client.get(
        "/recall?time_query=last 7 days&tags=cursor&limit=10&sort=time_desc",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data.get("sort") == "time_desc"
    assert data["results"], "Expected at least one result"
    assert data["results"][0]["id"] == "10000000-0000-0000-0000-000000000002"


def test_recall_time_window_defaults_to_time_desc_sort(client, mock_state, auth_headers):
    """If time window is provided with no query, recall should default to newest-first ordering."""
    mock_state.memory_graph.memories.clear()

    now = datetime.now(timezone.utc)
    older_ts = (now - timedelta(days=2)).isoformat()
    newer_ts = (now - timedelta(days=1)).isoformat()

    mock_state.memory_graph.memories["10000000-0000-0000-0000-000000000001"] = {
        "id": "10000000-0000-0000-0000-000000000001",
        "content": "Older memory default sort",
        "tags": ["cursor"],
        "importance": 0.1,
        "timestamp": older_ts,
        "updated_at": older_ts,
        "last_accessed": older_ts,
    }
    mock_state.memory_graph.memories["10000000-0000-0000-0000-000000000002"] = {
        "id": "10000000-0000-0000-0000-000000000002",
        "content": "Newer memory default sort",
        "tags": ["cursor"],
        "importance": 0.1,
        "timestamp": newer_ts,
        "updated_at": newer_ts,
        "last_accessed": newer_ts,
    }

    response = client.get(
        "/recall?time_query=last 7 days&tags=cursor&limit=10", headers=auth_headers
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data.get("sort") == "time_desc"
    assert data["results"][0]["id"] == "10000000-0000-0000-0000-000000000002"


def test_recall_with_explicit_timestamps(client, mock_state, auth_headers):
    """Test memory recall with explicit start and end timestamps."""
    start = (datetime.now(timezone.utc) - timedelta(days=7)).replace(tzinfo=None).isoformat()
    end = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

    response = client.get(f"/recall?start={start}&end={end}", headers=auth_headers)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "time_window" in data


def test_recall_metadata_roundtrip(client, mock_state, auth_headers):
    """Custom metadata and timestamps stored via POST /memory surface in /recall (#111)."""
    with_metadata = {
        "content": "Memory with provenance metadata",
        "tags": ["metadata-roundtrip", "with-metadata"],
        "importance": 0.8,
        "metadata": {"created_by": "test-agent", "task": "synthetic-task"},
    }
    without_metadata = {
        "content": "Memory without metadata",
        "tags": ["metadata-roundtrip", "no-metadata"],
        "importance": 0.7,
    }

    memory_ids = {}
    for key, payload in (("with", with_metadata), ("without", without_metadata)):
        store_response = client.post("/memory", json=payload, headers=auth_headers)
        assert store_response.status_code == 201
        store_data = store_response.get_json()
        assert store_data["status"] == "success"
        memory_ids[key] = store_data["memory_id"]

    response = client.get("/recall?tags=metadata-roundtrip&limit=10", headers=auth_headers)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"

    results = {result["id"]: result["memory"] for result in data.get("results", [])}
    assert set(results) == set(memory_ids.values())

    enriched = results[memory_ids["with"]]
    assert isinstance(enriched["metadata"], dict)
    assert enriched["metadata"]["created_by"] == "test-agent"
    assert enriched["metadata"]["task"] == "synthetic-task"
    # POST /memory defaults updated_at to created_at and last_accessed to updated_at
    assert enriched["updated_at"]
    assert enriched["last_accessed"]

    # Backward compat: memories stored without metadata round-trip without user
    # metadata. JIT enrichment may add server-side bookkeeping keys only
    # (written by jit_enrich_lightweight in automem/enrichment/runtime_orchestration.py).
    plain = results[memory_ids["without"]]
    plain_metadata = plain.get("metadata") or {}
    assert isinstance(plain_metadata, dict)
    assert set(plain_metadata) <= {"enrichment", "entities"}
    assert plain["updated_at"]
    assert plain["last_accessed"]


def test_recall_with_high_limit(client, mock_state, auth_headers):
    """Test recall with limit exceeding max - should clamp to 50."""
    response = client.get("/recall?limit=100", headers=auth_headers)
    assert response.status_code == 200
    # API clamps limit to 50 instead of returning error


def _store_memory(
    mock_state,
    memory_id,
    content,
    tags,
    importance,
    mem_type="Context",
    timestamp=None,
    **extra,
):
    mock_state.memory_graph.memories[memory_id] = {
        "id": memory_id,
        "content": content,
        "tags": tags,
        "importance": importance,
        "type": mem_type,
        "timestamp": timestamp or utc_now(),
        **extra,
    }


def test_recall_current_only_filters_temporal_state(client, mock_state, auth_headers):
    mock_state.memory_graph.memories.clear()
    now = datetime.now(timezone.utc)
    active_id = "cc000000-0000-0000-0000-000000000001"
    expired_id = "cc000000-0000-0000-0000-000000000002"
    future_id = "cc000000-0000-0000-0000-000000000003"

    _store_memory(mock_state, active_id, "Active current state", ["state"], 0.9)
    _store_memory(
        mock_state,
        expired_id,
        "Expired stale state",
        ["state"],
        0.95,
        t_invalid=(now - timedelta(days=1)).isoformat(),
    )
    _store_memory(
        mock_state,
        future_id,
        "Future state",
        ["state"],
        0.98,
        t_valid=(now + timedelta(days=1)).isoformat(),
    )

    response = client.get(
        "/recall?tags=state&limit=10&state_debug=true",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.get_json()
    ids = [result["id"] for result in data["results"]]
    assert ids == [active_id]
    assert data["state_filter"]["suppressed_count"] == 2
    reasons = {item["reason"] for item in data["state_filter"]["suppressed"]}
    assert reasons == {"expired", "not_yet_valid"}

    response = client.get(
        "/recall?tags=state&limit=10&current_only=false",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.get_json()
    ids = {result["id"] for result in data["results"]}
    assert {active_id, expired_id, future_id}.issubset(ids)
    assert "state_filter" not in data


def test_recall_state_mode_defaults_to_current_and_echoes_mode(client, mock_state, auth_headers):
    mock_state.memory_graph.memories.clear()
    now = datetime.now(timezone.utc)
    active_id = "cc000000-0000-0000-0000-000000000032"
    expired_id = "cc000000-0000-0000-0000-000000000033"

    _store_memory(mock_state, active_id, "Active state mode current", ["state"], 0.9)
    _store_memory(
        mock_state,
        expired_id,
        "Expired state mode current",
        ["state"],
        0.95,
        t_invalid=(now - timedelta(days=1)).isoformat(),
    )

    response = client.get("/recall?tags=state&limit=10", headers=auth_headers)

    assert response.status_code == 200
    data = response.get_json()
    assert data["state_mode"] == "current"
    assert [result["id"] for result in data["results"]] == [active_id]
    assert data["state_filter"]["suppressed_count"] == 1


def test_recall_state_mode_history_preserves_state_history(client, mock_state, auth_headers):
    mock_state.memory_graph.memories.clear()
    now = datetime.now(timezone.utc)
    active_id = "cc000000-0000-0000-0000-000000000034"
    expired_id = "cc000000-0000-0000-0000-000000000035"
    future_id = "cc000000-0000-0000-0000-000000000036"

    _store_memory(mock_state, active_id, "Active state mode history", ["state"], 0.9)
    _store_memory(
        mock_state,
        expired_id,
        "Expired state mode history",
        ["state"],
        0.95,
        t_invalid=(now - timedelta(days=1)).isoformat(),
    )
    _store_memory(
        mock_state,
        future_id,
        "Future state mode history",
        ["state"],
        0.98,
        t_valid=(now + timedelta(days=1)).isoformat(),
    )

    response = client.get(
        "/recall?tags=state&limit=10&state_mode=history&state_debug=true",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["state_mode"] == "history"
    ids = {result["id"] for result in data["results"]}
    assert {active_id, expired_id, future_id}.issubset(ids)
    assert "state_filter" not in data


@pytest.mark.parametrize(
    ("state_mode", "current_only", "expected_mode", "expects_current"),
    [
        ("history", "true", "current", True),
        ("current", "false", "history", False),
    ],
)
def test_recall_current_only_takes_precedence_over_state_mode(
    client,
    mock_state,
    auth_headers,
    state_mode,
    current_only,
    expected_mode,
    expects_current,
):
    mock_state.memory_graph.memories.clear()
    now = datetime.now(timezone.utc)
    active_id = "cc000000-0000-0000-0000-000000000037"
    expired_id = "cc000000-0000-0000-0000-000000000038"

    _store_memory(mock_state, active_id, "Active precedence state", ["state"], 0.9)
    _store_memory(
        mock_state,
        expired_id,
        "Expired precedence state",
        ["state"],
        0.95,
        t_invalid=(now - timedelta(days=1)).isoformat(),
    )

    response = client.get(
        f"/recall?tags=state&limit=10&state_mode={state_mode}&current_only={current_only}",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["state_mode"] == expected_mode
    ids = {result["id"] for result in data["results"]}
    if expects_current:
        assert ids == {active_id}
        assert data["state_filter"]["suppressed_count"] == 1
    else:
        assert {active_id, expired_id}.issubset(ids)
        assert "state_filter" not in data


def test_recall_state_mode_rejects_unknown_value(client, auth_headers):
    response = client.get("/recall?state_mode=latest", headers=auth_headers)

    assert response.status_code == 400
    assert "state_mode" in response.get_data(as_text=True)


def test_recall_current_only_filters_archived_state(client, mock_state, auth_headers):
    mock_state.memory_graph.memories.clear()
    active_id = "cc000000-0000-0000-0000-000000000004"
    archived_id = "cc000000-0000-0000-0000-000000000005"

    _store_memory(mock_state, active_id, "Active retained state", ["state"], 0.9)
    _store_memory(
        mock_state,
        archived_id,
        "Archived stale state",
        ["state"],
        0.95,
        archived=True,
    )

    response = client.get(
        "/recall?tags=state&limit=10&state_debug=true",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.get_json()
    assert [result["id"] for result in data["results"]] == [active_id]
    assert data["state_filter"]["suppressed_count"] == 1
    assert data["state_filter"]["suppressed"][0]["id"] == archived_id
    assert data["state_filter"]["suppressed"][0]["reason"] == "archived"


@pytest.mark.parametrize("relation_type", ["INVALIDATED_BY", "EVOLVED_INTO"])
def test_recall_current_only_injects_active_replacement(
    client, mock_state, auth_headers, relation_type
):
    mock_state.memory_graph.memories.clear()
    mock_state.memory_graph.relationships.clear()
    old_id = "cc000000-0000-0000-0000-000000000010"
    replacement_id = "cc000000-0000-0000-0000-000000000011"

    _store_memory(mock_state, old_id, "Legacy favorite editor was Vim", ["state"], 1.0)
    _store_memory(
        mock_state,
        replacement_id,
        "Current favorite editor is Zed",
        ["replacement"],
        0.1,
    )
    mock_state.memory_graph.relationships.append(
        {"id1": old_id, "id2": replacement_id, "type": relation_type, "strength": 0.9}
    )

    response = client.get("/recall?limit=1&state_debug=true", headers=auth_headers)

    assert response.status_code == 200
    data = response.get_json()
    assert [result["id"] for result in data["results"]] == [replacement_id]
    assert data["results"][0]["match_type"] == "state_replacement"
    assert data["state_filter"]["suppressed_count"] == 1
    assert data["state_filter"]["replacement_count"] == 1
    assert data["state_filter"]["suppressed"][0]["replacement_id"] == replacement_id


def test_recall_current_only_batch_loads_relation_replacements(client, mock_state, auth_headers):
    mock_state.memory_graph.memories.clear()
    mock_state.memory_graph.relationships.clear()

    old_ids = [
        "cc000000-0000-0000-0000-000000000016",
        "cc000000-0000-0000-0000-000000000017",
        "cc000000-0000-0000-0000-000000000018",
    ]
    replacement_ids = [
        "cc000000-0000-0000-0000-000000000116",
        "cc000000-0000-0000-0000-000000000117",
        "cc000000-0000-0000-0000-000000000118",
    ]
    for idx, (old_id, replacement_id) in enumerate(zip(old_ids, replacement_ids)):
        _store_memory(mock_state, old_id, f"Legacy state {idx}", ["state"], 1.0 - idx / 10)
        _store_memory(mock_state, replacement_id, f"Current state {idx}", ["state"], 0.1)
        mock_state.memory_graph.relationships.append(
            {
                "id1": old_id,
                "id2": replacement_id,
                "type": "INVALIDATED_BY",
                "strength": 0.9,
            }
        )

    response = client.get(
        "/recall?tags=state&limit=3&state_debug=true",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.get_json()
    assert [result["id"] for result in data["results"]] == replacement_ids
    assert data["state_filter"]["suppressed_count"] == 3
    replacement_queries = [
        query
        for query, params in mock_state.memory_graph.queries
        if "RETURN source_id" in query and set(params.get("ids") or []) == set(old_ids)
    ]
    assert len(replacement_queries) == 1


def test_recall_current_only_keeps_replacement_score_order(client, mock_state, auth_headers):
    mock_state.memory_graph.memories.clear()
    mock_state.memory_graph.relationships.clear()
    old_id = "cc000000-0000-0000-0000-000000000019"
    active_id = "cc000000-0000-0000-0000-000000000119"
    replacement_id = "cc000000-0000-0000-0000-000000000219"

    _store_memory(mock_state, old_id, "Highest scoring legacy state", ["state"], 1.0)
    _store_memory(mock_state, active_id, "Lower scoring active state", ["state"], 0.9)
    _store_memory(mock_state, replacement_id, "Current replacement state", ["state"], 0.1)
    mock_state.memory_graph.relationships.append(
        {"id1": old_id, "id2": replacement_id, "type": "INVALIDATED_BY", "strength": 0.9}
    )

    response = client.get(
        "/recall?tags=state&limit=2&state_debug=true",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.get_json()
    assert [result["id"] for result in data["results"]] == [replacement_id, active_id]


def test_recall_current_only_replacement_respects_tag_filter(client, mock_state, auth_headers):
    mock_state.memory_graph.memories.clear()
    mock_state.memory_graph.relationships.clear()
    old_id = "cc000000-0000-0000-0000-000000000012"
    replacement_id = "cc000000-0000-0000-0000-000000000013"

    _store_memory(mock_state, old_id, "Legacy gated billing plan was Basic", ["gated-old"], 1.0)
    _store_memory(
        mock_state,
        replacement_id,
        "Current gated billing plan is Pro",
        ["gated-current"],
        0.1,
    )
    mock_state.memory_graph.relationships.append(
        {"id1": old_id, "id2": replacement_id, "type": "INVALIDATED_BY", "strength": 0.9}
    )

    response = client.get(
        "/recall?tags=gated-old&tag_match=exact&limit=1&state_debug=true",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["results"] == []
    assert data["state_filter"]["suppressed_count"] == 1
    assert data["state_filter"]["replacement_count"] == 0
    assert data["state_filter"]["suppressed"][0]["id"] == old_id
    assert data["state_filter"]["suppressed"][0]["replacement_id"] == replacement_id
    assert data["state_filter"]["replacements"] == []


@pytest.mark.parametrize("relation_type", ["INVALIDATED_BY", "EVOLVED_INTO"])
def test_recall_current_only_false_keeps_relation_history(
    client, mock_state, auth_headers, relation_type
):
    mock_state.memory_graph.memories.clear()
    mock_state.memory_graph.relationships.clear()
    old_id = "cc000000-0000-0000-0000-000000000014"
    replacement_id = "cc000000-0000-0000-0000-000000000015"

    _store_memory(mock_state, old_id, "Historical tracker was Jira", ["state"], 1.0)
    _store_memory(mock_state, replacement_id, "Current tracker is Linear", ["state"], 0.9)
    mock_state.memory_graph.relationships.append(
        {"id1": old_id, "id2": replacement_id, "type": relation_type, "strength": 0.9}
    )

    response = client.get(
        "/recall?tags=state&limit=2&current_only=false&state_debug=true",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.get_json()
    assert [result["id"] for result in data["results"]] == [old_id, replacement_id]
    assert "state_filter" not in data


def test_recall_current_only_does_not_suppress_contradictions(client, mock_state, auth_headers):
    mock_state.memory_graph.memories.clear()
    mock_state.memory_graph.relationships.clear()
    old_id = "cc000000-0000-0000-0000-000000000020"
    conflicting_id = "cc000000-0000-0000-0000-000000000021"

    _store_memory(mock_state, old_id, "Legacy plan used SQLite", ["state"], 1.0)
    _store_memory(mock_state, conflicting_id, "Conflicting plan used Postgres", ["state"], 0.1)
    mock_state.memory_graph.relationships.append(
        {"id1": old_id, "id2": conflicting_id, "type": "CONTRADICTS", "strength": 0.9}
    )

    response = client.get("/recall?limit=2&state_debug=true", headers=auth_headers)

    assert response.status_code == 200
    data = response.get_json()
    assert [result["id"] for result in data["results"]] == [old_id, conflicting_id]
    assert data["state_filter"]["suppressed_count"] == 0


def test_recall_current_only_injects_replacement_for_expired_memory(
    client, mock_state, auth_headers
):
    mock_state.memory_graph.memories.clear()
    mock_state.memory_graph.relationships.clear()
    now = datetime.now(timezone.utc)
    old_id = "cc000000-0000-0000-0000-000000000025"
    replacement_id = "cc000000-0000-0000-0000-000000000026"

    _store_memory(
        mock_state,
        old_id,
        "Expired favorite editor was Vim",
        ["state"],
        1.0,
        t_invalid=(now - timedelta(days=1)).isoformat(),
    )
    _store_memory(mock_state, replacement_id, "Current favorite editor is Zed", ["state"], 0.1)
    mock_state.memory_graph.relationships.append(
        {"id1": old_id, "id2": replacement_id, "type": "INVALIDATED_BY", "strength": 0.9}
    )

    response = client.get("/recall?limit=1&state_debug=true", headers=auth_headers)

    assert response.status_code == 200
    data = response.get_json()
    assert [result["id"] for result in data["results"]] == [replacement_id]
    assert data["state_filter"]["suppressed"][0]["reason"] == "expired"
    assert data["state_filter"]["replacements"][0]["replaces_id"] == old_id


def test_recall_current_only_filters_vector_results(client, mock_state, auth_headers):
    now = datetime.now(timezone.utc)

    def custom_search(
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        *,
        with_payload: bool = True,
        with_vectors: bool = False,
        query_filter=None,
    ) -> list[Any]:
        _ = collection_name, query_vector, limit, with_payload, with_vectors, query_filter
        return [
            SimpleNamespace(
                id="vec-active",
                score=0.75,
                payload={
                    "id": "vec-active",
                    "content": "Active vector state",
                    "tags": ["state"],
                    "importance": 0.5,
                    "timestamp": utc_now(),
                },
            ),
            SimpleNamespace(
                id="vec-expired",
                score=0.74,
                payload={
                    "id": "vec-expired",
                    "content": "Expired vector state",
                    "tags": ["state"],
                    "importance": 0.5,
                    "timestamp": utc_now(),
                    "t_invalid": (now - timedelta(days=1)).isoformat(),
                },
            ),
        ]

    mock_state.qdrant.search = custom_search

    response = client.get("/recall?query=state&limit=2", headers=auth_headers)

    assert response.status_code == 200
    data = response.get_json()
    assert [result["id"] for result in data["results"]] == ["vec-active"]
    assert data["state_filter"]["suppressed_count"] == 1

    response = client.get(
        "/recall?query=state&limit=2&current_only=false",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.get_json()
    assert {result["id"] for result in data["results"]} == {"vec-active", "vec-expired"}


def test_recall_current_only_filters_tag_only_results(client, mock_state, auth_headers):
    now = datetime.now(timezone.utc)
    mock_state.memory_graph = None
    mock_state.qdrant.points = {
        "tag-active": {
            "vector": [0.1] * 3,
            "payload": {
                "id": "tag-active",
                "content": "Active tag state",
                "tags": ["state"],
                "importance": 0.5,
                "timestamp": utc_now(),
            },
        },
        "tag-expired": {
            "vector": [0.1] * 3,
            "payload": {
                "id": "tag-expired",
                "content": "Expired tag state",
                "tags": ["state"],
                "importance": 0.6,
                "timestamp": utc_now(),
                "t_invalid": (now - timedelta(days=1)).isoformat(),
            },
        },
    }

    response = client.get("/recall?tags=state&tag_match=exact&limit=10", headers=auth_headers)

    assert response.status_code == 200
    data = response.get_json()
    assert [result["id"] for result in data["results"]] == ["tag-active"]
    assert data["state_filter"]["suppressed_count"] == 1


def test_recall_current_only_filters_relation_expansion(client, mock_state, auth_headers):
    mock_state.memory_graph.memories.clear()
    mock_state.memory_graph.relationships.clear()
    now = datetime.now(timezone.utc)
    seed_id = "cc000000-0000-0000-0000-000000000030"
    expired_related_id = "cc000000-0000-0000-0000-000000000031"

    _store_memory(mock_state, seed_id, "Seed state", ["state"], 0.9)
    _store_memory(
        mock_state,
        expired_related_id,
        "Expired related state",
        ["related"],
        0.8,
        t_invalid=(now - timedelta(days=1)).isoformat(),
    )
    mock_state.memory_graph.relationships.append(
        {"id1": seed_id, "id2": expired_related_id, "type": "RELATES_TO", "strength": 0.9}
    )

    response = client.get(
        "/recall?tags=state&limit=1&expand_relations=true&relation_limit=5",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.get_json()
    assert [result["id"] for result in data["results"]] == [seed_id]
    assert data["state_filter"]["suppressed_count"] == 1


def test_recall_prioritizes_style_context(client, mock_state, auth_headers):
    """Ensure coding-style memories rise to the top when editing Python."""
    mock_state.memory_graph.memories.clear()
    style_id = "aa000000-0000-0000-0000-000000000001"
    other_id = "aa000000-0000-0000-0000-000000000002"
    _store_memory(
        mock_state,
        other_id,
        "Recent brainstorming about product metrics",
        ["planning"],
        0.98,
        mem_type="Context",
    )
    _store_memory(
        mock_state,
        style_id,
        "Python coding style: prefer dataclasses, avoid mutable defaults.",
        ["coding-style", "python", "style-guide"],
        0.82,
        mem_type="Style",
    )

    response = client.get("/recall?limit=2&active_path=/tmp/sample.py", headers=auth_headers)
    assert response.status_code == 200
    data = response.get_json()
    assert data["results"][0]["id"] == style_id
    context_info = data.get("context_priority") or {}
    assert context_info.get("language") == "python"
    assert context_info.get("injected") is False


def test_recall_injects_style_when_limit_small(client, mock_state, auth_headers):
    """Ensure style memory appears even when limit would normally omit it."""
    mock_state.memory_graph.memories.clear()
    style_id = "aa000000-0000-0000-0000-000000000003"
    other_id = "aa000000-0000-0000-0000-000000000004"
    _store_memory(
        mock_state,
        other_id,
        "High importance launch checklist",
        ["launch"],
        0.99,
        mem_type="Insight",
    )
    _store_memory(
        mock_state,
        style_id,
        "Python naming rules: prefer snake_case and explicit imports.",
        ["coding-style", "python"],
        0.7,
        mem_type="Style",
    )

    response = client.get("/recall?limit=1&active_path=/workspace/tool.py", headers=auth_headers)
    assert response.status_code == 200
    data = response.get_json()
    assert data["results"][0]["id"] == style_id
    context_info = data.get("context_priority") or {}
    assert context_info.get("injected") is True


def test_recall_priority_ids_fetches_specific_memory(client, mock_state, auth_headers):
    """priority_ids should fetch a memory directly even when the query does not match it."""
    mock_state.memory_graph.memories.clear()
    target_id = "aa000000-0000-0000-0000-000000000010"
    query_match_id = "aa000000-0000-0000-0000-000000000011"

    _store_memory(
        mock_state,
        target_id,
        "Completely unrelated anchored memory",
        ["anchor"],
        0.2,
        timestamp="2026-03-01T00:00:00+00:00",
    )
    _store_memory(
        mock_state,
        query_match_id,
        "Python query match memory",
        ["python"],
        0.9,
        timestamp="2026-03-02T00:00:00+00:00",
    )

    response = client.get(
        f"/recall?query=python&priority_ids={target_id}&limit=2",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.get_json()
    ids = [result["id"] for result in data["results"]]
    assert target_id in ids
    assert query_match_id in ids


def test_recall_priority_ids_are_guaranteed_with_small_limit(client, mock_state, auth_headers):
    """priority_ids should survive score sorting and limit truncation."""
    mock_state.memory_graph.memories.clear()
    target_id = "aa000000-0000-0000-0000-000000000012"
    query_match_id = "aa000000-0000-0000-0000-000000000013"

    _store_memory(
        mock_state,
        target_id,
        "Anchored memory with unrelated content",
        ["anchor"],
        0.1,
        timestamp="2026-03-01T00:00:00+00:00",
    )
    _store_memory(
        mock_state,
        query_match_id,
        "Python query match that would normally rank first",
        ["python"],
        1.0,
        timestamp="2026-03-02T00:00:00+00:00",
    )

    response = client.get(
        f"/recall?query=python&priority_ids={target_id}&limit=1",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["count"] == 1
    assert data["results"][0]["id"] == target_id


def test_recall_vector_results_include_keyword_score_from_content(client, mock_state, auth_headers):
    def custom_search(
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        *,
        with_payload: bool = True,
        with_vectors: bool = False,
        query_filter=None,
    ) -> list[Any]:
        _ = collection_name, query_vector, limit, with_payload, with_vectors, query_filter
        return [
            SimpleNamespace(
                id="vec-1",
                score=0.71,
                payload={
                    "id": "vec-1",
                    "content": "AutoJack recall investigation memory",
                    "tags": ["debugging"],
                    "importance": 0.3,
                    "timestamp": utc_now(),
                },
            )
        ]

    mock_state.qdrant.search = custom_search
    response = client.get("/recall?query=AutoJack recall&limit=1", headers=auth_headers)

    assert response.status_code == 200
    data = response.get_json()
    assert data["count"] == 1
    assert data["results"][0]["score_components"]["keyword"] == 1.0


def test_recall_semantic_query_hydrates_summary_from_graph(mock_state):
    memory_with_summary_id = "11111111-1111-1111-1111-111111111111"
    memory_without_summary_id = "22222222-2222-2222-2222-222222222222"

    for memory_id, summary in (
        (memory_with_summary_id, "Stored graph summary for issue 180."),
        (memory_without_summary_id, None),
    ):
        mock_state.memory_graph.memories[memory_id] = {
            "id": memory_id,
            "content": f"Issue 180 semantic recall memory {memory_id}",
            "summary": summary,
            "tags": ["issue180"],
            "tag_prefixes": ["issue180"],
            "importance": 0.8,
            "timestamp": utc_now(),
            "type": "Context",
            "confidence": 0.9,
            "metadata": "{}",
        }
        mock_state.qdrant.points[memory_id] = {
            "vector": [0.1] * 3,
            "payload": {
                "id": memory_id,
                "content": f"Issue 180 semantic recall memory {memory_id}",
                "tags": ["issue180"],
                "tag_prefixes": ["issue180"],
                "importance": 0.8,
                "timestamp": utc_now(),
                "type": "Context",
                "confidence": 0.9,
                "metadata": {},
            },
        }

    def custom_search(
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        *,
        with_payload: bool = True,
        with_vectors: bool = False,
        query_filter=None,
    ) -> list[Any]:
        _ = collection_name, query_vector, limit, with_payload, with_vectors, query_filter
        return [
            SimpleNamespace(
                id=memory_with_summary_id,
                score=0.91,
                payload=mock_state.qdrant.points[memory_with_summary_id]["payload"],
            ),
            SimpleNamespace(
                id=memory_without_summary_id,
                score=0.9,
                payload=mock_state.qdrant.points[memory_without_summary_id]["payload"],
            ),
        ]

    mock_state.qdrant.search = custom_search

    with app.app.test_request_context("/recall?query=issue180&limit=2&current_only=false"):
        response = handle_recall(
            get_memory_graph=lambda: mock_state.memory_graph,
            get_qdrant_client=lambda: mock_state.qdrant,
            normalize_tag_list=app._normalize_tag_list,
            normalize_timestamp=app._normalize_timestamp,
            parse_time_expression=app._parse_time_expression,
            extract_keywords=app._extract_keywords,
            compute_metadata_score=app._compute_metadata_score,
            result_passes_filters=app._result_passes_filters,
            graph_keyword_search=app._graph_keyword_search,
            vector_search=app._vector_search,
            vector_filter_only_tag_search=app._vector_filter_only_tag_search,
            recall_max_limit=50,
            logger=Mock(),
            jit_enrich_fn=None,
        )

    data = response.get_json()
    results_by_id = {result["id"]: result["memory"] for result in data["results"]}

    assert results_by_id[memory_with_summary_id]["summary"] == "Stored graph summary for issue 180."
    assert "summary" not in results_by_id[memory_without_summary_id]


# ==================== Tag Scope Diagnostics + Relevance Gate (issue #130) ====


def _scoped_pool_search(hits: list[dict]) -> Any:
    """Build a qdrant search stub returning the given (id, score, payload) hits."""

    def custom_search(
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        *,
        with_payload: bool = True,
        with_vectors: bool = False,
        query_filter=None,
    ) -> list[Any]:
        _ = collection_name, query_vector, limit, with_payload, with_vectors, query_filter
        return [
            SimpleNamespace(id=hit["id"], score=hit["score"], payload=hit["payload"])
            for hit in hits
        ]

    return custom_search


def _issue130_pool(mock_state) -> None:
    """Two flint-tagged memories surviving the tag gate: one off-topic but
    high-importance, one on-topic. Contents avoid the query tokens so the
    keyword fallback contributes no evidence."""
    mock_state.qdrant.search = _scoped_pool_search(
        [
            {
                "id": "off-topic-important",
                "score": 0.04,
                "payload": {
                    "id": "off-topic-important",
                    "content": "Critical lessons learned the hard way",
                    "tags": ["flint"],
                    "importance": 1.0,
                    "confidence": 1.0,
                    "timestamp": utc_now(),
                },
            },
            {
                "id": "on-topic",
                "score": 0.6,
                "payload": {
                    "id": "on-topic",
                    "content": "Semantically close result content",
                    "tags": ["flint"],
                    "importance": 0.0,
                    "timestamp": _timestamp_days_ago(200),
                },
            },
        ]
    )


def test_recall_tag_scope_gate_off_keeps_legacy_ranking(
    client, mock_state, auth_headers, monkeypatch
):
    """Default gate (0.0): off-topic high-importance wins; tag_scope echoes zero gated."""
    _pin_default_scoring(monkeypatch, gate=0.0)
    _issue130_pool(mock_state)

    response = client.get(
        "/recall",
        query_string={
            "query": "quarterly metrics dashboard",
            "tags": "flint",
            "tag_match": "exact",
            "limit": 5,
            "min_score": 0,
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.get_json()
    assert [r["id"] for r in data["results"]] == ["off-topic-important", "on-topic"]
    assert data["tag_scope"] == {
        "filtered": True,
        "pool_size_hint": 2,
        "gated_low_evidence": 0,
    }


def test_recall_tag_scope_gate_reranks_off_topic_high_importance(
    client, mock_state, auth_headers, monkeypatch
):
    """Gate on: on-topic ranks first and the gated result is counted in tag_scope."""
    _pin_default_scoring(monkeypatch, gate=0.2)
    _issue130_pool(mock_state)

    response = client.get(
        "/recall",
        query_string={
            "query": "quarterly metrics dashboard",
            "tags": "flint",
            "tag_match": "exact",
            "limit": 5,
            "min_score": 0,
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.get_json()
    assert [r["id"] for r in data["results"]] == ["on-topic", "off-topic-important"]
    assert data["tag_scope"]["filtered"] is True
    assert data["tag_scope"]["gated_low_evidence"] >= 1
    gated = next(r for r in data["results"] if r["id"] == "off-topic-important")
    assert gated["score_components"]["relevance_gated"] is True
    assert "evidence" in gated["score_components"]


def test_recall_tag_scope_absent_without_tags(client, mock_state, auth_headers, monkeypatch):
    """Backward compat: no tags passed -> no tag_scope (or scope_fallback) echo."""
    _pin_default_scoring(monkeypatch, gate=0.2)
    _issue130_pool(mock_state)

    response = client.get(
        "/recall",
        query_string={"query": "quarterly metrics dashboard", "limit": 5, "min_score": 0},
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.get_json()
    assert "tag_scope" not in data
    assert "scope_fallback" not in data


def test_recall_tag_scope_pool_size_hint_null_without_semantic_query(
    client, mock_state, auth_headers, monkeypatch
):
    """Tag-only recall (no query/embedding) has no comparable vector pool
    count, so pool_size_hint must be null."""
    _pin_default_scoring(monkeypatch, gate=0.0)
    _store_memory(
        mock_state,
        "ee000000-0000-0000-0000-000000000001",
        "Tag-only scoped memory",
        ["flint"],
        0.5,
    )

    response = client.get(
        "/recall",
        query_string={"tags": "flint", "tag_match": "exact", "limit": 5},
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["tag_scope"]["filtered"] is True
    assert data["tag_scope"]["pool_size_hint"] is None


def _scope_fallback_pool(mock_state) -> None:
    """One scoped memory plus two outside the tag scope, all vector matches."""
    mock_state.qdrant.search = _scoped_pool_search(
        [
            {
                "id": "fill-strong",
                "score": 0.9,
                "payload": {
                    "id": "fill-strong",
                    "content": "Strong unscoped vector match",
                    "tags": ["other"],
                    "importance": 0.1,
                    "timestamp": utc_now(),
                },
            },
            {
                "id": "fill-weak",
                "score": 0.8,
                "payload": {
                    "id": "fill-weak",
                    "content": "Weaker unscoped vector match",
                    "tags": ["other"],
                    "importance": 0.1,
                    "timestamp": utc_now(),
                },
            },
            {
                "id": "scoped-1",
                "score": 0.5,
                "payload": {
                    "id": "scoped-1",
                    "content": "Scoped vector match",
                    "tags": ["scoped"],
                    "importance": 0.1,
                    "timestamp": utc_now(),
                },
            },
        ]
    )


def test_recall_scope_fallback_fills_after_scoped_results(
    client, mock_state, auth_headers, monkeypatch
):
    _pin_default_scoring(monkeypatch, gate=0.0)
    _scope_fallback_pool(mock_state)

    response = client.get(
        "/recall",
        query_string={
            "query": "quarterly metrics dashboard",
            "tags": "scoped",
            "tag_match": "exact",
            "limit": 3,
            "min_score": 0,
            "scope_fallback": "true",
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.get_json()
    # Scoped results come first regardless of score; fills are appended after.
    assert [r["id"] for r in data["results"]] == ["scoped-1", "fill-strong", "fill-weak"]
    assert "outside_tag_scope" not in data["results"][0]
    assert data["results"][1]["outside_tag_scope"] is True
    assert data["results"][2]["outside_tag_scope"] is True
    assert data["scope_fallback"] is True
    assert data["tag_scope"]["filtered"] is True


def test_recall_scope_fallback_respects_exclude_tags(client, mock_state, auth_headers, monkeypatch):
    _pin_default_scoring(monkeypatch, gate=0.0)
    _scope_fallback_pool(mock_state)

    response = client.get(
        "/recall",
        query_string={
            "query": "quarterly metrics dashboard",
            "tags": "scoped",
            "tag_match": "exact",
            "limit": 3,
            "min_score": 0,
            "scope_fallback": "true",
            "exclude_tags": "other",
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.get_json()
    assert [r["id"] for r in data["results"]] == ["scoped-1"]
    assert data["scope_fallback"] is True


def test_recall_scope_fallback_defaults_off(client, mock_state, auth_headers, monkeypatch):
    _pin_default_scoring(monkeypatch, gate=0.0)
    _scope_fallback_pool(mock_state)

    response = client.get(
        "/recall",
        query_string={
            "query": "quarterly metrics dashboard",
            "tags": "scoped",
            "tag_match": "exact",
            "limit": 3,
            "min_score": 0,
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.get_json()
    assert [r["id"] for r in data["results"]] == ["scoped-1"]
    assert "scope_fallback" not in data
    assert all("outside_tag_scope" not in r for r in data["results"])


def test_recall_scope_fallback_does_not_resurrect_min_score_dropped_scoped_result(
    client, mock_state, auth_headers, monkeypatch
):
    """A scoped result dropped by min_score carries the scope tag, so it must
    not re-enter via the unscoped fill search mislabeled outside_tag_scope."""
    _pin_default_scoring(monkeypatch, gate=0.0)
    mock_state.qdrant.search = _scoped_pool_search(
        [
            {
                "id": "scoped-strong",
                "score": 0.9,
                "payload": {
                    "id": "scoped-strong",
                    "content": "Scoped vector match",
                    "tags": ["scoped"],
                    "importance": 0.1,
                    "timestamp": utc_now(),
                },
            },
            {
                # Final score ~0.13 with pinned defaults: dropped by min_score
                # in the scoped pass, then re-fetched by the fallback search.
                "id": "scoped-weak",
                "score": 0.05,
                "payload": {
                    "id": "scoped-weak",
                    "content": "Weak scoped vector match",
                    "tags": ["scoped"],
                    "importance": 0.1,
                    "timestamp": utc_now(),
                },
            },
            {
                "id": "fill-ok",
                "score": 0.8,
                "payload": {
                    "id": "fill-ok",
                    "content": "Strong unscoped vector match",
                    "tags": ["other"],
                    "importance": 0.1,
                    "timestamp": utc_now(),
                },
            },
        ]
    )

    response = client.get(
        "/recall",
        query_string={
            "query": "quarterly metrics dashboard",
            "tags": "scoped",
            "tag_match": "exact",
            "limit": 3,
            "min_score": 0.2,
            "scope_fallback": "true",
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.get_json()
    assert [r["id"] for r in data["results"]] == ["scoped-strong", "fill-ok"]
    assert data["results"][1]["outside_tag_scope"] is True
    assert data["scope_fallback"] is True


def test_recall_scope_fallback_fill_below_min_score_excluded(
    client, mock_state, auth_headers, monkeypatch
):
    """Fills get min_score filter parity with the scoped path."""
    _pin_default_scoring(monkeypatch, gate=0.0)
    mock_state.qdrant.search = _scoped_pool_search(
        [
            {
                "id": "scoped-1",
                "score": 0.9,
                "payload": {
                    "id": "scoped-1",
                    "content": "Scoped vector match",
                    "tags": ["scoped"],
                    "importance": 0.1,
                    "timestamp": utc_now(),
                },
            },
            {
                # Final score ~0.13 with pinned defaults: below min_score, so
                # it must not fill the open slots.
                "id": "fill-weak",
                "score": 0.05,
                "payload": {
                    "id": "fill-weak",
                    "content": "Weak unscoped vector match",
                    "tags": ["other"],
                    "importance": 0.1,
                    "timestamp": utc_now(),
                },
            },
        ]
    )

    response = client.get(
        "/recall",
        query_string={
            "query": "quarterly metrics dashboard",
            "tags": "scoped",
            "tag_match": "exact",
            "limit": 3,
            "min_score": 0.2,
            "scope_fallback": "true",
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.get_json()
    assert [r["id"] for r in data["results"]] == ["scoped-1"]
    # The fallback ran (echoed); it just found nothing eligible.
    assert data["scope_fallback"] is True


def test_recall_scope_fallback_edge_superseded_fill_replaced_under_current_state(
    client, mock_state, auth_headers, monkeypatch
):
    """Filter parity for state mode: an edge-superseded memory
    (INVALIDATED_BY) must not re-enter as a fill under current state mode;
    its active replacement fills instead, matching the main path."""
    _pin_default_scoring(monkeypatch, gate=0.0)
    mock_state.memory_graph.memories.clear()
    mock_state.memory_graph.relationships.clear()
    superseded_id = "dd000000-0000-0000-0000-000000000001"
    replacement_id = "dd000000-0000-0000-0000-000000000002"
    _store_memory(mock_state, superseded_id, "Old unscoped fact", ["other"], 0.9)
    _store_memory(
        mock_state,
        replacement_id,
        "Current unscoped fact",
        ["other-current"],
        0.1,
    )
    mock_state.memory_graph.relationships.append(
        {"id1": superseded_id, "id2": replacement_id, "type": "INVALIDATED_BY", "strength": 0.9}
    )
    mock_state.qdrant.search = _scoped_pool_search(
        [
            {
                "id": "scoped-1",
                "score": 0.5,
                "payload": {
                    "id": "scoped-1",
                    "content": "Scoped vector match",
                    "tags": ["scoped"],
                    "importance": 0.1,
                    "timestamp": utc_now(),
                },
            },
            {
                "id": superseded_id,
                "score": 0.9,
                "payload": {
                    "id": superseded_id,
                    "content": "Old unscoped fact",
                    "tags": ["other"],
                    "importance": 0.9,
                    "timestamp": utc_now(),
                },
            },
        ]
    )

    response = client.get(
        "/recall",
        query_string={
            "query": "quarterly metrics dashboard",
            "tags": "scoped",
            "tag_match": "exact",
            "limit": 3,
            "min_score": 0,
            "scope_fallback": "true",
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.get_json()
    ids = [r["id"] for r in data["results"]]
    assert superseded_id not in ids
    assert ids == ["scoped-1", replacement_id]
    fill = data["results"][1]
    assert fill["outside_tag_scope"] is True
    assert fill["match_type"] == "state_replacement"


def test_recall_scope_fallback_in_scope_state_replacement_not_resurrected(
    client, mock_state, auth_headers, monkeypatch
):
    """An out-of-scope fill superseded by an IN-scope replacement must not
    smuggle that replacement back in mislabeled outside_tag_scope: in-scope
    rows belong to the scoped pass, where min_score already rejected this one."""
    _pin_default_scoring(monkeypatch, gate=0.0)
    mock_state.memory_graph.memories.clear()
    mock_state.memory_graph.relationships.clear()
    superseded_id = "dd000000-0000-0000-0000-000000000011"
    replacement_id = "dd000000-0000-0000-0000-000000000012"
    _store_memory(mock_state, superseded_id, "Old unscoped fact", ["other"], 0.9)
    _store_memory(
        mock_state,
        replacement_id,
        "Replacement fact with unrelated wording",
        ["scoped"],
        0.1,
    )
    mock_state.memory_graph.relationships.append(
        {"id1": superseded_id, "id2": replacement_id, "type": "INVALIDATED_BY", "strength": 0.9}
    )
    mock_state.qdrant.search = _scoped_pool_search(
        [
            {
                "id": "scoped-1",
                "score": 0.9,
                "payload": {
                    "id": "scoped-1",
                    "content": "Scoped vector match",
                    "tags": ["scoped"],
                    "importance": 0.1,
                    "timestamp": utc_now(),
                },
            },
            {
                "id": superseded_id,
                "score": 0.9,
                "payload": {
                    "id": superseded_id,
                    "content": "Old unscoped fact",
                    "tags": ["other"],
                    "importance": 0.9,
                    "timestamp": utc_now(),
                },
            },
        ]
    )

    response = client.get(
        "/recall",
        query_string={
            "query": "quarterly metrics dashboard",
            "tags": "scoped",
            "tag_match": "exact",
            "limit": 3,
            "min_score": 0.3,
            "scope_fallback": "true",
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.get_json()
    ids = [r["id"] for r in data["results"]]
    # The superseded fill is suppressed; its in-scope replacement must not
    # take its slot as a fill (it is in scope, so the fill label would lie).
    assert superseded_id not in ids
    assert replacement_id not in ids
    assert ids == ["scoped-1"]
    assert all("outside_tag_scope" not in r for r in data["results"])


def _filter_aware_pool_search(scoped_hits: list[dict], unscoped_hits: list[dict]) -> Any:
    """Qdrant search stub that respects tag scoping: the scoped pass (tag
    query_filter set) sees ``scoped_hits``; the unscoped fallback pass
    (query_filter None) sees ``unscoped_hits``."""

    def custom_search(
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        *,
        with_payload: bool = True,
        with_vectors: bool = False,
        query_filter=None,
    ) -> list[Any]:
        _ = collection_name, query_vector, limit, with_payload, with_vectors
        hits = unscoped_hits if query_filter is None else scoped_hits
        return [
            SimpleNamespace(id=hit["id"], score=hit["score"], payload=hit["payload"])
            for hit in hits
        ]

    return custom_search


def test_recall_scope_fallback_rejects_in_scope_fill_above_min_score(
    client, mock_state, auth_headers, monkeypatch
):
    """An in-scope candidate surfaced only by the unscoped fallback search must
    be rejected as a direct fill even when its recomputed fill score clears
    min_score. Runs under state_mode=history so the state-filter pass (which
    re-checks tag scope on its own rows) cannot mask the direct-fill guard."""
    _pin_default_scoring(monkeypatch, gate=0.0)
    scoped_hit = {
        "id": "scoped-1",
        "score": 0.9,
        "payload": {
            "id": "scoped-1",
            "content": "Scoped vector match",
            "tags": ["scoped"],
            "importance": 0.1,
            "timestamp": utc_now(),
        },
    }
    in_scope_victim = {
        # Strong vector match + high importance: fill score well above
        # min_score, so only the in-scope rejection can keep it out.
        "id": "scoped-hidden",
        "score": 0.9,
        "payload": {
            "id": "scoped-hidden",
            "content": "Scoped match missing from the scoped pass",
            "tags": ["scoped"],
            "importance": 0.9,
            "timestamp": utc_now(),
        },
    }
    fill_ok = {
        "id": "fill-ok",
        "score": 0.8,
        "payload": {
            "id": "fill-ok",
            "content": "Strong unscoped vector match",
            "tags": ["other"],
            "importance": 0.1,
            "timestamp": utc_now(),
        },
    }
    mock_state.qdrant.search = _filter_aware_pool_search(
        [scoped_hit], [scoped_hit, in_scope_victim, fill_ok]
    )

    response = client.get(
        "/recall",
        query_string={
            "query": "quarterly metrics dashboard",
            "tags": "scoped",
            "tag_match": "exact",
            "limit": 3,
            "min_score": 0.3,
            "scope_fallback": "true",
            "state_mode": "history",
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.get_json()
    ids = [r["id"] for r in data["results"]]
    # The in-scope candidate must not be resurrected as a fill; the genuine
    # out-of-scope fill still lands.
    assert "scoped-hidden" not in ids
    assert ids == ["scoped-1", "fill-ok"]
    assert data["results"][1]["outside_tag_scope"] is True


def test_recall_adaptive_floor_keeps_clustered_relevant_tail():
    data = _call_handle_recall_for_scores(
        "AutoJack",
        [0.76, 0.75, 0.73, 0.55, 0.54, 0.53, 0.52, 0.51],
    )

    assert data["count"] == 8
    assert "score_filter" not in data


def test_recall_adaptive_floor_applies_on_large_drop_when_half_remain():
    data = _call_handle_recall_for_scores(
        "AutoJack",
        [1.0, 0.99, 0.98, 0.4, 0.39, 0.38, 0.37, 0.36],
    )

    assert data["count"] == 4
    assert data["score_filter"]["adaptive_floor"] == 0.4
    assert data["score_filter"]["filtered_count"] == 4


def test_recall_adaptive_floor_skips_cut_that_would_remove_more_than_half():
    data = _call_handle_recall_for_scores(
        "AutoJack",
        [1.0, 0.99, 0.3, 0.29, 0.28, 0.27, 0.26],
    )

    assert data["count"] == 7
    assert "score_filter" not in data


# ==================== Test Memory Update ====================


def test_get_memory_success_parses_metadata_and_updates_access(client, mock_state, auth_headers):
    """Test successful memory retrieval by ID with parsed metadata and access tracking."""
    memory_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
    mock_state.memory_graph.memories[memory_id] = {
        "id": memory_id,
        "content": "Memory content",
        "tags": ["api", "test"],
        "importance": 0.7,
        "metadata": json.dumps({"source": "unit-test", "count": 2}),
        "timestamp": utc_now(),
    }

    response = client.get(f"/memory/{memory_id}", headers=auth_headers)
    assert response.status_code == 200
    data = response.get_json()

    assert data["status"] == "success"
    assert data["memory"]["id"] == memory_id
    assert data["memory"]["metadata"] == {"source": "unit-test", "count": 2}

    # Verify last_accessed update query was issued via on_access callback
    last_accessed_queries = [
        query
        for query, _params in mock_state.memory_graph.queries
        if "SET m.last_accessed" in query
    ]
    assert last_accessed_queries


@pytest.mark.usefixtures("mock_state")
def test_get_memory_not_found(client, auth_headers):
    """Test retrieving a non-existent memory by ID."""
    response = client.get("/memory/00000000-0000-0000-0000-000000000000", headers=auth_headers)
    assert response.status_code == 404


@pytest.mark.usefixtures("mock_state")
def test_get_memory_invalid_id(client, auth_headers):
    """Test retrieving a memory with an invalid (non-UUID) ID returns 400."""
    response = client.get("/memory/not-a-uuid", headers=auth_headers)
    assert response.status_code == 400


def test_get_memory_graph_unavailable(client, mock_state, auth_headers):
    """Test get memory endpoint returns 503 when graph is unavailable."""
    mock_state.memory_graph = None
    response = client.get("/memory/00000000-0000-0000-0000-000000000001", headers=auth_headers)
    assert response.status_code == 503


def test_get_memory_query_failure(client, mock_state, auth_headers):
    """Test get memory endpoint returns 500 when graph query fails."""
    mock_graph = mock_state.memory_graph
    mock_graph.query = Mock(side_effect=RuntimeError("query failed"))
    response = client.get("/memory/00000000-0000-0000-0000-000000000002", headers=auth_headers)
    assert response.status_code == 500


def test_update_memory_success(client, mock_state, auth_headers):
    """Test successful memory update."""
    memory_id = "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"

    # Create initial memory
    mock_state.memory_graph.memories[memory_id] = {
        "id": memory_id,
        "content": "Original content",
        "tags": ["original"],
        "importance": 0.5,
        "timestamp": utc_now(),
    }

    update_data = {
        "content": "Updated content",
        "tags": ["updated", "modified"],
        "importance": 0.9,
    }

    response = client.patch(f"/memory/{memory_id}", json=update_data, headers=auth_headers)

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["memory_id"] == memory_id


def test_update_memory_preserves_temporal_validity_in_graph_and_qdrant(
    client, mock_state, auth_headers
):
    memory_id = "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeef"
    original_valid = "2026-01-01T00:00:00+00:00"
    original_invalid = "2026-06-01T00:00:00+00:00"
    mock_state.memory_graph.memories[memory_id] = {
        "id": memory_id,
        "content": "Temporal memory",
        "tags": ["test"],
        "importance": 0.5,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "metadata": "{}",
        "t_valid": original_valid,
        "t_invalid": original_invalid,
    }
    mock_state.qdrant.points[memory_id] = {
        "vector": [0.1] * 768,
        "payload": {
            "content": "Temporal memory",
            "tags": ["test"],
            "importance": 0.5,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "t_valid": original_valid,
            "t_invalid": original_invalid,
        },
    }

    response = client.patch(
        f"/memory/{memory_id}",
        json={"content": "Temporal memory updated"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    memory = mock_state.memory_graph.memories[memory_id]
    assert memory["t_valid"] == original_valid
    assert memory["t_invalid"] == original_invalid
    payload = mock_state.qdrant.points[memory_id]["payload"]
    assert payload["t_valid"] == original_valid
    assert payload["t_invalid"] == original_invalid


@pytest.mark.usefixtures("mock_state")
def test_update_memory_invalid_id(client, auth_headers):
    """Test updating with an invalid (non-UUID) memory ID returns 400."""
    response = client.patch(
        "/memory/not-a-uuid",
        json={"content": "New content"},
        headers=auth_headers,
    )
    assert response.status_code == 400


@pytest.mark.usefixtures("mock_state")
def test_update_memory_not_found(client, auth_headers):
    """Test updating non-existent memory."""
    response = client.patch(
        "/memory/00000000-0000-0000-0000-000000000099",
        json={"content": "New content"},
        headers=auth_headers,
    )
    assert response.status_code == 404


def test_update_memory_partial_fields(client, mock_state, auth_headers):
    """Test updating only some fields of a memory."""
    memory_id = "ffffffff-ffff-ffff-ffff-ffffffffffff"

    # Create initial memory
    mock_state.memory_graph.memories[memory_id] = {
        "id": memory_id,
        "content": "Original content",
        "tags": ["original"],
        "importance": 0.5,
        "metadata": json.dumps({"source": "test"}),
    }

    # Update only tags
    response = client.patch(
        f"/memory/{memory_id}", json={"tags": ["new", "tags"]}, headers=auth_headers
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"


# ==================== Test Memory Delete ====================


def test_delete_memory_success(client, mock_state, auth_headers):
    """Test successful memory deletion."""
    memory_id = "dd000000-0000-0000-0000-000000000001"

    # Create memory to delete
    mock_state.memory_graph.memories[memory_id] = {
        "id": memory_id,
        "content": "To be deleted",
        "timestamp": utc_now(),
    }

    # Also add to Qdrant
    mock_state.qdrant.points[memory_id] = {
        "vector": [0.1] * 768,
        "payload": {"content": "To be deleted"},
    }

    response = client.delete(f"/memory/{memory_id}", headers=auth_headers)

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["memory_id"] == memory_id
    assert memory_id not in mock_state.memory_graph.memories


@pytest.mark.usefixtures("mock_state")
def test_delete_memory_invalid_id(client, auth_headers):
    """Test deleting with an invalid (non-UUID) memory ID returns 400."""
    response = client.delete("/memory/not-a-uuid", headers=auth_headers)
    assert response.status_code == 400


@pytest.mark.usefixtures("mock_state")
def test_delete_memory_not_found(client, auth_headers):
    """Test deleting non-existent memory."""
    response = client.delete("/memory/00000000-0000-0000-0000-000000000098", headers=auth_headers)
    assert response.status_code == 404


# ==================== Test Memory By Tag ====================


def test_memory_by_tag_single(client, mock_state, auth_headers):
    """Test retrieving memories by a single tag."""
    # Add some memories with tags
    mock_state.memory_graph.memories["aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"] = {
        "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "content": "Python tutorial",
        "tags": ["python", "tutorial"],
        "importance": 0.8,
        "timestamp": utc_now(),
    }

    response = client.get("/memory/by-tag?tags=python", headers=auth_headers)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "memories" in data  # API returns 'memories' not 'results'
    assert data["limit"] == 20
    assert data["offset"] == 0
    assert data["has_more"] is False


def test_memory_by_tag_multiple(client, mock_state, auth_headers):
    """Test retrieving memories by multiple tags."""
    response = client.get("/memory/by-tag?tags=python&tags=ai&tags=ml", headers=auth_headers)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"


def test_memory_by_tag_no_tags(client, mock_state, auth_headers):
    """Test error when no tags provided."""
    response = client.get("/memory/by-tag", headers=auth_headers)
    assert response.status_code == 400


def test_memory_by_tag_pagination(client, mock_state, auth_headers):
    """Test offset pagination for /memory/by-tag."""
    tag = "paged-tag"
    timestamp = utc_now()
    for i in range(205):
        memory_id = f"00000000-0000-0000-0000-{i:012d}"
        memory = {
            "id": memory_id,
            "content": f"Paged memory {i}",
            "tags": [tag],
            "importance": 0.5,
            "timestamp": timestamp,
        }
        mock_state.memory_graph.memories[memory_id] = memory

    first = client.get("/memory/by-tag?tags=paged-tag&limit=200", headers=auth_headers)
    assert first.status_code == 200
    first_data = first.get_json()
    assert first_data["count"] == 200
    assert first_data["limit"] == 200
    assert first_data["offset"] == 0
    assert first_data["has_more"] is True

    second = client.get("/memory/by-tag?tags=paged-tag&limit=200&offset=200", headers=auth_headers)
    assert second.status_code == 200
    second_data = second.get_json()
    assert second_data["count"] == 5
    assert second_data["limit"] == 200
    assert second_data["offset"] == 200
    assert second_data["has_more"] is False

    first_ids = {memory["id"] for memory in first_data["memories"]}
    second_ids = {memory["id"] for memory in second_data["memories"]}
    assert first_ids.isdisjoint(second_ids)


def test_memory_by_tag_invalid_offset_normalized(client, mock_state, auth_headers):
    """Test invalid or negative offset normalizes to zero."""
    mock_state.memory_graph.memories["aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaab"] = {
        "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaab",
        "content": "Offset memory",
        "tags": ["offset-tag"],
        "importance": 0.8,
        "timestamp": utc_now(),
    }

    invalid = client.get("/memory/by-tag?tags=offset-tag&offset=nope", headers=auth_headers)
    assert invalid.status_code == 200
    invalid_data = invalid.get_json()
    assert invalid_data["offset"] == 0
    assert invalid_data["count"] == 1

    negative = client.get("/memory/by-tag?tags=offset-tag&offset=-10", headers=auth_headers)
    assert negative.status_code == 200
    negative_data = negative.get_json()
    assert negative_data["offset"] == 0
    assert negative_data["count"] == 1


def test_delete_memory_by_tag_bulk(client, mock_state, auth_headers):
    """Test bulk delete by tag removes graph and vector entries."""
    tag = "bulk-delete-tag"
    for i in range(3):
        memory_id = f"00000000-0000-0000-0000-0000000001{i:02d}"
        payload = {
            "content": f"Bulk delete memory {i}",
            "tags": [tag],
            "importance": 0.7,
            "timestamp": utc_now(),
        }
        mock_state.memory_graph.memories[memory_id] = {"id": memory_id, **payload}
        mock_state.qdrant.points[memory_id] = {"vector": [0.1] * 3, "payload": payload}

    response = client.delete("/memory/by-tag?tags=bulk-delete-tag", headers=auth_headers)
    assert response.status_code == 200
    data = response.get_json()
    assert data == {"status": "success", "tags": [tag], "deleted_count": 3}

    follow_up = client.get("/memory/by-tag?tags=bulk-delete-tag", headers=auth_headers)
    assert follow_up.status_code == 200
    follow_up_data = follow_up.get_json()
    assert follow_up_data["count"] == 0
    assert follow_up_data["has_more"] is False
    deleted_ids = [f"00000000-0000-0000-0000-0000000001{i:02d}" for i in range(3)]
    assert all(point_id not in mock_state.qdrant.points for point_id in deleted_ids)


def test_delete_memory_by_tag_no_matches(client, mock_state, auth_headers):
    """Test bulk delete by tag succeeds with zero matches."""
    response = client.delete("/memory/by-tag?tags=missing-tag", headers=auth_headers)
    assert response.status_code == 200
    data = response.get_json()
    assert data == {"status": "success", "tags": ["missing-tag"], "deleted_count": 0}


# ==================== Test Admin Reembed ====================


def test_admin_reembed_success(client, mock_state, admin_headers):
    """Test successful re-embedding of memories."""
    # Add memories to reembed
    mock_state.memory_graph.memories["aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"] = {
        "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "content": "First memory to reembed",
        "tags": ["test"],
        "importance": 0.7,
    }
    mock_state.memory_graph.memories["bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"] = {
        "id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        "content": "Second memory to reembed",
        "tags": ["test"],
        "importance": 0.8,
    }

    # Mock OpenAI client - return one embedding per input (batch processing)
    mock_state.openai_client = Mock()
    mock_state.openai_client.embeddings.create.return_value = Mock(
        data=[Mock(embedding=[0.2] * 768), Mock(embedding=[0.3] * 768)]
    )

    response = client.post(
        "/admin/reembed", json={"batch_size": 10, "limit": 2}, headers=admin_headers
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "complete"
    assert data["processed"] == 2
    assert data["total"] == 2


def test_admin_reembed_no_openai(client, mock_state, admin_headers):
    """Test reembed returns appropriate error when OpenAI not configured."""
    mock_state.openai_client = None

    # Add memories to reembed
    mock_state.memory_graph.memories["aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"] = {
        "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "content": "Memory to reembed with fallback",
        "tags": ["test"],
    }

    response = client.post(
        "/admin/reembed", json={"batch_size": 10, "limit": 1}, headers=admin_headers
    )

    # Should return 503 when OpenAI is not available (reembed requires OpenAI)
    assert response.status_code == 503
    data = response.get_json()
    assert "OpenAI" in data["message"]


def test_admin_reembed_no_qdrant(client, mock_state, admin_headers):
    """Test reembed fails when Qdrant not available."""
    mock_state.qdrant = None

    response = client.post("/admin/reembed", json={"batch_size": 10}, headers=admin_headers)

    assert response.status_code == 503
    data = response.get_json()
    assert "Qdrant is not available" in data["message"]


def test_admin_reembed_no_admin_token(client, mock_state, auth_headers):
    """Test reembed requires admin token."""
    response = client.post(
        "/admin/reembed", json={"batch_size": 10}, headers=auth_headers
    )  # Regular auth, not admin

    assert response.status_code in [401, 403]


def test_admin_reembed_force_flag(client, mock_state, admin_headers):
    """Test force reembedding with force flag."""
    # Add memory
    mock_state.memory_graph.memories["aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"] = {
        "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "content": "Memory with existing embedding",
        "tags": ["test"],
    }

    # Mock OpenAI - return one embedding per input (batch processing)
    mock_state.openai_client = Mock()
    mock_state.openai_client.embeddings.create.return_value = Mock(
        data=[Mock(embedding=[0.3] * 768)]  # One memory = one embedding
    )

    response = client.post(
        "/admin/reembed", json={"force": True, "limit": 1}, headers=admin_headers
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["processed"] == 1


# ==================== Test Consolidation ====================


def test_consolidate_trigger(client, mock_state, auth_headers):
    """Test triggering consolidation tasks."""
    response = client.post(
        "/consolidate",
        json={"mode": "full"},
        headers=auth_headers,  # API uses 'mode' not 'task'
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"  # API returns 'success' not 'triggered'
    # API doesn't return the task/mode field


def test_consolidate_invalid_task(client, mock_state, auth_headers):
    """Test consolidation with invalid task name - API accepts any task."""
    response = client.post("/consolidate", json={"task": "invalid_task"}, headers=auth_headers)
    assert response.status_code == 200  # API doesn't validate task names


def test_consolidate_status(client, mock_state, auth_headers):
    """Test consolidation status endpoint."""
    response = client.get("/consolidate/status", headers=auth_headers)
    assert response.status_code == 200
    data = response.get_json()
    assert "status" in data
    assert "next_runs" in data  # API returns 'next_runs' not 'tasks'


# ==================== Test Enrichment ====================


def test_enrichment_status(client, mock_state, auth_headers):
    """Test enrichment status endpoint."""
    response = client.get("/enrichment/status", headers=auth_headers)
    assert response.status_code == 200
    data = response.get_json()
    assert "status" in data
    assert "queue_size" in data  # API returns 'queue_size' not 'queue'
    assert "stats" in data


def test_enrichment_status_includes_classification_metrics(client, monkeypatch, auth_headers):
    """/enrichment/status exposes classification fallback metrics."""
    monkeypatch.setattr(app, "API_TOKEN", "test-token")

    # The enrichment blueprint captured the module-level service state at
    # import time, so drive the classifier against that same stats object.
    stats = app.state.classification_stats
    baseline = stats.to_dict()

    def _build_classifier(create_fn):
        completions = SimpleNamespace(create=create_fn)
        client_stub = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        return app.MemoryClassifier(
            normalize_memory_type=lambda raw: (raw, False),
            ensure_openai_client=lambda: None,
            get_openai_client=lambda: client_stub,
            classification_model="gpt-4o-mini",
            logger=app.logger,
            stats=stats,
        )

    def _raise(*args, **kwargs):
        raise RuntimeError("429 insufficient_quota")

    def _succeed(*args, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"type": "Insight", "confidence": 0.9}')
                )
            ]
        )

    # Content matches no regex pattern, forcing the LLM path.
    assert _build_classifier(_raise).classify("qwxz flibber jabberwock") == ("Memory", 0.3)
    assert _build_classifier(_succeed).classify("qwxz flibber jabberwock") == ("Insight", 0.9)

    response = client.get("/enrichment/status", headers=auth_headers)
    assert response.status_code == 200
    data = response.get_json()
    assert "classification" in data

    block = data["classification"]
    assert block["llm_attempts"] == baseline["llm_attempts"] + 2
    assert block["llm_successes"] == baseline["llm_successes"] + 1
    assert block["fallbacks"] == baseline["fallbacks"] + 1
    assert block["pattern_classifications"] == baseline["pattern_classifications"]
    assert "429" in (block["last_error"] or "")
    assert block["last_error_at"]


def test_enrichment_reprocess(client, mock_state, admin_headers):
    """Test reprocessing memories for enrichment."""
    response = client.post(
        "/enrichment/reprocess",
        json={
            "ids": [
                "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                "cccccccc-cccc-cccc-cccc-cccccccccccc",
            ]
        },
        headers=admin_headers,
    )

    assert response.status_code == 202
    data = response.get_json()
    assert data["status"] == "queued"
    assert data["count"] == 3


def test_enrichment_reprocess_no_ids(client, mock_state, admin_headers):
    """Test reprocess with no IDs provided."""
    response = client.post("/enrichment/reprocess", json={}, headers=admin_headers)
    assert response.status_code == 400


# ==================== Test Association Creation ====================


def test_create_association_all_relationship_types(client, mock_state, auth_headers):
    """Test creating associations with all public authorable relationship types."""
    # Create two memories
    mock_state.memory_graph.memories["aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"] = {
        "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "content": "Memory 1",
    }
    mock_state.memory_graph.memories["bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"] = {
        "id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        "content": "Memory 2",
    }

    relationship_types = sorted(config.AUTHORABLE_RELATIONS)

    for rel_type in relationship_types:
        response = client.post(
            "/associate",
            json={
                "memory1_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "memory2_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                "type": rel_type,
                "strength": 0.8,
            },
            headers=auth_headers,
        )
        assert response.status_code == 201, f"Failed for relationship type: {rel_type}"
        data = response.get_json()
        assert data["status"] == "success"


def test_create_association_rejects_system_generated_relationship_types(
    client, mock_state, auth_headers
):
    """Public association writes should reject system/internal relation labels."""
    mock_state.memory_graph.memories["aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"] = {
        "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "content": "Memory 1",
    }
    mock_state.memory_graph.memories["bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"] = {
        "id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        "content": "Memory 2",
    }

    for rel_type in ("SIMILAR_TO", "PRECEDED_BY", "DISCOVERED", "EXPLAINS"):
        response = client.post(
            "/associate",
            json={
                "memory1_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "memory2_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                "type": rel_type,
                "strength": 0.8,
            },
            headers=auth_headers,
        )
        assert response.status_code == 400, f"Expected rejection for {rel_type}"
        data = response.get_json()
        assert "Relation type must be one of" in data["message"]


def test_create_association_with_properties(client, mock_state, auth_headers):
    """Test creating association with additional properties."""
    mem1_id = "11111111-1111-1111-1111-111111111111"
    mem2_id = "22222222-2222-2222-2222-222222222222"
    mock_state.memory_graph.memories[mem1_id] = {
        "id": mem1_id,
        "content": "Preferred method",
    }
    mock_state.memory_graph.memories[mem2_id] = {
        "id": mem2_id,
        "content": "Alternative method",
    }

    response = client.post(
        "/associate",
        json={
            "memory1_id": mem1_id,
            "memory2_id": mem2_id,
            "type": "PREFERS_OVER",
            "strength": 0.9,
            "reason": "Better performance",
            "context": "Production environment",
        },
        headers=auth_headers,
    )

    assert response.status_code == 201
    data = response.get_json()
    assert data["status"] == "success"
    assert data["reason"] == "Better performance"
    assert data["context"] == "Production environment"


def test_create_association_batch_all_success(client, mock_state, auth_headers):
    """Batch association should create valid relationships in one request."""
    mem_ids = [
        "11111111-1111-1111-1111-111111111111",
        "22222222-2222-2222-2222-222222222222",
        "33333333-3333-3333-3333-333333333333",
    ]
    for index, memory_id in enumerate(mem_ids, start=1):
        mock_state.memory_graph.memories[memory_id] = {
            "id": memory_id,
            "content": f"Memory {index}",
        }

    response = client.post(
        "/associate",
        json={
            "associations": [
                {
                    "memory1_id": mem_ids[0],
                    "memory2_id": mem_ids[1],
                    "type": "RELATES_TO",
                    "strength": 0.8,
                },
                {
                    "memory1_id": mem_ids[1],
                    "memory2_id": mem_ids[2],
                    "type": "PREFERS_OVER",
                    "strength": 0.9,
                    "reason": "Clearer",
                    "context": "Test",
                },
            ]
        },
        headers=auth_headers,
    )

    assert response.status_code == 201
    data = response.get_json()
    assert data["status"] == "success"
    assert data["created_count"] == 2
    assert data["failed_count"] == 0
    assert data["summary"] == "2/2 associations created successfully"
    assert [item["index"] for item in data["succeeded"]] == [0, 1]
    assert data["failed"] == []
    assert len(mock_state.memory_graph.relationships) == 2
    prefers = [
        rel for rel in mock_state.memory_graph.relationships if rel["type"] == "PREFERS_OVER"
    ]
    assert prefers[0]["reason"] == "Clearer"
    assert prefers[0]["context"] == "Test"


def test_create_association_batch_partial_success(client, mock_state, auth_headers):
    """Batch association should report per-item failures without rolling back successes."""
    mem_a = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    mem_b = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    missing = "cccccccc-cccc-cccc-cccc-cccccccccccc"
    for memory_id in (mem_a, mem_b):
        mock_state.memory_graph.memories[memory_id] = {
            "id": memory_id,
            "content": f"Memory {memory_id}",
        }

    response = client.post(
        "/associate",
        json={
            "associations": [
                {
                    "memory1_id": mem_a,
                    "memory2_id": mem_b,
                    "type": "RELATES_TO",
                    "strength": 0.8,
                },
                {
                    "memory1_id": "not-a-uuid",
                    "memory2_id": mem_b,
                    "type": "RELATES_TO",
                    "strength": 0.8,
                },
                {
                    "memory1_id": mem_a,
                    "memory2_id": mem_a,
                    "type": "RELATES_TO",
                    "strength": 0.8,
                },
                {
                    "memory1_id": mem_a,
                    "memory2_id": mem_b,
                    "type": "SIMILAR_TO",
                    "strength": 0.8,
                },
                {
                    "memory1_id": mem_a,
                    "memory2_id": missing,
                    "type": "RELATES_TO",
                    "strength": 0.8,
                },
            ]
        },
        headers=auth_headers,
    )

    assert response.status_code == 207
    data = response.get_json()
    assert data["status"] == "partial_success"
    assert data["created_count"] == 1
    assert data["failed_count"] == 4
    assert data["summary"] == "1/5 associations created successfully"
    assert [item["index"] for item in data["succeeded"]] == [0]
    failures = {item["index"]: item["reason"] for item in data["failed"]}
    assert "'memory1_id' must be a valid UUID" in failures[1]
    assert "Cannot associate a memory with itself" in failures[2]
    assert "Relation type must be one of" in failures[3]
    assert "One or both memories do not exist" in failures[4]
    assert len(mock_state.memory_graph.relationships) == 1


def test_create_association_batch_all_item_failures(client, mock_state, auth_headers):
    """A valid batch request with no valid items should still return per-item failures."""
    response = client.post(
        "/associate",
        json={
            "associations": [
                {
                    "memory1_id": "not-a-uuid",
                    "memory2_id": "also-not-a-uuid",
                    "type": "RELATES_TO",
                    "strength": 0.8,
                }
            ]
        },
        headers=auth_headers,
    )

    assert response.status_code == 207
    data = response.get_json()
    assert data["status"] == "partial_success"
    assert data["created_count"] == 0
    assert data["failed_count"] == 1
    assert data["summary"] == "0/1 associations created successfully"


def test_create_association_batch_rejects_empty_array(client, mock_state, auth_headers):
    """Malformed batch envelopes should remain hard 400 errors."""
    response = client.post("/associate", json={"associations": []}, headers=auth_headers)

    assert response.status_code == 400
    data = response.get_json()
    assert "'associations' must be a non-empty array" in data["message"]


def test_create_association_single_ignores_non_list_associations_key(
    client, mock_state, auth_headers
):
    """Legacy single-association payloads should tolerate an unrelated associations key."""
    mem1_id = "44444444-4444-4444-4444-444444444444"
    mem2_id = "55555555-5555-5555-5555-555555555555"
    mock_state.memory_graph.memories[mem1_id] = {
        "id": mem1_id,
        "content": "Memory 1",
    }
    mock_state.memory_graph.memories[mem2_id] = {
        "id": mem2_id,
        "content": "Memory 2",
    }

    response = client.post(
        "/associate",
        json={
            "memory1_id": mem1_id,
            "memory2_id": mem2_id,
            "type": "RELATES_TO",
            "strength": 0.8,
            "associations": None,
        },
        headers=auth_headers,
    )

    assert response.status_code == 201
    data = response.get_json()
    assert data["status"] == "success"
    assert len(mock_state.memory_graph.relationships) == 1


# ==================== Test Startup Recall ====================


def test_startup_recall(client, mock_state, auth_headers):
    """Test startup recall endpoint."""
    # Add some recent high-importance memories
    now = utc_now()
    mock_state.memory_graph.memories["20000000-0000-0000-0000-000000000001"] = {
        "id": "20000000-0000-0000-0000-000000000001",
        "content": "Critical information",
        "importance": 0.95,
        "timestamp": now,
        "tags": ["critical"],
    }

    response = client.get("/startup-recall", headers=auth_headers)
    assert response.status_code == 200
    data = response.get_json()
    assert "status" in data
    # API returns different field structure for startup recall
    assert "has_critical" in data or "critical_lessons" in data


# ==================== Test Analyze Endpoint ====================


def test_analyze_endpoint(client, mock_state, auth_headers):
    """Test analytics endpoint."""
    # Add varied memories for analytics
    mock_state.memory_graph.memories["30000000-0000-0000-0000-000000000001"] = {
        "id": "30000000-0000-0000-0000-000000000001",
        "content": "Architectural decision",
        "type": "Decision",
        "confidence": 0.9,
        "importance": 0.85,
    }
    mock_state.memory_graph.memories["30000000-0000-0000-0000-000000000002"] = {
        "id": "30000000-0000-0000-0000-000000000002",
        "content": "Common pattern",
        "type": "Pattern",
        "confidence": 0.7,
        "importance": 0.6,
    }

    response = client.get("/analyze", headers=auth_headers)
    assert response.status_code == 200
    data = response.get_json()
    assert "status" in data
    # API returns analytics in nested structure
    assert "analytics" in data
    if "analytics" in data:
        assert "memory_types" in data["analytics"]


# ==================== Test Error Handling ====================


def test_falkordb_unavailable(client, mock_state, auth_headers):
    """Test graceful handling when FalkorDB is unavailable."""
    mock_state.memory_graph = None

    response = client.post("/memory", json={"content": "Test memory"}, headers=auth_headers)
    assert response.status_code == 503
    data = response.get_json()
    assert "FalkorDB is unavailable" in data["message"]


def test_invalid_json_payload(client, mock_state, auth_headers):
    """Test handling of invalid JSON payload."""
    response = client.post(
        "/memory",
        data="This is not JSON",
        headers=auth_headers,
        content_type="application/json",
    )
    assert response.status_code == 400


def test_missing_required_fields(client, mock_state, auth_headers):
    """Test validation of required fields."""
    # Missing content field
    response = client.post("/memory", json={"tags": ["test"]}, headers=auth_headers)
    assert response.status_code == 400
    data = response.get_json()
    assert "'content' is required" in data["message"]


def test_authorization_required(client, mock_state):
    """Test that authorization is required for protected endpoints."""
    # No auth headers
    response = client.post("/memory", json={"content": "Test"})
    # Should return 401 or pass through (depending on API_TOKEN config)
    # Since we're not setting API_TOKEN in tests, it should pass
    assert response.status_code in [200, 201, 400, 401, 403]


def test_invalid_timestamp_format(client, mock_state, auth_headers):
    """Test handling of invalid timestamp formats."""
    response = client.post(
        "/memory",
        json={"content": "Test memory", "timestamp": "not-a-valid-timestamp"},
        headers=auth_headers,
    )
    assert response.status_code == 400
    data = response.get_json()
    assert "timestamp" in data["message"].lower()


def test_embedding_dimension_validation(client, mock_state, auth_headers):
    """Test validation of embedding dimensions."""
    response = client.post(
        "/memory",
        json={"content": "Test memory", "embedding": [0.1, 0.2]},  # Wrong dimension
        headers=auth_headers,
    )
    assert response.status_code == 400
    data = response.get_json()
    assert str(config.VECTOR_SIZE) in data["message"]


def test_expand_related_memories_filters_strength_and_importance():
    seed_id = "11111111-0000-0000-0000-000000000001"
    keep_id = "11111111-0000-0000-0000-000000000002"
    weak_id = "11111111-0000-0000-0000-000000000003"
    low_imp_id = "11111111-0000-0000-0000-000000000004"
    seed_results = [{"id": seed_id, "final_score": 0.8, "memory": {"id": seed_id}}]

    class _Node(SimpleNamespace):
        pass

    class Graph:
        def query(self, _query: str, _params: dict) -> SimpleNamespace:
            related = [
                (
                    "RELATES_TO",
                    0.2,
                    _Node(properties={"id": keep_id, "importance": 0.9}),
                ),
                (
                    "RELATES_TO",
                    0.05,
                    _Node(properties={"id": weak_id, "importance": 0.9}),
                ),
                (
                    "RELATES_TO",
                    0.8,
                    _Node(properties={"id": low_imp_id, "importance": 0.1}),
                ),
            ]
            return SimpleNamespace(result_set=related)

    graph = Graph()
    seen_ids: set[str] = set()

    def _passes(*args, **kwargs):
        return True

    def _score(*args, **kwargs):
        return 0.5, {}

    results = _expand_related_memories(
        graph=graph,
        seed_results=seed_results,
        seen_ids=seen_ids,
        result_passes_filters=_passes,
        compute_metadata_score=_score,
        query_text="",
        query_tokens=[],
        context_profile=None,
        start_time=None,
        end_time=None,
        tag_filters=None,
        tag_mode="any",
        tag_match="prefix",
        per_seed_limit=5,
        expansion_limit=10,
        allowed_relations={"RELATES_TO"},
        logger=Mock(),
        expand_min_strength=0.1,
        expand_min_importance=0.2,
    )

    ids = {res["id"] for res in results}
    assert ids == {keep_id}


def test_expand_related_memories_normalizes_legacy_discovered_relations():
    seed_id = "22222222-0000-0000-0000-000000000001"
    related_id = "22222222-0000-0000-0000-000000000002"
    seed_results = [{"id": seed_id, "final_score": 0.8, "memory": {"id": seed_id}}]

    class _Node(SimpleNamespace):
        pass

    class Graph:
        def query(self, _query: str, _params: dict) -> SimpleNamespace:
            related = [
                (
                    "EXPLAINS",
                    0.7,
                    None,
                    _Node(properties={"id": related_id, "importance": 0.9}),
                ),
            ]
            return SimpleNamespace(result_set=related)

    results = _expand_related_memories(
        graph=Graph(),
        seed_results=seed_results,
        seen_ids=set(),
        result_passes_filters=lambda *args, **kwargs: True,
        compute_metadata_score=lambda *args, **kwargs: (0.5, {}),
        query_text="",
        query_tokens=[],
        context_profile=None,
        start_time=None,
        end_time=None,
        tag_filters=None,
        tag_mode="any",
        tag_match="prefix",
        per_seed_limit=5,
        expansion_limit=10,
        allowed_relations={"DISCOVERED"},
        logger=Mock(),
    )

    assert len(results) == 1
    relation_info = results[0]["relations"][0]
    assert relation_info["type"] == "DISCOVERED"
    assert relation_info["kind"] == "explains"


def test_expand_related_memories_bypasses_tag_filter_by_default():
    seed_id = "33333333-0000-0000-0000-000000000001"
    related_id = "33333333-0000-0000-0000-000000000002"
    seed_results = [{"id": seed_id, "final_score": 0.8, "memory": {"id": seed_id}}]
    seen_filter_args = []

    class _Node(SimpleNamespace):
        pass

    class Graph:
        def query(self, _query: str, _params: dict) -> SimpleNamespace:
            return SimpleNamespace(
                result_set=[
                    (
                        "EXEMPLIFIES",
                        0.8,
                        _Node(properties={"id": related_id, "importance": 0.9}),
                    )
                ]
            )

    def _passes(*args):
        seen_filter_args.append(args)
        return args[3] is None

    results = _expand_related_memories(
        graph=Graph(),
        seed_results=seed_results,
        seen_ids=set(),
        result_passes_filters=_passes,
        compute_metadata_score=lambda *args, **kwargs: (0.5, {}),
        query_text="rate limiter redis scan",
        query_tokens=["rate", "limiter"],
        context_profile=None,
        start_time=None,
        end_time=None,
        tag_filters=["tensor-pipeline"],
        tag_mode="any",
        tag_match="exact",
        per_seed_limit=5,
        expansion_limit=10,
        allowed_relations={"EXEMPLIFIES"},
        logger=Mock(),
        expand_respect_tags=False,
    )

    assert [res["id"] for res in results] == [related_id]
    assert seen_filter_args[0][3] is None


def test_expand_related_memories_respects_tags_when_opted_in():
    seed_id = "44444444-0000-0000-0000-000000000001"
    related_id = "44444444-0000-0000-0000-000000000002"
    seed_results = [{"id": seed_id, "final_score": 0.8, "memory": {"id": seed_id}}]

    class _Node(SimpleNamespace):
        pass

    class Graph:
        def query(self, _query: str, _params: dict) -> SimpleNamespace:
            return SimpleNamespace(
                result_set=[
                    (
                        "DERIVED_FROM",
                        0.8,
                        _Node(properties={"id": related_id, "importance": 0.9}),
                    )
                ]
            )

    def _passes(*args):
        return args[3] is None

    results = _expand_related_memories(
        graph=Graph(),
        seed_results=seed_results,
        seen_ids=set(),
        result_passes_filters=_passes,
        compute_metadata_score=lambda *args, **kwargs: (0.5, {}),
        query_text="auth generic pattern",
        query_tokens=["auth", "pattern"],
        context_profile=None,
        start_time=None,
        end_time=None,
        tag_filters=["tensor-pipeline"],
        tag_mode="any",
        tag_match="exact",
        per_seed_limit=5,
        expansion_limit=10,
        allowed_relations={"DERIVED_FROM"},
        logger=Mock(),
        expand_respect_tags=True,
    )

    assert results == []


def test_expand_related_memories_still_honors_exclude_tags():
    seed_id = "55555555-0000-0000-0000-000000000001"
    related_id = "55555555-0000-0000-0000-000000000002"
    seed_results = [{"id": seed_id, "final_score": 0.8, "memory": {"id": seed_id}}]
    seen_filter_args = []

    class _Node(SimpleNamespace):
        pass

    class Graph:
        def query(self, _query: str, _params: dict) -> SimpleNamespace:
            return SimpleNamespace(
                result_set=[
                    (
                        "REINFORCES",
                        0.8,
                        _Node(properties={"id": related_id, "importance": 0.9}),
                    )
                ]
            )

    def _passes(*args):
        seen_filter_args.append(args)
        return args[6] == ["archived"] and args[3] is None

    results = _expand_related_memories(
        graph=Graph(),
        seed_results=seed_results,
        seen_ids=set(),
        result_passes_filters=_passes,
        compute_metadata_score=lambda *args, **kwargs: (0.5, {}),
        query_text="cross project logging",
        query_tokens=["logging"],
        context_profile=None,
        start_time=None,
        end_time=None,
        tag_filters=["tensor-pipeline"],
        tag_mode="any",
        tag_match="exact",
        per_seed_limit=5,
        expansion_limit=10,
        allowed_relations={"REINFORCES"},
        logger=Mock(),
        exclude_tags=["archived"],
        expand_respect_tags=False,
    )

    assert [res["id"] for res in results] == [related_id]
    assert seen_filter_args[0][6] == ["archived"]


def test_expand_related_memories_still_honors_time_window():
    seed_id = "66666666-0000-0000-0000-000000000001"
    related_id = "66666666-0000-0000-0000-000000000002"
    seed_results = [{"id": seed_id, "final_score": 0.8, "memory": {"id": seed_id}}]
    seen_filter_args = []

    class _Node(SimpleNamespace):
        pass

    class Graph:
        def query(self, _query: str, _params: dict) -> SimpleNamespace:
            return SimpleNamespace(
                result_set=[
                    (
                        "EXEMPLIFIES",
                        0.8,
                        _Node(properties={"id": related_id, "importance": 0.9}),
                    )
                ]
            )

    def _passes(*args):
        seen_filter_args.append(args)
        return (
            args[1] == "2026-01-01T00:00:00Z"
            and args[2] == "2026-12-31T23:59:59Z"
            and args[3] is None
        )

    results = _expand_related_memories(
        graph=Graph(),
        seed_results=seed_results,
        seen_ids=set(),
        result_passes_filters=_passes,
        compute_metadata_score=lambda *args, **kwargs: (0.5, {}),
        query_text="rate limiter redis scan",
        query_tokens=["redis"],
        context_profile=None,
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-12-31T23:59:59Z",
        tag_filters=["tensor-pipeline"],
        tag_mode="any",
        tag_match="exact",
        per_seed_limit=5,
        expansion_limit=10,
        allowed_relations={"EXEMPLIFIES"},
        logger=Mock(),
        expand_respect_tags=False,
    )

    assert [res["id"] for res in results] == [related_id]
    assert seen_filter_args[0][1] == "2026-01-01T00:00:00Z"
    assert seen_filter_args[0][2] == "2026-12-31T23:59:59Z"


def test_relation_taxonomy_sets_are_consistent():
    assert config.AUTHORABLE_RELATIONS == {
        "RELATES_TO",
        "LEADS_TO",
        "OCCURRED_BEFORE",
        "PREFERS_OVER",
        "EXEMPLIFIES",
        "CONTRADICTS",
        "REINFORCES",
        "INVALIDATED_BY",
        "EVOLVED_INTO",
        "DERIVED_FROM",
        "PART_OF",
    }
    assert config.DEFAULT_EXPAND_RELATIONS == config.AUTHORABLE_RELATIONS
    assert config.PUBLIC_RELATIONS == config.AUTHORABLE_RELATIONS | {"SIMILAR_TO", "PRECEDED_BY"}
    assert config.FILTERABLE_RELATIONS == config.PUBLIC_RELATIONS | {"DISCOVERED"}
    assert set(config.RELATION_COLORS) == config.PUBLIC_RELATIONS


def test_related_memories_defaults_to_authorable_relations(client, mock_state, auth_headers):
    class Graph:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []

        def query(self, query: str, params: dict[str, Any] | None = None) -> SimpleNamespace:
            self.calls.append((query, params or {}))
            return SimpleNamespace(result_set=[])

    graph = Graph()
    mock_state.memory_graph = graph

    response = client.get(
        "/memories/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa/related",
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert graph.calls
    query, _params = graph.calls[0]
    assert "RELATES_TO" in query
    assert "PART_OF" in query
    assert "SIMILAR_TO" not in query
    assert "PRECEDED_BY" not in query
    assert "DISCOVERED" not in query


def test_related_memories_supports_explicit_system_relation_opt_ins(
    client, mock_state, auth_headers
):
    class Graph:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []

        def query(self, query: str, params: dict[str, Any] | None = None) -> SimpleNamespace:
            self.calls.append((query, params or {}))
            return SimpleNamespace(result_set=[])

    graph = Graph()
    mock_state.memory_graph = graph

    response = client.get(
        "/memories/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa/related"
        "?relationship_types=SIMILAR_TO,DISCOVERED",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["relationship_types"] == ["SIMILAR_TO", "DISCOVERED"]
    query, _params = graph.calls[0]
    assert "SIMILAR_TO" in query
    assert "DISCOVERED" in query
    assert "EXPLAINS" in query
    assert "SHARES_THEME" in query
    assert "PARALLEL_CONTEXT" in query


def test_related_memories_fallback_inlines_sanitized_depth(client, mock_state, auth_headers):
    class Graph:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []

        def query(self, query: str, params: dict[str, Any] | None = None) -> SimpleNamespace:
            self.calls.append((query, params or {}))
            if len(self.calls) == 1:
                raise RuntimeError("apoc unavailable")
            if "$max_depth" in query:
                raise RuntimeError("FalkorDB rejects parameterized variable-length ranges")
            return SimpleNamespace(result_set=[])

    graph = Graph()
    mock_state.memory_graph = graph

    response = client.get(
        "/memories/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa/related?max_depth=2",
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert len(graph.calls) == 2
    fallback_query, fallback_params = graph.calls[1]
    assert "*1..2" in fallback_query
    assert "$max_depth" not in fallback_query
    assert "max_depth" not in fallback_params
    assert fallback_params["id"] == "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    assert fallback_params["limit"] == 5


# ==================== Test Rate Limiting (if implemented) ====================


@pytest.mark.skip(reason="Rate limiting not yet implemented")
def test_rate_limiting(client, mock_state, auth_headers):
    """Test rate limiting functionality."""
    # Make many requests quickly
    for i in range(100):
        response = client.get("/health")

    # Should eventually get rate limited
    response = client.get("/health")
    assert response.status_code == 429  # Too Many Requests


@pytest.mark.usefixtures("mock_state")
def test_recall_with_exclude_tags_single(client, auth_headers):
    """Test memory recall with exclude_tags parameter - single tag exclusion."""
    # Create memories with different tags
    mem1_data = {
        "content": "Python function in conversation 5",
        "tags": ["python", "conversation_5", "user"],
        "importance": 0.8,
    }
    mem2_data = {
        "content": "JavaScript function in conversation 3",
        "tags": ["javascript", "conversation_3", "user"],
        "importance": 0.7,
    }
    mem3_data = {
        "content": "Python class in conversation 5",
        "tags": ["python", "conversation_5", "assistant"],
        "importance": 0.9,
    }

    for payload in (mem1_data, mem2_data, mem3_data):
        store_response = client.post("/memory", json=payload, headers=auth_headers)
        assert store_response.status_code == 201
        assert store_response.get_json()["status"] == "success"

    # Recall without exclusion - should get user-tagged memories
    response = client.get("/recall?tags=user&limit=10", headers=auth_headers)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"

    results = data.get("results", [])
    assert len(results) == 2

    # Recall with exclude_tags=conversation_5 - should only get conversation_3
    response = client.get(
        "/recall?tags=user&exclude_tags=conversation_5&limit=10",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"

    # Check that conversation_5 memories are excluded
    results = data.get("results", [])
    assert len(results) == 1
    for mem in results:
        mem_tags = mem.get("memory", {}).get("tags", [])
        assert "conversation_5" not in mem_tags
        assert "conversation_3" in mem_tags


@pytest.mark.usefixtures("mock_state")
def test_recall_with_exclude_tags_multiple(client, auth_headers):
    """Test memory recall with exclude_tags parameter - multiple tag exclusions."""
    # Create memories with different conversation tags
    mem1_data = {
        "content": "Memory from conversation 1",
        "tags": ["user_1", "conversation_1"],
        "importance": 0.8,
    }
    mem2_data = {
        "content": "Memory from conversation 2",
        "tags": ["user_1", "conversation_2"],
        "importance": 0.7,
    }
    mem3_data = {
        "content": "Memory from conversation 3",
        "tags": ["user_1", "conversation_3"],
        "importance": 0.9,
    }
    mem4_data = {
        "content": "Memory from conversation 4",
        "tags": ["user_1", "conversation_4"],
        "importance": 0.6,
    }

    for payload in (mem1_data, mem2_data, mem3_data, mem4_data):
        store_response = client.post("/memory", json=payload, headers=auth_headers)
        assert store_response.status_code == 201
        assert store_response.get_json()["status"] == "success"

    # Exclude multiple conversations (1 and 2)
    response = client.get(
        "/recall?tags=user_1&exclude_tags=conversation_1,conversation_2&limit=10",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"

    # Check that excluded conversation memories are not present
    results = data.get("results", [])
    assert len(results) == 2
    conversation_tags = set()
    for mem in results:
        mem_tags = mem.get("memory", {}).get("tags", [])
        assert "conversation_1" not in mem_tags
        assert "conversation_2" not in mem_tags
        conversation_tags.update({tag for tag in mem_tags if tag.startswith("conversation_")})
    assert conversation_tags == {"conversation_3", "conversation_4"}


@pytest.mark.usefixtures("mock_state")
def test_recall_with_exclude_tags_prefix_matching(client, auth_headers):
    """Test that exclude_tags works with prefix matching."""
    # Create memories with prefixed tags
    mem1_data = {
        "content": "Temporary note 1",
        "tags": ["user_1", "temp:draft"],
        "importance": 0.5,
    }
    mem2_data = {
        "content": "Temporary note 2",
        "tags": ["user_1", "temp:scratch"],
        "importance": 0.4,
    }
    mem3_data = {
        "content": "Permanent note",
        "tags": ["user_1", "permanent"],
        "importance": 0.9,
    }

    for payload in (mem1_data, mem2_data, mem3_data):
        store_response = client.post("/memory", json=payload, headers=auth_headers)
        assert store_response.status_code == 201
        assert store_response.get_json()["status"] == "success"

    # Exclude all temp: prefixed tags
    response = client.get(
        "/recall?tags=user_1&exclude_tags=temp&limit=10",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"

    # Should only get permanent note
    results = data.get("results", [])
    assert len(results) == 1
    for mem in results:
        mem_tags = mem.get("memory", {}).get("tags", [])
        # No tag should start with "temp"
        assert not any(tag.startswith("temp") for tag in mem_tags)
        assert "permanent" in mem_tags


@pytest.mark.usefixtures("mock_state")
def test_recall_with_tags_and_exclude_tags_combined(client, auth_headers):
    """Test combining tags filter with exclude_tags."""
    # Create memories
    mem1_data = {
        "content": "User message in current conversation",
        "tags": ["user_1", "conversation_5", "user"],
        "importance": 0.8,
    }
    mem2_data = {
        "content": "Assistant message in current conversation",
        "tags": ["user_1", "conversation_5", "assistant"],
        "importance": 0.7,
    }
    mem3_data = {
        "content": "User message in past conversation",
        "tags": ["user_1", "conversation_3", "user"],
        "importance": 0.9,
    }

    for payload in (mem1_data, mem2_data, mem3_data):
        store_response = client.post("/memory", json=payload, headers=auth_headers)
        assert store_response.status_code == 201
        assert store_response.get_json()["status"] == "success"

    # Include user_1 but exclude conversation_5
    response = client.get(
        "/recall?tags=user_1&exclude_tags=conversation_5&limit=10",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"

    # Should only get memory from conversation_3
    results = data.get("results", [])
    assert len(results) == 1
    for mem in results:
        mem_tags = mem.get("memory", {}).get("tags", [])
        assert "user_1" in mem_tags
        assert "conversation_5" not in mem_tags
        assert "conversation_3" in mem_tags


@pytest.mark.usefixtures("mock_state")
def test_recall_exclude_tags_with_no_results(client, auth_headers):
    """Test that exclude_tags returns empty when all memories are excluded."""
    # Create memory
    mem_data = {
        "content": "Only memory",
        "tags": ["user_1", "conversation_5"],
        "importance": 0.8,
    }

    store_response = client.post("/memory", json=mem_data, headers=auth_headers)
    assert store_response.status_code == 201
    assert store_response.get_json()["status"] == "success"

    # Exclude the only conversation
    response = client.get(
        "/recall?tags=user_1&exclude_tags=conversation_5&limit=10",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert len(data.get("results", [])) == 0


# ==================== Score-sort timestamp tiebreak (issue #159 part 1) ====================


def _sortable_result(final_score, timestamp, importance=0.5, source="graph"):
    return {
        "final_score": final_score,
        "original_score": final_score,
        "source": source,
        "memory": {"importance": importance, "timestamp": timestamp},
    }


def test_score_sort_key_breaks_exact_ties_newest_first():
    """Exact score ties order newest-first deterministically."""
    older = _sortable_result(0.5, "2025-01-01T00:00:00+00:00")
    newer = _sortable_result(0.5, "2025-06-01T00:00:00+00:00")

    ordered = sorted([older, newer], key=recall_module._score_sort_key)

    assert ordered[0] is newer
    assert ordered[1] is older


def test_score_sort_key_score_still_dominates_timestamp():
    """A higher-scored older result still beats a lower-scored newer one."""
    older_high = _sortable_result(0.9, "2024-01-01T00:00:00+00:00")
    newer_low = _sortable_result(0.5, "2025-06-01T00:00:00+00:00")

    ordered = sorted([newer_low, older_high], key=recall_module._score_sort_key)

    assert ordered[0] is older_high


def test_score_sort_key_importance_breaks_ties_before_timestamp():
    """Existing keys (importance) are preserved ahead of the timestamp tiebreak."""
    older_important = _sortable_result(0.5, "2024-01-01T00:00:00+00:00", importance=0.9)
    newer_plain = _sortable_result(0.5, "2025-06-01T00:00:00+00:00", importance=0.1)

    ordered = sorted([newer_plain, older_important], key=recall_module._score_sort_key)

    assert ordered[0] is older_important


def test_score_sort_key_unparseable_timestamp_falls_back_to_epoch():
    """Unparseable timestamps sort as epoch (oldest) among exact ties, without raising."""
    garbage = _sortable_result(0.5, "not-a-timestamp")
    missing = _sortable_result(0.5, None)
    dated = _sortable_result(0.5, "2025-06-01T00:00:00+00:00")

    ordered = sorted([garbage, missing, dated], key=recall_module._score_sort_key)

    assert ordered[0] is dated


# ==================== Temporal-intent detection (issue #158/#159 part 2) ====================


@pytest.mark.parametrize(
    "query",
    [
        "What is my current favorite editor?",
        "what's the latest deployment status",
        "Most Recent decision on auth",
        "which framework do I prefer now",
        "what changed today",
        "what was updated in the schema",
        "what happened last time we deployed",
        "the newest API version",
        "Currently using which database?",
    ],
)
def test_query_has_temporal_intent_positive(query):
    assert query_has_temporal_intent(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "favorite editor preferences",
        "nowhere plans for the trip",
        "currency exchange rates",
        "the lasting impact of the refactor",
        "",
        None,
    ],
)
def test_query_has_temporal_intent_negative(query):
    assert query_has_temporal_intent(query) is False


# ==================== recency_bias re-rank (issues #158/#159 part 2) ====================


def test_recall_recency_bias_off_by_default_keeps_importance_order(
    client, mock_state, auth_headers
):
    """Default (env off, no param): older high-importance fact stays first; no echo."""
    mock_state.memory_graph.memories.clear()
    now = datetime.now(timezone.utc)
    old_id = "dd000000-0000-0000-0000-000000000001"
    new_id = "dd000000-0000-0000-0000-000000000002"

    _store_memory(
        mock_state,
        old_id,
        "Favorite color is blue",
        ["fact"],
        0.65,
        timestamp=(now - timedelta(days=10)).isoformat(),
    )
    _store_memory(mock_state, new_id, "Favorite color is green", ["fact"], 0.5)

    response = client.get("/recall?tags=fact&limit=10", headers=auth_headers)

    assert response.status_code == 200
    data = response.get_json()
    assert [result["id"] for result in data["results"]] == [old_id, new_id]
    assert "recency_bias" not in data
    for result in data["results"]:
        assert "temporal" not in (result.get("score_components") or {})


def test_recall_recency_bias_on_promotes_newer_conflicting_fact(client, mock_state, auth_headers):
    """recency_bias=on: the newer fact outranks the older higher-importance one.

    Importance gap is 0.15: tag-only recall weights importance via both the
    keyword (trending match_score) and importance components (~0.45 combined),
    so the default 0.1 temporal weight flips moderate gaps, not arbitrary ones.
    """
    mock_state.memory_graph.memories.clear()
    now = datetime.now(timezone.utc)
    old_id = "dd000000-0000-0000-0000-000000000003"
    new_id = "dd000000-0000-0000-0000-000000000004"

    _store_memory(
        mock_state,
        old_id,
        "Favorite color is blue",
        ["fact"],
        0.65,
        timestamp=(now - timedelta(days=10)).isoformat(),
    )
    _store_memory(mock_state, new_id, "Favorite color is green", ["fact"], 0.5)

    response = client.get("/recall?tags=fact&limit=10&recency_bias=on", headers=auth_headers)

    assert response.status_code == 200
    data = response.get_json()
    assert data.get("recency_bias") == "on"
    assert [result["id"] for result in data["results"]] == [new_id, old_id]
    components = {result["id"]: result["score_components"] for result in data["results"]}
    assert components[new_id]["temporal"] == pytest.approx(1.0)
    assert components[old_id]["temporal"] == pytest.approx(0.0)


def test_recall_recency_bias_auto_fires_on_temporal_query(client, mock_state, auth_headers):
    mock_state.memory_graph.memories.clear()
    now = datetime.now(timezone.utc)
    old_id = "dd000000-0000-0000-0000-000000000005"
    new_id = "dd000000-0000-0000-0000-000000000006"

    _store_memory(
        mock_state,
        old_id,
        "Favorite editor is Vim",
        ["fact"],
        0.9,
        timestamp=(now - timedelta(days=10)).isoformat(),
    )
    _store_memory(mock_state, new_id, "Favorite editor is Zed", ["fact"], 0.1)

    response = client.get(
        "/recall?query=what is my current favorite editor&tags=fact&limit=10&recency_bias=auto",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data.get("recency_bias") == "on"
    assert any("temporal" in (result.get("score_components") or {}) for result in data["results"])


def test_recall_recency_bias_auto_skips_non_temporal_query(client, mock_state, auth_headers):
    mock_state.memory_graph.memories.clear()
    now = datetime.now(timezone.utc)
    old_id = "dd000000-0000-0000-0000-000000000007"
    new_id = "dd000000-0000-0000-0000-000000000008"

    _store_memory(
        mock_state,
        old_id,
        "Favorite editor is Vim",
        ["fact"],
        0.9,
        timestamp=(now - timedelta(days=10)).isoformat(),
    )
    _store_memory(mock_state, new_id, "Favorite editor is Zed", ["fact"], 0.1)

    response = client.get(
        "/recall?query=favorite editor preference history&tags=fact&limit=10&recency_bias=auto",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.get_json()
    assert "recency_bias" not in data
    for result in data["results"]:
        assert "temporal" not in (result.get("score_components") or {})


def test_recall_recency_bias_all_same_timestamp_is_safe(client, mock_state, auth_headers):
    """Degenerate spread (all candidates share a timestamp) contributes nothing, no crash."""
    mock_state.memory_graph.memories.clear()
    shared_ts = datetime.now(timezone.utc).isoformat()
    first_id = "dd000000-0000-0000-0000-000000000009"
    second_id = "dd000000-0000-0000-0000-000000000010"

    _store_memory(mock_state, first_id, "Same-time fact A", ["fact"], 0.9, timestamp=shared_ts)
    _store_memory(mock_state, second_id, "Same-time fact B", ["fact"], 0.1, timestamp=shared_ts)

    response = client.get("/recall?tags=fact&limit=10&recency_bias=on", headers=auth_headers)

    assert response.status_code == 200
    data = response.get_json()
    assert data.get("recency_bias") == "on"
    assert [result["id"] for result in data["results"]] == [first_id, second_id]
    for result in data["results"]:
        assert "temporal" not in (result.get("score_components") or {})


def test_recall_recency_bias_skips_timestamp_conversion_errors(
    monkeypatch, client, mock_state, auth_headers
):
    """Extreme platform date failures are skipped instead of 500ing the request."""
    mock_state.memory_graph.memories.clear()
    now = datetime.now(timezone.utc)
    old_id = "dd000000-0000-0000-0000-000000000013"
    new_id = "dd000000-0000-0000-0000-000000000014"
    bad_id = "dd000000-0000-0000-0000-000000000015"
    bad_timestamp = "platform-date-range-error"
    original_parse = recall_module._parse_iso_datetime

    class _TimestampRaises:
        def timestamp(self):
            raise OSError("timestamp out of range")

    def fake_parse(value):
        if value == bad_timestamp:
            return _TimestampRaises()
        return original_parse(value)

    monkeypatch.setattr(recall_module, "_parse_iso_datetime", fake_parse)
    _store_memory(
        mock_state,
        old_id,
        "Favorite color is blue",
        ["fact"],
        0.65,
        timestamp=(now - timedelta(days=10)).isoformat(),
    )
    _store_memory(mock_state, new_id, "Favorite color is green", ["fact"], 0.5)
    _store_memory(
        mock_state,
        bad_id,
        "Favorite color needs a timestamp fallback",
        ["fact"],
        0.1,
        timestamp=bad_timestamp,
    )

    response = client.get("/recall?tags=fact&limit=10&recency_bias=on", headers=auth_headers)

    assert response.status_code == 200
    data = response.get_json()
    assert data.get("recency_bias") == "on"
    components = {result["id"]: result["score_components"] for result in data["results"]}
    assert components[new_id]["temporal"] == pytest.approx(1.0)
    assert components[old_id]["temporal"] == pytest.approx(0.0)
    assert "temporal" not in components[bad_id]


def test_recall_recency_bias_invalid_param_falls_back_to_default(client, mock_state, auth_headers):
    mock_state.memory_graph.memories.clear()
    now = datetime.now(timezone.utc)
    old_id = "dd000000-0000-0000-0000-000000000016"
    new_id = "dd000000-0000-0000-0000-000000000017"

    _store_memory(
        mock_state,
        old_id,
        "Favorite color is blue",
        ["fact"],
        0.9,
        timestamp=(now - timedelta(days=10)).isoformat(),
    )
    _store_memory(mock_state, new_id, "Favorite color is green", ["fact"], 0.1)

    response = client.get("/recall?tags=fact&limit=10&recency_bias=bogus", headers=auth_headers)

    assert response.status_code == 200
    data = response.get_json()
    assert "recency_bias" not in data
    assert [result["id"] for result in data["results"]] == [old_id, new_id]


# ==================== Supersession chain-walk (issue #159 part 3) ====================


def test_recall_current_only_resolves_supersession_chain_to_head(client, mock_state, auth_headers):
    """A→INVALIDATED_BY→B→EVOLVED_INTO→C surfaces C with provenance pointing at A."""
    mock_state.memory_graph.memories.clear()
    mock_state.memory_graph.relationships.clear()
    a_id = "ee000000-0000-0000-0000-00000000000a"
    b_id = "ee000000-0000-0000-0000-00000000000b"
    c_id = "ee000000-0000-0000-0000-00000000000c"

    _store_memory(mock_state, a_id, "Editor was Vim", ["state"], 1.0)
    _store_memory(mock_state, b_id, "Editor became VS Code", ["middle"], 0.1)
    _store_memory(mock_state, c_id, "Editor is now Zed", ["head"], 0.1)
    mock_state.memory_graph.relationships.append(
        {"id1": a_id, "id2": b_id, "type": "INVALIDATED_BY", "strength": 0.9}
    )
    mock_state.memory_graph.relationships.append(
        {"id1": b_id, "id2": c_id, "type": "EVOLVED_INTO", "strength": 0.8}
    )

    response = client.get("/recall?limit=1&state_debug=true", headers=auth_headers)

    assert response.status_code == 200
    data = response.get_json()
    assert [result["id"] for result in data["results"]] == [c_id]
    head = data["results"][0]
    assert head["match_type"] == "state_replacement"
    assert head["state_replaces"] == a_id
    assert head["relations"][0]["from"] == a_id
    assert data["state_filter"]["suppressed"][0]["replacement_id"] == c_id
    assert data["state_filter"]["replacements"][0] == {
        "id": c_id,
        "replaces_id": a_id,
        "relation_type": "INVALIDATED_BY",
    }


def test_recall_current_only_supersession_cycle_terminates(client, mock_state, auth_headers):
    """A→B→A cycles terminate and surface the first replacement."""
    mock_state.memory_graph.memories.clear()
    mock_state.memory_graph.relationships.clear()
    a_id = "ee000000-0000-0000-0000-000000000010"
    b_id = "ee000000-0000-0000-0000-000000000011"

    _store_memory(mock_state, a_id, "Cyclic fact A", ["state"], 1.0)
    _store_memory(mock_state, b_id, "Cyclic fact B", ["cycle"], 0.1)
    mock_state.memory_graph.relationships.append(
        {"id1": a_id, "id2": b_id, "type": "INVALIDATED_BY", "strength": 0.9}
    )
    mock_state.memory_graph.relationships.append(
        {"id1": b_id, "id2": a_id, "type": "INVALIDATED_BY", "strength": 0.9}
    )

    response = client.get("/recall?limit=1&state_debug=true", headers=auth_headers)

    assert response.status_code == 200
    data = response.get_json()
    assert [result["id"] for result in data["results"]] == [b_id]
    assert data["results"][0]["state_replaces"] == a_id


def test_recall_current_only_supersession_chain_depth_bounded(client, mock_state, auth_headers):
    """A 7-hop chain stops at STATE_REPLACEMENT_MAX_DEPTH hops without error."""
    mock_state.memory_graph.memories.clear()
    mock_state.memory_graph.relationships.clear()

    source_id = "ee000000-0000-0000-0000-000000000020"
    chain_ids = [f"ee000000-0000-0000-0000-0000000000{30 + idx}" for idx in range(7)]

    _store_memory(mock_state, source_id, "Deep chain source", ["state"], 1.0)
    previous = source_id
    for idx, chain_id in enumerate(chain_ids):
        _store_memory(mock_state, chain_id, f"Deep chain hop {idx + 1}", ["deep-chain"], 0.1)
        mock_state.memory_graph.relationships.append(
            {"id1": previous, "id2": chain_id, "type": "INVALIDATED_BY", "strength": 0.9}
        )
        previous = chain_id

    response = client.get("/recall?limit=1&state_debug=true", headers=auth_headers)

    assert response.status_code == 200
    data = response.get_json()
    expected_head = chain_ids[recall_module.STATE_REPLACEMENT_MAX_DEPTH - 1]
    assert [result["id"] for result in data["results"]] == [expected_head]
    assert data["results"][0]["state_replaces"] == source_id


def test_recall_current_only_single_hop_fires_exactly_one_chain_round(
    client, mock_state, auth_headers
):
    """A single-hop replacement costs exactly one extra chain query.

    The resolver can't know the chain ended until it asks: after the
    first-hop batch resolves old -> replacement, one additional (empty)
    chain round runs against the replacement head before resolution
    stops. So the single-hop case is first-hop + 1, not first-hop only.
    """
    mock_state.memory_graph.memories.clear()
    mock_state.memory_graph.relationships.clear()
    old_id = "ee000000-0000-0000-0000-000000000040"
    replacement_id = "ee000000-0000-0000-0000-000000000041"

    _store_memory(mock_state, old_id, "Single hop legacy", ["state"], 1.0)
    _store_memory(mock_state, replacement_id, "Single hop current", ["hop"], 0.1)
    mock_state.memory_graph.relationships.append(
        {"id1": old_id, "id2": replacement_id, "type": "INVALIDATED_BY", "strength": 0.9}
    )

    response = client.get("/recall?limit=1", headers=auth_headers)

    assert response.status_code == 200
    replacement_queries = [
        query for query, _params in mock_state.memory_graph.queries if "RETURN source_id" in query
    ]
    # one first-hop batch + one (empty) chain round for the replacement head
    assert len(replacement_queries) == 2


def test_recall_current_only_no_replacements_fires_single_query(client, mock_state, auth_headers):
    """Without any supersession edges, only the first-hop batch query runs."""
    mock_state.memory_graph.memories.clear()
    mock_state.memory_graph.relationships.clear()
    only_id = "ee000000-0000-0000-0000-000000000050"
    _store_memory(mock_state, only_id, "Standalone fact", ["state"], 0.9)

    response = client.get("/recall?limit=1", headers=auth_headers)

    assert response.status_code == 200
    replacement_queries = [
        query for query, _params in mock_state.memory_graph.queries if "RETURN source_id" in query
    ]
    assert len(replacement_queries) == 1


# ==================== Preference latest-wins acceptance (issue #158) ====================


def test_recall_preference_latest_wins_with_recency_bias(client, mock_state, auth_headers):
    """Two conflicting Preference memories: recency_bias=on surfaces the latest first."""
    mock_state.memory_graph.memories.clear()
    now = datetime.now(timezone.utc)
    old_pref = "ff000000-0000-0000-0000-000000000001"
    new_pref = "ff000000-0000-0000-0000-000000000002"

    _store_memory(
        mock_state,
        old_pref,
        "Prefers tabs for indentation",
        ["preference"],
        0.65,
        mem_type="Preference",
        timestamp=(now - timedelta(days=30)).isoformat(),
    )
    _store_memory(
        mock_state,
        new_pref,
        "Prefers spaces for indentation",
        ["preference"],
        0.5,
        mem_type="Preference",
    )

    response = client.get("/recall?tags=preference&limit=10&recency_bias=on", headers=auth_headers)

    assert response.status_code == 200
    data = response.get_json()
    assert [result["id"] for result in data["results"]] == [new_pref, old_pref]


def test_recall_preference_explicit_invalidation_returns_replacement_only(
    client, mock_state, auth_headers
):
    """An INVALIDATED_BY edge suppresses the old preference entirely (no recency_bias needed)."""
    mock_state.memory_graph.memories.clear()
    mock_state.memory_graph.relationships.clear()
    now = datetime.now(timezone.utc)
    old_pref = "ff000000-0000-0000-0000-000000000003"
    new_pref = "ff000000-0000-0000-0000-000000000004"

    _store_memory(
        mock_state,
        old_pref,
        "Prefers tabs for indentation",
        ["preference"],
        0.9,
        mem_type="Preference",
        timestamp=(now - timedelta(days=30)).isoformat(),
    )
    _store_memory(
        mock_state,
        new_pref,
        "Prefers spaces for indentation",
        ["preference"],
        0.1,
        mem_type="Preference",
    )
    mock_state.memory_graph.relationships.append(
        {"id1": old_pref, "id2": new_pref, "type": "INVALIDATED_BY", "strength": 0.9}
    )

    response = client.get("/recall?tags=preference&limit=10&state_debug=true", headers=auth_headers)

    assert response.status_code == 200
    data = response.get_json()
    assert [result["id"] for result in data["results"]] == [new_pref]
    assert data["state_filter"]["suppressed_count"] == 1
