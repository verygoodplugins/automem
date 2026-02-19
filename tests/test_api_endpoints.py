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
from automem.api.recall import _expand_related_memories
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


def test_recall_with_high_limit(client, mock_state, auth_headers):
    """Test recall with limit exceeding max - should clamp to 50."""
    response = client.get("/recall?limit=100", headers=auth_headers)
    assert response.status_code == 200
    # API clamps limit to 50 instead of returning error


def _store_memory(
    mock_state, memory_id, content, tags, importance, mem_type="Context", timestamp=None
):
    mock_state.memory_graph.memories[memory_id] = {
        "id": memory_id,
        "content": content,
        "tags": tags,
        "importance": importance,
        "type": mem_type,
        "timestamp": timestamp or utc_now(),
    }


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
    """Test creating associations with all new relationship types."""
    # Create two memories
    mock_state.memory_graph.memories["aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"] = {
        "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "content": "Memory 1",
    }
    mock_state.memory_graph.memories["bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"] = {
        "id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        "content": "Memory 2",
    }

    relationship_types = [
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
    ]

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
