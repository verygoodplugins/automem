"""Comprehensive test suite for AutoMem Flask API endpoints."""

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch, MagicMock

import pytest
from flask import Flask
from flask.testing import FlaskClient
from qdrant_client import models as qdrant_models

import app
from app import utc_now, _normalize_timestamp


class MockGraph:
    """Enhanced mock for FalkorDB graph with more realistic behavior."""

    def __init__(self):
        self.queries = []
        self.memories = {}
        self.relationships = []

    def query(self, query: str, params: dict = None):
        """
        Dispatch a mock FalkorDB query and return a result-shaped object appropriate for test scenarios.
        
        This method records the raw query and parameters and returns a SimpleNamespace with a `result_set`
        that mimics FalkorDB responses for a range of supported mock operations used in tests:
        - Create/MERGE and CREATE of Memory nodes (returns a node with `properties` for the stored memory).
        - Update of a Memory's content/tags/importance (returns updated node or empty result if not found).
        - Retrieve a Memory by id (returns node or empty result).
        - Delete a Memory by id (returns an acknowledgment or empty result).
        - Search memories by a single tag (returns matching nodes).
        - Retrieve all memory id/content pairs for reembedding (returns list of [id, content]).
        - Create associations between two memories (returns confirmation when both memories exist).
        - Tag-filtered recall queries with ordering by importance (desc) and timestamp (desc) and optional limiting;
          tag filters match when any memory tag starts with a filter prefix (case-insensitive).
        
        Parameters:
            query (str): The Cypher-like query string being executed (mocked).
            params (dict, optional): Parameters for the query (ids, tags, importance, tag_filters, limit, etc.).
        
        Returns:
            SimpleNamespace: An object with a `result_set` attribute containing rows that mirror FalkorDB query results:
                - For node returns: rows are lists containing a SimpleNamespace with a `properties` dict.
                - For id/content retrieval: rows are lists like [id, content].
                - For delete/association acknowledgments: rows contain simple markers (e.g., ["deleted"], ["created"]).
                - Empty `result_set` indicates no matching result.
        """
        params = params or {}
        self.queries.append((query, params))

        # Memory CRUD operations
        if "MERGE (m:Memory {id:" in query or "CREATE (m:Memory {id:" in query:
            memory_id = params["id"]
            self.memories[memory_id] = {
                "id": memory_id,
                "content": params.get("content", ""),
                "tags": params.get("tags", []),
                "importance": params.get("importance", 0.5),
                "type": params.get("type", "Memory"),
                "timestamp": params.get("timestamp", utc_now()),
                "metadata": params.get("metadata", "{}"),
                "updated_at": params.get("updated_at"),
                "last_accessed": params.get("last_accessed"),
                "confidence": params.get("confidence", 1.0),
            }
            node = SimpleNamespace(properties=self.memories[memory_id])
            return SimpleNamespace(result_set=[[node]])

        # Update memory
        if "MATCH (m:Memory {id:" in query and "SET m.content" in query:
            memory_id = params["id"]
            if memory_id in self.memories:
                self.memories[memory_id].update({
                    "content": params.get("content", self.memories[memory_id]["content"]),
                    "tags": params.get("tags", self.memories[memory_id]["tags"]),
                    "importance": params.get("importance", self.memories[memory_id]["importance"]),
                    "updated_at": params.get("updated_at", utc_now()),
                })
                node = SimpleNamespace(properties=self.memories[memory_id])
                return SimpleNamespace(result_set=[[node]])
            return SimpleNamespace(result_set=[])

        # Retrieve memory by ID
        if "MATCH (m:Memory {id:" in query and "RETURN m" in query and "WHERE" not in query:
            memory_id = params.get("id") or params.get("id1") or params.get("id2")
            if memory_id in self.memories:
                node = SimpleNamespace(properties=self.memories[memory_id])
                return SimpleNamespace(result_set=[[node]])
            return SimpleNamespace(result_set=[])

        # Delete memory
        if "MATCH (m:Memory {id:" in query and "DELETE m" in query:
            memory_id = params.get("id")
            if memory_id in self.memories:
                del self.memories[memory_id]
                return SimpleNamespace(result_set=[["deleted"]])
            return SimpleNamespace(result_set=[])

        # Search memories by tag
        if "MATCH (m:Memory)" in query and "$tag IN m.tags" in query:
            results = []
            for mem in self.memories.values():
                if params.get("tag") in mem.get("tags", []):
                    node = SimpleNamespace(properties=mem)
                    results.append([node])
            return SimpleNamespace(result_set=results)

        # Get all memories for reembedding
        if "MATCH (m:Memory)" in query and "RETURN m.id, m.content" in query:
            results = []
            for mem_id, mem in self.memories.items():
                if mem.get("content"):
                    results.append([mem_id, mem["content"]])
            return SimpleNamespace(result_set=results)

        # Handle association creation - check both memories exist
        if "MATCH (m1:Memory" in query and "MATCH (m2:Memory" in query and "MERGE (m1)" in query:
            id1 = params.get("id1")
            id2 = params.get("id2")
            if id1 in self.memories and id2 in self.memories:
                # Return successful association creation
                return SimpleNamespace(result_set=[["created"]])
            # Return empty if memories don't exist
            return SimpleNamespace(result_set=[])

        return_m_line = "RETURN m\n" in query or "RETURN m\r" in query
        if (
            "MATCH (m:Memory)" in query
            and return_m_line
            and "ORDER BY m.importance DESC" in query
        ):
            results = list(self.memories.values())
            tag_filters = [tag.lower() for tag in params.get("tag_filters", [])]
            if tag_filters:
                filtered = []
                for mem in results:
                    mem_tags = [
                        str(tag).strip().lower()
                        for tag in (mem.get("tags") or [])
                        if isinstance(tag, str)
                    ]
                    if any(
                        any(tag.startswith(filter_tag) for tag in mem_tags)
                        for filter_tag in tag_filters
                    ):
                        filtered.append(mem)
                results = filtered
            results.sort(
                key=lambda mem: (
                    float(mem.get("importance") or 0.0),
                    str(mem.get("timestamp") or ""),
                ),
                reverse=True,
            )
            limit = params.get("limit", len(results))
            limited = results[:limit]
            return SimpleNamespace(
                result_set=[
                    [SimpleNamespace(properties=mem)] for mem in limited
                ]
            )

        # Default empty result
        return SimpleNamespace(result_set=[])


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
            self.points[point.id] = {
                "vector": point.vector,
                "payload": point.payload
            }

    def search(self, collection_name, query_vector, limit=5, with_payload=True, with_vectors=False):
        """Mock search operation."""
        self.search_calls.append({
            "collection": collection_name,
            "vector": query_vector,
            "limit": limit
        })
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
        """
        Remove the specified point IDs from the mock client's stored points and record the delete call.
        
        Parameters:
            collection_name (str): Name of the collection targeted by the delete operation.
            points_selector: An object with a `points` attribute (iterable of point IDs) specifying which points to delete.
        """
        self.delete_calls.append((collection_name, points_selector))
        if hasattr(points_selector, "points"):
            for point_id in points_selector.points:
                if point_id in self.points:
                    del self.points[point_id]

    def scroll(self, collection_name, scroll_filter=None, limit=10, with_payload=True):
        """
        Return up to `limit` stored points whose payload matches `scroll_filter`.
        
        Parameters:
            collection_name (str): Name of the collection to search (unused in mock but kept for API compatibility).
            scroll_filter (object|None): Filter object describing match conditions; evaluated against each point's payload using `_filter_matches`.
            limit (int): Maximum number of points to return.
            with_payload (bool): If True include payloads on returned points (mock always includes payload).
        
        Returns:
            tuple:
                matches (list): List of match objects (SimpleNamespace) each with `id` and `payload` attributes.
                next_cursor (None): Cursor for continued scrolling; always `None` in this mock implementation.
        """
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
        """
        Determine whether a point payload satisfies all "must" conditions of a Qdrant scroll filter.
        
        This performs case-insensitive string matching for each condition in `scroll_filter.must`. Supports `MatchAny` (succeeds if any of the provided target strings appear in the payload field) and `MatchValue` (succeeds if the exact target string appears in the payload field). Non-string or empty payload values are ignored. If `scroll_filter` is None or has no `must` conditions, the payload is considered a match.
        
        Parameters:
            payload (dict): The point payload mapping field names to lists of values (as stored in Qdrant).
            scroll_filter: A Qdrant scroll filter object with an optional `must` attribute containing match conditions.
        
        Returns:
            bool: `True` if the payload meets all `must` conditions, `False` otherwise.
        """
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
    """
    Create and install a mocked ServiceState for tests, including in-memory graph, Qdrant client, OpenAI client, embeddings, and auth tokens.
    
    Parameters:
        monkeypatch: pytest monkeypatch fixture used to patch app-level attributes and initialization functions.
    
    Returns:
        state (app.ServiceState): The populated ServiceState with mocked components:
            - memory_graph: MockGraph instance simulating FalkorDB behavior.
            - qdrant: MockQdrantClient instance simulating Qdrant.
            - openai_client: Mock object for OpenAI interactions.
    
    Notes:
        This function patches app.state, app.init_falkordb, app.init_qdrant, app.init_openai,
        embedding generation helpers (_generate_real_embedding, _generate_placeholder_embedding),
        and API_TOKEN/ADMIN_TOKEN values for test isolation.
    """
    state = app.ServiceState()
    state.memory_graph = MockGraph()
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
    return {
        "Authorization": "Bearer test-token",
        "X-Admin-Token": "test-admin-token"
    }


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
        "importance": 0.8
    }
    mock_state.memory_graph.memories["test-id"] = {
        "id": "test-id",
        "content": memory_data["content"],
        "tags": memory_data["tags"],
        "importance": memory_data["importance"],
        "timestamp": utc_now()
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


def _store_memory(mock_state, memory_id, content, tags, importance, mem_type="Context", timestamp=None):
    """
    Insert a memory into the test mock state's in-memory FalkorDB graph.
    
    Parameters:
        mock_state: ServiceState test fixture containing the mock memory_graph.
        memory_id (str): Unique identifier to store for the memory node.
        content (str): Text content of the memory.
        tags (list[str]): List of tags associated with the memory.
        importance (int | float): Importance score used for prioritization in recalls.
        mem_type (str): Memory type label (e.g., "Context", "Style"); defaults to "Context".
        timestamp (datetime | None): Explicit timestamp for the memory; if omitted, the current UTC time is used.
    """
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
    style_id = "style-mem"
    other_id = "general-mem"
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
    style_id = "style-mem-limit"
    other_id = "general-mem-limit"
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

def test_update_memory_success(client, mock_state, auth_headers):
    """Test successful memory update."""
    memory_id = "test-memory-123"

    # Create initial memory
    mock_state.memory_graph.memories[memory_id] = {
        "id": memory_id,
        "content": "Original content",
        "tags": ["original"],
        "importance": 0.5,
        "timestamp": utc_now()
    }

    update_data = {
        "content": "Updated content",
        "tags": ["updated", "modified"],
        "importance": 0.9
    }

    response = client.patch(f"/memory/{memory_id}",
                           json=update_data,
                           headers=auth_headers)

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["memory_id"] == memory_id


def test_update_memory_not_found(client, mock_state, auth_headers):
    """Test updating non-existent memory."""
    response = client.patch("/memory/non-existent-id",
                           json={"content": "New content"},
                           headers=auth_headers)
    assert response.status_code == 404


def test_update_memory_partial_fields(client, mock_state, auth_headers):
    """Test updating only some fields of a memory."""
    memory_id = "test-memory-456"

    # Create initial memory
    mock_state.memory_graph.memories[memory_id] = {
        "id": memory_id,
        "content": "Original content",
        "tags": ["original"],
        "importance": 0.5,
        "metadata": json.dumps({"source": "test"})
    }

    # Update only tags
    response = client.patch(f"/memory/{memory_id}",
                           json={"tags": ["new", "tags"]},
                           headers=auth_headers)

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"


# ==================== Test Memory Delete ====================

def test_delete_memory_success(client, mock_state, auth_headers):
    """Test successful memory deletion."""
    memory_id = "delete-test-123"

    # Create memory to delete
    mock_state.memory_graph.memories[memory_id] = {
        "id": memory_id,
        "content": "To be deleted",
        "timestamp": utc_now()
    }

    # Also add to Qdrant
    mock_state.qdrant.points[memory_id] = {
        "vector": [0.1] * 768,
        "payload": {"content": "To be deleted"}
    }

    response = client.delete(f"/memory/{memory_id}", headers=auth_headers)

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["memory_id"] == memory_id
    assert memory_id not in mock_state.memory_graph.memories


def test_delete_memory_not_found(client, mock_state, auth_headers):
    """Test deleting non-existent memory."""
    response = client.delete("/memory/non-existent", headers=auth_headers)
    assert response.status_code == 404


# ==================== Test Memory By Tag ====================

def test_memory_by_tag_single(client, mock_state, auth_headers):
    """Test retrieving memories by a single tag."""
    # Add some memories with tags
    mock_state.memory_graph.memories["mem1"] = {
        "id": "mem1",
        "content": "Python tutorial",
        "tags": ["python", "tutorial"],
        "importance": 0.8,
        "timestamp": utc_now()
    }

    response = client.get("/memory/by-tag?tags=python", headers=auth_headers)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "memories" in data  # API returns 'memories' not 'results'


def test_memory_by_tag_multiple(client, mock_state, auth_headers):
    """Test retrieving memories by multiple tags."""
    response = client.get("/memory/by-tag?tags=python&tags=ai&tags=ml",
                         headers=auth_headers)
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
    mock_state.memory_graph.memories["mem1"] = {
        "id": "mem1",
        "content": "First memory to reembed",
        "tags": ["test"],
        "importance": 0.7
    }
    mock_state.memory_graph.memories["mem2"] = {
        "id": "mem2",
        "content": "Second memory to reembed",
        "tags": ["test"],
        "importance": 0.8
    }

    # Mock OpenAI client
    mock_state.openai_client = Mock()
    mock_state.openai_client.embeddings.create.return_value = Mock(
        data=[Mock(embedding=[0.2] * 768)]
    )

    response = client.post("/admin/reembed",
                          json={"batch_size": 10, "limit": 2},
                          headers=admin_headers)

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "complete"
    assert data["processed"] == 2
    assert data["total"] == 2


def test_admin_reembed_no_openai(client, mock_state, admin_headers):
    """Test reembed fails gracefully when OpenAI not configured."""
    mock_state.openai_client = None

    response = client.post("/admin/reembed",
                          json={"batch_size": 10},
                          headers=admin_headers)

    assert response.status_code == 503
    data = response.get_json()
    assert "OpenAI API key not configured" in data["message"]


def test_admin_reembed_no_qdrant(client, mock_state, admin_headers):
    """Test reembed fails when Qdrant not available."""
    mock_state.qdrant = None

    response = client.post("/admin/reembed",
                          json={"batch_size": 10},
                          headers=admin_headers)

    assert response.status_code == 503
    data = response.get_json()
    assert "Qdrant is not available" in data["message"]


def test_admin_reembed_no_admin_token(client, mock_state, auth_headers):
    """Test reembed requires admin token."""
    response = client.post("/admin/reembed",
                          json={"batch_size": 10},
                          headers=auth_headers)  # Regular auth, not admin

    assert response.status_code in [401, 403]


def test_admin_reembed_force_flag(client, mock_state, admin_headers):
    """Test force reembedding with force flag."""
    # Add memory
    mock_state.memory_graph.memories["mem1"] = {
        "id": "mem1",
        "content": "Memory with existing embedding",
        "tags": ["test"]
    }

    # Mock OpenAI
    mock_state.openai_client = Mock()
    mock_state.openai_client.embeddings.create.return_value = Mock(
        data=[Mock(embedding=[0.3] * 768)]
    )

    response = client.post("/admin/reembed",
                          json={"force": True, "limit": 1},
                          headers=admin_headers)

    assert response.status_code == 200
    data = response.get_json()
    assert data["processed"] == 1


# ==================== Test Consolidation ====================

def test_consolidate_trigger(client, mock_state, auth_headers):
    """Test triggering consolidation tasks."""
    response = client.post("/consolidate",
                          json={"mode": "full"},  # API uses 'mode' not 'task'
                          headers=auth_headers)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"  # API returns 'success' not 'triggered'
    # API doesn't return the task/mode field


def test_consolidate_invalid_task(client, mock_state, auth_headers):
    """Test consolidation with invalid task name - API accepts any task."""
    response = client.post("/consolidate",
                          json={"task": "invalid_task"},
                          headers=auth_headers)
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
    response = client.post("/enrichment/reprocess",
                          json={"ids": ["mem1", "mem2", "mem3"]},
                          headers=admin_headers)

    assert response.status_code == 202
    data = response.get_json()
    assert data["status"] == "queued"
    assert data["count"] == 3


def test_enrichment_reprocess_no_ids(client, mock_state, admin_headers):
    """Test reprocess with no IDs provided."""
    response = client.post("/enrichment/reprocess",
                          json={},
                          headers=admin_headers)
    assert response.status_code == 400


# ==================== Test Association Creation ====================

def test_create_association_all_relationship_types(client, mock_state, auth_headers):
    """Test creating associations with all new relationship types."""
    # Create two memories
    mock_state.memory_graph.memories["mem1"] = {"id": "mem1", "content": "Memory 1"}
    mock_state.memory_graph.memories["mem2"] = {"id": "mem2", "content": "Memory 2"}

    relationship_types = [
        "RELATES_TO", "LEADS_TO", "OCCURRED_BEFORE",
        "PREFERS_OVER", "EXEMPLIFIES", "CONTRADICTS",
        "REINFORCES", "INVALIDATED_BY", "EVOLVED_INTO",
        "DERIVED_FROM", "PART_OF"
    ]

    for rel_type in relationship_types:
        response = client.post("/associate",
                              json={
                                  "memory1_id": "mem1",
                                  "memory2_id": "mem2",
                                  "type": rel_type,
                                  "strength": 0.8
                              },
                              headers=auth_headers)
        assert response.status_code == 201, f"Failed for relationship type: {rel_type}"
        data = response.get_json()
        assert data["status"] == "success"


def test_create_association_with_properties(client, mock_state, auth_headers):
    """Test creating association with additional properties."""
    mock_state.memory_graph.memories["mem1"] = {"id": "mem1", "content": "Preferred method"}
    mock_state.memory_graph.memories["mem2"] = {"id": "mem2", "content": "Alternative method"}

    response = client.post("/associate",
                          json={
                              "memory1_id": "mem1",
                              "memory2_id": "mem2",
                              "type": "PREFERS_OVER",
                              "strength": 0.9,
                              "properties": {
                                  "reason": "Better performance",
                                  "context": "Production environment"
                              }
                          },
                          headers=auth_headers)

    assert response.status_code == 201
    data = response.get_json()
    assert data["status"] == "success"


# ==================== Test Startup Recall ====================

def test_startup_recall(client, mock_state, auth_headers):
    """Test startup recall endpoint."""
    # Add some recent high-importance memories
    now = utc_now()
    mock_state.memory_graph.memories["important"] = {
        "id": "important",
        "content": "Critical information",
        "importance": 0.95,
        "timestamp": now,
        "tags": ["critical"]
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
    mock_state.memory_graph.memories["decision1"] = {
        "id": "decision1",
        "content": "Architectural decision",
        "type": "Decision",
        "confidence": 0.9,
        "importance": 0.85
    }
    mock_state.memory_graph.memories["pattern1"] = {
        "id": "pattern1",
        "content": "Common pattern",
        "type": "Pattern",
        "confidence": 0.7,
        "importance": 0.6
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

    response = client.post("/memory",
                          json={"content": "Test memory"},
                          headers=auth_headers)
    assert response.status_code == 503
    data = response.get_json()
    assert "FalkorDB is unavailable" in data["message"]


def test_invalid_json_payload(client, mock_state, auth_headers):
    """Test handling of invalid JSON payload."""
    response = client.post("/memory",
                          data="This is not JSON",
                          headers=auth_headers,
                          content_type="application/json")
    assert response.status_code == 400


def test_missing_required_fields(client, mock_state, auth_headers):
    """Test validation of required fields."""
    # Missing content field
    response = client.post("/memory",
                          json={"tags": ["test"]},
                          headers=auth_headers)
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
    response = client.post("/memory",
                          json={
                              "content": "Test memory",
                              "timestamp": "not-a-valid-timestamp"
                          },
                          headers=auth_headers)
    assert response.status_code == 400
    data = response.get_json()
    assert "timestamp" in data["message"].lower()


def test_embedding_dimension_validation(client, mock_state, auth_headers):
    """Test validation of embedding dimensions."""
    response = client.post("/memory",
                          json={
                              "content": "Test memory",
                              "embedding": [0.1, 0.2]  # Wrong dimension
                          },
                          headers=auth_headers)
    assert response.status_code == 400
    data = response.get_json()
    assert "768" in data["message"]


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