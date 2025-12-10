import json
from types import SimpleNamespace

import pytest

import app


class DummyGraph:
    """Minimal fake FalkorDB graph interface for tests."""

    def __init__(self):
        self.queries = []
        self.nodes: set[str] = set()
        self.memories = []

    def query(self, query, params=None):
        params = params or {}
        self.queries.append((query, params))

        # Store memory creation
        if "MERGE (m:Memory {id:" in query:
            memory_id = params["id"]
            self.nodes.add(memory_id)
            self.memories.append(
                {
                    "id": memory_id,
                    "content": params.get("content", ""),
                    "type": params.get("type", "Memory"),
                    "confidence": params.get("confidence", 0.5),
                    "importance": params.get("importance", 0.5),
                }
            )
            return SimpleNamespace(result_set=[[SimpleNamespace(properties={"id": memory_id})]])

        # Analytics queries
        if "MATCH (m:Memory)" in query and "RETURN m.type, COUNT(m)" in query:
            # Return memory type distribution
            types_count = {}
            for mem in self.memories:
                mem_type = mem.get("type", "Memory")
                if mem_type not in types_count:
                    types_count[mem_type] = {"count": 0, "total_conf": 0}
                types_count[mem_type]["count"] += 1
                types_count[mem_type]["total_conf"] += mem.get("confidence", 0.5)

            result_set = []
            for mem_type, data in types_count.items():
                avg_conf = data["total_conf"] / data["count"] if data["count"] > 0 else 0
                result_set.append([mem_type, data["count"], avg_conf])
            return SimpleNamespace(result_set=result_set)

        # Pattern queries
        if "MATCH (p:Pattern)" in query:
            return SimpleNamespace(result_set=[])

        # Preference queries
        if "MATCH (m1:Memory)-[r:PREFERS_OVER]" in query:
            return SimpleNamespace(result_set=[])

        # Temporal insights query
        if "toInteger(substring(m.timestamp" in query:
            return SimpleNamespace(result_set=[])

        # Confidence distribution query
        if "WHEN m.confidence" in query:
            return SimpleNamespace(result_set=[["medium", len(self.memories)]])

        # Entity extraction query
        if "MATCH (m:Memory)" in query and "RETURN m.content" in query:
            result_set = [[mem["content"]] for mem in self.memories[:100]]
            return SimpleNamespace(result_set=result_set)

        # Simulate an association creation returning a stub relation
        if "MERGE (m1)-[r:" in query:
            return SimpleNamespace(
                result_set=[
                    [
                        "RELATES_TO",
                        params.get("strength", 0.5),
                        {"properties": {"id": params.get("id2", "")}},
                    ]
                ]
            )

        # Graph recall relations query
        if "MATCH (m:Memory {id:" in query and "RETURN type" in query:
            return SimpleNamespace(result_set=[])

        # Text search query should return stored node
        if "MATCH (m:Memory)" in query and "RETURN m" in query and "WHERE" in query:
            data = {
                "id": params.get("query", "memory-1"),
                "content": "Example",
                "importance": 0.9,
            }
            return SimpleNamespace(result_set=[[SimpleNamespace(properties=data)]])

        return SimpleNamespace(result_set=[])


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    state = app.ServiceState()
    graph = DummyGraph()
    state.memory_graph = graph
    monkeypatch.setattr(app, "state", state)
    monkeypatch.setattr(app, "init_falkordb", lambda: None)
    monkeypatch.setattr(app, "init_qdrant", lambda: None)
    # Mock API tokens for auth
    monkeypatch.setattr(app, "API_TOKEN", "test-token")
    monkeypatch.setattr(app, "ADMIN_TOKEN", "test-admin-token")
    yield graph


@pytest.fixture
def client():
    return app.app.test_client()


@pytest.fixture
def auth_headers():
    """Provide authorization headers for testing."""
    return {"Authorization": "Bearer test-token"}


def test_store_memory_without_content_returns_400(client, auth_headers):
    response = client.post(
        "/memory", data=json.dumps({}), content_type="application/json", headers=auth_headers
    )
    assert response.status_code == 400
    body = response.get_json()
    assert body["status"] == "error"


def test_store_memory_success(client, reset_state, auth_headers):
    response = client.post(
        "/memory",
        data=json.dumps({"content": "Hello", "tags": ["test"], "importance": 0.7}),
        content_type="application/json",
        headers=auth_headers,
    )
    assert response.status_code == 201
    body = response.get_json()
    assert body["status"] == "success"
    assert body["qdrant"] in {"unconfigured", "stored", "failed"}


def test_create_association_validates_payload(client, reset_state, auth_headers):
    response = client.post(
        "/associate",
        data=json.dumps({"memory1_id": "a", "memory2_id": "a"}),
        content_type="application/json",
        headers=auth_headers,
    )
    assert response.status_code == 400


def test_create_association_success(client, reset_state, auth_headers):
    for memory_id in ("a", "b"):
        response = client.post(
            "/memory",
            data=json.dumps({"id": memory_id, "content": f"Memory {memory_id}"}),
            content_type="application/json",
            headers=auth_headers,
        )
        assert response.status_code == 201

    response = client.post(
        "/associate",
        data=json.dumps(
            {
                "memory1_id": "a",
                "memory2_id": "b",
                "type": "relates_to",
                "strength": 0.9,
            }
        ),
        content_type="application/json",
        headers=auth_headers,
    )
    assert response.status_code == 201
    body = response.get_json()
    assert body["relation_type"] == "RELATES_TO"


def test_memory_classification(client, reset_state, auth_headers):
    """Test that memories are automatically classified."""
    # Decision memory
    response = client.post(
        "/memory",
        data=json.dumps({"content": "I decided to use FalkorDB over ArangoDB"}),
        content_type="application/json",
        headers=auth_headers,
    )
    assert response.status_code == 201
    body = response.get_json()
    assert body["type"] == "Decision"
    assert body["confidence"] >= 0.6

    # Preference memory
    response = client.post(
        "/memory",
        data=json.dumps({"content": "I prefer Railway for deployments"}),
        content_type="application/json",
        headers=auth_headers,
    )
    assert response.status_code == 201
    body = response.get_json()
    assert body["type"] == "Preference"

    # Pattern memory
    response = client.post(
        "/memory",
        data=json.dumps({"content": "I usually write tests before implementation"}),
        content_type="application/json",
        headers=auth_headers,
    )
    assert response.status_code == 201
    body = response.get_json()
    assert body["type"] == "Pattern"


def test_temporal_validity_fields(client, reset_state, auth_headers):
    """Test temporal validity fields t_valid and t_invalid."""
    response = client.post(
        "/memory",
        data=json.dumps(
            {
                "content": "This was valid in 2023",
                "t_valid": "2023-01-01T00:00:00Z",
                "t_invalid": "2024-01-01T00:00:00Z",
            }
        ),
        content_type="application/json",
        headers=auth_headers,
    )
    assert response.status_code == 201
    body = response.get_json()
    assert body["status"] == "success"


def test_new_relationship_types(client, reset_state, auth_headers):
    """Test new PKG relationship types with properties."""
    # Create memories for preference relationship
    response = client.post(
        "/memory",
        data=json.dumps({"id": "tool1", "content": "FalkorDB"}),
        content_type="application/json",
        headers=auth_headers,
    )
    assert response.status_code == 201

    response = client.post(
        "/memory",
        data=json.dumps({"id": "tool2", "content": "ArangoDB"}),
        content_type="application/json",
        headers=auth_headers,
    )
    assert response.status_code == 201

    # Create PREFERS_OVER relationship with properties
    response = client.post(
        "/associate",
        data=json.dumps(
            {
                "memory1_id": "tool1",
                "memory2_id": "tool2",
                "type": "PREFERS_OVER",
                "strength": 0.95,
                "context": "cost-effectiveness",
                "reason": "30x cost difference",
            }
        ),
        content_type="application/json",
        headers=auth_headers,
    )
    assert response.status_code == 201
    body = response.get_json()
    assert body["relation_type"] == "PREFERS_OVER"
    assert body["context"] == "cost-effectiveness"
    assert body["reason"] == "30x cost difference"


def test_analytics_endpoint(client, reset_state, auth_headers):
    """Test the analytics endpoint."""
    # Add some test memories first
    memories = [
        {"content": "I decided to use Python", "tags": ["decision", "language"]},
        {"content": "I prefer dark mode", "tags": ["preference"]},
        {"content": "I usually code at night", "tags": ["pattern", "habit"]},
    ]

    for memory in memories:
        response = client.post(
            "/memory",
            data=json.dumps(memory),
            content_type="application/json",
            headers=auth_headers,
        )
        assert response.status_code == 201

    # Get analytics
    response = client.get("/analyze", headers=auth_headers)
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "success"
    assert "analytics" in body
    analytics = body["analytics"]

    # Check analytics structure
    assert "memory_types" in analytics
    assert "patterns" in analytics
    assert "preferences" in analytics
    assert "temporal_insights" in analytics
    assert "entity_frequency" in analytics
    assert "confidence_distribution" in analytics


def test_recall_updates_last_accessed(client, reset_state, auth_headers):
    """Test that /recall updates last_accessed for retrieved memories."""
    # Store a memory
    response = client.post(
        "/memory",
        data=json.dumps({"content": "Test memory for recall", "tags": ["test"]}),
        content_type="application/json",
        headers=auth_headers,
    )
    assert response.status_code == 201

    # Do a recall - this should trigger last_accessed update
    response = client.get("/recall?query=test&limit=5", headers=auth_headers)
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "success"

    # Verify the query was issued (check DummyGraph recorded queries)
    # Look for the UNWIND query that updates last_accessed
    last_accessed_queries = [q for q, p in reset_state.queries if "SET m.last_accessed" in q]
    assert len(last_accessed_queries) >= 1, "last_accessed update query should have been issued"


def test_by_tag_updates_last_accessed(client, reset_state, auth_headers):
    """Test that /memory/by-tag updates last_accessed for retrieved memories."""
    # Store a memory with a specific tag
    response = client.post(
        "/memory",
        data=json.dumps({"content": "Tagged memory", "tags": ["unique-tag"]}),
        content_type="application/json",
        headers=auth_headers,
    )
    assert response.status_code == 201

    # Query by tag - this should trigger last_accessed update
    response = client.get("/memory/by-tag?tags=unique-tag", headers=auth_headers)
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "success"

    # Verify the last_accessed update query was issued
    last_accessed_queries = [q for q, p in reset_state.queries if "SET m.last_accessed" in q]
    assert len(last_accessed_queries) >= 1, "last_accessed update query should have been issued"


def test_update_last_accessed_handles_empty_list():
    """Test that update_last_accessed gracefully handles empty list."""
    # Should not raise, should be a no-op
    app.update_last_accessed([])


def test_update_last_accessed_handles_none_graph(monkeypatch):
    """Test that update_last_accessed handles missing graph gracefully."""
    # Set graph to None
    state = app.ServiceState()
    state.memory_graph = None
    monkeypatch.setattr(app, "state", state)

    # Should not raise, should be a no-op
    app.update_last_accessed(["some-id"])
