import json
from types import SimpleNamespace

import pytest

import app


class DummyGraph:
    """Minimal fake FalkorDB graph interface for tests."""

    def __init__(self):
        self.queries = []
        self.nodes: set[str] = set()

    def query(self, query, params):
        self.queries.append((query, params))
        if "MERGE (m:Memory {id:" in query:
            self.nodes.add(params["id"])
            return SimpleNamespace(result_set=[[SimpleNamespace(properties={"id": params["id"]})]])
        # Simulate an association creation returning a stub relation
        if "MERGE (m1)-[r:" in query:
            return SimpleNamespace(result_set=[["RELATES_TO", params["strength"], {"properties": {"id": params["id2"]}}]])
        # Graph recall relations query
        if "MATCH (m:Memory {id:" in query and "RETURN type" in query:
            return SimpleNamespace(result_set=[])
        # Text search query should return stored node
        if "MATCH (m:Memory)" in query and "RETURN m" in query:
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
    yield graph


@pytest.fixture
def client():
    return app.app.test_client()


def test_store_memory_without_content_returns_400(client):
    response = client.post("/memory", data=json.dumps({}), content_type="application/json")
    assert response.status_code == 400
    body = response.get_json()
    assert body["status"] == "error"


def test_store_memory_success(client, reset_state):
    response = client.post(
        "/memory",
        data=json.dumps({"content": "Hello", "tags": ["test"], "importance": 0.7}),
        content_type="application/json",
    )
    assert response.status_code == 201
    body = response.get_json()
    assert body["status"] == "success"
    assert body["qdrant"] in {"unconfigured", "stored", "failed"}


def test_create_association_validates_payload(client, reset_state):
    response = client.post(
        "/associate",
        data=json.dumps({"memory1_id": "a", "memory2_id": "a"}),
        content_type="application/json",
    )
    assert response.status_code == 400


def test_create_association_success(client, reset_state):
    for memory_id in ("a", "b"):
        response = client.post(
            "/memory",
            data=json.dumps({"id": memory_id, "content": f"Memory {memory_id}"}),
            content_type="application/json",
        )
        assert response.status_code == 201

    response = client.post(
        "/associate",
        data=json.dumps({
            "memory1_id": "a",
            "memory2_id": "b",
            "type": "relates_to",
            "strength": 0.9,
        }),
        content_type="application/json",
    )
    assert response.status_code == 201
    body = response.get_json()
    assert body["relation_type"] == "RELATES_TO"
