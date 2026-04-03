from __future__ import annotations

import pytest

import app
from tests.support.fake_graph import FakeGraph


def _make_state():
    state = app.ServiceState()
    state.memory_graph = FakeGraph()
    state.qdrant = None
    return state


def _auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-token"}


def _preseed_headers() -> dict[str, str]:
    return {
        "Authorization": "Bearer test-token",
        "X-AutoMem-Internal": "preseed",
    }


@pytest.fixture
def client():
    app.app.config["TESTING"] = True
    return app.app.test_client()


def test_preseed_creates_memories_and_associations(client, monkeypatch):
    state = _make_state()
    monkeypatch.setattr(app, "state", state)
    monkeypatch.setattr(app, "API_TOKEN", "test-token")

    response = client.post(
        "/api/v1/preseed",
        json={
            "memories": [
                {
                    "content": "User is a senior engineer at Acme",
                    "importance": 0.9,
                    "tags": ["role"],
                    "type": "fact",
                },
                {
                    "content": "User prefers concise communication",
                    "importance": 0.85,
                    "tags": ["preferences"],
                    "type": "preference",
                },
            ],
            "associations": [
                {
                    "from_index": 0,
                    "to_index": 1,
                    "relationship": "RELATES_TO",
                    "strength": 0.7,
                }
            ],
        },
        headers=_preseed_headers(),
    )

    assert response.status_code == 201
    body = response.get_json()
    assert body["status"] == "success"
    assert body["memories_created"] == 2
    assert body["associations_created"] == 1
    assert len(body["memory_ids"]) == 2

    stored = list(state.memory_graph.memories.values())
    assert len(stored) == 2
    assert all("onboarding" in (memory.get("tags") or []) for memory in stored)

    relationship = state.memory_graph.relationships[0]
    assert relationship["type"] == "RELATES_TO"
    assert relationship["id1"] == body["memory_ids"][0]
    assert relationship["id2"] == body["memory_ids"][1]


def test_preseed_requires_internal_header(client, monkeypatch):
    state = _make_state()
    monkeypatch.setattr(app, "state", state)
    monkeypatch.setattr(app, "API_TOKEN", "test-token")

    response = client.post(
        "/api/v1/preseed",
        json={"memories": [{"content": "Test", "type": "fact"}]},
        headers=_auth_headers(),
    )

    assert response.status_code == 403
    assert "Internal preseed authorization required" in response.get_json()["message"]


def test_api_v1_stats_returns_dashboard_shape(client, monkeypatch):
    state = _make_state()
    state.memory_graph.memories["11111111-1111-1111-1111-111111111111"] = {
        "id": "11111111-1111-1111-1111-111111111111",
        "content": "Onboarding role memory",
        "tags": ["onboarding", "role"],
        "importance": 0.9,
        "type": "fact",
        "timestamp": "2026-04-01T10:00:00Z",
        "updated_at": "2026-04-02T10:00:00Z",
    }
    state.memory_graph.memories["22222222-2222-2222-2222-222222222222"] = {
        "id": "22222222-2222-2222-2222-222222222222",
        "content": "Preference memory",
        "tags": ["onboarding", "preferences"],
        "importance": 0.8,
        "type": "preference",
        "timestamp": "2026-04-03T10:00:00Z",
        "last_accessed": "2026-04-03T12:00:00Z",
    }
    state.memory_graph.relationships.append(
        {
            "id1": "11111111-1111-1111-1111-111111111111",
            "id2": "22222222-2222-2222-2222-222222222222",
            "type": "RELATES_TO",
            "strength": 0.7,
        }
    )
    monkeypatch.setattr(app, "state", state)
    monkeypatch.setattr(app, "API_TOKEN", "test-token")

    response = client.get("/api/v1/stats", headers=_auth_headers())

    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "success"
    assert body["memories_stored"] == 2
    assert body["associations"] == 1
    assert body["memory_types"] == {"fact": 1, "preference": 1}
    assert body["top_tags"][0] == {"tag": "onboarding", "count": 2}
    assert body["last_activity"] == "2026-04-03T12:00:00Z"
    assert body["graph_density"] == 0.5
