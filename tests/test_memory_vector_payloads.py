"""Exercise the registered memory blueprint's three vector payload writers."""

from tests.test_api_endpoints import (  # noqa: F401
    FakeEmbeddingProvider,
    auth_headers,
    client,
    mock_state,
)

STATE_KEYS = {"t_valid", "t_invalid", "archived", "tag_prefixes", "relevance_score"}


def test_create_vector_payload(client, mock_state, auth_headers):
    mock_state.effective_vector_size = 768
    response = client.post(
        "/memory",
        headers=auth_headers,
        json={
            "content": "alpha",
            "type": "Insight",
            "tags": ["auth/login"],
            "embedding": [0.1] * 768,
            "t_invalid": "2026-10-22T00:00:00Z",
        },
    )
    assert response.status_code == 201, response.json
    payload = next(iter(mock_state.qdrant.points.values()))["payload"]
    assert STATE_KEYS <= payload.keys()
    assert payload["t_invalid"] == "2026-10-22T00:00:00+00:00"
    assert payload["tag_prefixes"] == ["auth", "auth:login"]


def test_patch_archived_vector_stays_archived(client, mock_state, auth_headers):
    memory_id = "00000000-0000-0000-0000-000000000001"
    mock_state.memory_graph.memories[memory_id] = {
        "id": memory_id,
        "content": "alpha",
        "tags": ["auth/login"],
        "archived": True,
        "relevance_score": 0.3,
        "t_invalid": "2026-10-22T00:00:00Z",
    }
    mock_state.qdrant.points[memory_id] = {"vector": [0.1] * 768, "payload": {}}
    response = client.patch(f"/memory/{memory_id}", headers=auth_headers, json={"importance": 0.8})
    assert response.status_code == 200, response.json
    payload = mock_state.qdrant.points[memory_id]["payload"]
    assert STATE_KEYS <= payload.keys()
    assert payload["archived"] is True
    assert payload["relevance_score"] == 0.3


def test_batch_vector_payload(client, mock_state, auth_headers):
    mock_state.effective_vector_size = 768
    mock_state.embedding_provider = FakeEmbeddingProvider([[0.1] * 768])
    response = client.post(
        "/memory/batch",
        headers=auth_headers,
        json={"memories": [{"content": "alpha", "type": "Insight", "tags": ["auth/login"]}]},
    )
    assert response.status_code == 201, response.json
    payload = next(iter(mock_state.qdrant.points.values()))["payload"]
    assert STATE_KEYS <= payload.keys()
    assert payload["t_valid"] == payload["timestamp"]
    assert payload["archived"] is False


def test_embedding_failures_visible_in_health(mock_state):
    from flask import Flask

    from automem.api.health import create_health_blueprint

    app = Flask(__name__)
    app.register_blueprint(
        create_health_blueprint(
            get_memory_graph=lambda: None,
            get_qdrant_client=lambda: None,
            state=mock_state,
            graph_name="custom",
            collection_name="custom",
            utc_now=lambda: "now",
        )
    )
    client = app.test_client()
    mock_state.embedding_stats.update(failures=3, exhausted=1, retries=2, last_error="unavailable")
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json["embedding"]["failures"] == 3
    assert response.json["embedding"]["exhausted"] == 1
    assert response.json["embedding"]["last_error"] == "unavailable"
