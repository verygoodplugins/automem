"""Regression coverage for explicit types on the registered memory write routes."""

import logging

import pytest

import app
from tests.test_api_endpoints import (  # noqa: F401 - shared pytest fixtures
    FakeEmbeddingProvider,
    auth_headers,
    client,
    mock_state,
)


@pytest.fixture
def typed_api(client, mock_state, auth_headers, monkeypatch):
    mock_state.effective_vector_size = 768
    mock_state.embedding_provider = FakeEmbeddingProvider([[0.1] * 768 for _ in range(10)])
    monkeypatch.setattr(app.memory_classifier, "classify", lambda _: ("Context", 0.7))
    return client, mock_state, auth_headers


def write_type(api, operation, value, *, omitted=False):
    client, state, headers = api
    payload = {} if omitted else {"type": value}
    if operation == "patch":
        created = client.post(
            "/memory",
            json={"content": "existing memory", "type": "Decision", "embedding": [0.1] * 768},
            headers=headers,
        )
        assert created.status_code == 201
        memory_id = created.json["memory_id"]
        response = client.patch(f"/memory/{memory_id}", json=payload, headers=headers)
    elif operation == "batch":
        response = client.post(
            "/memory/batch",
            json={"memories": [{"content": "batch memory", **payload}]},
            headers=headers,
        )
        memory_id = response.json.get("memory_ids", [None])[0]
    else:
        response = client.post(
            "/memory",
            json={"content": "new memory", "embedding": [0.1] * 768, **payload},
            headers=headers,
        )
        memory_id = response.json.get("memory_id")
    return response, memory_id


@pytest.mark.parametrize("operation", ["post", "patch", "batch"])
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("insight", "Insight"),
        ("decision", "Decision"),
        ("analysis", "Insight"),
        ("Pattern", "Pattern"),
    ],
)
def test_known_type_aliases_persist_canonically(typed_api, operation, raw, expected):
    response, memory_id = write_type(typed_api, operation, raw)
    assert response.status_code in (200, 201), response.json
    _, state, _ = typed_api
    assert state.memory_graph.memories[memory_id]["type"] == expected
    assert state.qdrant.points[memory_id]["payload"]["type"] == expected


@pytest.mark.parametrize("operation", ["post", "patch", "batch"])
def test_custom_string_type_is_preserved_with_warning(typed_api, operation, caplog):
    with caplog.at_level(logging.WARNING):
        response, memory_id = write_type(typed_api, operation, "CustomTicket")
    assert response.status_code in (200, 201), response.json
    assert typed_api[1].memory_graph.memories[memory_id]["type"] == "CustomTicket"
    assert typed_api[1].qdrant.points[memory_id]["payload"]["type"] == "CustomTicket"
    assert "custom memory type" in caplog.text.lower()


@pytest.mark.parametrize("operation", ["post", "patch", "batch"])
@pytest.mark.parametrize("raw", [None, False, 3, [], {"type": "Insight"}, "", "  "])
def test_invalid_explicit_types_return_400_before_mutation(typed_api, operation, raw):
    response, memory_id = write_type(typed_api, operation, raw)
    assert response.status_code == 400, response.json
    assert "'type' must be a non-empty string" in response.json["message"]
    state = typed_api[1]
    if operation == "patch":
        assert state.memory_graph.memories[memory_id]["type"] == "Decision"
        assert state.qdrant.points[memory_id]["payload"]["type"] == "Decision"
    else:
        assert not state.memory_graph.memories
        assert not state.qdrant.points


@pytest.mark.parametrize("operation", ["post", "patch", "batch"])
def test_omitted_type_retains_classification_or_existing_value(typed_api, operation):
    response, memory_id = write_type(typed_api, operation, None, omitted=True)
    assert response.status_code in (200, 201), response.json
    expected = "Decision" if operation == "patch" else "Context"
    assert typed_api[1].memory_graph.memories[memory_id]["type"] == expected


def test_invalid_type_later_in_batch_leaves_all_items_unwritten(typed_api):
    client, state, headers = typed_api
    response = client.post(
        "/memory/batch",
        json={
            "memories": [
                {"content": "valid first", "type": "decision"},
                {"content": "invalid second", "type": []},
            ]
        },
        headers=headers,
    )
    assert response.status_code == 400
    assert not state.memory_graph.memories
    assert not state.qdrant.points
