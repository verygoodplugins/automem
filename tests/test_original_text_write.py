"""Write and update contract for the internal original-text search property."""

import json

import pytest

from automem import config
from automem.utils.graph import _serialize_node
from tests.support.fake_graph import FakeGraph, FakeNode
from tests.test_api_endpoints import (  # noqa: F401
    FakeEmbeddingProvider,
    auth_headers,
    client,
    mock_state,
)


class OriginalTextGraph(FakeGraph):
    """Extend the shared graph double with the derived field and index label."""

    def __init__(self):
        super().__init__()
        self.original_text_labels = set()

    def query(self, query, params=None, **kwargs):
        result = super().query(query, params, **kwargs)
        params = params or {}
        if "SET m:MemoryOriginalText" in query and params.get("id") in self.memories:
            memory_id = params["id"]
            self.memories[memory_id]["original_search_text"] = params["original_search_text"]
            self.original_text_labels.add(memory_id)
        if "SET node:MemoryOriginalText" in query:
            for memory in params.get("memories", []):
                if memory["id"] in self.memories:
                    self.memories[memory["id"]]["original_search_text"] = memory[
                        "original_search_text"
                    ]
                    self.original_text_labels.add(memory["id"])
        return result


@pytest.fixture
def indexed_state(mock_state):
    mock_state.memory_graph = OriginalTextGraph()
    return mock_state


@pytest.fixture
def summarizer(monkeypatch):
    monkeypatch.setattr("automem.api.memory.MEMORY_CONTENT_SOFT_LIMIT", 50)
    monkeypatch.setattr("automem.api.memory.MEMORY_CONTENT_HARD_LIMIT", 2000)
    monkeypatch.setattr("automem.api.memory.MEMORY_AUTO_SUMMARIZE", True)
    monkeypatch.setattr("automem.api.memory.summarize_content", lambda *_args: "Brief summary.")


def store_summarized(client, indexed_state, auth_headers):
    original = "Detailed saved context " * 6 + "identifier hidden at the end"
    response = client.post(
        "/memory",
        json={
            "content": original,
            "metadata": {"source": "test"},
            "type": "Decision",
            "embedding": [0.1] * config.VECTOR_SIZE,
        },
        headers=auth_headers,
    )
    assert response.status_code == 201
    memory_id = response.get_json()["memory_id"]
    return memory_id, original, indexed_state.memory_graph.memories[memory_id]


def test_single_summarized_write_indexes_original_without_public_field(
    client, indexed_state, auth_headers, summarizer
):
    memory_id, original, stored = store_summarized(client, indexed_state, auth_headers)
    assert stored["content"] == "Brief summary."
    assert stored["original_search_text"] == original
    assert memory_id in indexed_state.memory_graph.original_text_labels
    metadata = json.loads(stored["metadata"])
    assert metadata == {
        "source": "test",
        "original_content": original,
        "was_summarized": True,
        "original_length": len(original),
    }
    response = client.get(f"/memory/{memory_id}", headers=auth_headers)
    assert response.status_code == 200
    assert "original_search_text" not in response.get_json()["memory"]
    assert "original_search_text" not in indexed_state.qdrant.points[memory_id]["payload"]


def test_short_and_failed_summary_write_the_original_body(
    client, indexed_state, auth_headers, summarizer, monkeypatch
):
    monkeypatch.setattr("automem.api.memory.summarize_content", lambda *_args: None)
    for content in ("A short body", "long body " * 10):
        response = client.post(
            "/memory", json={"content": content, "type": "Decision"}, headers=auth_headers
        )
        assert response.status_code == 201
        memory_id = response.get_json()["memory_id"]
        stored = indexed_state.memory_graph.memories[memory_id]
        assert stored["content"] == stored["original_search_text"] == content.strip()
        assert "original_content" not in json.loads(stored["metadata"])


def test_batch_preserves_each_original_and_metadata(
    client, indexed_state, auth_headers, summarizer
):
    indexed_state.embedding_provider = FakeEmbeddingProvider(
        [[0.1] * config.VECTOR_SIZE, [0.2] * config.VECTOR_SIZE]
    )
    original = "Detailed batch original " * 5 + "private identifier"
    response = client.post(
        "/memory/batch",
        json={
            "memories": [
                {"content": original, "metadata": {"source": "batch"}, "type": "Decision"},
                {"content": "short body", "type": "Decision"},
            ]
        },
        headers=auth_headers,
    )
    assert response.status_code == 201
    first_id, second_id = response.get_json()["memory_ids"]
    first = indexed_state.memory_graph.memories[first_id]
    second = indexed_state.memory_graph.memories[second_id]
    assert first["content"] == "Brief summary."
    assert first["original_search_text"] == original
    assert json.loads(first["metadata"]) == {
        "source": "batch",
        "original_content": original,
        "was_summarized": True,
        "original_length": len(original),
    }
    assert second["content"] == second["original_search_text"] == "short body"
    assert {first_id, second_id} <= indexed_state.memory_graph.original_text_labels
    assert "original_search_text" not in indexed_state.qdrant.points[first_id]["payload"]


def test_patch_content_clears_stale_original_and_indexes_replacement(
    client, indexed_state, auth_headers, summarizer
):
    memory_id, original, stored = store_summarized(client, indexed_state, auth_headers)
    response = client.patch(
        f"/memory/{memory_id}",
        json={
            "content": "Replacement content",
            "metadata": {"source": "new", "original_content": original},
        },
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert stored["content"] == stored["original_search_text"] == "Replacement content"
    assert json.loads(stored["metadata"]) == {"source": "new"}
    assert indexed_state.qdrant.points[memory_id]["payload"]["metadata"] == {"source": "new"}


def test_metadata_replacement_removal_replaces_index_with_safe_content(
    client, indexed_state, auth_headers, summarizer
):
    memory_id, _, stored = store_summarized(client, indexed_state, auth_headers)
    response = client.patch(
        f"/memory/{memory_id}", json={"metadata": {"source": "new"}}, headers=auth_headers
    )
    assert response.status_code == 200
    assert stored["original_search_text"] == stored["content"] == "Brief summary."
    assert json.loads(stored["metadata"]) == {"source": "new"}


def test_unrelated_patch_retains_original_and_legacy_without_index_uses_content(
    client, indexed_state, auth_headers, summarizer
):
    memory_id, original, stored = store_summarized(client, indexed_state, auth_headers)
    response = client.patch(f"/memory/{memory_id}", json={"tags": ["new"]}, headers=auth_headers)
    assert response.status_code == 200
    assert stored["original_search_text"] == original
    del stored["original_search_text"]
    response = client.patch(f"/memory/{memory_id}", json={"importance": 0.3}, headers=auth_headers)
    assert response.status_code == 200
    assert stored["original_search_text"] == stored["content"] == "Brief summary."


def test_metadata_altering_original_is_not_trusted_as_index_text(
    client, indexed_state, auth_headers, summarizer
):
    memory_id, _, stored = store_summarized(client, indexed_state, auth_headers)
    response = client.patch(
        f"/memory/{memory_id}",
        json={"metadata": {"original_content": "Unverified other text", "was_summarized": True}},
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert stored["original_search_text"] == "Brief summary."


def test_serialize_hides_index_field_without_mutating_graph_data():
    properties = {"id": "id", "content": "summary", "original_search_text": "hidden"}
    for node in (properties, FakeNode(properties)):
        assert _serialize_node(node) == {"id": "id", "content": "summary"}
        assert properties["original_search_text"] == "hidden"


def test_invalid_batch_is_rejected_before_graph_writes(client, indexed_state, auth_headers):
    response = client.post(
        "/memory/batch", json={"memories": [{"content": ""}]}, headers=auth_headers
    )
    assert response.status_code == 400
    assert indexed_state.memory_graph.memories == {}


def test_content_patch_without_metadata_preserves_other_fields(
    client, indexed_state, auth_headers, summarizer
):
    memory_id, _, stored = store_summarized(client, indexed_state, auth_headers)
    response = client.patch(
        f"/memory/{memory_id}", json={"content": "fresh facts"}, headers=auth_headers
    )
    assert response.status_code == 200
    assert stored["original_search_text"] == "fresh facts"
    assert json.loads(stored["metadata"]) == {"source": "test"}


def test_metadata_patch_with_unchanged_original_preserves_index(
    client, indexed_state, auth_headers, summarizer
):
    memory_id, original, stored = store_summarized(client, indexed_state, auth_headers)
    metadata = json.loads(stored["metadata"])
    metadata["source"] = "another"
    response = client.patch(
        f"/memory/{memory_id}", json={"metadata": metadata}, headers=auth_headers
    )
    assert response.status_code == 200
    assert stored["original_search_text"] == original
    assert json.loads(stored["metadata"])["source"] == "another"


def test_batch_duplicate_input_produces_distinct_indexed_records(
    client, indexed_state, auth_headers, summarizer
):
    indexed_state.embedding_provider = FakeEmbeddingProvider(
        [[0.1] * config.VECTOR_SIZE, [0.2] * config.VECTOR_SIZE]
    )
    content = "identical original " * 6
    response = client.post(
        "/memory/batch",
        json={"memories": [{"content": content, "type": "Decision"}] * 2},
        headers=auth_headers,
    )
    assert response.status_code == 201
    first_id, second_id = response.get_json()["memory_ids"]
    assert first_id != second_id
    for memory_id in (first_id, second_id):
        stored = indexed_state.memory_graph.memories[memory_id]
        assert stored["original_search_text"] == content.strip()
        assert json.loads(stored["metadata"])["original_content"] == content.strip()
