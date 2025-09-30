"""Integration tests that run against real Docker services."""

import json
import time
import uuid
from datetime import datetime, timedelta, timezone

import pytest
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@pytest.fixture(scope="module")
def api_client():
    """Create a session with retries for the real API."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Set auth header for all requests
    session.headers.update({
        "Authorization": "Bearer test-token",
        "Content-Type": "application/json"
    })

    # Base URL for the API
    session.base_url = "http://localhost:8001"

    # Wait for API to be ready
    max_retries = 10
    for i in range(max_retries):
        try:
            response = session.get(f"{session.base_url}/health")
            if response.status_code == 200:
                break
        except requests.ConnectionError:
            if i == max_retries - 1:
                pytest.skip("API not available")
            time.sleep(1)

    return session


@pytest.fixture
def admin_headers():
    """Headers with admin token."""
    return {
        "Authorization": "Bearer test-token",
        "X-Admin-Token": "test-admin-token",
        "Content-Type": "application/json"
    }


def test_health_check_real(api_client):
    """Test health endpoint returns correct status."""
    response = api_client.get(f"{api_client.base_url}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert data.get("falkordb") == "connected"
    assert data.get("qdrant") == "connected"


def test_memory_lifecycle_real(api_client):
    """Test complete memory lifecycle: create, recall, update, delete."""
    # 1. Store a memory
    memory_content = f"Integration test memory {uuid.uuid4()}"
    store_response = api_client.post(
        f"{api_client.base_url}/memory",
        json={
            "content": memory_content,
            "tags": ["test", "integration"],
            "importance": 0.8,
            "metadata": {"source": "integration_test"}
        }
    )
    assert store_response.status_code == 201
    store_data = store_response.json()
    assert "id" in store_data
    memory_id = store_data["id"]

    # 2. Recall the memory by content
    recall_response = api_client.get(
        f"{api_client.base_url}/recall",
        params={"query": memory_content, "limit": 5}
    )
    assert recall_response.status_code == 200
    recall_data = recall_response.json()
    assert recall_data["status"] == "success"
    assert len(recall_data["results"]) > 0

    # Find our memory in results
    found = False
    for result in recall_data["results"]:
        if result["memory"]["id"] == memory_id:
            found = True
            assert result["memory"]["content"] == memory_content
            assert "test" in result["memory"]["tags"]
            break
    assert found, f"Memory {memory_id} not found in recall results"

    # 3. Update the memory
    updated_content = f"Updated: {memory_content}"
    update_response = api_client.patch(
        f"{api_client.base_url}/memory/{memory_id}",
        json={
            "content": updated_content,
            "importance": 0.9
        }
    )
    assert update_response.status_code == 200
    update_data = update_response.json()
    assert update_data["status"] == "success"

    # 4. Verify update by recalling
    recall2_response = api_client.get(
        f"{api_client.base_url}/recall",
        params={"query": updated_content, "limit": 5}
    )
    assert recall2_response.status_code == 200
    recall2_data = recall2_response.json()

    found_updated = False
    for result in recall2_data["results"]:
        if result["memory"]["id"] == memory_id:
            found_updated = True
            assert result["memory"]["content"] == updated_content
            assert result["memory"]["importance"] == 0.9
            break
    assert found_updated, "Updated memory not found"

    # 5. Delete the memory
    delete_response = api_client.delete(
        f"{api_client.base_url}/memory/{memory_id}"
    )
    assert delete_response.status_code == 200
    delete_data = delete_response.json()
    assert delete_data["status"] == "success"

    # 6. Verify deletion
    recall3_response = api_client.get(
        f"{api_client.base_url}/recall",
        params={"query": updated_content, "limit": 5}
    )
    assert recall3_response.status_code == 200
    recall3_data = recall3_response.json()

    # Should not find the deleted memory
    for result in recall3_data["results"]:
        assert result["memory"]["id"] != memory_id


def test_association_real(api_client):
    """Test creating associations between memories."""
    # Create two memories
    memory1_response = api_client.post(
        f"{api_client.base_url}/memory",
        json={
            "content": f"Association test memory 1 - {uuid.uuid4()}",
            "importance": 0.7
        }
    )
    assert memory1_response.status_code == 201
    memory1_id = memory1_response.json()["id"]

    memory2_response = api_client.post(
        f"{api_client.base_url}/memory",
        json={
            "content": f"Association test memory 2 - {uuid.uuid4()}",
            "importance": 0.7
        }
    )
    assert memory2_response.status_code == 201
    memory2_id = memory2_response.json()["id"]

    # Create association
    assoc_response = api_client.post(
        f"{api_client.base_url}/associate",
        json={
            "memory1_id": memory1_id,
            "memory2_id": memory2_id,
            "type": "RELATES_TO",
            "strength": 0.8
        }
    )
    assert assoc_response.status_code == 201
    assoc_data = assoc_response.json()
    assert assoc_data["relation_type"] == "RELATES_TO"
    assert assoc_data["strength"] == 0.8

    # Clean up
    api_client.delete(f"{api_client.base_url}/memory/{memory1_id}")
    api_client.delete(f"{api_client.base_url}/memory/{memory2_id}")


def test_tag_filtering_real(api_client):
    """Test filtering memories by tags."""
    unique_tag = f"tag_{uuid.uuid4().hex[:8]}"

    # Create memories with unique tag
    memory_ids = []
    for i in range(3):
        response = api_client.post(
            f"{api_client.base_url}/memory",
            json={
                "content": f"Tagged memory {i} with {unique_tag}",
                "tags": [unique_tag, f"index_{i}"],
                "importance": 0.5 + (i * 0.1)
            }
        )
        assert response.status_code == 201
        memory_ids.append(response.json()["id"])

    # Filter by tag
    tag_response = api_client.get(
        f"{api_client.base_url}/memory/by-tag",
        params={"tags": unique_tag, "limit": 10}
    )
    assert tag_response.status_code == 200
    tag_data = tag_response.json()
    assert tag_data["status"] == "success"
    assert len(tag_data["memories"]) == 3

    # Verify all memories have the tag
    for memory in tag_data["memories"]:
        assert unique_tag in memory["tags"]

    # Clean up
    for memory_id in memory_ids:
        api_client.delete(f"{api_client.base_url}/memory/{memory_id}")


def test_time_range_recall_real(api_client):
    """Test recalling memories within time ranges."""
    # Store a memory with current timestamp
    now = datetime.now(timezone.utc)
    memory_response = api_client.post(
        f"{api_client.base_url}/memory",
        json={
            "content": f"Time-based memory {uuid.uuid4()}",
            "timestamp": now.isoformat(),
            "importance": 0.6
        }
    )
    assert memory_response.status_code == 201
    memory_id = memory_response.json()["id"]

    # Recall with time range that includes the memory
    start_time = (now - timedelta(hours=1)).isoformat()
    end_time = (now + timedelta(hours=1)).isoformat()

    recall_response = api_client.get(
        f"{api_client.base_url}/recall",
        params={
            "start": start_time,
            "end": end_time,
            "limit": 10
        }
    )
    assert recall_response.status_code == 200
    recall_data = recall_response.json()

    # Check if our memory is in the results
    found = any(r["memory"]["id"] == memory_id for r in recall_data.get("results", []))
    assert found, "Memory not found in time range"

    # Clean up
    api_client.delete(f"{api_client.base_url}/memory/{memory_id}")


def test_embedding_handling_real(api_client):
    """Test that embeddings are handled correctly."""
    # Create a 768-dimensional embedding
    embedding = [0.1] * 768

    memory_response = api_client.post(
        f"{api_client.base_url}/memory",
        json={
            "content": f"Memory with embedding {uuid.uuid4()}",
            "embedding": embedding,
            "importance": 0.7
        }
    )
    assert memory_response.status_code == 201
    memory_id = memory_response.json()["id"]

    # Clean up
    api_client.delete(f"{api_client.base_url}/memory/{memory_id}")


def test_admin_reembed_real(api_client, admin_headers):
    """Test admin re-embedding endpoint."""
    # First create a memory without embedding
    memory_response = api_client.post(
        f"{api_client.base_url}/memory",
        json={
            "content": f"Memory for re-embedding {uuid.uuid4()}",
            "importance": 0.6
        }
    )
    assert memory_response.status_code == 201
    memory_id = memory_response.json()["id"]

    # Try re-embedding (will use placeholder if no OpenAI key)
    reembed_response = requests.post(
        f"{api_client.base_url}/admin/reembed",
        headers=admin_headers,
        json={
            "limit": 1,
            "batch_size": 1
        }
    )

    # Should either succeed or return 503 if OpenAI not configured
    assert reembed_response.status_code in [200, 503]

    # Clean up
    api_client.delete(f"{api_client.base_url}/memory/{memory_id}")


def test_error_handling_real(api_client):
    """Test various error conditions."""
    # 1. Invalid memory ID
    response = api_client.get(
        f"{api_client.base_url}/memory/invalid-uuid-format"
    )
    assert response.status_code == 404

    # 2. Missing required field
    response = api_client.post(
        f"{api_client.base_url}/memory",
        json={"importance": 0.5}  # Missing 'content'
    )
    assert response.status_code == 400

    # 3. Invalid association (same memory)
    memory_response = api_client.post(
        f"{api_client.base_url}/memory",
        json={"content": "Test memory for association"}
    )
    memory_id = memory_response.json()["id"]

    assoc_response = api_client.post(
        f"{api_client.base_url}/associate",
        json={
            "memory1_id": memory_id,
            "memory2_id": memory_id,  # Same as memory1
            "type": "RELATES_TO",
            "strength": 0.5
        }
    )
    assert assoc_response.status_code == 400

    # Clean up
    api_client.delete(f"{api_client.base_url}/memory/{memory_id}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])