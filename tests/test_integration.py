"""Integration tests that run against real Docker services.

These tests are opt-in. Enable with:
  AUTOMEM_RUN_INTEGRATION_TESTS=1

Optional helpers:
  AUTOMEM_START_DOCKER=1     # start docker compose automatically
  AUTOMEM_STOP_DOCKER=1      # stop docker compose after tests (default 1)
  AUTOMEM_TEST_BASE_URL=...  # override API base URL (default http://localhost:8001)
  AUTOMEM_ALLOW_LIVE=1       # required when AUTOMEM_TEST_BASE_URL is not localhost
"""

import json
import os
import re
import subprocess
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import pytest
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_INTEGRATION_ENABLED = os.getenv("AUTOMEM_RUN_INTEGRATION_TESTS")
pytestmark = [pytest.mark.integration]
if not _INTEGRATION_ENABLED:
    pytestmark.append(
        pytest.mark.skip(
            reason="Integration tests disabled. Set AUTOMEM_RUN_INTEGRATION_TESTS=1 to enable."
        )
    )

# Resolve test base URL and safety for live targets
_DEFAULT_BASE = "http://localhost:8001"
_BASE_URL = os.getenv("AUTOMEM_TEST_BASE_URL", _DEFAULT_BASE)
_parsed = urlparse(_BASE_URL)
_is_local = _parsed.hostname in {"localhost", "127.0.0.1"}
if _INTEGRATION_ENABLED and not _is_local and os.getenv("AUTOMEM_ALLOW_LIVE") != "1":
    pytestmark.append(
        pytest.mark.skip(
            reason=(
                f"Live server tests skipped for {_BASE_URL}. Set AUTOMEM_ALLOW_LIVE=1 to enable "
                "against a non-local endpoint."
            )
        )
    )


@pytest.fixture(scope="session", autouse=True)
def maybe_start_docker():
    """Optionally start/stop docker-compose for the integration suite.

    - Starts only when AUTOMEM_START_DOCKER=1 and the API is not already healthy.
    - Stops after the session by default (unless AUTOMEM_STOP_DOCKER is 0/false).
    """
    if not _INTEGRATION_ENABLED:
        yield
        return

    root = Path(__file__).resolve().parents[1]
    start = os.getenv("AUTOMEM_START_DOCKER") == "1"
    stop_flag = os.getenv("AUTOMEM_STOP_DOCKER", "1").lower() not in {"0", "false", "no"}

    # Quick health probe to detect an already-running API
    def _api_healthy() -> bool:
        try:
            r = requests.get(f"{_BASE_URL}/health", timeout=2)
            return r.status_code == 200
        except requests.RequestException:
            return False

    started_here = False
    if start and not _api_healthy():
        try:
            subprocess.run(["docker", "compose", "up", "-d"], cwd=root, check=True)
            started_here = True
        except Exception:
            # If docker isn't available, fall back and let the health loop skip later
            started_here = False

    try:
        yield
    finally:
        if started_here and stop_flag:
            try:
                subprocess.run(["docker", "compose", "down"], cwd=root, check=True)
            except Exception:
                pass


@pytest.fixture(scope="module")
def api_client():
    """Create a session with retries for the real API."""
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Set auth header for all requests (configurable)
    api_token = os.getenv("AUTOMEM_TEST_API_TOKEN", "test-token")
    session.headers.update(
        {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }
    )

    # Base URL for the API (override via AUTOMEM_TEST_BASE_URL)
    session.base_url = _BASE_URL

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
        "Authorization": f"Bearer {os.getenv('AUTOMEM_TEST_API_TOKEN', 'test-token')}",
        "X-Admin-Token": os.getenv("AUTOMEM_TEST_ADMIN_TOKEN", "test-admin-token"),
        "Content-Type": "application/json",
    }


def test_health_check_real(api_client):
    """Test health endpoint returns correct status."""
    response = api_client.get(f"{api_client.base_url}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in {"healthy", "degraded"}
    assert "timestamp" in data
    assert data.get("falkordb") == "connected"
    assert data.get("qdrant") == "connected"


def test_store_then_recall_real_smoke(api_client):
    """Store then recall should return the same memory id/content."""
    memory_content = f"Store-then-recall smoke {uuid.uuid4()}"
    unique_tag = f"itest-smoke-{uuid.uuid4().hex[:10]}"
    store_response = api_client.post(
        f"{api_client.base_url}/memory",
        json={
            "content": memory_content,
            "tags": ["test", "integration", "smoke", unique_tag],
            "importance": 0.7,
        },
    )
    assert store_response.status_code == 201
    memory_id = store_response.json()["memory_id"]

    found = False
    for _ in range(5):
        recall_response = api_client.get(
            f"{api_client.base_url}/recall",
            params={"query": memory_content, "tags": unique_tag, "limit": 50},
        )
        assert recall_response.status_code == 200
        recall_data = recall_response.json()
        for result in recall_data.get("results", []):
            if result.get("id") == memory_id:
                assert result.get("memory", {}).get("content") == memory_content
                found = True
                break
        if found:
            break
        time.sleep(0.5)

    assert found, f"Stored memory {memory_id} was not recalled"

    api_client.delete(f"{api_client.base_url}/memory/{memory_id}")


def test_recall_jit_related_fields_real(api_client):
    """Recall response should remain compatible with optional JIT enrichment fields."""
    memory_content = f"JIT compatibility integration {uuid.uuid4()}"
    unique_tag = f"itest-jit-{uuid.uuid4().hex[:10]}"
    store_response = api_client.post(
        f"{api_client.base_url}/memory",
        json={
            "content": memory_content,
            "tags": ["test", "integration", "jit-check", unique_tag],
            "importance": 0.65,
        },
    )
    assert store_response.status_code == 201
    memory_id = store_response.json()["memory_id"]

    data = {}
    matching = []
    for _ in range(5):
        response = api_client.get(
            f"{api_client.base_url}/recall",
            params={"query": memory_content, "tags": unique_tag, "limit": 50},
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "success"
        matching = [result for result in data.get("results", []) if result.get("id") == memory_id]
        if matching:
            break
        time.sleep(0.5)

    if "jit_enriched_count" in data:
        assert isinstance(data["jit_enriched_count"], int)
        assert data["jit_enriched_count"] >= 0

    assert matching, f"Stored memory {memory_id} missing from recall response"

    sample = matching[0]
    assert sample.get("memory", {}).get("content") == memory_content
    if "jit_enriched" in sample:
        assert isinstance(sample["jit_enriched"], bool)

    api_client.delete(f"{api_client.base_url}/memory/{memory_id}")


def test_memory_lifecycle_real(api_client):
    """Test complete memory lifecycle: create, recall, update, delete."""
    # 1. Store a memory
    memory_content = f"Integration test memory {uuid.uuid4()}"
    unique_tag = f"itest-lifecycle-{uuid.uuid4().hex[:10]}"
    store_response = api_client.post(
        f"{api_client.base_url}/memory",
        json={
            "content": memory_content,
            "tags": ["test", "integration", unique_tag],
            "importance": 0.8,
            "metadata": {"source": "integration_test"},
        },
    )
    assert store_response.status_code == 201
    store_data = store_response.json()
    assert "memory_id" in store_data
    memory_id = store_data["memory_id"]

    # 2. Recall the memory by content (with retry for async embedding)
    import time

    found = False
    for attempt in range(5):  # Retry up to 5 times
        recall_response = api_client.get(
            f"{api_client.base_url}/recall",
            params={"query": memory_content, "tags": unique_tag, "limit": 50},
        )
        assert recall_response.status_code == 200
        recall_data = recall_response.json()
        assert recall_data["status"] == "success"

        # Find our memory in results
        for result in recall_data["results"]:
            if result["id"] == memory_id:
                found = True
                assert result["memory"]["content"] == memory_content
                assert "test" in result["memory"]["tags"]
                break

        if found:
            break

        # Wait before retrying (embeddings are async)
        if attempt < 4:
            time.sleep(0.5)

    assert found, f"Memory {memory_id} not found in recall results after {attempt + 1} attempts"

    # 3. Update the memory
    updated_content = f"Updated: {memory_content}"
    update_response = api_client.patch(
        f"{api_client.base_url}/memory/{memory_id}",
        json={"content": updated_content, "importance": 0.9},
    )
    assert update_response.status_code == 200
    update_data = update_response.json()
    assert update_data["status"] == "success"

    # 4. Verify update by recalling
    recall2_response = api_client.get(
        f"{api_client.base_url}/recall",
        params={"query": updated_content, "tags": unique_tag, "limit": 50},
    )
    assert recall2_response.status_code == 200
    recall2_data = recall2_response.json()

    found_updated = False
    for result in recall2_data["results"]:
        if result["id"] == memory_id:
            found_updated = True
            assert result["memory"]["content"] == updated_content
            assert result["memory"]["importance"] == 0.9
            break
    assert found_updated, "Updated memory not found"

    # 5. Delete the memory
    delete_response = api_client.delete(f"{api_client.base_url}/memory/{memory_id}")
    assert delete_response.status_code == 200
    delete_data = delete_response.json()
    assert delete_data["status"] == "success"

    # 6. Verify deletion
    recall3_response = api_client.get(
        f"{api_client.base_url}/recall", params={"query": updated_content, "limit": 5}
    )
    assert recall3_response.status_code == 200
    recall3_data = recall3_response.json()

    # Should not find the deleted memory
    for result in recall3_data["results"]:
        assert result["id"] != memory_id


def test_association_real(api_client):
    """Test creating associations between memories."""
    # Create two memories
    memory1_response = api_client.post(
        f"{api_client.base_url}/memory",
        json={"content": f"Association test memory 1 - {uuid.uuid4()}", "importance": 0.7},
    )
    assert memory1_response.status_code == 201
    memory1_id = memory1_response.json()["memory_id"]

    memory2_response = api_client.post(
        f"{api_client.base_url}/memory",
        json={"content": f"Association test memory 2 - {uuid.uuid4()}", "importance": 0.7},
    )
    assert memory2_response.status_code == 201
    memory2_id = memory2_response.json()["memory_id"]

    # Create association
    assoc_response = api_client.post(
        f"{api_client.base_url}/associate",
        json={
            "memory1_id": memory1_id,
            "memory2_id": memory2_id,
            "type": "RELATES_TO",
            "strength": 0.8,
        },
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
                "importance": 0.5 + (i * 0.1),
            },
        )
        assert response.status_code == 201
        memory_ids.append(response.json()["memory_id"])

    # Filter by tag
    tag_response = api_client.get(
        f"{api_client.base_url}/memory/by-tag", params={"tags": unique_tag, "limit": 10}
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
            "importance": 0.6,
        },
    )
    assert memory_response.status_code == 201
    memory_id = memory_response.json()["memory_id"]

    # Recall with time range that includes the memory
    start_time = (now - timedelta(hours=1)).isoformat()
    end_time = (now + timedelta(hours=1)).isoformat()

    recall_response = api_client.get(
        f"{api_client.base_url}/recall", params={"start": start_time, "end": end_time, "limit": 10}
    )
    assert recall_response.status_code == 200
    recall_data = recall_response.json()

    # Check if our memory is in the results
    found = any(r["id"] == memory_id for r in recall_data.get("results", []))
    assert found, "Memory not found in time range"

    # Clean up
    api_client.delete(f"{api_client.base_url}/memory/{memory_id}")


def test_embedding_handling_real(api_client):
    """Test that embeddings are handled correctly."""
    # Create a 768-dimensional embedding
    embedding = [0.1] * 768

    payload = {
        "content": f"Memory with embedding {uuid.uuid4()}",
        "embedding": embedding,
        "importance": 0.7,
    }

    memory_response = api_client.post(f"{api_client.base_url}/memory", json=payload)
    if memory_response.status_code == 400:
        body = memory_response.json()
        message = str(body.get("message", ""))
        match = re.search(r"exactly\s+(\d+)\s+values", message, flags=re.IGNORECASE)
        if match:
            adjusted_dim = int(match.group(1))
            print(f"Adjusting embedding dimension from {len(embedding)} to {adjusted_dim}")
            payload["embedding"] = [0.1] * adjusted_dim
            memory_response = api_client.post(f"{api_client.base_url}/memory", json=payload)

    assert memory_response.status_code == 201
    memory_id = memory_response.json()["memory_id"]

    # Clean up
    api_client.delete(f"{api_client.base_url}/memory/{memory_id}")


def test_admin_reembed_real(api_client, admin_headers):
    """Test admin re-embedding endpoint."""
    # First create a memory without embedding
    memory_response = api_client.post(
        f"{api_client.base_url}/memory",
        json={"content": f"Memory for re-embedding {uuid.uuid4()}", "importance": 0.6},
    )
    assert memory_response.status_code == 201
    memory_id = memory_response.json()["memory_id"]

    # Try re-embedding (will use placeholder if no OpenAI key)
    reembed_response = requests.post(
        f"{api_client.base_url}/admin/reembed",
        headers=admin_headers,
        json={"limit": 1, "batch_size": 1},
    )

    # Should either succeed or return 503 if OpenAI not configured
    assert reembed_response.status_code in [200, 503]

    # Clean up
    api_client.delete(f"{api_client.base_url}/memory/{memory_id}")


def test_error_handling_real(api_client):
    """Test various error conditions."""
    # 1. Non-existent memory delete should 404
    response = api_client.delete(f"{api_client.base_url}/memory/{uuid.uuid4()}")
    assert response.status_code == 404

    # 2. Missing required field
    response = api_client.post(
        f"{api_client.base_url}/memory", json={"importance": 0.5}  # Missing 'content'
    )
    assert response.status_code == 400

    # 3. Invalid association (same memory)
    memory_response = api_client.post(
        f"{api_client.base_url}/memory", json={"content": "Test memory for association"}
    )
    memory_id = memory_response.json()["memory_id"]

    assoc_response = api_client.post(
        f"{api_client.base_url}/associate",
        json={
            "memory1_id": memory_id,
            "memory2_id": memory_id,  # Same as memory1
            "type": "RELATES_TO",
            "strength": 0.5,
        },
    )
    assert assoc_response.status_code == 400

    # Clean up
    api_client.delete(f"{api_client.base_url}/memory/{memory_id}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
