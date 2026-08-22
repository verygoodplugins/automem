import json
from queue import Empty, Queue

import pytest

import app
from automem.api import stream as stream_mod
from automem.api.stream import event_count, is_single_memory_store
from tests.support.fake_graph import FakeGraph


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    state = app.ServiceState()
    graph = FakeGraph()
    state.memory_graph = graph
    monkeypatch.setattr(app, "state", state)
    monkeypatch.setattr(app, "init_falkordb", lambda: None)
    monkeypatch.setattr(app, "init_qdrant", lambda: None)
    monkeypatch.setattr(app, "API_TOKEN", "test-token")
    monkeypatch.setattr(app, "ADMIN_TOKEN", "test-admin-token")
    yield graph


@pytest.fixture
def client():
    return app.app.test_client()


@pytest.fixture
def auth_headers():
    return {"Authorization": "Bearer test-token"}


@pytest.fixture
def sse_queue():
    queue: Queue = Queue(maxsize=100)
    with stream_mod._subscribers_lock:
        stream_mod._subscribers.append(queue)
    try:
        yield queue
    finally:
        with stream_mod._subscribers_lock:
            if queue in stream_mod._subscribers:
                stream_mod._subscribers.remove(queue)


def _drain(queue: Queue) -> list[dict]:
    events: list[dict] = []
    while True:
        try:
            raw = queue.get_nowait()
        except Empty:
            break
        assert raw.startswith("data: ")
        events.append(json.loads(raw[len("data: ") :].strip()))
    return events


def _store(client, auth_headers, content: str, **fields) -> str:
    payload = {"content": content, **fields}
    response = client.post(
        "/memory",
        data=json.dumps(payload),
        content_type="application/json",
        headers=auth_headers,
    )
    assert response.status_code == 201, response.get_json()
    return response.get_json()["memory_id"]


def test_emit_event_delivers_json_payload_to_subscribers(sse_queue):
    stream_mod.emit_event("memory.store", {"id": "abc"}, lambda: "2026-08-23T00:00:00Z")
    events = _drain(sse_queue)
    assert events == [
        {
            "type": "memory.store",
            "timestamp": "2026-08-23T00:00:00Z",
            "data": {"id": "abc"},
        }
    ]


def test_store_emits_memory_store_event(client, auth_headers, sse_queue):
    memory_id = _store(
        client,
        auth_headers,
        "SSE store coverage probe",
        tags=["sse", "automem"],
        importance=0.8,
        type="Insight",
    )
    events = [event for event in _drain(sse_queue) if event["type"] == "memory.store"]
    assert len(events) == 1
    data = events[0]["data"]
    assert data["id"] == memory_id
    assert data["content_preview"] == "SSE store coverage probe"
    assert data["type"] == "Insight"
    assert data["importance"] == 0.8
    assert data["tags"][:2] == ["sse", "automem"]
    assert data["size_bytes"] == len("SSE store coverage probe".encode("utf-8"))
    assert "elapsed_ms" in data


def test_store_event_size_bytes_uses_utf8_length(client, auth_headers, sse_queue):
    memory_id = _store(client, auth_headers, "café")
    events = [event for event in _drain(sse_queue) if event["type"] == "memory.store"]
    assert events[0]["data"]["id"] == memory_id
    assert events[0]["data"]["size_bytes"] == len("café".encode("utf-8"))
    assert events[0]["data"]["size_bytes"] > len("café")


def test_failed_store_does_not_emit(client, auth_headers, sse_queue):
    response = client.post(
        "/memory",
        data=json.dumps({}),
        content_type="application/json",
        headers=auth_headers,
    )
    assert response.status_code == 400
    assert _drain(sse_queue) == []


def test_recall_emits_memory_recall_event(client, auth_headers, sse_queue):
    response = client.get("/recall?query=hello+world&limit=7&tags=sse", headers=auth_headers)
    assert response.status_code == 200
    events = [event for event in _drain(sse_queue) if event["type"] == "memory.recall"]
    assert len(events) == 1
    data = events[0]["data"]
    assert data["query"] == "hello world"
    assert data["limit"] == 7
    assert data["result_count"] == 0
    assert data["tags"] == ["sse"]
    assert "elapsed_ms" in data


def test_recall_event_includes_multi_query_text(client, auth_headers, sse_queue):
    response = client.get("/recall?queries=alpha&queries=beta", headers=auth_headers)
    assert response.status_code == 200
    events = [event for event in _drain(sse_queue) if event["type"] == "memory.recall"]
    assert len(events) == 1
    assert events[0]["data"]["query"] == "alpha | beta"


def test_associate_emits_memory_associate_event(client, auth_headers, sse_queue):
    memory1_id = _store(client, auth_headers, "Source memory for SSE associate")
    memory2_id = _store(client, auth_headers, "Target memory for SSE associate")
    _drain(sse_queue)

    response = client.post(
        "/associate",
        data=json.dumps(
            {
                "memory1_id": memory1_id,
                "memory2_id": memory2_id,
                "type": "RELATES_TO",
                "strength": 0.9,
            }
        ),
        content_type="application/json",
        headers=auth_headers,
    )
    assert response.status_code == 201
    events = [event for event in _drain(sse_queue) if event["type"] == "memory.associate"]
    assert len(events) == 1
    data = events[0]["data"]
    assert data["memory1_id"] == memory1_id
    assert data["memory2_id"] == memory2_id
    assert data["relation_type"] == "RELATES_TO"
    assert data["strength"] == 0.9
    assert data["count"] == 1


def test_failed_associate_does_not_emit(client, auth_headers, sse_queue):
    same_id = "a0000000-0000-0000-0000-000000000001"
    response = client.post(
        "/associate",
        data=json.dumps({"memory1_id": same_id, "memory2_id": same_id}),
        content_type="application/json",
        headers=auth_headers,
    )
    assert response.status_code == 400
    assert _drain(sse_queue) == []


def test_batch_associate_emits_one_summary_event(client, auth_headers, sse_queue):
    first_id = _store(client, auth_headers, "Batch associate source")
    second_id = _store(client, auth_headers, "Batch associate target")
    third_id = _store(client, auth_headers, "Batch associate extra")
    _drain(sse_queue)

    response = client.post(
        "/associate",
        data=json.dumps(
            {
                "associations": [
                    {
                        "memory1_id": first_id,
                        "memory2_id": second_id,
                        "type": "RELATES_TO",
                        "strength": 0.8,
                    },
                    {
                        "memory1_id": second_id,
                        "memory2_id": third_id,
                        "type": "LEADS_TO",
                        "strength": 0.7,
                    },
                ]
            }
        ),
        content_type="application/json",
        headers=auth_headers,
    )
    assert response.status_code == 201
    events = [event for event in _drain(sse_queue) if event["type"] == "memory.associate"]
    assert len(events) == 1
    data = events[0]["data"]
    assert data["count"] == 2
    assert data["failed_count"] == 0
    assert sorted(data["relation_types"]) == ["LEADS_TO", "RELATES_TO"]


def test_update_emits_memory_update_event(client, auth_headers, sse_queue):
    memory_id = _store(client, auth_headers, "Original content", tags=["sse"])
    _drain(sse_queue)

    response = client.patch(
        f"/memory/{memory_id}",
        data=json.dumps({"content": "Updated content", "importance": 0.95}),
        content_type="application/json",
        headers=auth_headers,
    )
    assert response.status_code == 200
    events = [event for event in _drain(sse_queue) if event["type"] == "memory.update"]
    assert len(events) == 1
    data = events[0]["data"]
    assert data["id"] == memory_id
    assert data["content_preview"] == "Updated content"
    assert "content" in data["fields"]
    assert "importance" in data["fields"]


def test_delete_emits_memory_delete_event(client, auth_headers, sse_queue):
    memory_id = _store(client, auth_headers, "Delete me")
    _drain(sse_queue)

    response = client.delete(f"/memory/{memory_id}", headers=auth_headers)
    assert response.status_code == 200
    events = [event for event in _drain(sse_queue) if event["type"] == "memory.delete"]
    assert len(events) == 1
    assert events[0]["data"]["id"] == memory_id
    assert events[0]["data"]["count"] == 1


def test_batch_store_emits_one_store_event(client, auth_headers, sse_queue):
    response = client.post(
        "/memory/batch",
        json={
            "memories": [
                {"content": "Batch SSE one", "tags": ["sse"]},
                {"content": "Batch SSE two", "tags": ["sse"]},
            ]
        },
        headers=auth_headers,
    )
    assert response.status_code == 201, response.get_json()
    events = [event for event in _drain(sse_queue) if event["type"] == "memory.store"]
    assert len(events) == 1
    data = events[0]["data"]
    assert data["count"] == 2
    assert data["content_preview"].startswith("2 memories stored")


def test_delete_by_tag_emits_memory_delete_event(client, auth_headers, sse_queue):
    _store(client, auth_headers, "Tagged for bulk delete", tags=["sse-bulk"])
    _drain(sse_queue)

    response = client.delete("/memory/by-tag?tags=sse-bulk", headers=auth_headers)
    assert response.status_code == 200
    body = response.get_json()
    assert body["deleted_count"] == 1
    events = [event for event in _drain(sse_queue) if event["type"] == "memory.delete"]
    assert len(events) == 1
    assert events[0]["data"]["count"] == 1
    assert events[0]["data"]["tags"] == ["sse-bulk"]


def test_event_count_preserves_explicit_zero():
    assert event_count({}) == 1
    assert event_count({"count": 2}) == 2
    assert event_count({"count": 0}) == 0
    assert is_single_memory_store({"id": "abc", "count": 1}) is True
    assert is_single_memory_store({"count": 2, "ids": ["a", "b"]}) is False
    assert is_single_memory_store({"content_preview": "2 memories stored"}) is False


def test_stream_status_reports_subscriber_count(client, auth_headers, sse_queue):
    response = client.get("/stream/status", headers=auth_headers)
    assert response.status_code == 200
    assert response.get_json()["subscribers"] >= 1
