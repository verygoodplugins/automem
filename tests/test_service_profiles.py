from __future__ import annotations

import gzip
import json
from types import SimpleNamespace

import pytest

import app
from automem.config import resolve_runtime_profile
from tests.support.fake_graph import FakeGraph


class DummyEmbeddingProvider:
    def provider_name(self) -> str:
        return "openai"


class ExportQdrantClient:
    def __init__(self) -> None:
        self.points: dict[str, dict[str, object]] = {}

    def upsert(self, collection_name, points):
        del collection_name
        for point in points:
            self.points[str(point.id)] = {
                "payload": point.payload,
                "vector": point.vector,
            }

    def scroll(self, collection_name, limit=100, offset=None, with_payload=True, with_vectors=True):
        del collection_name, limit, offset, with_payload, with_vectors
        rows = []
        for point_id, point in self.points.items():
            rows.append(
                SimpleNamespace(
                    id=point_id,
                    payload=point["payload"],
                    vector=point["vector"],
                )
            )
        return rows, None


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    state = app.ServiceState()
    state.memory_graph = FakeGraph()
    state.qdrant = ExportQdrantClient()
    state.embedding_provider = DummyEmbeddingProvider()
    monkeypatch.setattr(app, "state", state)
    monkeypatch.setattr(app, "init_falkordb", lambda: None)
    monkeypatch.setattr(app, "init_qdrant", lambda: None)
    monkeypatch.setattr(app, "API_TOKEN", "test-token")
    monkeypatch.setattr(app, "ADMIN_TOKEN", "test-admin-token")
    return state


@pytest.fixture
def client():
    app.app.config["TESTING"] = True
    return app.app.test_client()


@pytest.fixture
def auth_headers():
    return {"Authorization": "Bearer test-token"}


@pytest.fixture
def admin_headers():
    return {"Authorization": "Bearer test-token", "X-Admin-Token": "test-admin-token"}


def test_resolve_runtime_profile_defaults_and_overrides() -> None:
    trial = resolve_runtime_profile({"AUTOMEM_SERVICE_TIER": "trial"})
    assert trial.tier == "trial"
    assert trial.mode == "active"
    assert trial.writes_enabled is True
    assert trial.self_service_export_enabled is False

    archived = resolve_runtime_profile(
        {
            "AUTOMEM_SERVICE_TIER": "ultimate",
            "AUTOMEM_SERVICE_MODE": "archived",
            "AUTOMEM_SELF_SERVICE_EXPORT_ENABLED": "false",
        }
    )
    assert archived.mode == "archived"
    assert archived.writes_enabled is False
    assert archived.admin_mutations_enabled is False
    assert archived.self_service_export_enabled is False


def test_service_profile_endpoint_reports_runtime_details(
    client, auth_headers, reset_state
) -> None:
    reset_state.service_tier = "trial"
    reset_state.service_mode = "active"
    reset_state.service_profile = resolve_runtime_profile(
        {"AUTOMEM_SERVICE_TIER": "trial"}
    ).to_dict()

    response = client.get("/service/profile", headers=auth_headers)

    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "success"
    assert body["service"]["tier"] == "trial"
    assert body["service"]["mode"] == "active"
    assert body["service"]["profile"]["capabilities"]["writes_enabled"] is True
    assert body["service"]["vector_store"]["available"] is True
    assert body["service"]["embedding"]["provider"] == "openai"


def test_read_only_mode_blocks_writes_but_allows_recall(
    client, auth_headers, admin_headers, reset_state
) -> None:
    reset_state.memory_graph.memories["11111111-1111-1111-1111-111111111111"] = {
        "id": "11111111-1111-1111-1111-111111111111",
        "content": "Read only memories can still be recalled",
        "tags": ["locked"],
        "importance": 0.7,
        "type": "Context",
        "timestamp": "2026-04-03T00:00:00Z",
        "metadata": "{}",
        "confidence": 0.8,
    }
    reset_state.service_mode = "read_only"
    reset_state.service_profile = resolve_runtime_profile(
        {"AUTOMEM_SERVICE_TIER": "pro", "AUTOMEM_SERVICE_MODE": "read_only"}
    ).to_dict()
    reset_state.qdrant = None

    write_response = client.post(
        "/memory",
        json={"content": "blocked"},
        headers=auth_headers,
    )
    assert write_response.status_code == 423
    write_body = write_response.get_json()
    assert write_body["error"] == "service_locked"
    assert write_body["service_mode"] == "read_only"

    recall_response = client.get("/recall?query=recalled", headers=auth_headers)
    assert recall_response.status_code == 200
    assert recall_response.get_json()["status"] == "success"

    admin_response = client.post("/admin/reembed", json={}, headers=admin_headers)
    assert admin_response.status_code == 423


def test_archived_mode_allows_admin_export_flow(
    client, admin_headers, reset_state, monkeypatch, tmp_path
) -> None:
    export_id = "trial-expiry-export"
    reset_state.service_tier = "trial"
    reset_state.service_mode = "archived"
    reset_state.service_profile = resolve_runtime_profile(
        {"AUTOMEM_SERVICE_TIER": "trial", "AUTOMEM_SERVICE_MODE": "archived"}
    ).to_dict()
    reset_state.memory_graph.memories["aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"] = {
        "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "content": "Trial memory to archive",
        "tags": ["trial"],
        "importance": 0.9,
        "type": "Context",
        "timestamp": "2026-04-03T00:00:00Z",
        "metadata": "{}",
        "confidence": 0.9,
    }
    reset_state.memory_graph.relationships.append(
        {
            "id1": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "id2": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "type": "RELATES_TO",
            "strength": 1.0,
        }
    )
    reset_state.qdrant.upsert(
        "memories",
        [
            SimpleNamespace(
                id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                vector=[0.1, 0.2],
                payload={"content": "Trial memory to archive"},
            )
        ],
    )
    monkeypatch.setenv("AUTOMEM_EXPORT_DIR", str(tmp_path))

    create_response = client.post(
        "/admin/exports",
        json={"export_id": export_id, "reason": "trial_expired"},
        headers=admin_headers,
    )
    assert create_response.status_code == 201
    created = create_response.get_json()
    assert created["status"] == "complete"
    assert created["export_id"] == export_id
    assert created["service"]["mode"] == "archived"
    assert created["graph"]["node_count"] == 1
    assert created["include_vectors"] is True

    status_response = client.get(f"/admin/exports/{export_id}", headers=admin_headers)
    assert status_response.status_code == 200
    assert status_response.get_json()["export_id"] == export_id

    download_response = client.get(f"/admin/exports/{export_id}/download", headers=admin_headers)
    assert download_response.status_code == 200
    bundle = json.loads(gzip.decompress(download_response.data).decode("utf-8"))
    assert bundle["reason"] == "trial_expired"
    assert bundle["service"]["mode"] == "archived"
    assert bundle["graph"]["stats"]["node_count"] == 1


def test_archived_trial_mode_returns_upgrade_payload(client, auth_headers, reset_state) -> None:
    reset_state.service_tier = "trial"
    reset_state.service_mode = "archived"
    reset_state.service_profile = resolve_runtime_profile(
        {"AUTOMEM_SERVICE_TIER": "trial", "AUTOMEM_SERVICE_MODE": "archived"}
    ).to_dict()

    response = client.post("/memory", json={"content": "blocked"}, headers=auth_headers)

    assert response.status_code == 423
    body = response.get_json()
    assert body["error"] == "trial_expired"
    assert body["reason"] == "trial_expired"
    assert body["upgrade_url"] == "https://automem.ai/subscribe"
    assert "30-day trial" in body["message"]
