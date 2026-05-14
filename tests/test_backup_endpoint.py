from __future__ import annotations

import gzip
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import app
from scripts.restore_from_backup import AutoMemRestore, resolve_backup_dir
from tests.support.fake_graph import FakeGraph


class FakeQdrantClient:
    def __init__(self, vector_size: int = 3) -> None:
        self.points: dict[str, dict[str, Any]] = {}
        self.vector_size = vector_size

    def scroll(
        self,
        collection_name: str,
        limit: int = 100,
        offset: int | None = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        **_kwargs: Any,
    ) -> tuple[list[Any], int | None]:
        del collection_name
        ids = sorted(self.points)
        start = int(offset or 0)
        selected = ids[start : start + limit]
        points = []
        for point_id in selected:
            point = self.points[point_id]
            points.append(
                SimpleNamespace(
                    id=point_id,
                    vector=point["vector"] if with_vectors else None,
                    payload=point["payload"] if with_payload else None,
                )
            )
        next_offset = start + len(selected)
        if next_offset >= len(ids):
            return points, None
        return points, next_offset

    def get_collection(self, collection_name: str) -> Any:
        del collection_name
        return SimpleNamespace(
            points_count=len(self.points),
            config=SimpleNamespace(
                params=SimpleNamespace(vectors=SimpleNamespace(size=self.vector_size))
            ),
        )


@pytest.fixture
def backup_state(monkeypatch: pytest.MonkeyPatch) -> Any:
    state = app.ServiceState()
    state.memory_graph = FakeGraph()
    state.qdrant = FakeQdrantClient()

    monkeypatch.setattr(app, "state", state)
    monkeypatch.setattr(app, "init_falkordb", lambda: None)
    monkeypatch.setattr(app, "init_qdrant", lambda: None)
    monkeypatch.setattr(app, "init_openai", lambda: None)
    monkeypatch.setattr(app, "API_TOKEN", "test-token")
    monkeypatch.setattr(app, "ADMIN_TOKEN", "test-admin-token")

    original_testing = app.app.config.get("TESTING")
    app.app.config["TESTING"] = True
    yield state
    app.app.config["TESTING"] = original_testing


@pytest.fixture
def client(backup_state: Any) -> Any:
    del backup_state
    with app.app.test_client() as test_client:
        yield test_client


def _admin_headers() -> dict[str, str]:
    return {"X-Admin-Token": "test-admin-token"}


def _archive_members(raw: bytes) -> dict[str, bytes]:
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        return {
            member.name: archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isfile()
        }


def _read_backup_json(members: dict[str, bytes], prefix: str) -> dict[str, Any]:
    matching = [name for name in members if name.startswith(prefix)]
    assert len(matching) == 1
    return json.loads(gzip.decompress(members[matching[0]]).decode("utf-8"))


def test_backup_requires_admin_token_not_api_token(client: Any) -> None:
    response = client.get("/backup", headers={"Authorization": "Bearer test-token"})

    assert response.status_code == 401


def test_backup_missing_admin_token_returns_401(client: Any) -> None:
    response = client.get("/backup")

    assert response.status_code == 401


def test_backup_unconfigured_admin_token_returns_403(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app, "ADMIN_TOKEN", None)

    response = client.get("/backup", headers=_admin_headers())

    assert response.status_code == 403


def test_backup_happy_path_tar_contains_falkordb_and_qdrant(client: Any, backup_state: Any) -> None:
    memory_id = "10000000-0000-0000-0000-000000000001"
    backup_state.memory_graph.memories[memory_id] = {
        "id": memory_id,
        "content": "Backup me",
        "tags": ["backup"],
        "importance": 0.8,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "type": "Context",
    }
    backup_state.qdrant.points[memory_id] = {
        "vector": [0.1, 0.2, 0.3],
        "payload": {"content": "Backup me", "tags": ["backup"]},
    }

    response = client.get("/backup", headers=_admin_headers())

    assert response.status_code == 200
    assert response.mimetype == "application/gzip"
    assert "automem-backup-" in response.headers["Content-Disposition"]

    members = _archive_members(response.get_data())
    assert any(name.startswith("falkordb/falkordb_") for name in members)
    assert any(name.startswith("qdrant/qdrant_") for name in members)

    falkordb = _read_backup_json(members, "falkordb/falkordb_")
    qdrant = _read_backup_json(members, "qdrant/qdrant_")
    assert falkordb["stats"]["node_count"] == 1
    assert falkordb["nodes"][0]["properties"]["content"] == "Backup me"
    assert qdrant["stats"]["points_count"] == 1
    assert qdrant["points"][0]["vector"] == [0.1, 0.2, 0.3]


def test_backup_include_falkordb_only_does_not_require_qdrant(
    client: Any, backup_state: Any
) -> None:
    backup_state.qdrant = None

    response = client.get("/backup?include=falkordb", headers=_admin_headers())

    assert response.status_code == 200
    members = _archive_members(response.get_data())
    assert any(name.startswith("falkordb/falkordb_") for name in members)
    assert not any(name.startswith("qdrant/qdrant_") for name in members)


@pytest.mark.parametrize("include", ["", "unknown", "falkordb,,qdrant"])
def test_backup_invalid_include_returns_400(client: Any, include: str) -> None:
    response = client.get(f"/backup?include={include}", headers=_admin_headers())

    assert response.status_code == 400


def test_backup_empty_corpus(client: Any) -> None:
    response = client.get("/backup", headers=_admin_headers())

    assert response.status_code == 200
    members = _archive_members(response.get_data())
    falkordb = _read_backup_json(members, "falkordb/falkordb_")
    qdrant = _read_backup_json(members, "qdrant/qdrant_")
    assert falkordb["stats"]["node_count"] == 0
    assert falkordb["stats"]["relationship_count"] == 0
    assert qdrant["stats"]["points_count"] == 0


def test_restore_accepts_downloaded_tar_gz_backup(tmp_path: Path) -> None:
    falkordb_payload = {
        "timestamp": "20260101_000000",
        "graph_name": "memories",
        "nodes": [],
        "relationships": [],
        "stats": {"node_count": 0, "relationship_count": 0},
    }
    compressed = gzip.compress(json.dumps(falkordb_payload).encode("utf-8"))
    archive_path = tmp_path / "snapshot.tar.gz"

    with tarfile.open(archive_path, mode="w:gz") as archive:
        info = tarfile.TarInfo("falkordb/falkordb_20260101_000000.json.gz")
        info.size = len(compressed)
        archive.addfile(info, io.BytesIO(compressed))

    with resolve_backup_dir(archive_path) as backup_dir:
        restore = AutoMemRestore(backup_dir=backup_dir, dry_run=True, force=True)
        result = restore.run_restore(falkordb_only=True)

    assert result["falkordb"]["nodes"] == 0
    assert result["falkordb"]["relationships"] == 0
