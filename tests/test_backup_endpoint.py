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
import scripts.restore_from_backup as restore_module
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


def test_backup_trailing_slash_uses_admin_auth(client: Any) -> None:
    response = client.get("/backup/", headers={"Authorization": "Bearer test-token"})

    assert response.status_code == 401

    response = client.get("/backup/", headers=_admin_headers())

    assert response.status_code == 200


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


def test_restore_rejects_unsafe_tar_member(tmp_path: Path) -> None:
    archive_path = tmp_path / "snapshot.tar.gz"

    with tarfile.open(archive_path, mode="w:gz") as archive:
        payload = b"unsafe"
        info = tarfile.TarInfo("../escape.txt")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    with pytest.raises(ValueError, match="unsafe backup path"):
        with resolve_backup_dir(archive_path):
            pass


def test_restore_relationships_use_batched_unwind_queries() -> None:
    from scripts.restore_from_backup import _restore_relationship_batches

    class RecordingGraph:
        def __init__(self) -> None:
            self.queries: list[tuple[str, dict[str, Any]]] = []

        def query(
            self,
            query: str,
            params: dict[str, Any] | None = None,
            **_kwargs: Any,
        ) -> Any:
            self.queries.append((query, params or {}))
            return SimpleNamespace(result_set=[])

    graph = RecordingGraph()
    node_backup_id_to_props = {
        1: ("Memory", {"id": "m1"}),
        2: ("Memory", {"id": "m2"}),
        3: ("Entity", {"id": "entity:tools:atlas-db"}),
    }
    relationships = [
        {
            "type": "RELATES_TO",
            "source_id": 1,
            "target_id": 2,
            "properties": {"strength": 0.7},
        },
        {
            "type": "RELATES_TO",
            "source_id": 2,
            "target_id": 1,
            "properties": {},
        },
        {
            "type": "REFERENCED_IN",
            "source_id": 3,
            "target_id": 1,
            "properties": {},
        },
    ]

    stats = _restore_relationship_batches(
        graph,
        relationships,
        node_backup_id_to_props,
        merge=False,
        existing_rels=set(),
        batch_size=2,
    )

    assert stats["created"] == 3
    assert stats["skipped"] == 0
    unwind_queries = [item for item in graph.queries if "UNWIND $rows AS row" in item[0]]
    assert len(unwind_queries) == 2
    assert len(unwind_queries[0][1]["rows"]) == 2
    assert unwind_queries[0][1]["rows"][0] == {
        "source_id": "m1",
        "target_id": "m2",
        "props": {"strength": 0.7},
    }
    assert "CREATE (a)-[r:`RELATES_TO`]->(b)" in unwind_queries[0][0]
    assert "CREATE (a)-[r:`REFERENCED_IN`]->(b)" in unwind_queries[1][0]


def test_restore_nodes_use_batched_unwind_queries() -> None:
    from scripts.restore_from_backup import _restore_node_batches

    class RecordingGraph:
        def __init__(self) -> None:
            self.queries: list[tuple[str, dict[str, Any]]] = []

        def query(
            self,
            query: str,
            params: dict[str, Any] | None = None,
            **_kwargs: Any,
        ) -> Any:
            self.queries.append((query, params or {}))
            return SimpleNamespace(result_set=[])

    graph = RecordingGraph()
    nodes = [
        {
            "id": 1,
            "labels": ["Memory"],
            "properties": {"id": "m1", "content": "First", "importance": 0.9},
        },
        {
            "id": 2,
            "labels": ["Memory"],
            "properties": {"id": "m2", "content": "Second"},
        },
        {
            "id": 3,
            "labels": ["Entity"],
            "properties": {"id": "entity:tools:atlas-db", "slug": "atlas-db"},
        },
    ]

    stats = _restore_node_batches(
        graph,
        nodes,
        merge=False,
        existing_uuids=set(),
        restore_time="2026-01-01T00:00:00+00:00",
        batch_size=2,
    )

    assert stats["created"] == 3
    assert stats["skipped"] == 0
    assert stats["node_backup_id_to_props"][1][0] == "Memory"
    unwind_queries = [item for item in graph.queries if "UNWIND $rows AS row" in item[0]]
    assert len(unwind_queries) == 2
    assert "CREATE (n:`Memory`)" in unwind_queries[0][0]
    assert len(unwind_queries[0][1]["rows"]) == 2
    assert unwind_queries[0][1]["rows"][1]["props"]["last_accessed"] == (
        "2026-01-01T00:00:00+00:00"
    )
    assert unwind_queries[0][1]["rows"][1]["props"]["relevance_score"] == 0.5
    assert "CREATE (n:`Entity`)" in unwind_queries[1][0]


def test_restore_creates_id_indexes_once_per_label() -> None:
    from scripts.restore_from_backup import _create_restore_id_indexes

    class RecordingGraph:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def query(
            self,
            query: str,
            params: dict[str, Any] | None = None,
            **_kwargs: Any,
        ) -> Any:
            del params
            self.queries.append(query)
            if "Entity" in query:
                raise RuntimeError("Attribute 'id' is already indexed")
            return SimpleNamespace(result_set=[])

    graph = RecordingGraph()

    _create_restore_id_indexes(graph, ["Memory", "Memory", "Entity"])

    assert graph.queries == [
        "CREATE INDEX FOR (n:`Entity`) ON (n.id)",
        "CREATE INDEX FOR (n:`Memory`) ON (n.id)",
    ]


def test_restore_falkordb_queries_use_timeout_and_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FlakyGraph:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any], int | None]] = []

        def query(
            self,
            query: str,
            params: dict[str, Any] | None = None,
            timeout: int | None = None,
        ) -> Any:
            self.calls.append((query, params or {}, timeout))
            if len(self.calls) == 1:
                raise RuntimeError("Query timed out")
            return SimpleNamespace(result_set=[[7]])

    graph = FlakyGraph()
    monkeypatch.setattr(restore_module, "FALKORDB_RESTORE_QUERY_TIMEOUT_MS", 1234)
    monkeypatch.setattr(restore_module, "FALKORDB_RESTORE_RETRIES", 2)
    monkeypatch.setattr(restore_module, "FALKORDB_RESTORE_RETRY_DELAY_SECONDS", 0)
    monkeypatch.setattr(restore_module.time, "sleep", lambda _seconds: None)

    result = restore_module._query_falkordb(graph, "MATCH (n) RETURN count(n)", {"x": 1})

    assert result.result_set == [[7]]
    assert graph.calls == [
        ("MATCH (n) RETURN count(n)", {"x": 1}, 1234),
        ("MATCH (n) RETURN count(n)", {"x": 1}, 1234),
    ]


def test_lab_clone_uses_paced_qdrant_restore_defaults() -> None:
    script = Path("scripts/lab/clone_production.sh").read_text()

    assert 'QDRANT_RESTORE_BATCH_SIZE="${QDRANT_RESTORE_BATCH_SIZE:-250}"' in script
    assert 'QDRANT_RESTORE_WAIT="${QDRANT_RESTORE_WAIT:-true}"' in script
    assert (
        'QDRANT_RESTORE_BATCH_DELAY_SECONDS="${QDRANT_RESTORE_BATCH_DELAY_SECONDS:-0}"'
        in script
    )
    assert restore_module.QDRANT_RESTORE_BATCH_SIZE == 250
    assert restore_module.QDRANT_RESTORE_BATCH_DELAY_SECONDS == 0
    assert restore_module.QDRANT_RESTORE_WAIT is True
    assert 'QDRANT_RESTORE_INDEXING_THRESHOLD="${QDRANT_RESTORE_INDEXING_THRESHOLD:-0}"' in script
    assert (
        'QDRANT_RESTORE_DEFAULT_SEGMENT_NUMBER="${QDRANT_RESTORE_DEFAULT_SEGMENT_NUMBER:-1}"'
        in script
    )
    assert 'QDRANT_RESTORE_MEMMAP_THRESHOLD="${QDRANT_RESTORE_MEMMAP_THRESHOLD:-0}"' in script
    assert 'QDRANT_RESTORE_HNSW_M="${QDRANT_RESTORE_HNSW_M:-0}"' in script
    assert 'QDRANT_RESTORE_VECTOR_ON_DISK="${QDRANT_RESTORE_VECTOR_ON_DISK:-true}"' in script
    assert 'QDRANT_RESTORE_ON_DISK_PAYLOAD="${QDRANT_RESTORE_ON_DISK_PAYLOAD:-false}"' in script
    assert 'QDRANT_PREFER_GRPC="${QDRANT_PREFER_GRPC:-true}"' in script
    assert 'QDRANT_GRPC_PORT="$QDRANT_GRPC_HOST_PORT"' in script
    assert "--qdrant-grpc-port PORT" in script
    assert '"${QDRANT_GRPC_HOST_PORT:-6334}:6334"' in Path("docker-compose.yml").read_text()


def test_lab_clone_disables_falkordb_persistence_during_restore() -> None:
    compose = Path("docker-compose.yml").read_text()
    script = Path("scripts/lab/clone_production.sh").read_text()

    assert (
        "REDIS_ARGS=${FALKORDB_REDIS_ARGS:---save 60 1 --appendonly yes "
        "--appendfsync everysec}"
    ) in compose
    assert 'FALKORDB_REDIS_ARGS="${FALKORDB_REDIS_ARGS:---save \\"\\" --appendonly no}"' in script
    assert (
        'FALKORDB_RESTORE_QUERY_TIMEOUT_MS="${FALKORDB_RESTORE_QUERY_TIMEOUT_MS:-300000}"'
        in script
    )


def test_lab_clone_can_skip_api_start_after_restore() -> None:
    script = Path("scripts/lab/clone_production.sh").read_text()

    assert "--skip-api" in script
    assert 'if [ "$SKIP_API" = true ]; then' in script
    assert "Skipping API startup" in script


def test_restore_qdrant_retries_transient_batch_disconnect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backup_dir = tmp_path / "backup"
    qdrant_dir = backup_dir / "qdrant"
    qdrant_dir.mkdir(parents=True)
    backup_file = qdrant_dir / "qdrant_20260101_000000.json.gz"
    points = [
        {
            "id": f"00000000-0000-0000-0000-00000000000{i}",
            "vector": [0.1, 0.2, 0.3],
            "payload": {"tags": ["automem"]},
        }
        for i in range(3)
    ]
    payload = {"stats": {"vector_size": 3}, "points": points}
    with gzip.open(backup_file, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    class FlakyQdrantClient:
        points_by_id: dict[str, dict[str, Any]] = {}
        upsert_calls = 0
        wait_values: list[bool | None] = []
        collection_created = False

        def __init__(self, **_kwargs: Any) -> None:
            pass

        def get_collection(self, _collection_name: str) -> Any:
            if not type(self).collection_created:
                raise RuntimeError("collection missing")
            return SimpleNamespace(points_count=len(type(self).points_by_id))

        def create_collection(self, **_kwargs: Any) -> None:
            type(self).collection_created = True

        def upsert(
            self,
            collection_name: str,
            points: list[Any],
            wait: bool | None = None,
        ) -> None:
            del collection_name
            type(self).wait_values.append(wait)
            type(self).upsert_calls += 1
            if type(self).upsert_calls == 2:
                raise ConnectionError("server disconnected")
            for point in points:
                type(self).points_by_id[str(point.id)] = {
                    "vector": point.vector,
                    "payload": point.payload,
                }

    monkeypatch.setattr(restore_module, "QdrantClient", FlakyQdrantClient)
    monkeypatch.setattr(restore_module, "QDRANT_RESTORE_BATCH_SIZE", 1, raising=False)
    monkeypatch.setattr(restore_module, "QDRANT_RESTORE_RETRIES", 2, raising=False)
    monkeypatch.setattr(
        restore_module, "QDRANT_RESTORE_RETRY_DELAY_SECONDS", 0, raising=False
    )
    monkeypatch.setattr(restore_module.time, "sleep", lambda _seconds: None)

    restore = AutoMemRestore(backup_dir=backup_dir, dry_run=False, force=True)
    result = restore.restore_qdrant(backup_file)

    assert result["points"] == 3
    assert FlakyQdrantClient.upsert_calls == 4
    assert FlakyQdrantClient.wait_values == [True, True, True, True]
    assert set(FlakyQdrantClient.points_by_id) == {point["id"] for point in points}


def test_restore_qdrant_client_uses_grpc_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class RecordingQdrantClient:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(restore_module, "QdrantClient", RecordingQdrantClient)
    monkeypatch.setattr(restore_module, "QDRANT_PREFER_GRPC", True)
    monkeypatch.setattr(restore_module, "QDRANT_GRPC_PORT", 7654)

    restore = AutoMemRestore(backup_dir=Path("backups"), dry_run=False, force=True)
    restore._qdrant_client()

    assert calls[0]["prefer_grpc"] is True
    assert calls[0]["grpc_port"] == 7654


def test_restore_qdrant_uses_optional_collection_tuning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backup_dir = tmp_path / "backup"
    qdrant_dir = backup_dir / "qdrant"
    qdrant_dir.mkdir(parents=True)
    backup_file = qdrant_dir / "qdrant_20260101_000000.json.gz"
    payload = {
        "stats": {"vector_size": 3},
        "points": [
            {
                "id": "00000000-0000-0000-0000-000000000001",
                "vector": [0.1, 0.2, 0.3],
                "payload": {"tags": ["automem"]},
            }
        ],
    }
    with gzip.open(backup_file, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    class RecordingQdrantClient:
        collection_created = False
        create_kwargs: dict[str, Any] = {}
        points_by_id: dict[str, dict[str, Any]] = {}

        def __init__(self, **_kwargs: Any) -> None:
            pass

        def get_collection(self, _collection_name: str) -> Any:
            if not type(self).collection_created:
                raise RuntimeError("collection missing")
            return SimpleNamespace(points_count=len(type(self).points_by_id))

        def create_collection(self, **kwargs: Any) -> None:
            type(self).collection_created = True
            type(self).create_kwargs = kwargs

        def upsert(
            self,
            collection_name: str,
            points: list[Any],
            wait: bool | None = None,
        ) -> None:
            del collection_name, wait
            for point in points:
                type(self).points_by_id[str(point.id)] = {
                    "vector": point.vector,
                    "payload": point.payload,
                }

    monkeypatch.setattr(restore_module, "QdrantClient", RecordingQdrantClient)
    monkeypatch.setattr(restore_module, "QDRANT_RESTORE_INDEXING_THRESHOLD", 0)
    monkeypatch.setattr(restore_module, "QDRANT_RESTORE_DEFAULT_SEGMENT_NUMBER", 1)
    monkeypatch.setattr(restore_module, "QDRANT_RESTORE_MEMMAP_THRESHOLD", 0)
    monkeypatch.setattr(restore_module, "QDRANT_RESTORE_HNSW_M", 0)
    monkeypatch.setattr(restore_module, "QDRANT_RESTORE_VECTOR_ON_DISK", True)
    monkeypatch.setattr(restore_module, "QDRANT_RESTORE_ON_DISK_PAYLOAD", False)

    restore = AutoMemRestore(backup_dir=backup_dir, dry_run=False, force=True)
    result = restore.restore_qdrant(backup_file)

    assert result["points"] == 1
    assert RecordingQdrantClient.create_kwargs["on_disk_payload"] is False
    optimizers = RecordingQdrantClient.create_kwargs["optimizers_config"]
    assert optimizers.indexing_threshold == 0
    assert optimizers.default_segment_number == 1
    assert optimizers.memmap_threshold == 0
    assert RecordingQdrantClient.create_kwargs["hnsw_config"].m == 0
    assert RecordingQdrantClient.create_kwargs["vectors_config"].on_disk is True
