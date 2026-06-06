from __future__ import annotations

import gzip
import io
import json
import queue
import tarfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional

VALID_BACKUP_INCLUDES = ("falkordb", "qdrant")
STREAM_QUEUE_MAX_CHUNKS = 8


class BackupError(RuntimeError):
    """Raised when backup creation fails."""


class InvalidBackupInclude(ValueError):
    """Raised when the backup include query parameter is invalid."""


@dataclass(frozen=True)
class BackupArtifact:
    service: str
    member_name: str
    data: bytes
    stats: dict[str, Any]


@dataclass(frozen=True)
class BackupFile:
    service: str
    path: Path
    stats: dict[str, Any]


def backup_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def parse_backup_include(raw_include: Optional[str]) -> tuple[str, ...]:
    """Parse a comma-separated include list, defaulting to both stores when absent."""
    if raw_include is None:
        return VALID_BACKUP_INCLUDES

    parts = [part.strip().lower() for part in raw_include.split(",")]
    includes = tuple(include for include in VALID_BACKUP_INCLUDES if include in parts)
    invalid = [part for part in parts if part and part not in VALID_BACKUP_INCLUDES]

    if invalid or not includes or any(not part for part in parts):
        valid = ",".join(VALID_BACKUP_INCLUDES)
        raise InvalidBackupInclude(f"include must be a comma-separated subset of: {valid}")

    return includes


def _gzip_json(data: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb") as gz:
        with io.TextIOWrapper(gz, encoding="utf-8") as writer:
            json.dump(data, writer, indent=2, default=str)
    return buffer.getvalue()


def _query_rows(result: Any) -> list[Any]:
    return list(getattr(result, "result_set", []) or [])


def export_falkordb_artifact(
    *,
    graph: Any,
    graph_name: str,
    timestamp: str,
    batch_size: int = 10000,
    logger: Any = None,
) -> BackupArtifact:
    """Export FalkorDB graph data as a compressed JSON backup artifact."""
    if graph is None:
        raise BackupError("FalkorDB is unavailable")

    nodes: list[dict[str, Any]] = []
    offset = 0

    while True:
        rows = _query_rows(
            graph.query(
                f"""
                MATCH (n)
                RETURN
                    id(n) as id,
                    labels(n) as labels,
                    properties(n) as props
                SKIP {offset} LIMIT {batch_size}
                """
            )
        )
        if not rows:
            break

        for row in rows:
            nodes.append({"id": row[0], "labels": row[1], "properties": row[2]})

        if logger:
            logger.info(
                "Exported FalkorDB node batch: %d nodes (total: %d)",
                len(rows),
                len(nodes),
            )
        if len(rows) < batch_size:
            break
        offset += batch_size

    relationships: list[dict[str, Any]] = []
    offset = 0

    while True:
        rows = _query_rows(
            graph.query(
                f"""
                MATCH (a)-[r]->(b)
                RETURN
                    id(a) as source_id,
                    type(r) as rel_type,
                    id(b) as target_id,
                    properties(r) as props
                SKIP {offset} LIMIT {batch_size}
                """
            )
        )
        if not rows:
            break

        for row in rows:
            relationships.append(
                {
                    "source_id": row[0],
                    "type": row[1],
                    "target_id": row[2],
                    "properties": row[3],
                }
            )

        if logger:
            logger.info(
                "Exported FalkorDB relationship batch: %d relationships (total: %d)",
                len(rows),
                len(relationships),
            )
        if len(rows) < batch_size:
            break
        offset += batch_size

    stats = {
        "node_count": len(nodes),
        "relationship_count": len(relationships),
    }
    backup_data = {
        "timestamp": timestamp,
        "graph_name": graph_name,
        "nodes": nodes,
        "relationships": relationships,
        "stats": stats,
    }
    return BackupArtifact(
        service="falkordb",
        member_name=f"falkordb/falkordb_{timestamp}.json.gz",
        data=_gzip_json(backup_data),
        stats=stats,
    )


def _vector_size_from_collection_info(collection_info: Any) -> Any:
    vectors = getattr(
        getattr(getattr(collection_info, "config", None), "params", None),
        "vectors",
        None,
    )
    if isinstance(vectors, dict):
        first = next(iter(vectors.values()), None)
        return getattr(first, "size", None)
    return getattr(vectors, "size", None)


def export_qdrant_artifact(
    *,
    client: Any,
    collection_name: str,
    timestamp: str,
    batch_size: int = 100,
    logger: Any = None,
) -> BackupArtifact:
    """Export Qdrant collection data as a compressed JSON backup artifact."""
    if client is None:
        raise BackupError("Qdrant is unavailable")

    all_points: list[dict[str, Any]] = []
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )

        for point in points:
            all_points.append(
                {
                    "id": point.id,
                    "vector": point.vector,
                    "payload": point.payload,
                }
            )

        if logger and points:
            logger.info(
                "Exported Qdrant point batch: %d points (total: %d)",
                len(points),
                len(all_points),
            )

        if next_offset is None:
            break
        offset = next_offset

    collection_info = client.get_collection(collection_name)
    stats = {
        "points_count": len(all_points),
        "vector_size": _vector_size_from_collection_info(collection_info),
    }
    backup_data = {
        "timestamp": timestamp,
        "collection_name": collection_name,
        "points": all_points,
        "stats": stats,
    }
    return BackupArtifact(
        service="qdrant",
        member_name=f"qdrant/qdrant_{timestamp}.json.gz",
        data=_gzip_json(backup_data),
        stats=stats,
    )


def create_backup_artifacts(
    *,
    includes: Iterable[str],
    timestamp: str,
    graph: Any = None,
    graph_name: str = "memories",
    qdrant_client: Any = None,
    collection_name: str = "memories",
    logger: Any = None,
) -> list[BackupArtifact]:
    artifacts: list[BackupArtifact] = []
    include_set = set(includes)

    if "falkordb" in include_set:
        artifacts.append(
            export_falkordb_artifact(
                graph=graph,
                graph_name=graph_name,
                timestamp=timestamp,
                logger=logger,
            )
        )
    if "qdrant" in include_set:
        artifacts.append(
            export_qdrant_artifact(
                client=qdrant_client,
                collection_name=collection_name,
                timestamp=timestamp,
                logger=logger,
            )
        )

    return artifacts


def write_backup_artifact(backup_dir: Path, artifact: BackupArtifact) -> BackupFile:
    path = backup_dir / artifact.member_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(artifact.data)
    return BackupFile(service=artifact.service, path=path, stats=artifact.stats)


def write_falkordb_backup_file(
    *,
    backup_dir: Path,
    graph: Any,
    graph_name: str,
    timestamp: str,
    logger: Any = None,
) -> BackupFile:
    return write_backup_artifact(
        backup_dir,
        export_falkordb_artifact(
            graph=graph,
            graph_name=graph_name,
            timestamp=timestamp,
            logger=logger,
        ),
    )


def write_qdrant_backup_file(
    *,
    backup_dir: Path,
    qdrant_client: Any,
    collection_name: str,
    timestamp: str,
    logger: Any = None,
) -> BackupFile:
    return write_backup_artifact(
        backup_dir,
        export_qdrant_artifact(
            client=qdrant_client,
            collection_name=collection_name,
            timestamp=timestamp,
            logger=logger,
        ),
    )


def cleanup_old_backup_files(*, backup_dir: Path, keep: int, logger: Any = None) -> None:
    for backup_type in VALID_BACKUP_INCLUDES:
        backup_path = backup_dir / backup_type
        backup_files = sorted(
            backup_path.glob("*.json.gz"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old_file in backup_files[keep:]:
            if logger:
                logger.info("Removing old %s backup: %s", backup_type, old_file.name)
            old_file.unlink()

        if logger:
            kept = min(len(backup_files), keep)
            removed = max(0, len(backup_files) - keep)
            logger.info("%s backup cleanup: kept %d, removed %d", backup_type, kept, removed)


class _QueueWriter:
    def __init__(self, output_queue: "queue.Queue[Any]") -> None:
        self.output_queue = output_queue
        self.bytes_written = 0

    def write(self, data: bytes) -> int:
        if data:
            chunk = bytes(data)
            self.bytes_written += len(chunk)
            self.output_queue.put(chunk)
        return len(data)

    def flush(self) -> None:
        return None


def _add_artifact_to_tar(tar: tarfile.TarFile, artifact: BackupArtifact) -> None:
    info = tarfile.TarInfo(artifact.member_name)
    info.size = len(artifact.data)
    info.mtime = int(datetime.now(timezone.utc).timestamp())
    tar.addfile(info, io.BytesIO(artifact.data))


def stream_backup_tar_gz(
    *,
    includes: Iterable[str],
    timestamp: str,
    graph: Any = None,
    graph_name: str = "memories",
    qdrant_client: Any = None,
    collection_name: str = "memories",
    logger: Any = None,
    on_complete: Optional[Callable[[dict[str, Any]], None]] = None,
) -> Iterator[bytes]:
    """Stream a tar.gz archive containing restore-compatible backup files."""
    output_queue: "queue.Queue[Any]" = queue.Queue(maxsize=STREAM_QUEUE_MAX_CHUNKS)

    def worker() -> None:
        writer = _QueueWriter(output_queue)
        stats: dict[str, Any] = {
            "status": "complete",
            "bytes": 0,
            "artifacts": {},
        }
        try:
            with tarfile.open(fileobj=writer, mode="w|gz") as tar:
                artifacts = create_backup_artifacts(
                    includes=includes,
                    timestamp=timestamp,
                    graph=graph,
                    graph_name=graph_name,
                    qdrant_client=qdrant_client,
                    collection_name=collection_name,
                    logger=logger,
                )
                for artifact in artifacts:
                    _add_artifact_to_tar(tar, artifact)
                    stats["artifacts"][artifact.service] = artifact.stats
        except Exception as exc:  # pragma: no cover - exercised by Flask streaming internals
            stats["status"] = "failed"
            stats["error"] = str(exc)
            output_queue.put(exc)
        finally:
            stats["bytes"] = writer.bytes_written
            if on_complete:
                on_complete(stats)
            output_queue.put(None)

    threading.Thread(target=worker, daemon=True).start()

    while True:
        item = output_queue.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        yield item
