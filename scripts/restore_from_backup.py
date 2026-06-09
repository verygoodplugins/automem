#!/usr/bin/env python3
"""Restore AutoMem from backup files.

Restores FalkorDB graph and Qdrant vectors from compressed JSON backups.

Usage:
    # Restore from latest backup
    python scripts/restore_from_backup.py

    # Restore from specific backup
    python scripts/restore_from_backup.py --backup-timestamp 20251019_085625

    # Restore from downloaded API backup tarball
    python scripts/restore_from_backup.py --backup-dir snapshot.tar.gz

    # Dry run (show what would be restored)
    python scripts/restore_from_backup.py --dry-run

    # Import/merge without deleting existing data
    python scripts/restore_from_backup.py --import
"""

import argparse
import gzip
import json
import logging
import os
import shutil
import sys
import tarfile
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator

from dotenv import load_dotenv
from falkordb import FalkorDB
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# Load environment
load_dotenv()
load_dotenv(Path.home() / ".config" / "automem" / ".env")


def _optional_int_env(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return int(value)


def _optional_bool_env(name: str) -> bool | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return value.lower().strip() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("automem.restore")

# Configuration
BACKUP_DIR = Path(os.getenv("AUTOMEM_BACKUP_DIR", "./backups"))
FALKORDB_HOST = os.getenv("FALKORDB_HOST", "localhost")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))
FALKORDB_PASSWORD = os.getenv("FALKORDB_PASSWORD")
FALKORDB_GRAPH = os.getenv("FALKORDB_GRAPH", "memories")
FALKORDB_RESTORE_QUERY_TIMEOUT_MS = int(os.getenv("FALKORDB_RESTORE_QUERY_TIMEOUT_MS", "300000"))
FALKORDB_RESTORE_RETRIES = int(os.getenv("FALKORDB_RESTORE_RETRIES", "3"))
FALKORDB_RESTORE_RETRY_DELAY_SECONDS = float(os.getenv("FALKORDB_RESTORE_RETRY_DELAY_SECONDS", "2"))
FALKORDB_RESTORE_NODE_BATCH_SIZE = int(os.getenv("FALKORDB_RESTORE_NODE_BATCH_SIZE", "250"))

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "memories")
QDRANT_TIMEOUT_SECONDS = _float_env("QDRANT_TIMEOUT_SECONDS", 60)
QDRANT_GRPC_PORT = _optional_int_env("QDRANT_GRPC_PORT")
QDRANT_PREFER_GRPC = os.getenv("QDRANT_PREFER_GRPC", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
QDRANT_RESTORE_BATCH_SIZE = int(os.getenv("QDRANT_RESTORE_BATCH_SIZE", "250"))
QDRANT_RESTORE_RETRIES = int(os.getenv("QDRANT_RESTORE_RETRIES", "5"))
QDRANT_RESTORE_RETRY_DELAY_SECONDS = float(os.getenv("QDRANT_RESTORE_RETRY_DELAY_SECONDS", "2"))
QDRANT_RESTORE_BATCH_DELAY_SECONDS = float(os.getenv("QDRANT_RESTORE_BATCH_DELAY_SECONDS", "0"))
QDRANT_RESTORE_WAIT = os.getenv("QDRANT_RESTORE_WAIT", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
QDRANT_RESTORE_INDEXING_THRESHOLD = _optional_int_env("QDRANT_RESTORE_INDEXING_THRESHOLD")
QDRANT_RESTORE_DEFAULT_SEGMENT_NUMBER = _optional_int_env("QDRANT_RESTORE_DEFAULT_SEGMENT_NUMBER")
QDRANT_RESTORE_MEMMAP_THRESHOLD = _optional_int_env("QDRANT_RESTORE_MEMMAP_THRESHOLD")
QDRANT_RESTORE_HNSW_M = _optional_int_env("QDRANT_RESTORE_HNSW_M")
QDRANT_RESTORE_VECTOR_ON_DISK = _optional_bool_env("QDRANT_RESTORE_VECTOR_ON_DISK")
QDRANT_RESTORE_ON_DISK_PAYLOAD = _optional_bool_env("QDRANT_RESTORE_ON_DISK_PAYLOAD")
QDRANT_RESTORE_READY_TIMEOUT_SECONDS = float(
    os.getenv("QDRANT_RESTORE_READY_TIMEOUT_SECONDS", "300")
)
FALKORDB_RESTORE_REL_BATCH_SIZE = int(os.getenv("FALKORDB_RESTORE_REL_BATCH_SIZE", "500"))


def _is_tar_gz(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".tar.gz") or name.endswith(".tgz")


def _safe_extract_tar_gz(archive_path: Path, target_dir: Path) -> None:
    target_root = target_dir.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if member.issym() or member.islnk():
                raise ValueError(f"Refusing to extract link from backup archive: {member.name}")
            if not (member.isdir() or member.isfile()):
                raise ValueError(
                    f"Refusing to extract special file from backup archive: {member.name}"
                )
            destination = (target_root / member.name).resolve()
            if not destination.is_relative_to(target_root):
                raise ValueError(f"Refusing to extract unsafe backup path: {member.name}")
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Unable to read backup archive member: {member.name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)


def _escape_cypher_identifier(value: str) -> str:
    return f"`{value.replace('`', '``')}`"


def _first_label(labels: str) -> str:
    return labels.split(":")[0].strip("`")


def _is_timeout_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "timed out" in message or "timeout" in message


def _query_falkordb(
    graph: Any,
    query: str,
    params: dict[str, Any] | None = None,
) -> Any:
    retries = max(0, FALKORDB_RESTORE_RETRIES)
    timeout_ms = max(1, FALKORDB_RESTORE_QUERY_TIMEOUT_MS)
    for attempt in range(0, retries + 1):
        try:
            try:
                return graph.query(query, params, timeout=timeout_ms)
            except TypeError as exc:
                if "timeout" not in str(exc).lower():
                    raise
                return graph.query(query, params)
        except Exception as exc:
            if attempt >= retries or not _is_timeout_error(exc):
                raise
            logger.warning(
                "      FalkorDB query timed out (attempt %s/%s); waiting to retry",
                attempt + 1,
                retries + 1,
            )
            time.sleep(FALKORDB_RESTORE_RETRY_DELAY_SECONDS)
    raise RuntimeError("FalkorDB query retry loop exited unexpectedly")


def _node_batch_query(labels: tuple[str, ...]) -> str:
    escaped_labels = ":".join(_escape_cypher_identifier(label) for label in labels)
    return f"""
        UNWIND $rows AS row
        CREATE (n:{escaped_labels})
        SET n += row.props
    """


def _execute_node_batch(
    graph: Any,
    *,
    labels: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> dict[str, int]:
    query = _node_batch_query(labels)
    try:
        _query_falkordb(graph, query, {"rows": rows})
        return {"created": len(rows), "skipped": 0}
    except Exception as exc:
        if len(rows) == 1:
            logger.warning("      Skipped node with labels %s: %s", ":".join(labels), exc)
            return {"created": 0, "skipped": 1}

    midpoint = max(1, len(rows) // 2)
    left = _execute_node_batch(graph, labels=labels, rows=rows[:midpoint])
    right = _execute_node_batch(graph, labels=labels, rows=rows[midpoint:])
    return {
        "created": left["created"] + right["created"],
        "skipped": left["skipped"] + right["skipped"],
    }


def _memory_restore_props(props: dict[str, Any], restore_time: str) -> dict[str, Any]:
    restored = props.copy()
    restored["last_accessed"] = restore_time
    if "relevance_score" not in restored or restored.get("relevance_score") is None:
        importance = restored.get("importance", 0.5) or 0.5
        restored["relevance_score"] = max(0.3, float(importance))
    return restored


def _restore_node_batches(
    graph: Any,
    nodes: list[dict[str, Any]],
    *,
    merge: bool,
    existing_uuids: set[str],
    restore_time: str,
    batch_size: int,
) -> dict[str, Any]:
    node_backup_id_to_props: dict[Any, tuple[str, dict[str, Any]]] = {}
    grouped_rows: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    nodes_skipped = 0

    for index, node in enumerate(nodes):
        if index % 100 == 0:
            logger.info("      Progress: %s/%s", index, len(nodes))

        labels = tuple(str(label) for label in node.get("labels", []) if str(label))
        if not labels:
            logger.warning("      Skipped node %s with no labels", index)
            nodes_skipped += 1
            continue

        props = node.get("properties", {}).copy()
        if "Memory" in labels:
            props = _memory_restore_props(props, restore_time)

        node_backup_id_to_props[node["id"]] = (":".join(labels), props)
        node_uuid = props.get("id")
        if merge and node_uuid and node_uuid in existing_uuids:
            nodes_skipped += 1
            continue

        grouped_rows[labels].append({"props": props})

    nodes_created = 0
    for labels, rows in grouped_rows.items():
        for start in range(0, len(rows), max(1, batch_size)):
            batch = rows[start : start + max(1, batch_size)]
            stats = _execute_node_batch(graph, labels=labels, rows=batch)
            nodes_created += stats["created"]
            nodes_skipped += stats["skipped"]

    logger.info("      Progress: %s/%s", len(nodes), len(nodes))
    return {
        "created": nodes_created,
        "skipped": nodes_skipped,
        "node_backup_id_to_props": node_backup_id_to_props,
    }


def _relationship_batch_query(
    *,
    source_label: str,
    target_label: str,
    rel_type: str,
    merge: bool,
) -> str:
    rel_operator = "MERGE" if merge else "CREATE"
    return f"""
        UNWIND $rows AS row
        MATCH (a:{_escape_cypher_identifier(source_label)} {{id: row.source_id}})
        MATCH (b:{_escape_cypher_identifier(target_label)} {{id: row.target_id}})
        {rel_operator} (a)-[r:{_escape_cypher_identifier(rel_type)}]->(b)
        SET r += row.props
    """


def _create_restore_id_indexes(graph: Any, labels: list[str]) -> None:
    for label in sorted(set(labels)):
        if not label:
            continue
        query = f"CREATE INDEX FOR (n:{_escape_cypher_identifier(label)}) ON (n.id)"
        try:
            _query_falkordb(graph, query)
        except Exception as exc:
            if "already indexed" in str(exc).lower():
                continue
            logger.debug("      Could not create restore id index for %s: %s", label, exc)


def _execute_relationship_batch(
    graph: Any,
    *,
    key: tuple[str, str, str],
    rows: list[dict[str, Any]],
    merge: bool,
) -> dict[str, int]:
    source_label, target_label, rel_type = key
    query = _relationship_batch_query(
        source_label=source_label,
        target_label=target_label,
        rel_type=rel_type,
        merge=merge,
    )
    try:
        _query_falkordb(graph, query, {"rows": rows})
        return {"created": len(rows), "skipped": 0}
    except Exception as exc:
        if len(rows) == 1:
            logger.debug("      Skipped relationship %s: %s", rel_type, exc)
            return {"created": 0, "skipped": 1}

    created = 0
    skipped = 0
    for row in rows:
        result = _execute_relationship_batch(graph, key=key, rows=[row], merge=merge)
        created += result["created"]
        skipped += result["skipped"]
    return {"created": created, "skipped": skipped}


def _restore_relationship_batches(
    graph: Any,
    relationships: list[dict[str, Any]],
    node_backup_id_to_props: dict[Any, tuple[str, dict[str, Any]]],
    *,
    merge: bool,
    existing_rels: set[tuple[str, str, str]],
    batch_size: int,
) -> dict[str, int]:
    batch_size = max(1, batch_size)
    batches: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    created = 0
    skipped = 0

    def flush(key: tuple[str, str, str]) -> None:
        nonlocal created, skipped
        rows = batches[key]
        if not rows:
            return
        result = _execute_relationship_batch(graph, key=key, rows=rows, merge=merge)
        created += result["created"]
        skipped += result["skipped"]
        batches[key] = []

    for i, rel in enumerate(relationships):
        if i % 1000 == 0:
            logger.info("      Progress: %s/%s", i, len(relationships))

        source_backup_id = rel["source_id"]
        target_backup_id = rel["target_id"]

        if (
            source_backup_id not in node_backup_id_to_props
            or target_backup_id not in node_backup_id_to_props
        ):
            logger.warning("      Skipping relationship %s - missing node IDs", rel["type"])
            skipped += 1
            continue

        source_labels, source_props = node_backup_id_to_props[source_backup_id]
        target_labels, target_props = node_backup_id_to_props[target_backup_id]
        source_uuid = source_props.get("id")
        target_uuid = target_props.get("id")

        if not source_uuid or not target_uuid:
            logger.warning("      Skipping relationship %s - missing UUID properties", rel["type"])
            skipped += 1
            continue

        rel_type = rel["type"]
        if merge and (rel_type, source_uuid, target_uuid) in existing_rels:
            skipped += 1
            continue

        key = (_first_label(source_labels), _first_label(target_labels), rel_type)
        batches[key].append(
            {
                "source_id": source_uuid,
                "target_id": target_uuid,
                "props": rel.get("properties", {}) or {},
            }
        )
        if len(batches[key]) >= batch_size:
            flush(key)

    for key in list(batches):
        flush(key)

    return {"created": created, "skipped": skipped}


@contextmanager
def resolve_backup_dir(backup_path: Path) -> Iterator[Path]:
    """Resolve a backup directory or extract a downloaded tar.gz backup to a temp dir."""
    if backup_path.is_file() and _is_tar_gz(backup_path):
        with tempfile.TemporaryDirectory(prefix="automem-restore-") as temp_dir:
            extracted = Path(temp_dir)
            _safe_extract_tar_gz(backup_path, extracted)
            yield extracted
        return

    yield backup_path


def _qdrant_collection_kwargs(vector_size: int) -> dict[str, Any]:
    try:
        from qdrant_client.http.models import HnswConfigDiff, OptimizersConfigDiff
    except Exception:
        from qdrant_client.models import HnswConfigDiff, OptimizersConfigDiff
    from qdrant_client.models import Distance, VectorParams

    vector_kwargs: dict[str, Any] = {"size": vector_size, "distance": Distance.COSINE}
    if QDRANT_RESTORE_VECTOR_ON_DISK is not None:
        vector_kwargs["on_disk"] = QDRANT_RESTORE_VECTOR_ON_DISK
    kwargs: dict[str, Any] = {
        "collection_name": QDRANT_COLLECTION,
        "vectors_config": VectorParams(**vector_kwargs),
    }
    optimizer_kwargs: dict[str, Any] = {}
    if QDRANT_RESTORE_INDEXING_THRESHOLD is not None:
        optimizer_kwargs["indexing_threshold"] = QDRANT_RESTORE_INDEXING_THRESHOLD
    if QDRANT_RESTORE_DEFAULT_SEGMENT_NUMBER is not None:
        optimizer_kwargs["default_segment_number"] = QDRANT_RESTORE_DEFAULT_SEGMENT_NUMBER
    if QDRANT_RESTORE_MEMMAP_THRESHOLD is not None:
        optimizer_kwargs["memmap_threshold"] = QDRANT_RESTORE_MEMMAP_THRESHOLD
    if optimizer_kwargs:
        kwargs["optimizers_config"] = OptimizersConfigDiff(**optimizer_kwargs)
    if QDRANT_RESTORE_HNSW_M is not None:
        kwargs["hnsw_config"] = HnswConfigDiff(m=QDRANT_RESTORE_HNSW_M)
    if QDRANT_RESTORE_ON_DISK_PAYLOAD is not None:
        kwargs["on_disk_payload"] = QDRANT_RESTORE_ON_DISK_PAYLOAD
    return kwargs


class AutoMemRestore:
    """Handles restoration of AutoMem data from backups."""

    def __init__(
        self,
        backup_dir: Path,
        dry_run: bool = False,
        force: bool = False,
        merge: bool = False,
    ):
        self.backup_dir = backup_dir
        self.dry_run = dry_run
        self.force = force
        self.merge = merge

    def _qdrant_client(self) -> QdrantClient:
        kwargs: dict[str, Any] = {
            "url": QDRANT_URL,
            "api_key": QDRANT_API_KEY,
            "timeout": QDRANT_TIMEOUT_SECONDS,
        }
        if QDRANT_PREFER_GRPC:
            kwargs["prefer_grpc"] = True
        if QDRANT_GRPC_PORT is not None:
            kwargs["grpc_port"] = QDRANT_GRPC_PORT
        return QdrantClient(
            **kwargs,
        )

    def _wait_for_qdrant_collection(self) -> QdrantClient:
        attempts = max(1, QDRANT_RESTORE_RETRIES * 3)
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            client = self._qdrant_client()
            try:
                client.get_collection(QDRANT_COLLECTION)
                return client
            except Exception as exc:
                last_error = exc
                if attempt == attempts:
                    break
                time.sleep(QDRANT_RESTORE_RETRY_DELAY_SECONDS)

        if last_error:
            raise last_error
        raise RuntimeError("Qdrant did not become ready")

    def _upsert_qdrant_batch(
        self,
        client: QdrantClient,
        points: list[PointStruct],
        *,
        offset: int,
        total: int,
    ) -> QdrantClient:
        retries = max(0, QDRANT_RESTORE_RETRIES)
        for attempt in range(0, retries + 1):
            try:
                try:
                    client.upsert(
                        collection_name=QDRANT_COLLECTION,
                        points=points,
                        wait=QDRANT_RESTORE_WAIT,
                    )
                except TypeError:
                    client.upsert(collection_name=QDRANT_COLLECTION, points=points)
                return client
            except Exception as exc:
                if attempt >= retries:
                    raise
                logger.warning(
                    "      Qdrant upsert failed at %s/%s " "(attempt %s/%s): %s; waiting to retry",
                    offset,
                    total,
                    attempt + 1,
                    retries + 1,
                    exc,
                )
                time.sleep(QDRANT_RESTORE_RETRY_DELAY_SECONDS)
                client = self._wait_for_qdrant_collection()

        return client

    def _wait_for_qdrant_points(self, client: QdrantClient, expected_count: int) -> None:
        deadline = time.monotonic() + max(1.0, QDRANT_RESTORE_READY_TIMEOUT_SECONDS)
        while True:
            info = client.get_collection(QDRANT_COLLECTION)
            current_count = int(getattr(info, "points_count", 0) or 0)
            if current_count >= expected_count:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Qdrant restore did not reach {expected_count} points "
                    f"within {QDRANT_RESTORE_READY_TIMEOUT_SECONDS:g}s "
                    f"(current={current_count})"
                )
            time.sleep(QDRANT_RESTORE_RETRY_DELAY_SECONDS)

    def find_latest_backup(self, backup_type: str) -> Path:
        """Find the most recent backup file."""
        backup_path = self.backup_dir / backup_type
        backup_files = sorted(
            backup_path.glob("*.json.gz"), key=lambda p: p.stat().st_mtime, reverse=True
        )

        if not backup_files:
            raise FileNotFoundError(f"No {backup_type} backups found in {backup_path}")

        return backup_files[0]

    def find_backup_by_timestamp(self, backup_type: str, timestamp: str) -> Path:
        """Find backup file by timestamp."""
        backup_file = self.backup_dir / backup_type / f"{backup_type}_{timestamp}.json.gz"
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup not found: {backup_file}")
        return backup_file

    def restore_falkordb(self, backup_file: Path) -> Dict[str, Any]:
        """Restore FalkorDB graph from JSON backup."""
        logger.info(f"📊 Restoring FalkorDB from {backup_file.name}...")

        # Load backup data
        with gzip.open(backup_file, "rt", encoding="utf-8") as f:
            backup_data = json.load(f)

        logger.info(
            f"   Backup contains {len(backup_data['nodes'])} nodes, "
            f"{len(backup_data['relationships'])} relationships"
        )

        if self.dry_run:
            logger.info("   [DRY RUN] Would restore to FalkorDB")
            return {
                "nodes": len(backup_data["nodes"]),
                "relationships": len(backup_data["relationships"]),
                "dry_run": True,
            }

        # Connect to FalkorDB
        db = FalkorDB(
            host=FALKORDB_HOST,
            port=FALKORDB_PORT,
            password=FALKORDB_PASSWORD,
            username="default" if FALKORDB_PASSWORD else None,
        )
        graph = db.select_graph(FALKORDB_GRAPH)

        # Warning about existing data
        existing_count = _query_falkordb(graph, "MATCH (n) RETURN count(*) as count")
        existing_nodes = existing_count.result_set[0][0] if existing_count.result_set else 0

        if existing_nodes > 0:
            if self.merge:
                logger.info(
                    f"📥 Import mode: Graph '{FALKORDB_GRAPH}' contains "
                    f"{existing_nodes} existing nodes - will merge with backup"
                )
            else:
                logger.warning(
                    f"⚠️  Graph '{FALKORDB_GRAPH}' contains {existing_nodes} existing nodes!"
                )
                if not self.force:
                    response = input("   Delete existing data and restore? [y/N]: ")
                    if response.lower() != "y":
                        logger.info("   Restore cancelled")
                        return {"cancelled": True}
                else:
                    logger.info("   --force flag set, proceeding with restore")

                # Clear existing data
                logger.info("   🗑️  Clearing existing graph data...")
                _query_falkordb(graph, "MATCH (n) DETACH DELETE n")

        logger.info("   📇 Ensuring FalkorDB id indexes...")
        restore_labels = [
            _first_label(":".join(node.get("labels", [])))
            for node in backup_data["nodes"]
            if node.get("labels")
        ]
        _create_restore_id_indexes(graph, restore_labels)

        # Restore nodes
        logger.info(f"   📥 Restoring {len(backup_data['nodes'])} nodes...")
        restore_time = datetime.now(timezone.utc).isoformat()

        # If merging, get existing node UUIDs to skip duplicates
        existing_uuids = set()
        if self.merge:
            existing_nodes_result = _query_falkordb(
                graph, "MATCH (n) WHERE n.id IS NOT NULL RETURN n.id as id"
            )
            if existing_nodes_result.result_set:
                existing_uuids = {row[0] for row in existing_nodes_result.result_set}

        node_stats = _restore_node_batches(
            graph,
            backup_data["nodes"],
            merge=self.merge,
            existing_uuids=existing_uuids,
            restore_time=restore_time,
            batch_size=FALKORDB_RESTORE_NODE_BATCH_SIZE,
        )
        node_backup_id_to_props = node_stats["node_backup_id_to_props"]
        nodes_created = node_stats["created"]
        nodes_skipped = node_stats["skipped"]

        logger.info(
            f"   ✅ Restored {nodes_created}/{len(backup_data['nodes'])} nodes"
            + (f" (skipped {nodes_skipped} existing)" if nodes_skipped > 0 else "")
        )

        # Restore relationships using UUID matching
        logger.info(f"   📥 Restoring {len(backup_data['relationships'])} relationships...")
        rel_created = 0
        rel_skipped = 0

        # If merging, get existing relationships to skip duplicates
        existing_rels = set()
        if self.merge:
            existing_rels_result = _query_falkordb(
                graph,
                """
                MATCH (a)-[r]->(b)
                WHERE a.id IS NOT NULL AND b.id IS NOT NULL
                RETURN type(r) as rel_type, a.id as source_id, b.id as target_id
            """,
            )
            if existing_rels_result.result_set:
                existing_rels = {
                    (row[0], row[1], row[2]) for row in existing_rels_result.result_set
                }

        rel_stats = _restore_relationship_batches(
            graph,
            backup_data["relationships"],
            node_backup_id_to_props,
            merge=self.merge,
            existing_rels=existing_rels,
            batch_size=FALKORDB_RESTORE_REL_BATCH_SIZE,
        )
        rel_created = rel_stats["created"]
        rel_skipped = rel_stats["skipped"]

        logger.info(
            f"   ✅ Restored {rel_created}/{len(backup_data['relationships'])} relationships"
            + (f" (skipped {rel_skipped} existing)" if rel_skipped > 0 else "")
        )

        return {
            "nodes_restored": nodes_created,
            "nodes_skipped": nodes_skipped if self.merge else 0,
            "nodes_attempted": len(backup_data["nodes"]),
            "relationships_restored": rel_created,
            "relationships_skipped": rel_skipped if self.merge else 0,
            "relationships_attempted": len(backup_data["relationships"]),
            "dry_run": False,
            "merge_mode": self.merge,
        }

    def restore_qdrant(self, backup_file: Path) -> Dict[str, Any]:
        """Restore Qdrant collection from JSON backup."""
        logger.info(f"🔍 Restoring Qdrant from {backup_file.name}...")

        # Load backup data
        with gzip.open(backup_file, "rt", encoding="utf-8") as f:
            backup_data = json.load(f)

        logger.info(f"   Backup contains {len(backup_data['points'])} points")

        if self.dry_run:
            logger.info("   [DRY RUN] Would restore to Qdrant")
            return {"points": len(backup_data["points"]), "dry_run": True}

        # Connect to Qdrant
        client = self._qdrant_client()

        # Check if collection exists
        try:
            collection_info = client.get_collection(QDRANT_COLLECTION)
            existing_points = collection_info.points_count

            if self.merge:
                logger.info(
                    f"📥 Import mode: Collection '{QDRANT_COLLECTION}' contains "
                    f"{existing_points} existing points - will merge with backup"
                )
            else:
                logger.warning(
                    f"⚠️  Collection '{QDRANT_COLLECTION}' contains "
                    f"{existing_points} existing points!"
                )
                if not self.force:
                    response = input("   Delete existing points and restore? [y/N]: ")
                    if response.lower() != "y":
                        logger.info("   Restore cancelled")
                        return {"cancelled": True}
                else:
                    logger.info("   --force flag set, proceeding with restore")

                # Clear existing points
                logger.info("   🗑️  Clearing existing collection...")
                client.delete_collection(QDRANT_COLLECTION)

                # Recreate collection
                client.create_collection(
                    **_qdrant_collection_kwargs(backup_data["stats"]["vector_size"])
                )
        except Exception as e:
            logger.info(f"   Creating new collection (previous: {e})")
            client.create_collection(
                **_qdrant_collection_kwargs(backup_data["stats"]["vector_size"])
            )

        # Restore points in batches
        logger.info(f"   📥 Restoring {len(backup_data['points'])} points...")
        batch_size = max(1, QDRANT_RESTORE_BATCH_SIZE)

        for i in range(0, len(backup_data["points"]), batch_size):
            batch = backup_data["points"][i : i + batch_size]
            logger.info(f"      Progress: {i}/{len(backup_data['points'])}")

            points = [
                PointStruct(id=point["id"], vector=point["vector"], payload=point["payload"])
                for point in batch
            ]

            client = self._upsert_qdrant_batch(
                client,
                points,
                offset=i,
                total=len(backup_data["points"]),
            )
            if QDRANT_RESTORE_BATCH_DELAY_SECONDS > 0 and i + batch_size < len(
                backup_data["points"]
            ):
                time.sleep(QDRANT_RESTORE_BATCH_DELAY_SECONDS)

        points_count = len(backup_data["points"])
        self._wait_for_qdrant_points(client, points_count)
        logger.info(f"   ✅ Restored {points_count} points (upserted - existing updated)")

        return {
            "points": len(backup_data["points"]),
            "dry_run": False,
            "merge_mode": self.merge,
        }

    def run_restore(
        self,
        timestamp: str = None,
        falkordb_only: bool = False,
        qdrant_only: bool = False,
    ) -> Dict[str, Any]:
        """Run full restore process."""
        logger.info("🚀 Starting AutoMem restore")

        results = {"falkordb": None, "qdrant": None}

        try:
            # Restore FalkorDB
            if not qdrant_only:
                if timestamp:
                    falkor_backup = self.find_backup_by_timestamp("falkordb", timestamp)
                else:
                    falkor_backup = self.find_latest_backup("falkordb")

                results["falkordb"] = self.restore_falkordb(falkor_backup)

            # Restore Qdrant
            if not falkordb_only:
                if timestamp:
                    qdrant_backup = self.find_backup_by_timestamp("qdrant", timestamp)
                else:
                    qdrant_backup = self.find_latest_backup("qdrant")

                results["qdrant"] = self.restore_qdrant(qdrant_backup)

            logger.info("✅ Restore completed successfully")
            return results

        except Exception as e:
            logger.error(f"❌ Restore failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="AutoMem restore tool - restores FalkorDB and Qdrant from backups",
        epilog="""
Examples:
  # Restore from latest backup
  python scripts/restore_from_backup.py

  # Restore from specific timestamp
  python scripts/restore_from_backup.py --backup-timestamp 20251019_085625

  # Restore from downloaded API backup tarball
  python scripts/restore_from_backup.py --backup-dir snapshot.tar.gz

  # Dry run (preview only)
  python scripts/restore_from_backup.py --dry-run

  # Restore only FalkorDB
  python scripts/restore_from_backup.py --falkordb-only

  # Restore only Qdrant
  python scripts/restore_from_backup.py --qdrant-only

  # Import/merge without deleting existing data
  python scripts/restore_from_backup.py --import
        """,
    )
    parser.add_argument(
        "--backup-dir",
        type=str,
        default=str(BACKUP_DIR),
        help="Directory containing backup files or downloaded .tar.gz (default: ./backups)",
    )
    parser.add_argument(
        "--backup-timestamp",
        type=str,
        help="Specific backup timestamp to restore (e.g., 20251019_085625)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview restore without making changes"
    )
    parser.add_argument(
        "--falkordb-only",
        action="store_true",
        help="Restore only FalkorDB (skip Qdrant)",
    )
    parser.add_argument(
        "--qdrant-only", action="store_true", help="Restore only Qdrant (skip FalkorDB)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force restore without confirmation prompts",
    )
    parser.add_argument(
        "--import",
        dest="merge",
        action="store_true",
        help="Import/merge backup data without deleting existing data (skips duplicates)",
    )

    args = parser.parse_args()

    try:
        with resolve_backup_dir(Path(args.backup_dir)) as backup_dir:
            restore = AutoMemRestore(
                backup_dir=backup_dir,
                dry_run=args.dry_run,
                force=args.force,
                merge=args.merge,
            )
            results = restore.run_restore(
                timestamp=args.backup_timestamp,
                falkordb_only=args.falkordb_only,
                qdrant_only=args.qdrant_only,
            )
        print(json.dumps(results, indent=2))
        sys.exit(0)
    except Exception as e:
        logger.error(f"Restore failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
