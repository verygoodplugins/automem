#!/usr/bin/env python3
"""Salvage missing FalkorDB Memory nodes from Qdrant payloads.

This inserts missing memories with their original IDs and the API's property
shape. Existing nodes, entities and relationships are preserved. Qdrant contains
no graph edges: use restore_from_backup.py with a FalkorDB backup for a full
restore. Re-enrichment can rebuild some derived relationships, not original ones.
"""

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from falkordb import FalkorDB
from qdrant_client import QdrantClient

# Support the documented `python scripts/recover_from_qdrant.py` invocation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from automem.utils.tags import _compute_tag_prefixes, _normalize_tag_list  # noqa: E402
from automem.utils.time import _normalize_timestamp, utc_now  # noqa: E402

load_dotenv()
load_dotenv(Path.home() / ".config" / "automem" / ".env")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "memories")
FALKORDB_HOST = os.getenv("FALKORDB_HOST", "localhost")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))
FALKORDB_PASSWORD = os.getenv("FALKORDB_PASSWORD")
FALKORDB_GRAPH = os.getenv("FALKORDB_GRAPH", "memories")
BATCH_SIZE = 50


def get_all_memories() -> List[Dict[str, Any]]:
    """Fetch the complete source before writing anything to FalkorDB."""
    print(f"Connecting to Qdrant collection '{QDRANT_COLLECTION}'")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    try:
        memories = []
        offset = None
        while True:
            points, next_offset = client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=BATCH_SIZE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            memories.extend({"id": point.id, "payload": point.payload} for point in points)
            if next_offset is None:
                break
            offset = next_offset
            time.sleep(0.1)
        return memories
    finally:
        client.close()


def memory_properties(memory: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a source point and use the API's canonical tag/time/metadata shape.

    Write directly to the graph because public POST /memory generates a new ID.
    Reusing the tag and time helpers keeps recovered nodes compatible with recall.
    """
    memory_id = memory.get("id")
    if isinstance(memory_id, bool) or not isinstance(memory_id, (str, int)):
        raise ValueError("Memory must have a string or integer ID")
    if not str(memory_id).strip():
        raise ValueError("Memory ID must not be empty")
    payload = memory.get("payload")
    if not isinstance(payload, dict) or not isinstance(payload.get("content"), str):
        raise ValueError(f"Memory {memory_id} must have a payload with string content")

    metadata = payload.get("metadata")
    if metadata is None:
        metadata = {}
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    if not isinstance(metadata, dict):
        raise ValueError(f"Memory {memory_id} metadata must be an object")
    tags = _normalize_tag_list(payload.get("tags"))
    timestamp = (
        _normalize_timestamp(payload["timestamp"]) if payload.get("timestamp") else utc_now()
    )
    updated_at = (
        _normalize_timestamp(payload["updated_at"]) if payload.get("updated_at") else timestamp
    )
    last_accessed = (
        _normalize_timestamp(payload["last_accessed"])
        if payload.get("last_accessed")
        else updated_at
    )
    properties = {
        "id": str(memory_id),
        "content": payload["content"],
        "timestamp": timestamp,
        "importance": payload.get("importance", 0.5),
        "type": payload.get("type", "Context"),
        "confidence": payload.get("confidence", 0.6),
        "tags": tags,
        "tag_prefixes": _compute_tag_prefixes(tags),
        "metadata": json.dumps(metadata, default=str),
        "t_valid": (
            _normalize_timestamp(payload["t_valid"]) if payload.get("t_valid") else timestamp
        ),
        "t_invalid": (
            _normalize_timestamp(payload["t_invalid"]) if payload.get("t_invalid") else None
        ),
        "updated_at": updated_at,
        "last_accessed": last_accessed,
        "archived": payload.get("archived", False),
        # New memories have no decay score yet; vector writers persist JSON null.
        "relevance_score": (
            1.0 if payload.get("relevance_score") is None else payload["relevance_score"]
        ),
        "processed": False,
    }
    for name in ("importance", "confidence", "relevance_score"):
        value = properties[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
        ):
            raise ValueError(f"Memory {memory_id} {name} must be a finite number")
    if not isinstance(properties["archived"], bool):
        raise ValueError(f"Memory {memory_id} archived must be a boolean")
    if not isinstance(properties["type"], str):
        raise ValueError(f"Memory {memory_id} type must be a string")
    return properties


def restore_memory_to_graph_only(memory: Dict[str, Any], client: FalkorDB) -> bool:
    """Insert a missing node; never overwrite a matching node or delete graph data."""
    try:
        properties = memory_properties(memory)
        graph = client.select_graph(FALKORDB_GRAPH)
        result = graph.query(
            """
            MERGE (m:Memory {id: $id})
            ON CREATE SET m += $properties
            RETURN m.id
            """,
            {"id": properties["id"], "properties": properties},
        )
        if not getattr(result, "result_set", None):
            raise RuntimeError("FalkorDB did not confirm the memory write")
        return True
    except Exception as exc:
        print(f"Failed to recover memory {memory.get('id')}: {exc}", file=sys.stderr)
        return False


def main() -> int:
    """Validate the entire source, then insert missing nodes without clearing data."""
    print(f"Recovering missing memories into FalkorDB graph '{FALKORDB_GRAPH}'")
    print(
        "Existing graph data is preserved. Qdrant cannot restore relationships or entities; "
        "use a FalkorDB backup for a full restore."
    )
    try:
        memories = get_all_memories()
        if not memories:
            raise ValueError("No memories found in Qdrant; graph left unchanged")
        for memory in memories:
            memory_properties(memory)
    except Exception as exc:
        print(f"Source validation failed; graph left unchanged: {exc}", file=sys.stderr)
        return 1

    try:
        client = FalkorDB(
            host=FALKORDB_HOST,
            port=FALKORDB_PORT,
            password=FALKORDB_PASSWORD,
            username="default" if FALKORDB_PASSWORD else None,
        )
    except Exception as exc:
        print(f"Failed to connect to FalkorDB: {exc}", file=sys.stderr)
        return 1

    failed_count = 0
    for memory in memories:
        if not restore_memory_to_graph_only(memory, client):
            failed_count += 1
    print(f"Confirmed: {len(memories) - failed_count}; failed: {failed_count}")
    print("Confirmed memories include existing nodes, which were left unchanged.")
    return 1 if failed_count else 0


if __name__ == "__main__":
    sys.exit(main())
