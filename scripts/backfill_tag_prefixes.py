from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from automem.utils.tags import _compute_tag_prefixes, _normalize_tag_list


def _parse_metadata(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except Exception:
            return {"_raw_metadata": value}
        if isinstance(parsed, dict):
            return parsed
        return {"_raw_metadata": parsed}
    return {"_raw_metadata": value}


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _connect_falkordb():
    try:
        from falkordb import FalkorDB  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing falkordb dependency; install requirements.txt") from exc

    host = _get_env("FALKORDB_HOST")
    port_raw = _get_env("FALKORDB_PORT")
    password = _get_env("FALKORDB_PASSWORD")
    username = _get_env("FALKORDB_USERNAME") or ("default" if password else None)
    graph_name = _get_env("FALKORDB_GRAPH", "memories") or "memories"

    if not host or not port_raw:
        raise RuntimeError("FALKORDB_HOST and FALKORDB_PORT are required")

    client = FalkorDB(host=host, port=int(port_raw), username=username, password=password)
    graph = client.select_graph(graph_name)
    return graph


def _connect_qdrant():
    try:
        from qdrant_client import QdrantClient  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing qdrant-client dependency; install requirements.txt") from exc

    url = _get_env("QDRANT_URL")
    api_key = _get_env("QDRANT_API_KEY")
    collection = _get_env("QDRANT_COLLECTION", "memories") or "memories"

    if not url:
        return None, None

    return QdrantClient(url=url, api_key=api_key), collection


def _iter_memories(graph: Any, batch_size: int) -> Sequence[Tuple[str, List[str], List[str], Any]]:
    skip = 0
    while True:
        result = graph.query(
            """
            MATCH (m:Memory)
            RETURN m.id, m.tags, m.tag_prefixes, m.metadata
            SKIP $skip
            LIMIT $limit
            """,
            {"skip": skip, "limit": batch_size},
        )
        rows = list(getattr(result, "result_set", []) or [])
        if not rows:
            return
        for row in rows:
            if not row or not row[0]:
                continue
            memory_id = str(row[0])
            tags = _normalize_tag_list(row[1] if len(row) > 1 else None)
            tag_prefixes = row[2] if len(row) > 2 else []
            if not isinstance(tag_prefixes, list):
                tag_prefixes = []
            metadata = row[3] if len(row) > 3 else None
            yield memory_id, tags, tag_prefixes, metadata
        skip += len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill tag_prefixes in FalkorDB and sync tags/tag_prefixes in Qdrant payload."
    )
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument(
        "--limit", type=int, default=0, help="Process at most N memories (0 = all)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute changes without writing to FalkorDB or Qdrant.",
    )
    parser.add_argument(
        "--no-qdrant",
        action="store_true",
        help="Skip syncing Qdrant payload (FalkorDB only).",
    )
    parser.add_argument(
        "--force-qdrant",
        action="store_true",
        help="Sync Qdrant payload even when FalkorDB tag_prefixes already match.",
    )
    args = parser.parse_args()

    graph = _connect_falkordb()
    qdrant_client, qdrant_collection = _connect_qdrant()
    sync_qdrant = (
        (not args.no_qdrant) and qdrant_client is not None and qdrant_collection is not None
    )

    start = time.time()
    processed = 0
    graph_updates = 0
    qdrant_updates = 0
    qdrant_failures = 0

    pending_graph_updates: List[Dict[str, Any]] = []

    def flush_graph_updates() -> None:
        nonlocal graph_updates, pending_graph_updates
        if args.dry_run or not pending_graph_updates:
            pending_graph_updates = []
            return
        graph.query(
            """
            UNWIND $rows AS row
            MATCH (m:Memory {id: row.id})
            SET m.tag_prefixes = row.tag_prefixes
            """,
            {"rows": pending_graph_updates},
        )
        graph_updates += len(pending_graph_updates)
        pending_graph_updates = []

    for memory_id, tags, existing_prefixes, metadata_raw in _iter_memories(graph, args.batch_size):
        processed += 1
        if args.limit and processed > args.limit:
            break

        computed_prefixes = _compute_tag_prefixes(tags)
        needs_graph_update = existing_prefixes != computed_prefixes

        if needs_graph_update:
            pending_graph_updates.append({"id": memory_id, "tag_prefixes": computed_prefixes})
            if len(pending_graph_updates) >= args.batch_size:
                flush_graph_updates()

        if sync_qdrant and (args.force_qdrant or needs_graph_update):
            try:
                qdrant_client.set_payload(  # type: ignore[union-attr]
                    collection_name=qdrant_collection,  # type: ignore[arg-type]
                    points=[memory_id],
                    payload={
                        "tags": tags,
                        "tag_prefixes": computed_prefixes,
                        "metadata": _parse_metadata(metadata_raw),
                    },
                )
                qdrant_updates += 1
            except Exception:
                qdrant_failures += 1

        if processed % 250 == 0:
            elapsed = time.time() - start
            rate = processed / elapsed if elapsed else 0.0
            print(
                f"processed={processed} graph_updates={graph_updates + len(pending_graph_updates)} "
                f"qdrant_updates={qdrant_updates} qdrant_failures={qdrant_failures} rate={rate:.1f}/s"
            )

    flush_graph_updates()

    elapsed = time.time() - start
    print(
        "done",
        json.dumps(
            {
                "processed": processed,
                "graph_updates": graph_updates,
                "qdrant_updates": qdrant_updates,
                "qdrant_failures": qdrant_failures,
                "elapsed_seconds": round(elapsed, 2),
                "dry_run": bool(args.dry_run),
                "qdrant_enabled": bool(sync_qdrant),
            }
        ),
    )

    if qdrant_failures:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
