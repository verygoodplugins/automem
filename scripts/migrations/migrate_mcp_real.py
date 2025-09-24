#!/usr/bin/env python3
"""SQLite → AutoMem migration utility.

This script extracts memories from the legacy MCP SQLite-vec database and
replays them into the AutoMem service so they benefit from the new graph and
vector infrastructure. It keeps original timestamps, rehydrates tags, and
stores the legacy metadata for future reference.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import requests

DEFAULT_AUTOMEM_URL = "https://automem.up.railway.app"
DEFAULT_DB_PATH = Path.home() / "Library" / "Application Support" / "mcp-memory" / "sqlite_vec.db"


IMPORTANCE_MAP: Dict[str, float] = {
    "critical": 0.95,
    "highest": 0.9,
    "high": 0.85,
    "medium": 0.6,
    "normal": 0.5,
    "low": 0.3,
    "lowest": 0.2,
}


@dataclass
class LegacyMemory:
    """Representation of a row stored by the old MCP SQLite backend."""

    row_id: int
    content_hash: str
    content: str
    tags_raw: Optional[str]
    memory_type: Optional[str]
    metadata_json: Optional[str]
    created_at_iso: str
    updated_at_iso: str
    created_at_epoch: Optional[float]
    updated_at_epoch: Optional[float]

    def parse_metadata(self) -> Dict[str, Any]:
        if not self.metadata_json:
            return {}
        try:
            decoded = json.loads(self.metadata_json)
        except json.JSONDecodeError:
            return {"raw_metadata": self.metadata_json}

        if isinstance(decoded, dict):
            return decoded

        # Keep non-dict payloads so nothing is lost during migration.
        return {"raw_metadata": decoded}


class AutoMemMigrator:
    """Streams memories from SQLite into AutoMem."""

    def __init__(
        self,
        db_path: Path,
        automem_url: str,
        batch_size: int = 25,
        sleep_seconds: float = 0.1,
        dry_run: bool = False,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> None:
        self.db_path = db_path
        self.automem_url = automem_url.rstrip("/")
        self.batch_size = max(1, batch_size)
        self.sleep_seconds = max(0.0, sleep_seconds)
        self.dry_run = dry_run
        self.limit = limit
        self.offset = max(0, offset)
        self.session = requests.Session()

    def run(self) -> int:
        if not self.db_path.exists():
            print(f"❌ SQLite database not found at {self.db_path}", file=sys.stderr)
            return 1

        print("🚀 Starting migration from SQLite to AutoMem")
        print(f"   • Database : {self.db_path}")
        print(f"   • AutoMem  : {self.automem_url}")
        print(f"   • Dry run  : {'yes' if self.dry_run else 'no'}")

        successes = 0
        failures = 0
        processed = 0

        for batch in self._stream_legacy_batches():
            payloads = [self._build_payload(memory) for memory in batch]

            for memory, payload in zip(batch, payloads):
                processed += 1

                if self.dry_run:
                    self._print_preview(memory, payload)
                    successes += 1
                    continue

                try:
                    response = self.session.post(
                        f"{self.automem_url}/memory",
                        json=payload,
                        timeout=30,
                    )
                except requests.RequestException as exc:
                    failures += 1
                    print(
                        f"❌ Failed to migrate {memory.content_hash[:8]}…: {exc}",
                        file=sys.stderr,
                    )
                    continue

                if response.status_code in (200, 201):
                    successes += 1
                else:
                    failures += 1
                    try:
                        details = response.json()
                    except ValueError:
                        details = {"body": response.text}
                    print(
                        "❌ AutoMem rejected memory {hash}: {status} {details}".format(
                            hash=memory.content_hash[:12],
                            status=response.status_code,
                            details=details,
                        ),
                        file=sys.stderr,
                    )

                if self.sleep_seconds:
                    time.sleep(self.sleep_seconds)

            print(
                f"… processed {processed} memories (ok: {successes}, failed: {failures})",
                flush=True,
            )

        print("\n✅ Migration complete")
        print(f"   • Total migrated : {successes}")
        if failures:
            print(f"   • Failed         : {failures}")

        return 0 if failures == 0 else 2

    def _stream_legacy_batches(self) -> Iterator[List[LegacyMemory]]:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row

        query = (
            "SELECT id, content_hash, content, tags, memory_type, metadata, "
            "created_at_iso, updated_at_iso, created_at, updated_at "
            "FROM memories ORDER BY id"
        )

        cursor = connection.execute(query)

        # Skip offset rows if requested.
        if self.offset:
            cursor.fetchmany(self.offset)

        rows_yielded = 0
        try:
            while True:
                if self.limit is not None and rows_yielded >= self.limit:
                    break

                remaining = None
                if self.limit is not None:
                    remaining = self.limit - rows_yielded
                    if remaining <= 0:
                        break

                fetch_size = self.batch_size
                if remaining is not None:
                    fetch_size = min(fetch_size, remaining)

                rows = cursor.fetchmany(fetch_size)
                if not rows:
                    break

                batch = [self._row_to_memory(row) for row in rows]
                rows_yielded += len(batch)
                yield batch
        finally:
            cursor.close()
            connection.close()

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> LegacyMemory:
        return LegacyMemory(
            row_id=row["id"],
            content_hash=row["content_hash"],
            content=row["content"],
            tags_raw=row["tags"],
            memory_type=row["memory_type"],
            metadata_json=row["metadata"],
            created_at_iso=row["created_at_iso"],
            updated_at_iso=row["updated_at_iso"],
            created_at_epoch=row["created_at"],
            updated_at_epoch=row["updated_at"],
        )

    def _build_payload(self, memory: LegacyMemory) -> Dict[str, Any]:
        metadata = memory.parse_metadata()
        tags = self._merge_tags(memory.tags_raw, metadata.get("tags"))

        importance = self._derive_importance(metadata)

        legacy_packet: Dict[str, Any] = {
            "storage": "mcp-sqlite-vec",
            "row_id": memory.row_id,
            "content_hash": memory.content_hash,
            "memory_type": memory.memory_type,
            "created_at_epoch": memory.created_at_epoch,
            "updated_at_epoch": memory.updated_at_epoch,
            "tags_raw": memory.tags_raw,
        }

        if metadata:
            legacy_packet["metadata"] = metadata

        last_accessed = metadata.get("last_accessed") if isinstance(metadata, dict) else None
        if not isinstance(last_accessed, str) or not last_accessed.strip():
            last_accessed = memory.updated_at_iso

        payload: Dict[str, Any] = {
            "id": memory.content_hash,
            "content": memory.content,
            "tags": tags,
            "importance": importance,
            "timestamp": memory.created_at_iso,
            "updated_at": memory.updated_at_iso,
            "last_accessed": last_accessed,
            "metadata": {
                "origin": "migration:mcp-sqlite-vec",
                "legacy": legacy_packet,
            },
        }

        # Preserve explicit `t_valid` or `t_invalid` markers if they exist in metadata.
        for key in ("t_valid", "t_invalid"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                payload[key] = value

        return payload

    @staticmethod
    def _merge_tags(primary: Optional[str], metadata_tags: Any) -> List[str]:
        collected: List[str] = []

        if primary:
            collected.extend(part.strip() for part in primary.split(",") if part.strip())

        collected.extend(AutoMemMigrator._coerce_tag_list(metadata_tags))

        seen = set()
        normalized: List[str] = []
        for tag in collected:
            lower = tag.lower()
            if lower not in seen:
                seen.add(lower)
                normalized.append(tag)
        return normalized

    @staticmethod
    def _coerce_tag_list(source: Any) -> List[str]:
        if source is None:
            return []

        if isinstance(source, list):
            return [str(item).strip() for item in source if str(item).strip()]

        if isinstance(source, str):
            stripped = source.strip()
            if not stripped:
                return []

            # Some entries store JSON inside the string; try to decode first.
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                decoded = None

            if isinstance(decoded, list):
                return [str(item).strip() for item in decoded if str(item).strip()]

            # Fall back to comma-separated parsing.
            return [part.strip() for part in stripped.split(",") if part.strip()]

        return []

    @staticmethod
    def _derive_importance(metadata: Dict[str, Any]) -> float:
        candidate = metadata.get("importance") or metadata.get("priority")
        if candidate is None:
            return 0.6

        if isinstance(candidate, (int, float)):
            return AutoMemMigrator._clamp_importance(float(candidate))

        if isinstance(candidate, str):
            lowered = candidate.strip().lower()
            if lowered in IMPORTANCE_MAP:
                return IMPORTANCE_MAP[lowered]

            try:
                numeric = float(lowered)
            except ValueError:
                return 0.6
            return AutoMemMigrator._clamp_importance(numeric)

        return 0.6

    @staticmethod
    def _clamp_importance(value: float) -> float:
        if value < 0:
            return 0.0
        if value > 1:
            return 1.0
        return value

    def _print_preview(self, memory: LegacyMemory, payload: Dict[str, Any]) -> None:
        print(
            f"• {memory.row_id}: {memory.content_hash[:12]} | tags={payload['tags']} | "
            f"timestamp={payload['timestamp']}"
        )
        print(f"    content : {memory.content[:120].replace('\n', ' ')}")
        print(
            f"    metadata: {{'origin': {payload['metadata']['origin']}, 'legacy_type': {memory.memory_type}}}"
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate MCP SQLite memories into AutoMem")
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to sqlite_vec.db exported by the MCP memory service",
    )
    parser.add_argument(
        "--automem-url",
        default=DEFAULT_AUTOMEM_URL,
        help="Base URL for the AutoMem service (default: http://localhost:8001)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Number of rows to fetch from SQLite for each iteration",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.1,
        help="Delay (in seconds) between POST requests to avoid overwhelming AutoMem",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of memories to migrate",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of initial rows to skip before migration",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the transformed payloads without calling AutoMem",
    )

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    migrator = AutoMemMigrator(
        db_path=args.db.expanduser(),
        automem_url=args.automem_url,
        batch_size=args.batch_size,
        sleep_seconds=args.sleep,
        dry_run=args.dry_run,
        limit=args.limit,
        offset=args.offset,
    )
    return migrator.run()


if __name__ == "__main__":
    sys.exit(main())
