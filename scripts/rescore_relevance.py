#!/usr/bin/env python3
"""One-time relevance_score rehabilitation for AutoMem production data.

Recalculates relevance_score for all memories using the corrected decay formula
(base_decay_rate=0.01 + importance floor) to undo damage from the overly aggressive
original rate (0.1). Also resets last_accessed for important memories to give them
a fair restart.

Usage:
    # Dry run against local Docker (default)
    python scripts/rescore_relevance.py --dry-run

    # Execute against local Docker
    python scripts/rescore_relevance.py

    # Dry run against Railway production
    python scripts/rescore_relevance.py --target railway --dry-run

    # Execute against Railway production
    python scripts/rescore_relevance.py --target railway
"""

import argparse
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()
load_dotenv(Path.home() / ".config" / "automem" / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("automem.rescore")

BASE_DECAY_RATE = float(os.getenv("CONSOLIDATION_BASE_DECAY_RATE", "0.01"))
IMPORTANCE_FLOOR_FACTOR = float(os.getenv("CONSOLIDATION_IMPORTANCE_FLOOR_FACTOR", "0.3"))
RELATIONSHIP_PRESERVATION = 0.3


def connect_graph(target: str):
    from falkordb import FalkorDB

    if target == "railway":
        host = os.getenv("FALKORDB_HOST", "localhost")
        port = int(os.getenv("FALKORDB_PORT", "6379"))
        password = os.getenv("FALKORDB_PASSWORD")
    else:
        host = "localhost"
        port = 6379
        password = None

    db = FalkorDB(
        host=host,
        port=port,
        password=password,
        username="default" if password else None,
    )
    graph_name = os.getenv("FALKORDB_GRAPH", "memories")
    return db.select_graph(graph_name)


def fetch_all_memories(graph) -> List[Dict[str, Any]]:
    """Fetch all Memory nodes with their properties."""
    memories = []
    offset = 0
    batch_size = 5000
    while True:
        result = graph.query(
            f"""
            MATCH (m:Memory)
            OPTIONAL MATCH (m)-[r]-(other:Memory)
            WITH m, COUNT(DISTINCT r) as rel_count
            RETURN m.id as id,
                   m.timestamp as timestamp,
                   m.last_accessed as last_accessed,
                   m.importance as importance,
                   m.confidence as confidence,
                   m.relevance_score as old_score,
                   m.type as type,
                   m.archived as archived,
                   rel_count
            SKIP {offset} LIMIT {batch_size}
            """
        )
        if not result.result_set:
            break
        for row in result.result_set:
            memories.append(
                {
                    "id": row[0],
                    "timestamp": row[1],
                    "last_accessed": row[2],
                    "importance": row[3],
                    "confidence": row[4],
                    "old_score": row[5],
                    "type": row[6],
                    "archived": row[7],
                    "rel_count": row[8],
                }
            )
        if len(result.result_set) < batch_size:
            break
        offset += batch_size
    return memories


def parse_ts(val) -> datetime:
    if val is None:
        return datetime.now(timezone.utc)
    if isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
    if isinstance(val, (int, float)):
        return datetime.fromtimestamp(val, tz=timezone.utc)
    s = str(val).strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            dt = datetime.strptime(s, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return datetime.now(timezone.utc)


def calculate_new_score(mem: Dict[str, Any], now: datetime) -> float:
    """Calculate relevance score using the corrected formula."""
    created_at = parse_ts(mem["timestamp"])
    last_accessed = parse_ts(mem["last_accessed"]) if mem["last_accessed"] else created_at

    age_days = max(0.0, (now - created_at).total_seconds() / 86400)
    decay_factor = math.exp(-BASE_DECAY_RATE * age_days)

    access_recency_days = max(0.0, (now - last_accessed).total_seconds() / 86400)
    access_factor = 1.0 if access_recency_days < 1 else math.exp(-0.05 * access_recency_days)

    rel_count = float(mem.get("rel_count") or 0)
    relationship_factor = 1.0 + (RELATIONSHIP_PRESERVATION * math.log1p(max(rel_count, 0)))

    importance = float(mem.get("importance") or 0.5)
    confidence = float(mem.get("confidence") or 0.5)

    relevance = (
        decay_factor
        * (0.3 + 0.3 * access_factor)
        * relationship_factor
        * (0.5 + importance)
        * (0.7 + 0.3 * confidence)
    )

    floor = importance * IMPORTANCE_FLOOR_FACTOR
    relevance = max(relevance, floor)

    return min(1.0, relevance)


def rescore(target: str, dry_run: bool, reset_access: bool = True) -> None:
    logger.info("Connecting to %s FalkorDB...", target)
    graph = connect_graph(target)

    logger.info("Fetching all memories...")
    memories = fetch_all_memories(graph)
    logger.info("Found %d memories", len(memories))

    now = datetime.now(timezone.utc)
    updates = []

    for mem in memories:
        new_score = calculate_new_score(mem, now)
        old_score = float(mem.get("old_score") or 0)

        should_reset_access = reset_access and float(mem.get("importance") or 0.5) >= 0.5

        updates.append(
            {
                "id": mem["id"],
                "old_score": old_score,
                "new_score": new_score,
                "delta": new_score - old_score,
                "importance": float(mem.get("importance") or 0.5),
                "type": mem.get("type", "?"),
                "reset_access": should_reset_access,
                "archived": mem.get("archived"),
            }
        )

    improved = [u for u in updates if u["delta"] > 0.01]
    degraded = [u for u in updates if u["delta"] < -0.01]
    unchanged = [u for u in updates if abs(u["delta"]) <= 0.01]

    avg_old = sum(u["old_score"] for u in updates) / len(updates) if updates else 0
    avg_new = sum(u["new_score"] for u in updates) / len(updates) if updates else 0
    access_resets = sum(1 for u in updates if u["reset_access"])
    unarchive_count = sum(1 for u in updates if u["archived"] and u["new_score"] > 0.05)

    logger.info("")
    logger.info("=" * 60)
    logger.info("  Rescore Summary")
    logger.info("=" * 60)
    logger.info("  Total memories:    %d", len(updates))
    logger.info("  Avg old score:     %.4f", avg_old)
    logger.info("  Avg new score:     %.4f", avg_new)
    logger.info("  Improved (>0.01):  %d", len(improved))
    logger.info("  Degraded (<-0.01): %d", len(degraded))
    logger.info("  Unchanged:         %d", len(unchanged))
    logger.info("  Access resets:     %d (importance >= 0.5)", access_resets)
    logger.info("  Would un-archive:  %d (new score > 0.05)", unarchive_count)

    top_improvements = sorted(improved, key=lambda u: u["delta"], reverse=True)[:10]
    if top_improvements:
        logger.info("")
        logger.info("  Top improvements:")
        for u in top_improvements:
            logger.info(
                "    [%s] imp=%.2f  %.4f -> %.4f (+%.4f)  %s",
                u["type"],
                u["importance"],
                u["old_score"],
                u["new_score"],
                u["delta"],
                u["id"][:12],
            )

    if dry_run:
        logger.info("")
        logger.info("  DRY RUN — no changes applied")
        logger.info("=" * 60)
        return

    logger.info("")
    logger.info("  Applying changes...")

    batch_size = 200
    applied = 0
    now_iso = now.isoformat()

    for i in range(0, len(updates), batch_size):
        batch = updates[i : i + batch_size]
        for u in batch:
            params = {"id": u["id"], "score": u["new_score"]}
            set_clauses = ["m.relevance_score = $score"]

            if u["reset_access"]:
                set_clauses.append("m.last_accessed = $ts")
                params["ts"] = now_iso

            if u["archived"] and u["new_score"] > 0.05:
                set_clauses.append("m.archived = false")

            set_clause = ", ".join(set_clauses)
            try:
                graph.query(
                    f"MATCH (m:Memory {{id: $id}}) SET {set_clause}",
                    params,
                )
                applied += 1
            except Exception:
                logger.exception("Failed to update %s", u["id"])

        logger.info("  Progress: %d/%d", min(i + batch_size, len(updates)), len(updates))

    logger.info("")
    logger.info("  Applied %d/%d updates", applied, len(updates))
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Rehabilitate relevance scores using corrected decay formula"
    )
    parser.add_argument(
        "--target",
        choices=["local", "railway"],
        default="local",
        help="Target FalkorDB instance (default: local)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without applying",
    )
    parser.add_argument(
        "--no-reset-access",
        action="store_true",
        help="Don't reset last_accessed for important memories",
    )
    args = parser.parse_args()

    rescore(
        target=args.target,
        dry_run=args.dry_run,
        reset_access=not args.no_reset_access,
    )


if __name__ == "__main__":
    main()
