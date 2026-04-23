#!/usr/bin/env python3
"""Audit relevance_score distribution in AutoMem production data.

Can run against:
  - A FalkorDB backup file (default, fast, no services needed)
  - A live FalkorDB instance (local Docker or Railway)

Usage:
    # From backup file (fast, no services)
    python scripts/audit_relevance.py

    # From backup with explicit path
    python scripts/audit_relevance.py --backup backups/falkordb/falkordb_20260302_172507.json.gz

    # From live FalkorDB (uses .env credentials)
    python scripts/audit_relevance.py --live
"""

import argparse
import gzip
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()
load_dotenv(Path.home() / ".config" / "automem" / ".env")


def load_from_backup(backup_path: Path) -> List[Dict[str, Any]]:
    """Load Memory nodes from a FalkorDB backup file."""
    with gzip.open(backup_path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    memories = []
    for node in data["nodes"]:
        if "Memory" in node.get("labels", []):
            memories.append(node["properties"])
    return memories


def load_from_live() -> List[Dict[str, Any]]:
    """Load Memory nodes from a live FalkorDB instance."""
    from falkordb import FalkorDB

    host = os.getenv("FALKORDB_HOST", "localhost")
    port = int(os.getenv("FALKORDB_PORT", "6379"))
    password = os.getenv("FALKORDB_PASSWORD")
    graph_name = os.getenv("FALKORDB_GRAPH", "memories")

    db = FalkorDB(
        host=host,
        port=port,
        password=password,
        username="default" if password else None,
    )
    graph = db.select_graph(graph_name)

    memories = []
    offset = 0
    batch_size = 5000
    while True:
        result = graph.query(
            f"""
            MATCH (m:Memory)
            RETURN properties(m) as props
            SKIP {offset} LIMIT {batch_size}
            """
        )
        if not result.result_set:
            break
        for row in result.result_set:
            memories.append(row[0])
        if len(result.result_set) < batch_size:
            break
        offset += batch_size
    return memories


def find_latest_backup(backup_dir: Path) -> Optional[Path]:
    """Find the most recent FalkorDB backup."""
    falkordb_dir = backup_dir / "falkordb"
    if not falkordb_dir.exists():
        return None
    backups = sorted(falkordb_dir.glob("*.json.gz"), key=lambda p: p.stat().st_mtime, reverse=True)
    return backups[0] if backups else None


def classify_relevance(score: float) -> str:
    if score >= 0.5:
        return "high"
    elif score >= 0.1:
        return "medium"
    elif score >= 0.05:
        return "low"
    else:
        return "archive"


def audit(memories: List[Dict[str, Any]]) -> None:
    total = len(memories)
    if total == 0:
        print("No memories found.")
        return

    scores = []
    importances = []
    classifications = Counter()
    type_counts = Counter()
    unfairly_killed = []
    archived_flag = 0
    protected_types = {"Decision", "Insight"}
    protected_in_archive = []
    buckets = defaultdict(int)
    bucket_ranges = [
        (0, 0.01, "0.00-0.01"),
        (0.01, 0.05, "0.01-0.05"),
        (0.05, 0.1, "0.05-0.10"),
        (0.1, 0.3, "0.10-0.30"),
        (0.3, 0.5, "0.30-0.50"),
        (0.5, 1.01, "0.50-1.00"),
    ]

    for m in memories:
        score = float(m.get("relevance_score") or 0)
        importance = float(m.get("importance") or 0.5)
        mem_type = m.get("type") or "Context"
        archived = m.get("archived")

        scores.append(score)
        importances.append(importance)
        classifications[classify_relevance(score)] += 1
        type_counts[mem_type] += 1

        if archived:
            archived_flag += 1

        if importance >= 0.7 and score < 0.1:
            unfairly_killed.append(m)

        if mem_type in protected_types and score < 0.1:
            protected_in_archive.append(m)

        for lo, hi, label in bucket_ranges:
            if lo <= score < hi:
                buckets[label] += 1
                break

    avg_score = sum(scores) / len(scores)
    avg_importance = sum(importances) / len(importances)

    print("=" * 60)
    print("  AutoMem Relevance Score Audit")
    print("=" * 60)
    print(f"\n  Total memories:       {total}")
    print(f"  Avg relevance_score:  {avg_score:.4f}")
    print(f"  Avg importance:       {avg_importance:.4f}")
    print(f"  Archived flag set:    {archived_flag} ({archived_flag/total*100:.1f}%)")

    print("\n--- Relevance Score Distribution ---")
    for lo, hi, label in bucket_ranges:
        count = buckets[label]
        bar = "#" * min(50, int(count / total * 200))
        print(f"  {label}: {count:>5} ({count/total*100:5.1f}%) {bar}")

    print("\n--- Classification ---")
    for cls in ["high", "medium", "low", "archive"]:
        count = classifications.get(cls, 0)
        print(f"  {cls:>8}: {count:>5} ({count/total*100:5.1f}%)")

    print("\n--- Memory Types ---")
    for mtype, count in type_counts.most_common():
        print(f"  {mtype:>15}: {count:>5}")

    print(f"\n--- Unfairly Killed (importance >= 0.7, relevance < 0.1) ---")
    print(f"  Count: {len(unfairly_killed)} ({len(unfairly_killed)/total*100:.1f}% of total)")
    if unfairly_killed:
        sorted_uk = sorted(
            unfairly_killed, key=lambda x: float(x.get("importance", 0)), reverse=True
        )
        for m in sorted_uk[:10]:
            content = (m.get("content") or "")[:80]
            imp = float(m.get("importance") or 0)
            rel = float(m.get("relevance_score") or 0)
            print(f"    imp={imp:.2f} rel={rel:.4f} | {content}")
        if len(unfairly_killed) > 10:
            print(f"    ... and {len(unfairly_killed) - 10} more")

    print(f"\n--- Protected Types in Archive (Decision/Insight with relevance < 0.1) ---")
    print(f"  Count: {len(protected_in_archive)}")
    if protected_in_archive:
        for m in protected_in_archive[:5]:
            content = (m.get("content") or "")[:80]
            mtype = m.get("type", "?")
            rel = float(m.get("relevance_score") or 0)
            print(f"    [{mtype}] rel={rel:.4f} | {content}")
        if len(protected_in_archive) > 5:
            print(f"    ... and {len(protected_in_archive) - 5} more")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Audit AutoMem relevance score distribution")
    parser.add_argument(
        "--backup",
        type=str,
        help="Path to FalkorDB backup .json.gz file (default: latest in ./backups/falkordb/)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Query a live FalkorDB instance (uses .env credentials)",
    )
    args = parser.parse_args()

    if args.live:
        print("Loading from live FalkorDB...")
        memories = load_from_live()
    else:
        if args.backup:
            backup_path = Path(args.backup)
        else:
            backup_path = find_latest_backup(Path("./backups"))
        if not backup_path or not backup_path.exists():
            print("ERROR: No backup found. Run scripts/backup_automem.py first, or use --live.")
            sys.exit(1)
        print(f"Loading from backup: {backup_path.name}")
        memories = load_from_backup(backup_path)

    audit(memories)


if __name__ == "__main__":
    main()
