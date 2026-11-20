"""Safely populate the graph's original-text search property and index label.

By default this command only reports changes. Existing legacy summarized originals
cannot be verified as current after older content PATCH operations; using those
originals requires an explicit opt-in. Never touches Qdrant or creates the index.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import sys
from typing import Any, Dict, Iterator, List, Optional, Tuple

SCAN_QUERY = """
    MATCH (m:Memory)
    WHERE m.id IS NOT NULL AND m.id > $after
    RETURN m.id, m.content, m.metadata, m.original_search_text, labels(m)
    ORDER BY m.id
    LIMIT $limit
"""
MISSING_ID_QUERY = """
    MATCH (m:Memory)
    WHERE m.id IS NULL OR m.id = ''
    RETURN count(m)
"""
DUPLICATE_ID_QUERY = """
    MATCH (m:Memory)
    WHERE m.id IS NOT NULL AND m.id <> ''
    WITH m.id AS memory_id, count(m) AS n
    WHERE n > 1
    RETURN count(memory_id)
"""
CAS_QUERY = """
    MATCH (m:Memory {id: $id})
    WITH collect(m) AS matches
    WHERE size(matches) = 1
    WITH matches[0] AS m
    WHERE ((m.content IS NULL AND $expected_content IS NULL) OR m.content = $expected_content)
      AND ((m.metadata IS NULL AND $expected_metadata IS NULL) OR m.metadata = $expected_metadata)
      AND ((m.original_search_text IS NULL AND $expected_indexed IS NULL)
           OR m.original_search_text = $expected_indexed)
    SET m.original_search_text = $search_text
    SET m:MemoryOriginalText
    RETURN m.id
"""
VERIFY_QUERY = """
    MATCH (m:Memory {id: $id})
    RETURN m.id, m.content, m.metadata, m.original_search_text, labels(m)
"""


def local_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def parse_metadata(raw: Any) -> Tuple[Optional[Dict[str, Any]], bool]:
    if raw is None or raw == "":
        return {}, False
    if isinstance(raw, dict):
        return raw, False
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return None, True
        if isinstance(parsed, dict):
            return parsed, False
    return None, True


def scan(graph: Any, batch_size: int, after_id: str, limit: int) -> Iterator[Tuple[Any, ...]]:
    cursor = after_id
    seen = 0
    while not limit or seen < limit:
        size = min(batch_size, limit - seen) if limit else batch_size
        result = graph.query(SCAN_QUERY, {"after": cursor, "limit": size})
        rows = list(getattr(result, "result_set", []) or [])
        if not rows:
            return
        for row in rows:
            if len(row) != 5 or not isinstance(row[0], str) or row[0] <= cursor:
                raise RuntimeError("scan returned invalid, duplicate or non-monotonic memory id")
            cursor = row[0]
            seen += 1
            yield tuple(row)
        if len(rows) < size:
            return


def run(
    graph: Any,
    *,
    batch_size: int = 250,
    after_id: str = "",
    limit: int = 0,
    apply: bool = False,
    assume_legacy_original_current: bool = False,
    verify_only: bool = False,
) -> Dict[str, Any]:
    if batch_size < 1 or batch_size > 1000 or limit < 0:
        raise ValueError("batch_size must be 1..1000 and limit must be >= 0")
    if apply and verify_only:
        raise ValueError("verify_only cannot be combined with apply")
    counts: Dict[str, Any] = {
        "dry_run": not apply,
        "verification_only": verify_only,
        "assume_legacy_original_current": assume_legacy_original_current,
        "scanned": 0,
        "indexed": 0,
        "missing_index": 0,
        "missing_label": 0,
        "unverified_legacy": 0,
        "index_uses_unverified_original": 0,
        "known_original_missing": 0,
        "malformed": 0,
        "preserved_existing": 0,
        "would_apply": 0,
        "applied": 0,
        "conflicts": 0,
        "conflict_ids": [],
        "verification_mismatches": 0,
        "verification_mismatch_ids": [],
        "last_id": after_id,
        "limit": limit,
        "legacy_batch_originals_unrecoverable_and_not_countable": True,
    }

    def count_query(query: str) -> int:
        rows = list(getattr(graph.query(query), "result_set", []) or [])
        if len(rows) != 1 or len(rows[0]) != 1 or not isinstance(rows[0][0], int):
            raise RuntimeError("could not verify memory ID integrity")
        return rows[0][0]

    counts["missing_ids"] = count_query(MISSING_ID_QUERY)
    counts["duplicate_id_groups"] = count_query(DUPLICATE_ID_QUERY)
    if counts["duplicate_id_groups"]:
        counts["aborted_reason"] = "duplicate memory IDs prevent safe keyset pagination"
        return counts

    for memory_id, content, raw_metadata, indexed, labels in scan(
        graph, batch_size, after_id, limit
    ):
        counts["scanned"] += 1
        counts["last_id"] = memory_id
        metadata, invalid_metadata = parse_metadata(raw_metadata)
        if (
            invalid_metadata
            or not isinstance(content, str)
            or not content
            or (indexed is not None and not isinstance(indexed, str))
            or not isinstance(labels, list)
        ):
            counts["malformed"] += 1
            continue
        summarized = metadata.get("was_summarized") is True
        original = metadata.get("original_content")
        has_original = summarized and isinstance(original, str) and bool(original)
        if has_original:
            counts["unverified_legacy"] += 1
        elif summarized:
            counts["known_original_missing"] += 1
        if has_original and indexed == original and original != content:
            counts["index_uses_unverified_original"] += 1
        has_index = isinstance(indexed, str) and bool(indexed)
        has_label = "MemoryOriginalText" in labels
        if has_index:
            counts["indexed"] += 1
        else:
            counts["missing_index"] += 1
        if not has_label:
            counts["missing_label"] += 1
        if verify_only:
            continue

        desired = original if has_original and assume_legacy_original_current else content
        # A preexisting field may have been written by a concurrent/current
        # server. Preserve it, except explicit opt-in can upgrade a previous
        # conservative current-content fallback to an unverified original.
        upgrade = (
            assume_legacy_original_current
            and has_original
            and indexed == content
            and desired != indexed
        )
        if has_index and not upgrade:
            desired = indexed
            counts["preserved_existing"] += 1
        if has_index and has_label and desired == indexed:
            continue
        counts["would_apply"] += 1
        if not apply:
            continue
        params = {
            "id": memory_id,
            "expected_content": content,
            "expected_metadata": raw_metadata,
            "expected_indexed": indexed,
            "search_text": desired,
        }
        rows = list(getattr(graph.query(CAS_QUERY, params), "result_set", []) or [])
        if len(rows) != 1:
            counts["conflicts"] += 1
            counts["conflict_ids"].append(memory_id)
            continue
        counts["applied"] += 1
        observed = list(
            getattr(graph.query(VERIFY_QUERY, {"id": memory_id}), "result_set", []) or []
        )
        if (
            len(observed) != 1
            or len(observed[0]) != 5
            or observed[0][1] != content
            or observed[0][2] != raw_metadata
            or observed[0][3] != desired
            or "MemoryOriginalText" not in (observed[0][4] or [])
        ):
            counts["verification_mismatches"] += 1
            counts["verification_mismatch_ids"].append(memory_id)
    return counts


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--apply", action="store_true", help="Apply graph writes; default is read-only"
    )
    mode.add_argument(
        "--dry-run", action="store_true", help="Explicitly select the default read-only scan"
    )
    mode.add_argument(
        "--verify", action="store_true", help="Read-only index-property and label coverage scan"
    )
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--limit", type=int, default=0, help="Maximum records; 0 means all")
    parser.add_argument("--after-id", default="", help="Resume strictly after this memory ID")
    parser.add_argument(
        "--host", default=os.getenv("FALKORDB_HOST"), help="FalkorDB host (or FALKORDB_HOST)"
    )
    parser.add_argument("--port", type=int, default=int(os.getenv("FALKORDB_PORT", "6379")))
    parser.add_argument("--graph", default=os.getenv("FALKORDB_GRAPH", "memories"))
    parser.add_argument(
        "--allow-remote", action="store_true", help="Permit a non-loopback database host"
    )
    parser.add_argument(
        "--assume-legacy-original-current",
        action="store_true",
        help="Index unverified legacy original_content despite possible stale content PATCH history",
    )
    args = parser.parse_args(argv)
    if not args.host:
        parser.error("--host or FALKORDB_HOST is required")
    if not local_host(args.host) and not args.allow_remote:
        parser.error("non-loopback host requires --allow-remote")
    if not 1 <= args.batch_size <= 1000 or args.limit < 0:
        parser.error("--batch-size must be 1..1000 and --limit must be >= 0")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be 1..65535")
    try:
        from falkordb import FalkorDB

        password = os.getenv("FALKORDB_PASSWORD") or None
        username = os.getenv("FALKORDB_USERNAME") or ("default" if password else None)
        graph = FalkorDB(
            host=args.host, port=args.port, password=password, username=username
        ).select_graph(args.graph)
        summary = run(
            graph,
            batch_size=args.batch_size,
            after_id=args.after_id,
            limit=args.limit,
            apply=args.apply,
            assume_legacy_original_current=args.assume_legacy_original_current,
            verify_only=args.verify,
        )
    except Exception as exc:
        print(f"backfill failed: {type(exc).__name__}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    if (
        summary["conflicts"]
        or summary["verification_mismatches"]
        or summary["malformed"]
        or summary["missing_ids"]
        or summary["duplicate_id_groups"]
    ):
        return 1
    if args.verify and (summary["missing_index"] or summary["missing_label"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
