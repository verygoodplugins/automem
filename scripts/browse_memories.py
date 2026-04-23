#!/usr/bin/env python3
"""Browse AutoMem production databases (FalkorDB graph + Qdrant vectors).

Usage:
    # Search for memories by text, date range, type
    python scripts/browse_memories.py search --text "some phrase"
    python scripts/browse_memories.py search --from 2025-10 --to 2025-11
    python scripts/browse_memories.py search --type Decision --min-importance 0.7

    # Inspect a specific memory (full UUID or 8+ char prefix)
    python scripts/browse_memories.py inspect abc12345

    # Overview statistics
    python scripts/browse_memories.py stats
    python scripts/browse_memories.py stats --full  # includes consistency check

    # Diagnose why a memory isn't surfacing in recall
    python scripts/browse_memories.py diagnose abc12345
"""

import argparse
import calendar
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()
load_dotenv(Path.home() / ".config" / "automem" / ".env")

# ---------------------------------------------------------------------------
# ANSI color helpers
# ---------------------------------------------------------------------------

USE_COLOR = sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    if not USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def red(t: str) -> str:
    return _c(t, "31")


def green(t: str) -> str:
    return _c(t, "32")


def yellow(t: str) -> str:
    return _c(t, "33")


def blue(t: str) -> str:
    return _c(t, "34")


def dim(t: str) -> str:
    return _c(t, "2")


def bold(t: str) -> str:
    return _c(t, "1")


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------


def connect_falkordb():
    """Connect to FalkorDB. Returns (graph, graph_name) or exits on failure."""
    from falkordb import FalkorDB

    host = os.getenv("FALKORDB_HOST", "localhost")
    port = int(os.getenv("FALKORDB_PORT", "6379"))
    password = os.getenv("FALKORDB_PASSWORD")
    graph_name = os.getenv("FALKORDB_GRAPH", "memories")

    try:
        db = FalkorDB(
            host=host,
            port=port,
            password=password,
            username="default" if password else None,
        )
        graph = db.select_graph(graph_name)
        # Quick connectivity check
        graph.query("RETURN 1")
        return graph, graph_name
    except Exception as exc:
        print(red(f"ERROR: Cannot connect to FalkorDB at {host}:{port} — {exc}"))
        sys.exit(1)


def connect_qdrant():
    """Connect to Qdrant. Returns (client, collection) or (None, None)."""
    url = os.getenv("QDRANT_URL")
    if not url:
        return None, None

    try:
        from qdrant_client import QdrantClient

        api_key = os.getenv("QDRANT_API_KEY")
        collection = os.getenv("QDRANT_COLLECTION", "memories")
        client = QdrantClient(url=url, api_key=api_key)
        client.get_collection(collection)  # connectivity check
        return client, collection
    except Exception as exc:
        print(yellow(f"WARNING: Cannot connect to Qdrant — {exc}"))
        return None, None


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def parse_ts(val) -> Optional[datetime]:
    """Parse various timestamp formats to datetime."""
    if not val:
        return None
    if isinstance(val, (int, float)):
        return datetime.fromtimestamp(val, tz=timezone.utc)
    s = str(val).strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def parse_date_start(val: str) -> str:
    """Convert date input to ISO start timestamp. Supports YYYY-MM shortcuts."""
    val = val.strip()
    if len(val) == 7 and val[4] == "-":  # YYYY-MM
        return f"{val}-01T00:00:00+00:00"
    if len(val) == 10 and val[4] == "-":  # YYYY-MM-DD
        return f"{val}T00:00:00+00:00"
    return val


def parse_date_end(val: str) -> str:
    """Convert date input to ISO end timestamp. YYYY-MM maps to end of month."""
    val = val.strip()
    if len(val) == 7 and val[4] == "-":  # YYYY-MM
        year, month = int(val[:4]), int(val[5:7])
        last_day = calendar.monthrange(year, month)[1]
        return f"{val}-{last_day:02d}T23:59:59+00:00"
    if len(val) == 10 and val[4] == "-":  # YYYY-MM-DD
        return f"{val}T23:59:59+00:00"
    return val


def trunc(text: str, width: int = 80) -> str:
    """Truncate text to width, appending ... if needed."""
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\r", "")
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


def resolve_memory_id(graph, prefix: str) -> Optional[str]:
    """Resolve a UUID prefix to a full memory ID. Returns None if ambiguous."""
    if len(prefix) >= 36:
        return prefix
    if len(prefix) < 4:
        print(red("ERROR: ID prefix must be at least 4 characters."))
        return None
    result = graph.query(
        "MATCH (m:Memory) WHERE m.id STARTS WITH $prefix RETURN m.id LIMIT 5",
        {"prefix": prefix},
    )
    rows = getattr(result, "result_set", []) or []
    if len(rows) == 0:
        print(red(f"ERROR: No memory found matching prefix '{prefix}'"))
        return None
    if len(rows) == 1:
        return rows[0][0]
    print(yellow(f"Ambiguous prefix '{prefix}' matches {len(rows)} memories:"))
    for row in rows:
        print(f"  {row[0]}")
    return None


# ---------------------------------------------------------------------------
# search subcommand
# ---------------------------------------------------------------------------


def _fetch_relationship_counts(graph, memory_ids: List[str]) -> Dict[str, int]:
    """Batch-fetch relationship counts for a list of memory IDs."""
    if not memory_ids:
        return {}
    # Process in batches to avoid oversized queries
    counts: Dict[str, int] = {}
    batch_size = 200
    for i in range(0, len(memory_ids), batch_size):
        batch = memory_ids[i : i + batch_size]
        try:
            result = graph.query(
                """
                MATCH (m:Memory)-[r]-(other:Memory)
                WHERE m.id IN $ids
                RETURN m.id, COUNT(r)
                """,
                {"ids": batch},
            )
            for row in getattr(result, "result_set", []) or []:
                counts[row[0]] = row[1]
        except Exception:
            pass
    return counts


def cmd_search(args, graph, qdrant_client, qdrant_collection):
    where_clauses = []
    params: Dict[str, Any] = {}

    if not args.include_archived:
        where_clauses.append("coalesce(m.archived, false) = false")

    if args.text:
        where_clauses.append("toLower(m.content) CONTAINS toLower($text_filter)")
        params["text_filter"] = args.text

    if getattr(args, "from_date", None):
        where_clauses.append("m.timestamp >= $from_time")
        params["from_time"] = parse_date_start(args.from_date)

    if args.to:
        where_clauses.append("m.timestamp <= $to_time")
        params["to_time"] = parse_date_end(args.to)

    if args.type:
        where_clauses.append("m.type = $type_filter")
        params["type_filter"] = args.type

    if args.tag:
        where_clauses.append(
            "ANY(tag IN coalesce(m.tags, []) WHERE toLower(tag) = toLower($tag_filter))"
        )
        params["tag_filter"] = args.tag

    if args.min_importance is not None:
        where_clauses.append("m.importance >= $min_importance")
        params["min_importance"] = args.min_importance

    where = " AND ".join(where_clauses) if where_clauses else "true"

    sort_map = {
        "date": "m.timestamp DESC",
        "importance": "m.importance DESC, m.timestamp DESC",
        "relevance": "coalesce(m.relevance_score, 0) DESC, m.importance DESC",
    }
    order = sort_map.get(args.sort, "m.timestamp DESC")

    limit_clause = f"LIMIT {args.limit}" if args.limit else ""

    query = f"""
        MATCH (m:Memory)
        WHERE {where}
        RETURN m.id AS id, m.timestamp AS ts, m.type AS type,
               m.importance AS imp, m.content AS content,
               coalesce(m.relevance_score, 0) AS rel,
               coalesce(m.archived, false) AS archived,
               coalesce(m.tags, []) AS tags,
               coalesce(m.confidence, 0) AS conf
        ORDER BY {order}
        {limit_clause}
    """

    try:
        result = graph.query(query, params)
    except Exception as exc:
        print(red(f"Query error: {exc}"))
        return

    rows = getattr(result, "result_set", []) or []
    if not rows:
        print(dim("No memories found matching your criteria."))
        return

    # Batch-fetch relationship counts for all results
    memory_ids = [row[0] for row in rows]
    rel_counts = _fetch_relationship_counts(graph, memory_ids)

    # Print each memory as a rich block
    for i, row in enumerate(rows):
        mid, ts, mtype, imp, content, rel, archived, tags, conf = row
        date_str = str(ts)[:19] if ts else "?"
        imp_val = float(imp or 0)
        rel_val = float(rel or 0)
        conf_val = float(conf or 0)
        mtype = str(mtype or "?")
        n_rels = rel_counts.get(mid, 0)

        # Filter out entity tags for cleaner display
        user_tags = [t for t in (tags or []) if not str(t).startswith("entity:")]
        entity_tags = [t for t in (tags or []) if str(t).startswith("entity:")]

        # Memory header line
        flag = " " + red("[ARCHIVED]") if archived else ""
        print(
            f"{bold(mid[:8])}  {date_str}  "
            f"{blue(mtype)}  "
            f"imp={imp_val:.2f}  rel={rel_val:.3f}  conf={conf_val:.2f}  "
            f"rels={n_rels}{flag}"
        )

        # Content (truncated to ~120 chars)
        content_str = trunc(str(content or ""), 120)
        print(f"  {content_str}")

        # Tags line
        if user_tags:
            tags_str = ", ".join(str(t) for t in user_tags[:15])
            if len(user_tags) > 15:
                tags_str += f" (+{len(user_tags) - 15} more)"
            print(f"  {dim('tags:')} {tags_str}")

        # Entity tags (dimmed, separate line)
        if entity_tags:
            # Show just the entity names, not the full prefix
            entities = []
            for et in entity_tags[:10]:
                parts = str(et).split(":")
                if len(parts) >= 3:
                    entities.append(f"{parts[1]}:{parts[2]}")
                else:
                    entities.append(str(et))
            ent_str = ", ".join(entities)
            if len(entity_tags) > 10:
                ent_str += f" (+{len(entity_tags) - 10} more)"
            print(f"  {dim('entities:')} {dim(ent_str)}")

        if i < len(rows) - 1:
            print()

    # Summary
    print()
    print(bold("--- Summary ---"))
    print(f"  Results: {len(rows)}")

    # Date range
    dates = [parse_ts(r[1]) for r in rows if r[1]]
    dates = [d for d in dates if d]
    if dates:
        print(
            f"  Date range: {min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}"
        )

    # Type distribution
    type_counts = Counter(str(r[2] or "?") for r in rows)
    type_parts = [f"{t}: {c}" for t, c in type_counts.most_common()]
    print(f"  Types: {', '.join(type_parts)}")

    # Importance stats
    imps = [float(r[3] or 0) for r in rows]
    if imps:
        print(
            f"  Importance: avg={sum(imps)/len(imps):.2f}, min={min(imps):.2f}, max={max(imps):.2f}"
        )

    # Top tags across results (excluding entity tags)
    all_tags: Counter = Counter()
    for row in rows:
        for t in row[7] or []:
            if not str(t).startswith("entity:"):
                all_tags[str(t)] += 1
    if all_tags:
        top = [f"{t} ({c})" for t, c in all_tags.most_common(15)]
        print(f"  Top tags: {', '.join(top)}")

    # Relationship stats
    rels_list = [rel_counts.get(r[0], 0) for r in rows]
    total_rels = sum(rels_list)
    isolated = sum(1 for r in rels_list if r == 0)
    if total_rels:
        print(
            f"  Relationships: {total_rels} total, avg={total_rels/len(rows):.1f}/memory, {isolated} isolated nodes"
        )

    print()


# ---------------------------------------------------------------------------
# inspect subcommand
# ---------------------------------------------------------------------------


def cmd_inspect(args, graph, qdrant_client, qdrant_collection):
    memory_id = resolve_memory_id(graph, args.id)
    if not memory_id:
        return

    # Fetch all properties
    result = graph.query(
        "MATCH (m:Memory {id: $id}) RETURN properties(m) as props",
        {"id": memory_id},
    )
    rows = getattr(result, "result_set", []) or []
    if not rows:
        print(red(f"Memory {memory_id} not found in FalkorDB."))
        return

    props = rows[0][0]

    print(bold("=== Memory Properties (FalkorDB) ===\n"))

    # Show content first, full
    content = props.pop("content", "")
    print(f"  {bold('content')}:")
    for line in str(content).split("\n"):
        print(f"    {line}")
    print()

    # Show other props in sorted order
    key_order = [
        "id",
        "type",
        "importance",
        "confidence",
        "relevance_score",
        "timestamp",
        "updated_at",
        "last_accessed",
        "tags",
        "tag_prefixes",
        "archived",
        "protected",
        "t_valid",
        "t_invalid",
        "processed",
    ]
    shown = set()
    for key in key_order:
        if key in props:
            val = props[key]
            print(f"  {blue(key):>30}: {val}")
            shown.add(key)

    # Remaining props (metadata, etc.)
    for key in sorted(props.keys()):
        if key not in shown:
            val = props[key]
            if isinstance(val, str) and len(val) > 200:
                val = val[:200] + "..."
            print(f"  {blue(key):>30}: {val}")

    # Qdrant check
    print(bold("\n=== Qdrant Status ===\n"))
    if qdrant_client:
        try:
            points = qdrant_client.retrieve(
                collection_name=qdrant_collection,
                ids=[memory_id],
                with_payload=True,
                with_vectors=False,
            )
            if points:
                print(f"  Status: {green('PRESENT')} in Qdrant")
                payload = points[0].payload or {}
                q_imp = payload.get("importance")
                q_type = payload.get("type")
                q_archived = payload.get("archived")
                print(f"  Payload importance: {q_imp}")
                print(f"  Payload type:       {q_type}")
                print(f"  Payload archived:   {q_archived}")
            else:
                print(
                    f"  Status: {red('MISSING')} from Qdrant — vector search will never find this memory"
                )
        except Exception as exc:
            print(yellow(f"  Qdrant check failed: {exc}"))
    else:
        print(dim("  Qdrant not connected"))

    # Relationships
    print(bold("\n=== Relationships ===\n"))
    outgoing = graph.query(
        """
        MATCH (m:Memory {id: $id})-[r]->(related:Memory)
        RETURN type(r) as rel_type,
               coalesce(r.strength, r.score, r.confidence, r.similarity, toFloat(r.count), 0.0) as strength,
               r.kind as kind,
               related.id as related_id,
               related.content as related_content
        ORDER BY strength DESC
        LIMIT 20
        """,
        {"id": memory_id},
    )
    incoming = graph.query(
        """
        MATCH (m:Memory {id: $id})<-[r]-(related:Memory)
        RETURN type(r) as rel_type,
               coalesce(r.strength, r.score, r.confidence, r.similarity, toFloat(r.count), 0.0) as strength,
               r.kind as kind,
               related.id as related_id,
               related.content as related_content
        ORDER BY strength DESC
        LIMIT 20
        """,
        {"id": memory_id},
    )

    out_rows = getattr(outgoing, "result_set", []) or []
    in_rows = getattr(incoming, "result_set", []) or []

    if not out_rows and not in_rows:
        print(dim("  No relationships found."))
    else:
        if out_rows:
            print(f"  {bold('Outgoing')} ({len(out_rows)}):")
            for row in out_rows:
                rel_type, strength, kind, rid, rcontent = row
                kind_str = f" ({kind})" if kind else ""
                print(
                    f"    --[{rel_type}{kind_str} str={float(strength):.2f}]--> "
                    f"{rid[:8]}  {trunc(str(rcontent or ''), 50)}"
                )
        if in_rows:
            print(f"\n  {bold('Incoming')} ({len(in_rows)}):")
            for row in in_rows:
                rel_type, strength, kind, rid, rcontent = row
                kind_str = f" ({kind})" if kind else ""
                print(
                    f"    <--[{rel_type}{kind_str} str={float(strength):.2f}]-- "
                    f"{rid[:8]}  {trunc(str(rcontent or ''), 50)}"
                )


# ---------------------------------------------------------------------------
# stats subcommand
# ---------------------------------------------------------------------------


def cmd_stats(args, graph, qdrant_client, qdrant_collection):
    print(bold("=== AutoMem Database Statistics ===\n"))

    # Total count
    result = graph.query("MATCH (m:Memory) RETURN COUNT(m)")
    total = (getattr(result, "result_set", []) or [[0]])[0][0]
    print(f"  Total memories: {bold(str(total))}")

    # Date range
    result = graph.query("MATCH (m:Memory) RETURN MIN(m.timestamp), MAX(m.timestamp)")
    row = (getattr(result, "result_set", []) or [[None, None]])[0]
    print(f"  Oldest: {row[0] or '?'}")
    print(f"  Newest: {row[1] or '?'}")

    # Archived count
    result = graph.query(
        "MATCH (m:Memory) WHERE coalesce(m.archived, false) = true RETURN COUNT(m)"
    )
    archived = (getattr(result, "result_set", []) or [[0]])[0][0]
    print(f"  Archived: {archived}")

    # By month
    print(bold("\n--- Memories by Month ---"))
    result = graph.query(
        """
        MATCH (m:Memory)
        RETURN substring(m.timestamp, 0, 7) as month, COUNT(*) as count
        ORDER BY month
    """
    )
    month_rows = getattr(result, "result_set", []) or []
    max_count = max((r[1] for r in month_rows), default=1)
    for month, count in month_rows:
        bar = "#" * int(count / max_count * 40)
        print(f"  {month or '?':7}  {count:>5}  {bar}")

    # By type
    print(bold("\n--- By Type ---"))
    result = graph.query(
        """
        MATCH (m:Memory)
        RETURN m.type as type, COUNT(*) as count
        ORDER BY count DESC
    """
    )
    for row in getattr(result, "result_set", []) or []:
        print(f"  {str(row[0] or '?'):>15}: {row[1]:>5}")

    # Importance buckets
    print(bold("\n--- Importance Distribution ---"))
    # Fetch all importances and bucket in Python (simpler than Cypher CASE)
    result = graph.query("MATCH (m:Memory) RETURN m.importance")
    importances = [float(r[0] or 0.5) for r in (getattr(result, "result_set", []) or [])]
    buckets = Counter()
    bucket_ranges = [
        (0.0, 0.2, "0.0-0.2"),
        (0.2, 0.4, "0.2-0.4"),
        (0.4, 0.6, "0.4-0.6"),
        (0.6, 0.8, "0.6-0.8"),
        (0.8, 1.01, "0.8-1.0"),
    ]
    for imp in importances:
        for lo, hi, label in bucket_ranges:
            if lo <= imp < hi:
                buckets[label] += 1
                break
    for _, _, label in bucket_ranges:
        print(f"  {label}: {buckets.get(label, 0):>5}")

    # Consistency check
    if args.full and qdrant_client:
        print(bold("\n--- FalkorDB / Qdrant Consistency ---"))
        # FalkorDB IDs
        falkor_ids = set()
        offset = 0
        while True:
            result = graph.query(f"MATCH (m:Memory) RETURN m.id SKIP {offset} LIMIT 5000")
            rows = getattr(result, "result_set", []) or []
            if not rows:
                break
            for r in rows:
                falkor_ids.add(r[0])
            if len(rows) < 5000:
                break
            offset += 5000

        # Qdrant IDs
        qdrant_ids = set()
        scroll_offset = None
        while True:
            points, next_offset = qdrant_client.scroll(
                collection_name=qdrant_collection,
                limit=100,
                offset=scroll_offset,
                with_payload=False,
                with_vectors=False,
            )
            if not points:
                break
            for p in points:
                qdrant_ids.add(str(p.id))
            if next_offset is None:
                break
            scroll_offset = next_offset

        print(f"  FalkorDB: {len(falkor_ids)}")
        print(f"  Qdrant:   {len(qdrant_ids)}")
        only_falkor = falkor_ids - qdrant_ids
        only_qdrant = qdrant_ids - falkor_ids
        if only_falkor:
            print(yellow(f"  In FalkorDB only: {len(only_falkor)}"))
            for mid in list(only_falkor)[:5]:
                print(f"    {mid}")
            if len(only_falkor) > 5:
                print(f"    ... and {len(only_falkor) - 5} more")
        if only_qdrant:
            print(yellow(f"  In Qdrant only: {len(only_qdrant)}"))
            for mid in list(only_qdrant)[:5]:
                print(f"    {mid}")
            if len(only_qdrant) > 5:
                print(f"    ... and {len(only_qdrant) - 5} more")
        if not only_falkor and not only_qdrant:
            print(green("  Databases are in sync!"))
    elif args.full:
        print(dim("\n  Qdrant not connected — skipping consistency check"))

    print()


# ---------------------------------------------------------------------------
# diagnose subcommand
# ---------------------------------------------------------------------------


def cmd_diagnose(args, graph, qdrant_client, qdrant_collection):
    memory_id = resolve_memory_id(graph, args.id)
    if not memory_id:
        return

    issues: List[Tuple[str, str, str]] = []  # (severity, title, detail)

    # Step 1: Fetch from FalkorDB
    result = graph.query(
        "MATCH (m:Memory {id: $id}) RETURN properties(m) as props",
        {"id": memory_id},
    )
    rows = getattr(result, "result_set", []) or []
    if not rows:
        print(red(f"CRITICAL: Memory {memory_id} does not exist in FalkorDB."))
        return

    props = rows[0][0]
    now = datetime.now(timezone.utc)

    print(bold(f"=== Diagnosis for {memory_id[:8]}... ===\n"))
    print(f"  Content: {trunc(str(props.get('content', '')), 80)}")
    print(f"  Type: {props.get('type', '?')}  |  Created: {str(props.get('timestamp', '?'))[:19]}")
    print()

    # Step 2: Archived?
    archived = props.get("archived", False)
    if archived:
        issues.append(("CRITICAL", "Archived", "Recall queries filter out archived memories"))

    # Step 3: Qdrant presence
    has_qdrant = False
    if qdrant_client:
        try:
            points = qdrant_client.retrieve(
                collection_name=qdrant_collection,
                ids=[memory_id],
                with_payload=True,
                with_vectors=True,
            )
            if points:
                has_qdrant = True
                vec = points[0].vector
                if vec:
                    magnitude = math.sqrt(sum(v * v for v in vec))
                    if magnitude < 0.01:
                        issues.append(
                            (
                                "WARNING",
                                "Zero/near-zero embedding",
                                f"Vector magnitude = {magnitude:.6f}",
                            )
                        )
                else:
                    issues.append(
                        ("WARNING", "No vector stored", "Point exists but vector is null")
                    )
            else:
                issues.append(
                    ("CRITICAL", "Missing from Qdrant", "Vector search will never find this memory")
                )
        except Exception as exc:
            issues.append(("WARNING", "Qdrant check failed", str(exc)))
    else:
        issues.append(("INFO", "Qdrant not connected", "Cannot verify vector presence"))

    # Step 4: Recency score (180-day linear decay)
    created = parse_ts(props.get("timestamp"))
    if created:
        age_days = max(0.0, (now - created).total_seconds() / 86400.0)
        recency = max(0.0, 1.0 - (age_days / 180.0))
    else:
        age_days = 0
        recency = 0.0

    if recency <= 0:
        issues.append(
            (
                "INFO",
                f"Recency score = 0.0 (age: {age_days:.0f} days)",
                "Linear 180-day decay — this memory is beyond the recency window",
            )
        )

    # Step 5: Relevance score simulation
    importance = float(props.get("importance", 0.5) or 0.5)
    confidence = float(props.get("confidence", 0.5) or 0.5)
    stored_relevance = float(props.get("relevance_score", 0) or 0)

    # Decay calculation (matches consolidation.py)
    base_decay_rate = float(os.getenv("CONSOLIDATION_BASE_DECAY_RATE", "0.01"))
    floor_factor = float(os.getenv("CONSOLIDATION_IMPORTANCE_FLOOR_FACTOR", "0.3"))

    decay_factor = math.exp(-base_decay_rate * age_days) if created else 0.0

    last_accessed = parse_ts(props.get("last_accessed"))
    if last_accessed:
        access_days = max(0.0, (now - last_accessed).total_seconds() / 86400.0)
        access_factor = 1.0 if access_days < 1 else math.exp(-0.05 * access_days)
    else:
        access_days = age_days
        access_factor = math.exp(-0.05 * age_days)

    # Relationship count
    rel_result = graph.query(
        """
        MATCH (m:Memory {id: $id})-[r]-(other:Memory)
        RETURN COUNT(r)
        """,
        {"id": memory_id},
    )
    rel_count = (getattr(rel_result, "result_set", []) or [[0]])[0][0]
    relationship_factor = 1.0 + (0.3 * math.log1p(max(float(rel_count), 0)))

    simulated_relevance = (
        decay_factor
        * (0.3 + 0.3 * access_factor)
        * relationship_factor
        * (0.5 + importance)
        * (0.7 + 0.3 * confidence)
    )
    floor = importance * floor_factor
    simulated_relevance = max(simulated_relevance, floor)
    simulated_relevance = min(1.0, simulated_relevance)

    # Step 6: Print scoring breakdown
    print(bold("--- Scoring Breakdown ---\n"))
    print(f"  {'Importance:':25} {importance:.3f}")
    print(f"  {'Confidence:':25} {confidence:.3f}")
    print(f"  {'Age (days):':25} {age_days:.1f}")
    print(f"  {'Decay factor:':25} {decay_factor:.6f}  (rate={base_decay_rate})")
    print(f"  {'Last accessed (days):':25} {access_days:.1f}")
    print(f"  {'Access factor:':25} {access_factor:.6f}")
    print(f"  {'Relationships:':25} {rel_count}")
    print(f"  {'Relationship factor:':25} {relationship_factor:.4f}")
    print(f"  {'Importance floor:':25} {floor:.4f}")
    print()
    print(f"  {'Stored relevance:':25} {stored_relevance:.6f}")
    print(f"  {'Simulated relevance:':25} {simulated_relevance:.6f}")
    print(f"  {'Recency score (recall):':25} {recency:.4f}")
    print(f"  {'In Qdrant:':25} {'Yes' if has_qdrant else 'No'}")
    print(f"  {'Archived:':25} {'Yes' if archived else 'No'}")

    # Search weight info
    w_recency = float(os.getenv("SEARCH_WEIGHT_RECENCY", "0.10"))
    w_relevance = float(os.getenv("SEARCH_WEIGHT_RELEVANCE", "0.0"))
    w_importance = float(os.getenv("SEARCH_WEIGHT_IMPORTANCE", "0.10"))

    if w_relevance == 0.0:
        issues.append(
            (
                "INFO",
                "SEARCH_WEIGHT_RELEVANCE = 0.0",
                "Relevance score is not used in recall scoring",
            )
        )

    if importance < 0.3:
        issues.append(
            (
                "WARNING",
                f"Low importance ({importance:.2f})",
                "This memory ranks low in importance-weighted scoring",
            )
        )

    if rel_count == 0:
        issues.append(("INFO", "No relationships", "Isolated node — no graph expansion benefit"))

    # Step 7: Issue summary
    print(bold("\n--- Issues ---\n"))
    if not issues:
        print(green("  No issues found! This memory should surface in recall."))
    else:
        for severity, title, detail in issues:
            if severity == "CRITICAL":
                tag = red(f"[{severity}]")
            elif severity == "WARNING":
                tag = yellow(f"[{severity}]")
            else:
                tag = dim(f"[{severity}]")
            print(f"  {tag} {title}")
            print(f"         {dim(detail)}")
    print()


# ---------------------------------------------------------------------------
# Argparse and main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Browse AutoMem production databases (FalkorDB + Qdrant)"
    )
    subparsers = parser.add_subparsers(dest="command")

    # -- search --
    sp_search = subparsers.add_parser("search", help="Search memories by text, date, type")
    sp_search.add_argument("-t", "--text", help="Substring to search in content")
    sp_search.add_argument("--from", dest="from_date", help="Start date (YYYY-MM or YYYY-MM-DD)")
    sp_search.add_argument("--to", help="End date (YYYY-MM or YYYY-MM-DD)")
    sp_search.add_argument("--type", help="Filter by memory type (Decision, Pattern, etc.)")
    sp_search.add_argument("--tag", help="Filter by tag")
    sp_search.add_argument("--min-importance", type=float, help="Minimum importance score")
    sp_search.add_argument(
        "-n", "--limit", type=int, default=0, help="Max results (0 = all, default: all)"
    )
    sp_search.add_argument(
        "-s",
        "--sort",
        choices=["date", "importance", "relevance"],
        default="date",
        help="Sort order (default: date)",
    )
    sp_search.add_argument(
        "--include-archived",
        action="store_true",
        help="Include archived memories (excluded by default)",
    )

    # -- inspect --
    sp_inspect = subparsers.add_parser("inspect", help="Inspect a single memory in detail")
    sp_inspect.add_argument("id", help="Memory UUID or prefix (min 4 chars)")

    # -- stats --
    sp_stats = subparsers.add_parser("stats", help="Database overview statistics")
    sp_stats.add_argument(
        "--full", action="store_true", help="Include FalkorDB/Qdrant consistency check (slower)"
    )

    # -- diagnose --
    sp_diagnose = subparsers.add_parser(
        "diagnose", help="Diagnose why a memory isn't surfacing in recall"
    )
    sp_diagnose.add_argument("id", help="Memory UUID or prefix (min 4 chars)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Connect
    print(dim("Connecting to databases..."))
    graph, graph_name = connect_falkordb()
    qdrant_client, qdrant_collection = connect_qdrant()
    status = f"FalkorDB: {green('OK')}"
    status += f"  |  Qdrant: {green('OK') if qdrant_client else yellow('N/A')}"
    print(dim(status))
    print()

    # Dispatch
    if args.command == "search":
        cmd_search(args, graph, qdrant_client, qdrant_collection)
    elif args.command == "inspect":
        cmd_inspect(args, graph, qdrant_client, qdrant_collection)
    elif args.command == "stats":
        cmd_stats(args, graph, qdrant_client, qdrant_collection)
    elif args.command == "diagnose":
        cmd_diagnose(args, graph, qdrant_client, qdrant_collection)


if __name__ == "__main__":
    main()
