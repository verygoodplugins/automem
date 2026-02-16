"""BM25 full-text search via SQLite FTS5.

Provides exact keyword matching to complement Qdrant vector search.
Memories are indexed on store and searched on recall, with results
fused into the existing pipeline via Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BM25_ENABLED = os.environ.get("BM25_ENABLED", "true").lower() in ("1", "true", "yes")
BM25_DB_PATH = os.environ.get("BM25_DB_PATH", "/app/data/bm25_index.db")

# RRF constant (k) — higher = less emphasis on top ranks
RRF_K = int(os.environ.get("BM25_RRF_K", "60"))

# How many BM25 results to fetch before fusion
BM25_FETCH_LIMIT = int(os.environ.get("BM25_FETCH_LIMIT", "50"))


# ---------------------------------------------------------------------------
# Thread-local connections (SQLite is not thread-safe by default)
# ---------------------------------------------------------------------------

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Return a thread-local SQLite connection, creating the DB if needed."""
    conn: Optional[sqlite3.Connection] = getattr(_local, "conn", None)
    if conn is not None:
        return conn

    db_path = BM25_DB_PATH
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    # Create tables if needed
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS memories (
            memory_id TEXT PRIMARY KEY,
            content   TEXT NOT NULL,
            tags      TEXT DEFAULT '',
            type      TEXT DEFAULT '',
            stored_at TEXT DEFAULT ''
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            content,
            tags,
            type,
            content='memories',
            content_rowid='rowid',
            tokenize='porter unicode61 remove_diacritics 2'
        );

        -- Triggers to keep FTS in sync with content table
        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, content, tags, type)
            VALUES (new.rowid, new.content, new.tags, new.type);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, tags, type)
            VALUES ('delete', old.rowid, old.content, old.tags, old.type);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, tags, type)
            VALUES ('delete', old.rowid, old.content, old.tags, old.type);
            INSERT INTO memories_fts(rowid, content, tags, type)
            VALUES (new.rowid, new.content, new.tags, new.type);
        END;
        """
    )
    conn.commit()

    _local.conn = conn
    return conn


# ---------------------------------------------------------------------------
# Index operations
# ---------------------------------------------------------------------------


def index_memory(
    memory_id: str,
    content: str,
    tags: Optional[List[str]] = None,
    memory_type: Optional[str] = None,
    stored_at: Optional[str] = None,
) -> None:
    """Add or update a memory in the FTS index."""
    if not BM25_ENABLED:
        return

    conn = _get_conn()
    tag_str = " ".join(tags) if tags else ""
    type_str = memory_type or ""
    stored_str = stored_at or ""

    try:
        conn.execute(
            """
            INSERT INTO memories (memory_id, content, tags, type, stored_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                content = excluded.content,
                tags = excluded.tags,
                type = excluded.type,
                stored_at = excluded.stored_at
            """,
            (memory_id, content, tag_str, type_str, stored_str),
        )
        conn.commit()
    except Exception:
        logger.exception("BM25 index_memory failed for %s", memory_id)


def remove_memory(memory_id: str) -> None:
    """Remove a memory from the FTS index."""
    if not BM25_ENABLED:
        return

    conn = _get_conn()
    try:
        conn.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
        conn.commit()
    except Exception:
        logger.exception("BM25 remove_memory failed for %s", memory_id)


def search(
    query: str,
    limit: int = BM25_FETCH_LIMIT,
    seen_ids: Optional[Set[str]] = None,
    tag_filters: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Search the FTS index. Returns list of {memory_id, score, snippet, source}."""
    if not BM25_ENABLED or not query or not query.strip():
        return []

    conn = _get_conn()
    seen = seen_ids or set()

    # Sanitize query for FTS5 — escape double quotes, strip operators
    fts_query = _build_fts_query(query)
    if not fts_query:
        return []

    try:
        t0 = time.monotonic()
        rows = conn.execute(
            """
            SELECT
                m.memory_id,
                m.content,
                m.tags,
                m.type,
                m.stored_at,
                rank
            FROM memories_fts
            JOIN memories m ON memories_fts.rowid = m.rowid
            WHERE memories_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (fts_query, limit),
        ).fetchall()
        elapsed_ms = (time.monotonic() - t0) * 1000

        results = []
        for memory_id, content, tags_str, mtype, stored_at, rank in rows:
            if memory_id in seen:
                continue

            # Apply tag filter if provided
            if tag_filters:
                mem_tags = set(tags_str.lower().split()) if tags_str else set()
                if not any(t.lower() in mem_tags for t in tag_filters):
                    continue

            # FTS5 rank is negative (more negative = better match)
            # Normalize to 0-1 range for fusion
            bm25_score = -rank if rank else 0.0

            results.append(
                {
                    "memory_id": memory_id,
                    "id": memory_id,
                    "score": bm25_score,
                    "source": "bm25",
                    "memory": {
                        "id": memory_id,
                        "memory_id": memory_id,
                        "content": content,
                        "tags": tags_str.split() if tags_str else [],
                        "type": mtype,
                        "stored_at": stored_at,
                    },
                }
            )
            seen.add(memory_id)

        logger.debug(
            "BM25 search: query=%r results=%d time=%.1fms",
            fts_query,
            len(results),
            elapsed_ms,
        )
        return results

    except Exception:
        logger.exception("BM25 search failed for query: %s", query)
        return []


def get_index_count() -> int:
    """Return the number of indexed memories."""
    if not BM25_ENABLED:
        return 0
    try:
        conn = _get_conn()
        row = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        return row[0] if row else 0
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Backfill: index all existing memories from the graph
# ---------------------------------------------------------------------------


def backfill_from_graph(graph: Any) -> int:
    """Index all memories from FalkorDB graph into BM25. Returns count indexed."""
    if not BM25_ENABLED:
        return 0

    try:
        result = graph.query("MATCH (m:Memory) RETURN m.id, m.content, m.tags, m.type, m.stored_at")
        count = 0
        for row in result.result_set:
            mid, content, tags_raw, mtype, stored_at = row
            if not mid or not content:
                continue
            tags = []
            if tags_raw:
                if isinstance(tags_raw, list):
                    tags = tags_raw
                elif isinstance(tags_raw, str):
                    # Could be JSON array or space-separated
                    tags_raw = tags_raw.strip()
                    if tags_raw.startswith("["):
                        import json

                        try:
                            tags = json.loads(tags_raw)
                        except Exception:
                            tags = tags_raw.split()
                    else:
                        tags = tags_raw.split()
            index_memory(mid, content, tags, mtype, stored_at)
            count += 1
        logger.info("BM25 backfill complete: %d memories indexed", count)
        return count
    except Exception:
        logger.exception("BM25 backfill failed")
        return 0


# ---------------------------------------------------------------------------
# RRF fusion helper
# ---------------------------------------------------------------------------


def fuse_rrf(
    *result_lists: List[Dict[str, Any]],
    k: int = RRF_K,
) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion across multiple ranked result lists.

    Each result must have 'memory_id' or 'id'. Returns merged list sorted
    by fused score descending. Original scores preserved in score_components.
    """
    scores: Dict[str, float] = {}
    results_by_id: Dict[str, Dict[str, Any]] = {}
    source_ranks: Dict[str, Dict[str, int]] = {}  # id -> {source: rank}

    for result_list in result_lists:
        for rank, result in enumerate(result_list):
            mid = result.get("memory_id") or result.get("id") or ""
            if not mid:
                continue

            rrf_score = 1.0 / (k + rank + 1)
            scores[mid] = scores.get(mid, 0.0) + rrf_score

            if mid not in results_by_id:
                results_by_id[mid] = result.copy()
                source_ranks[mid] = {}

            source = result.get("source", "unknown")
            source_ranks[mid][source] = rank + 1

    # Build final list
    fused = []
    for mid, score in sorted(scores.items(), key=lambda x: -x[1]):
        entry = results_by_id[mid]
        entry["rrf_score"] = score
        entry.setdefault("score_components", {})
        entry["score_components"]["rrf"] = score
        entry["score_components"]["source_ranks"] = source_ranks.get(mid, {})
        fused.append(entry)

    return fused


# ---------------------------------------------------------------------------
# FTS5 query builder
# ---------------------------------------------------------------------------

# Characters that are FTS5 operators or would break the query
_FTS_STRIP = re.compile(r'["\'\(\)\*\:\^\~\{\}\[\]\\]')
_WHITESPACE = re.compile(r"\s+")


def _build_fts_query(raw_query: str) -> str:
    """Convert a natural language query into an FTS5 query string.

    Strategy: extract words, join with implicit AND (FTS5 default).
    For multi-word queries we also add an OR-joined phrase match
    to boost results that have the words adjacent.
    """
    cleaned = _FTS_STRIP.sub(" ", raw_query)
    words = _WHITESPACE.split(cleaned.strip())
    words = [w for w in words if w and len(w) > 1]

    if not words:
        return ""

    if len(words) == 1:
        return words[0]

    # Individual terms (implicit AND) OR exact phrase
    terms_part = " ".join(words)
    phrase_part = '"' + " ".join(words) + '"'
    return f"({terms_part}) OR ({phrase_part})"
