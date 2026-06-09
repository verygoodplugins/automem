#!/usr/bin/env python3
"""
Migrate entity tags on Memory nodes into first-class Entity nodes in FalkorDB.

Scans all Memory nodes for `entity:{category}:{slug}` tags, creates Entity nodes,
and links them via REFERENCED_IN relationships. Idempotent (safe to re-run).

Usage:
    python scripts/migrate_entity_nodes.py [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from automem.utils.entity_quality import validate_entity_tag

load_dotenv()
load_dotenv(Path.home() / ".config" / "automem" / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ENTITY_TAG_RE = re.compile(r"^entity:([a-z]+):(.+)$")
MIGRATION_MIN_CONFIDENCE = 0.8

_MIGRATION_REVIEW_PREFIXES = {
    "current",
    "final",
    "first",
    "last",
    "next",
    "old",
    "new",
    "priority",
    "project",
    "review",
    "source",
    "status",
    "target",
    "task",
    "ticket",
    "today",
    "tomorrow",
}

_MIGRATION_REVIEW_SUFFIXES = {
    "hygiene",
    "priority",
    "review",
    "status",
    "support",
    "ticket",
}

_MIGRATION_DURABLE_SUFFIXES = (
    "ai",
    "api",
    "app",
    "bot",
    "cli",
    "cloud",
    "corp",
    "db",
    "hub",
    "kit",
    "labs",
    "sdk",
)


def slug_to_name(slug: str) -> str:
    """Convert a slug like 'alice-smith' to 'Alice Smith'."""
    return slug.replace("-", " ").title()


def _slug_tokens(slug: str) -> list[str]:
    return [token for token in slug.split("-") if token]


def _migration_review_reason(validation: Any) -> str | None:
    """Return why an accepted tag should stay audit-only for Entity migration.

    Historical tags lack original casing and extraction context. Single-token
    non-person tags like "socket" may be real products or ordinary nouns, so
    they remain review-only until an admin/assistant path can classify them.
    """
    if not validation.accepted or validation.category == "people":
        return None

    tokens = _slug_tokens(validation.canonical_slug)
    if not tokens:
        return "migration_review_low_signal_phrase"

    if len(tokens) == 1:
        token = tokens[0]
        if re.search(r"[a-z]+[0-9]+|[0-9]+[a-z]+", token):
            return None
        return "migration_review_single_token_nonperson"

    if any(len(token) == 1 for token in tokens):
        return "migration_review_low_signal_phrase"

    if tokens[0] in _MIGRATION_REVIEW_PREFIXES or tokens[-1] in _MIGRATION_REVIEW_SUFFIXES:
        return "migration_review_low_signal_phrase"

    if any(token.endswith(_MIGRATION_DURABLE_SUFFIXES) for token in tokens):
        return None

    if any(re.search(r"[a-z]+[0-9]+|[0-9]+[a-z]+", token) for token in tokens):
        return None

    return "migration_review_low_signal_phrase"


def connect_falkordb() -> Any:
    """Connect to FalkorDB and return the graph."""
    from automem.config import FALKORDB_PORT, GRAPH_NAME

    try:
        from falkordb import FalkorDB
    except ImportError:
        logger.error("falkordb package not installed")
        sys.exit(1)

    host = os.getenv("FALKORDB_HOST", "localhost")
    password = os.getenv("FALKORDB_PASSWORD")
    params = {"host": host, "port": FALKORDB_PORT}
    if password:
        params["password"] = password
        params["username"] = "default"

    db = FalkorDB(**params)
    return db.select_graph(GRAPH_NAME)


def collect_entity_tags(graph) -> dict[str, list[str]]:
    """Scan all Memory nodes and collect entity tags mapped to memory IDs.

    Returns:
        dict mapping entity tag (e.g. "entity:people:alice-smith") to list of memory IDs
    """
    result = graph.query("MATCH (m:Memory) WHERE m.tags IS NOT NULL RETURN m.id, m.tags")
    entity_to_memories: dict[str, list[str]] = defaultdict(list)
    for row in getattr(result, "result_set", []) or []:
        mem_id = row[0]
        tags = row[1]
        if not isinstance(tags, list):
            continue
        for tag in tags:
            if isinstance(tag, str) and ENTITY_TAG_RE.match(tag):
                entity_to_memories[tag].append(mem_id)
    return dict(entity_to_memories)


def _dedup_memory_ids(memory_ids: list[str]) -> list[str]:
    return list(dict.fromkeys(str(memory_id) for memory_id in memory_ids if memory_id))


def run_migration(
    graph, entity_to_memories: dict[str, list[str]], *, dry_run: bool
) -> dict[str, Any]:
    """Create Entity nodes and REFERENCED_IN edges."""
    now = datetime.now(timezone.utc).isoformat()
    created_entities = 0
    created_edges = 0
    accepted: dict[str, dict[str, Any]] = {}
    rejected_entities: list[dict[str, Any]] = []

    for entity_tag, memory_ids in sorted(entity_to_memories.items()):
        match = ENTITY_TAG_RE.match(entity_tag)
        if not match:
            continue
        validation = validate_entity_tag(entity_tag)
        if not validation.accepted:
            rejected_entities.append(
                {
                    "tag": entity_tag,
                    "category": validation.category,
                    "slug": validation.slug,
                    "reason": validation.reason,
                    "references": len(memory_ids),
                }
            )
            logger.info(
                "Rejected Entity: %s (reason=%s, references=%d)%s",
                entity_tag,
                validation.reason,
                len(memory_ids),
                " [DRY RUN]" if dry_run else "",
            )
            continue
        if validation.confidence < MIGRATION_MIN_CONFIDENCE:
            rejected_entities.append(
                {
                    "tag": entity_tag,
                    "category": validation.category,
                    "slug": validation.slug,
                    "reason": "migration_low_confidence",
                    "references": len(memory_ids),
                }
            )
            logger.info(
                "Rejected Entity: %s (reason=%s, confidence=%.2f, references=%d)%s",
                entity_tag,
                "migration_low_confidence",
                validation.confidence,
                len(memory_ids),
                " [DRY RUN]" if dry_run else "",
            )
            continue
        review_reason = _migration_review_reason(validation)
        if review_reason:
            rejected_entities.append(
                {
                    "tag": entity_tag,
                    "category": validation.category,
                    "slug": validation.slug,
                    "reason": review_reason,
                    "references": len(memory_ids),
                    "confidence": validation.confidence,
                }
            )
            logger.info(
                "Rejected Entity: %s (reason=%s, confidence=%.2f, references=%d)%s",
                entity_tag,
                review_reason,
                validation.confidence,
                len(memory_ids),
                " [DRY RUN]" if dry_run else "",
            )
            continue

        canonical_id = validation.canonical_tag
        entry = accepted.setdefault(
            canonical_id,
            {
                "id": canonical_id,
                "category": validation.category,
                "slug": validation.canonical_slug,
                "name": validation.name or slug_to_name(validation.canonical_slug),
                "memory_ids": [],
                "source_tags": [],
                "confidence": validation.confidence,
            },
        )
        entry["memory_ids"] = _dedup_memory_ids(entry["memory_ids"] + memory_ids)
        entry["source_tags"] = sorted(set(entry["source_tags"] + [entity_tag]))

    accepted_items = sorted(accepted.values(), key=lambda item: item["id"])

    for entry in accepted_items:
        entity_tag = entry["id"]
        category = entry["category"]
        slug = entry["slug"]
        name = entry["name"]
        memory_ids = entry["memory_ids"]

        logger.info(
            "Entity: %s (category=%s, slug=%s, references=%d)%s",
            entity_tag,
            category,
            slug,
            len(memory_ids),
            " [DRY RUN]" if dry_run else "",
        )

        if not dry_run:
            # MERGE Entity node (idempotent)
            graph.query(
                """
                MERGE (e:Entity {id: $id})
                ON CREATE SET
                    e.slug = $slug,
                    e.category = $category,
                    e.name = $name,
                    e.aliases = [],
                    e.identity = null,
                    e.identity_version = 0,
                    e.identity_updated_at = null,
                    e.identity_source_count = $ref_count,
                    e.created_at = $now,
                    e.last_referenced_at = $now
                ON MATCH SET
                    e.identity_source_count = $ref_count,
                    e.last_referenced_at = $now
                """,
                {
                    "id": entity_tag,
                    "slug": slug,
                    "category": category,
                    "name": name,
                    "ref_count": len(memory_ids),
                    "now": now,
                },
            )
            created_entities += 1

            # Create REFERENCED_IN edges (batched)
            graph.query(
                """
                MATCH (e:Entity {id: $entity_id})
                UNWIND $mem_ids AS mid
                MATCH (m:Memory {id: mid})
                MERGE (e)-[:REFERENCED_IN]->(m)
                """,
                {"entity_id": entity_tag, "mem_ids": memory_ids},
            )
            created_edges += len(memory_ids)
        else:
            created_entities += 1
            created_edges += len(memory_ids)

    logger.info(
        "Summary: %d accepted entities, %d rejected entities, %d edges%s",
        created_entities,
        len(rejected_entities),
        created_edges,
        " (dry run)" if dry_run else " created",
    )
    return {
        "accepted_entities": len(accepted_items),
        "rejected_entity_count": len(rejected_entities),
        "created_entities": created_entities,
        "created_edges": created_edges,
        "accepted": [
            {
                "id": item["id"],
                "category": item["category"],
                "slug": item["slug"],
                "name": item["name"],
                "references": len(item["memory_ids"]),
                "source_tags": item["source_tags"],
                "confidence": item["confidence"],
            }
            for item in accepted_items
        ],
        "rejected_entities": rejected_entities,
        "readiness_blockers": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate entity tags to Entity nodes")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be created")
    args = parser.parse_args()

    graph = connect_falkordb()
    entity_to_memories = collect_entity_tags(graph)
    logger.info("Found %d distinct entity tags across memories", len(entity_to_memories))

    if not entity_to_memories:
        logger.info("Nothing to migrate.")
        return

    run_migration(graph, entity_to_memories, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
