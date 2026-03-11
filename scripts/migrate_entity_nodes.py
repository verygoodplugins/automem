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

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()
load_dotenv(Path.home() / ".config" / "automem" / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ENTITY_TAG_RE = re.compile(r"^entity:([a-z]+):(.+)$")


def slug_to_name(slug: str) -> str:
    """Convert a slug like 'alice-smith' to 'Alice Smith'."""
    return slug.replace("-", " ").title()


def connect_falkordb():
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
    result = graph.query(
        "MATCH (m:Memory) WHERE m.tags IS NOT NULL RETURN m.id, m.tags"
    )
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


def run_migration(graph, entity_to_memories: dict[str, list[str]], *, dry_run: bool) -> None:
    """Create Entity nodes and REFERENCED_IN edges."""
    now = datetime.now(timezone.utc).isoformat()
    created_entities = 0
    created_edges = 0

    for entity_tag, memory_ids in sorted(entity_to_memories.items()):
        match = ENTITY_TAG_RE.match(entity_tag)
        if not match:
            continue
        category = match.group(1)
        slug = match.group(2)
        name = slug_to_name(slug)

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
        "Summary: %d entities, %d edges%s",
        created_entities,
        created_edges,
        " (dry run)" if dry_run else " created",
    )


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
