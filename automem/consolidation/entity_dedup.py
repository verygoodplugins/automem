"""Entity deduplication logic for AutoMem.

Identifies and merges duplicate entities by analysing:
- String similarity of slugs (substring, Levenshtein distance)
- Memory overlap: if >60% of memories for entity A are also tagged with entity B
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MergeCandidate:
    """A pair of entities that may be duplicates."""

    entity_a_id: str
    entity_b_id: str
    canonical_id: str
    alias_id: str
    confidence: float
    reason: str


@dataclass
class MergeResult:
    """Outcome of a merge operation."""

    canonical_id: str
    alias_id: str
    alias_slug: str
    edges_moved: int = 0


def _levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(curr_row[j] + 1, prev_row[j + 1] + 1, prev_row[j] + cost))
        prev_row = curr_row
    return prev_row[-1]


def _slug_similarity(slug_a: str, slug_b: str) -> float:
    """Heuristic similarity score (0-1) between two entity slugs."""
    if slug_a == slug_b:
        return 1.0
    # Substring match — strong signal, especially for hyphenated slugs
    # e.g. "alice" in "alice-smith" should score high
    if slug_a in slug_b or slug_b in slug_a:
        shorter = min(len(slug_a), len(slug_b))
        longer = max(len(slug_a), len(slug_b))
        # Boost substring matches: at least 0.6 similarity
        ratio = shorter / longer if longer > 0 else 0.0
        return max(0.6, ratio)
    # Levenshtein
    max_len = max(len(slug_a), len(slug_b))
    if max_len == 0:
        return 1.0
    dist = _levenshtein(slug_a, slug_b)
    return max(0.0, 1.0 - dist / max_len)


def _memory_overlap(memories_a: Set[str], memories_b: Set[str]) -> float:
    """Fraction of memories in the *smaller* set that also appear in the larger set."""
    if not memories_a or not memories_b:
        return 0.0
    smaller, larger = (
        (memories_a, memories_b) if len(memories_a) <= len(memories_b) else (memories_b, memories_a)
    )
    overlap = len(smaller & larger)
    return overlap / len(smaller)


def find_merge_candidates(
    graph: Any,
    *,
    min_slug_similarity: float = 0.5,
    min_overlap_for_auto: float = 0.6,
) -> Tuple[List[MergeCandidate], List[MergeCandidate]]:
    """Scan Entity nodes and find dedup candidates.

    Returns:
        (auto_merge, review) - two lists of MergeCandidate objects.
        auto_merge: high-confidence candidates that should be merged automatically.
        review: lower-confidence candidates for human review.
    """
    # Load all entities
    result = graph.query(
        "MATCH (e:Entity) WHERE e.merged_into IS NULL RETURN e.id, e.slug, e.category, e.aliases"
    )
    entities: List[Dict[str, Any]] = []
    for row in getattr(result, "result_set", []) or []:
        aliases = row[3] if row[3] else []
        if isinstance(aliases, str):
            try:
                aliases = json.loads(aliases)
            except Exception:
                aliases = []
        entities.append(
            {
                "id": row[0],
                "slug": row[1],
                "category": row[2],
                "aliases": aliases,
            }
        )

    if len(entities) < 2:
        return [], []

    # Gather memory sets per entity (single batch query)
    entity_memories: Dict[str, Set[str]] = defaultdict(set)
    all_edges = graph.query(
        "MATCH (e:Entity)-[:REFERENCED_IN]->(m:Memory) WHERE e.merged_into IS NULL RETURN e.id, m.id"
    )
    for row in getattr(all_edges, "result_set", []) or []:
        if row[0] and row[1]:
            entity_memories[row[0]].add(str(row[1]))

    auto_merge: List[MergeCandidate] = []
    review: List[MergeCandidate] = []

    # Compare pairs within the same category
    for i, ea in enumerate(entities):
        for eb in entities[i + 1 :]:
            if ea["category"] != eb["category"]:
                continue

            slug_sim = _slug_similarity(ea["slug"], eb["slug"])
            if slug_sim < min_slug_similarity:
                continue

            overlap = _memory_overlap(
                entity_memories.get(ea["id"], set()),
                entity_memories.get(eb["id"], set()),
            )

            # Determine canonical: longer slug wins
            if len(ea["slug"]) >= len(eb["slug"]):
                canonical_id, alias_id = ea["id"], eb["id"]
            else:
                canonical_id, alias_id = eb["id"], ea["id"]

            is_substring = ea["slug"] in eb["slug"] or eb["slug"] in ea["slug"]
            confidence = min(1.0, slug_sim * 0.4 + overlap * 0.6)

            candidate = MergeCandidate(
                entity_a_id=ea["id"],
                entity_b_id=eb["id"],
                canonical_id=canonical_id,
                alias_id=alias_id,
                confidence=confidence,
                reason=f"slug_sim={slug_sim:.2f}, overlap={overlap:.2f}, substring={is_substring}",
            )

            if is_substring and overlap > min_overlap_for_auto:
                auto_merge.append(candidate)
            elif confidence >= 0.5:
                review.append(candidate)

    return auto_merge, review


def merge_entities(graph: Any, canonical_id: str, alias_id: str) -> MergeResult:
    """Merge alias entity into canonical entity.

    - Moves REFERENCED_IN edges from alias to canonical
    - Adds alias slug to canonical's aliases list
    - Marks alias entity with merged_into
    """
    # Get alias slug
    res = graph.query(
        "MATCH (e:Entity {id: $id}) RETURN e.slug, e.aliases",
        {"id": alias_id},
    )
    alias_slug = ""
    alias_aliases: List[str] = []
    for row in getattr(res, "result_set", []) or []:
        alias_slug = row[0] or ""
        raw_aliases = row[1] or []
        if isinstance(raw_aliases, str):
            try:
                raw_aliases = json.loads(raw_aliases)
            except Exception:
                raw_aliases = []
        alias_aliases = raw_aliases

    # Move edges: collect memory IDs from alias and batch-create edges on canonical
    edge_res = graph.query(
        "MATCH (e:Entity {id: $id})-[:REFERENCED_IN]->(m:Memory) RETURN m.id",
        {"id": alias_id},
    )
    mem_ids: List[str] = []
    for row in getattr(edge_res, "result_set", []) or []:
        if row[0]:
            mem_ids.append(str(row[0]))

    edges_moved = len(mem_ids)
    if mem_ids:
        # Copy edges to canonical
        graph.query(
            """
            MATCH (e:Entity {id: $canonical_id})
            UNWIND $mem_ids AS mid
            MATCH (m:Memory {id: mid})
            MERGE (e)-[:REFERENCED_IN]->(m)
            """,
            {"canonical_id": canonical_id, "mem_ids": mem_ids},
        )
        # Remove old edges from alias
        graph.query(
            "MATCH (e:Entity {id: $alias_id})-[r:REFERENCED_IN]->() DELETE r",
            {"alias_id": alias_id},
        )

    # Update canonical aliases (deduplicated)
    # Fetch current canonical aliases first
    canon_res = graph.query(
        "MATCH (e:Entity {id: $id}) RETURN e.aliases",
        {"id": canonical_id},
    )
    current_aliases: List[str] = []
    for crow in getattr(canon_res, "result_set", []) or []:
        raw = crow[0] or []
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raw = []
        current_aliases = raw

    incoming_aliases = [alias_slug] + alias_aliases if alias_slug else alias_aliases
    merged_aliases = list(
        dict.fromkeys(current_aliases + incoming_aliases)
    )  # dedup, preserve order
    graph.query(
        """
        MATCH (e:Entity {id: $id})
        SET e.aliases = $aliases
        """,
        {"id": canonical_id, "aliases": merged_aliases},
    )

    # Mark alias as merged
    graph.query(
        "MATCH (e:Entity {id: $id}) SET e.merged_into = $canonical_id",
        {"id": alias_id, "canonical_id": canonical_id},
    )

    logger.info(
        "Merged entity %s into %s (slug=%s, edges=%d)",
        alias_id,
        canonical_id,
        alias_slug,
        edges_moved,
    )

    return MergeResult(
        canonical_id=canonical_id,
        alias_id=alias_id,
        alias_slug=alias_slug,
        edges_moved=edges_moved,
    )
