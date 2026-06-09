"""Identity synthesis for Entity nodes.

Gathers memories linked to an entity and uses an LLM to produce a stable
2-5 sentence identity definition. Stores the result on the Entity node.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from automem.config import IDENTITY_SYNTHESIS_MODEL
from automem.utils.time import _parse_iso_datetime

logger = logging.getLogger(__name__)

IDENTITY_PROMPT_TEMPLATE = """\
Given these memories about {entity_name} ({category}), synthesize a stable identity definition.

Focus on:
- What/who this is (core definition)
- Key relationships to other entities
- Enduring attributes (not recent events)
- Role/function in the user's life

Do NOT include:
- Specific dated events (those are episodic)
- Recent/temporary states
- Speculation beyond what the memories support

Memories:
{memories}

{previous_section}
Identity (2-5 sentences):"""

STATE_SUPPRESSING_RELATIONS: Set[str] = {"INVALIDATED_BY", "EVOLVED_INTO"}


def _build_previous_section(previous_identity: Optional[str], version: int) -> str:
    """Build the previous identity section for the prompt.

    Every 5th version (v5, v10, ...) triggers a full re-synthesis from scratch
    to prevent identity drift. The check uses the *current* version before
    increment, so v5 triggers re-synthesis and the result is stored as v6.
    """
    if not previous_identity:
        return ""
    if version > 0 and version % 5 == 0:
        return "(Full re-synthesis — ignore previous identity and synthesize fresh from memories.)"
    return f"Previous identity (refine, don't replace unless contradicted):\n{previous_identity}"


def _state_reason_for_memory(memory: Dict[str, Any], now: datetime) -> Optional[str]:
    if memory.get("archived") is True:
        return "archived"

    t_valid = _parse_iso_datetime(memory.get("t_valid"))
    if t_valid is not None and t_valid > now:
        return "not_yet_valid"

    t_invalid = _parse_iso_datetime(memory.get("t_invalid"))
    if t_invalid is not None and t_invalid <= now:
        return "expired"

    return None


def _active_suppressed_memory_ids(graph: Any, memory_ids: List[str], now: datetime) -> Set[str]:
    ordered_ids = list(dict.fromkeys(memory_id for memory_id in memory_ids if memory_id))
    if not ordered_ids:
        return set()

    try:
        records = graph.query(
            """
            UNWIND $ids AS source_id
            MATCH (m:Memory {id: source_id})-[r]->(related:Memory)
            WHERE type(r) IN $types
            RETURN source_id, related.archived, related.t_valid, related.t_invalid
            """,
            {"ids": ordered_ids, "types": sorted(STATE_SUPPRESSING_RELATIONS)},
        )
    except Exception:
        logger.exception("Failed to load identity current-state suppressions")
        return set()

    suppressed: Set[str] = set()
    for row in getattr(records, "result_set", []) or []:
        if not row:
            continue
        source_id = str(row[0] or "")
        if not source_id:
            continue
        replacement = {
            "archived": row[1] if len(row) > 1 else None,
            "t_valid": row[2] if len(row) > 2 else None,
            "t_invalid": row[3] if len(row) > 3 else None,
        }
        if _state_reason_for_memory(replacement, now) is None:
            suppressed.add(source_id)
    return suppressed


def _gather_entity_memories(
    graph: Any,
    entity_id: str,
    limit: Optional[int] = 50,
    *,
    current_only: bool = True,
) -> Tuple[List[Dict[str, Any]], int]:
    """Fetch memories linked to an entity, ordered by importance desc."""
    limit_clause = "LIMIT $limit" if limit is not None else ""
    params: Dict[str, Any] = {"id": entity_id}
    if limit is not None:
        params["limit"] = limit

    result = graph.query(
        f"""
        MATCH (e:Entity {{id: $id}})-[:REFERENCED_IN]->(m:Memory)
        RETURN m.id, m.content, m.importance, m.timestamp, m.type,
               m.t_valid, m.t_invalid, m.archived
        ORDER BY coalesce(m.importance, 0.0) DESC
        {limit_clause}
        """,
        params,
    )
    memories: List[Dict[str, Any]] = []
    for row in getattr(result, "result_set", []) or []:
        memories.append(
            {
                "id": row[0],
                "content": row[1] or "",
                "importance": row[2],
                "timestamp": row[3],
                "type": row[4],
                "t_valid": row[5] if len(row) > 5 else None,
                "t_invalid": row[6] if len(row) > 6 else None,
                "archived": row[7] if len(row) > 7 else None,
            }
        )

    if not current_only:
        return memories, len(memories)

    now = datetime.now(timezone.utc)
    current_memories = [
        memory for memory in memories if _state_reason_for_memory(memory, now) is None
    ]
    suppressed_ids = _active_suppressed_memory_ids(
        graph,
        [str(memory.get("id") or "") for memory in current_memories],
        now,
    )
    if suppressed_ids:
        current_memories = [
            memory
            for memory in current_memories
            if str(memory.get("id") or "") not in suppressed_ids
        ]

    if limit is not None:
        return current_memories[:limit], len(current_memories)
    return current_memories, len(current_memories)


def _count_current_entity_references(graph: Any, entity_id: str) -> int:
    _memories, total_count = _gather_entity_memories(graph, entity_id, limit=None)
    return total_count


def _format_memories_for_prompt(memories: List[Dict[str, Any]]) -> str:
    """Format memories into a text block for the LLM prompt."""
    lines: List[str] = []
    for i, mem in enumerate(memories, 1):
        content = (mem.get("content") or "").strip()
        if not content:
            continue
        ts = mem.get("timestamp") or "unknown"
        lines.append(f"{i}. [{ts[:10]}] {content[:300]}")
    return "\n".join(lines)


def synthesize_identity(
    graph: Any,
    entity_id: str,
    openai_client: Any,
    *,
    model: Optional[str] = None,
    memory_limit: int = 50,
) -> Optional[str]:
    """Synthesize identity for a single entity.

    Args:
        graph: FalkorDB graph instance.
        entity_id: The entity ID (e.g. "entity:people:alice-smith").
        openai_client: OpenAI-compatible client.
        model: Model override (defaults to IDENTITY_SYNTHESIS_MODEL config).
        memory_limit: Max memories to gather.

    Returns:
        The synthesized identity string, or None if insufficient data.
    """
    model = model or IDENTITY_SYNTHESIS_MODEL

    # Fetch entity details
    ent_result = graph.query(
        """
        MATCH (e:Entity {id: $id})
        RETURN e.name, e.category, e.identity, e.identity_version
        """,
        {"id": entity_id},
    )
    ent_rows = getattr(ent_result, "result_set", []) or []
    if not ent_rows:
        logger.warning("Entity %s not found", entity_id)
        return None

    entity_name = ent_rows[0][0] or entity_id
    category = ent_rows[0][1] or "unknown"
    previous_identity = ent_rows[0][2]
    current_version = int(ent_rows[0][3] or 0)

    # Gather all current references for the source count, then sample for the prompt.
    all_current_memories, total_ref_count = _gather_entity_memories(
        graph,
        entity_id,
        limit=None,
    )
    memories = all_current_memories[:memory_limit]
    if not memories:
        logger.info("No memories for entity %s, skipping synthesis", entity_id)
        return None

    formatted_memories = _format_memories_for_prompt(memories)
    previous_section = _build_previous_section(previous_identity, current_version)

    prompt = IDENTITY_PROMPT_TEMPLATE.format(
        entity_name=entity_name,
        category=category,
        memories=formatted_memories,
        previous_section=previous_section,
    )

    try:
        # Use max_completion_tokens (newer API) with max_tokens fallback
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You synthesize concise identity definitions from episodic memories.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_completion_tokens=500,
            )
        except TypeError:
            # Older clients that don't support max_completion_tokens
            response = openai_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You synthesize concise identity definitions from episodic memories.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )
        raw_content = response.choices[0].message.content or ""
        identity_text = raw_content.strip()
        if not identity_text:
            logger.warning("LLM returned empty content for entity %s", entity_id)
            return None
    except Exception:
        logger.exception("LLM call failed for entity %s", entity_id)
        return None

    # Store on entity node
    now = datetime.now(timezone.utc).isoformat()
    graph.query(
        """
        MATCH (e:Entity {id: $id})
        SET e.identity = $identity,
            e.identity_version = $version,
            e.identity_updated_at = $now,
            e.identity_source_count = $source_count
        """,
        {
            "id": entity_id,
            "identity": identity_text,
            "version": current_version + 1,
            "now": now,
            "source_count": int(total_ref_count),
        },
    )

    logger.info(
        "Synthesized identity for %s (v%d, %d current sources, %d chars)",
        entity_id,
        current_version + 1,
        int(total_ref_count),
        len(identity_text),
    )
    return identity_text


def run_identity_consolidation(
    graph: Any,
    openai_client: Any,
    *,
    model: Optional[str] = None,
    min_references: int = 1,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run identity synthesis for all eligible entities.

    This includes:
    1. Entity dedup (find and auto-merge candidates)
    2. Identity synthesis for entities needing it

    Returns:
        Summary dict with counts and details.
    """
    from automem.consolidation.entity_dedup import find_merge_candidates, merge_entities

    result: Dict[str, Any] = {
        "merges_performed": [],
        "merge_candidates_for_review": [],
        "identities_synthesized": 0,
        "entities_examined": 0,
        "errors": [],
    }

    # Step 1: Dedup
    try:
        auto_merge, review = find_merge_candidates(graph)
        result["merge_candidates_for_review"] = [
            {
                "canonical": c.canonical_id,
                "alias": c.alias_id,
                "confidence": c.confidence,
                "reason": c.reason,
            }
            for c in review
        ]

        if not dry_run:
            for candidate in auto_merge:
                try:
                    merge_result = merge_entities(graph, candidate.canonical_id, candidate.alias_id)
                    result["merges_performed"].append(
                        {
                            "canonical": merge_result.canonical_id,
                            "alias": merge_result.alias_id,
                            "alias_slug": merge_result.alias_slug,
                            "edges_moved": merge_result.edges_moved,
                        }
                    )
                except Exception as exc:
                    logger.exception(
                        "Failed to merge %s into %s",
                        candidate.alias_id,
                        candidate.canonical_id,
                    )
                    result["errors"].append(f"merge {candidate.alias_id}: {exc}")
    except Exception as exc:
        logger.exception("Entity dedup failed")
        result["errors"].append(f"dedup: {exc}")

    # Step 2: Synthesize identities (only for new or changed entities)
    ent_result = graph.query(
        """
        MATCH (e:Entity)
        WHERE e.merged_into IS NULL
        OPTIONAL MATCH (e)-[ref:REFERENCED_IN]->()
        WITH e, count(ref) as ref_count
        RETURN e.id, e.identity_source_count, e.identity, ref_count
        """
    )
    for row in getattr(ent_result, "result_set", []) or []:
        entity_id = row[0]
        stored_source_count = int(row[1] or 0)
        existing_identity = row[2]
        actual_ref_count = _count_current_entity_references(graph, entity_id)
        result["entities_examined"] += 1

        if actual_ref_count < min_references:
            continue

        # Only re-synthesize if entity has no identity or reference count changed
        needs_synthesis = existing_identity is None or actual_ref_count != stored_source_count
        if not needs_synthesis:
            continue

        if dry_run:
            result["identities_synthesized"] += 1
            continue

        try:
            identity = synthesize_identity(
                graph,
                entity_id,
                openai_client,
                model=model,
            )
            if identity:
                result["identities_synthesized"] += 1
        except Exception as exc:
            logger.exception("Identity synthesis failed for %s", entity_id)
            result["errors"].append(f"synthesis {entity_id}: {exc}")

    return result
