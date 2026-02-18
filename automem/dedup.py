"""Write-time deduplication gate for AutoMem.

Before storing a new memory, checks for semantically similar existing memories
and uses an LLM to classify the appropriate action:

- ADD: Genuinely new information, store normally
- UPDATE: Refines/adds detail to an existing memory, merge into it
- SUPERSEDE: Replaces an outdated memory (delete old, store new)
- NOOP: Already known, skip entirely

Inspired by Helixir's decision engine approach.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Minimum vector similarity to even consider dedup (below this, always ADD)
SIMILARITY_THRESHOLD = 0.70

# Maximum candidates to evaluate
MAX_CANDIDATES = 3

DEDUP_PROMPT = """You are a memory deduplication system. Given a NEW memory and EXISTING memories, decide what to do.

NEW MEMORY:
{new_content}

EXISTING MEMORIES:
{existing_memories}

For each existing memory, decide the relationship to the new memory. Then output ONE action for the new memory:

- ADD: The new memory contains genuinely new information not covered by any existing memory. Store it.
- UPDATE <id>: The new memory refines, corrects, or adds meaningful detail to an existing memory. The existing memory should be updated to incorporate the new information. Output the merged content.
- SUPERSEDE <id>: The new memory replaces an outdated existing memory (e.g., a decision changed, a status updated). The old one should be deleted and the new one stored.
- NOOP: The new memory is already fully covered by existing memories. Skip it.

Rules:
- If the new memory has ANY meaningful new information beyond what exists, prefer ADD or UPDATE over NOOP.
- UPDATE means the existing memory's content should be expanded/corrected. Provide the merged text.
- SUPERSEDE means the old memory is wrong/outdated and should be replaced entirely.
- NOOP only if the new memory is truly redundant — same facts, same level of detail.
- When in doubt, ADD. False negatives (storing a near-dupe) are less harmful than false positives (losing information).

Respond with ONLY valid JSON:
{{"action": "ADD"}}
or
{{"action": "UPDATE", "target_id": "<id>", "merged_content": "<full merged text>"}}
or
{{"action": "SUPERSEDE", "target_id": "<id>"}}
or
{{"action": "NOOP", "reason": "<brief reason>"}}"""


def check_dedup(
    new_content: str,
    generate_embedding: Callable[[str], List[float]],
    qdrant_client: Any,
    collection_name: str,
    openai_client: Any,
    model: str = "gpt-4o-mini",
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> Dict[str, Any]:
    """Check if a new memory should be added, merged, or skipped.

    Returns:
        Dict with keys:
        - action: "ADD" | "UPDATE" | "SUPERSEDE" | "NOOP"
        - target_id: (for UPDATE/SUPERSEDE) the existing memory ID to modify
        - merged_content: (for UPDATE) the merged text
        - reason: (for NOOP) why it was skipped
        - candidates: list of similar memories found (for debugging)
    """
    result: Dict[str, Any] = {"action": "ADD", "candidates": []}

    if not qdrant_client or not openai_client:
        return result

    # Step 1: Generate embedding for the new content
    try:
        embedding = generate_embedding(new_content)
    except Exception:
        logger.warning("Failed to generate embedding for dedup check, defaulting to ADD")
        return result

    if not embedding:
        return result

    # Step 2: Search for similar existing memories
    try:
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=embedding,
            limit=MAX_CANDIDATES,
            score_threshold=similarity_threshold,
        )
    except Exception:
        logger.warning("Qdrant search failed during dedup check, defaulting to ADD")
        return result

    if not search_results:
        return result

    # Step 3: Format candidates for LLM
    candidates = []
    for hit in search_results:
        payload = hit.payload or {}
        candidates.append(
            {
                "id": str(hit.id),
                "content": payload.get("content", ""),
                "score": round(hit.score, 3),
                "type": payload.get("type", ""),
                "importance": payload.get("importance", 0),
            }
        )

    result["candidates"] = candidates

    existing_text = "\n\n".join(
        f"[ID: {c['id']}] (similarity: {c['score']})\n{c['content']}"
        for c in candidates
    )

    # Step 4: Ask LLM to classify
    prompt = DEDUP_PROMPT.format(
        new_content=new_content,
        existing_memories=existing_text,
    )

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1000,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        if not raw:
            logger.warning("LLM returned empty content for dedup, defaulting to ADD")
            return result
        raw = raw.strip()
        decision = json.loads(raw)

        action = decision.get("action", "ADD").upper()
        if action not in ("ADD", "UPDATE", "SUPERSEDE", "NOOP"):
            action = "ADD"

        result["action"] = action

        # Validate target_id against actual candidates to prevent LLM hallucination
        candidate_ids = {c["id"] for c in candidates}
        target_id = decision.get("target_id", "")
        if target_id and target_id not in candidate_ids:
            logger.warning(
                "Dedup target_id %s not in candidates %s, falling back to ADD",
                target_id, candidate_ids,
            )
            result["action"] = "ADD"
            return result

        if action == "UPDATE":
            result["target_id"] = decision.get("target_id", "")
            result["merged_content"] = decision.get("merged_content", "")
            if not result["target_id"] or not result["merged_content"]:
                # Invalid UPDATE response, fall back to ADD
                result["action"] = "ADD"
                logger.warning("LLM returned UPDATE without target_id or merged_content, falling back to ADD")

        elif action == "SUPERSEDE":
            result["target_id"] = decision.get("target_id", "")
            if not result["target_id"]:
                result["action"] = "ADD"
                logger.warning("LLM returned SUPERSEDE without target_id, falling back to ADD")

        elif action == "NOOP":
            result["reason"] = decision.get("reason", "duplicate")

        logger.info(
            "Dedup decision: %s (candidates: %d, top_score: %.3f)",
            result["action"],
            len(candidates),
            candidates[0]["score"] if candidates else 0,
        )

    except Exception:
        logger.warning("LLM dedup classification failed, defaulting to ADD", exc_info=True)
        result["action"] = "ADD"

    return result
