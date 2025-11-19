from __future__ import annotations

import re
import json
from typing import Any, Dict, List, Optional, Set, Tuple

from automem.utils.time import _parse_iso_datetime
from automem.config import (
    SEARCH_WEIGHT_VECTOR,
    SEARCH_WEIGHT_KEYWORD,
    SEARCH_WEIGHT_TAG,
    SEARCH_WEIGHT_IMPORTANCE,
    SEARCH_WEIGHT_CONFIDENCE,
    SEARCH_WEIGHT_RECENCY,
    SEARCH_WEIGHT_EXACT,
)


def _parse_metadata_field(value: Any) -> Any:
    """Convert stored metadata value back into a dictionary when possible."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            decoded = json.loads(value)
            if isinstance(decoded, dict):
                return decoded
        except Exception:
            return value
    return value


def _collect_metadata_terms(metadata: Dict[str, Any]) -> Set[str]:
    terms: Set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, str):
            trimmed = item.strip()
            if not trimmed:
                return
            if len(trimmed) <= 256:
                lower = trimmed.lower()
                terms.add(lower)
                for token in re.findall(r"[a-z0-9_\-]+", lower):
                    terms.add(token)
        elif isinstance(item, (list, tuple, set)):
            for sub in item:
                visit(sub)
        elif isinstance(item, dict):
            for sub in item.values():
                visit(sub)

    visit(metadata)
    return terms


def _compute_recency_score(timestamp: Optional[str]) -> float:
    """
    Compute a recency score for an ISO-8601 timestamp with a linear decay over 180 days.
    
    Parameters:
        timestamp (Optional[str]): ISO-8601 timestamp string representing the item's time; if falsy or unparsable, treated as absent.
    
    Returns:
        float: A score between 0.0 and 1.0 where 1.0 means the timestamp is now or in the future, values linearly decline to 0.0 at 180 days of age, and 0.0 is returned for missing or unparsable timestamps.
    """
    if not timestamp:
        return 0.0
    parsed = _parse_iso_datetime(timestamp)
    if not parsed:
        return 0.0
    from datetime import datetime, timezone  # local import to avoid cycles

    age_days = max((datetime.now(timezone.utc) - parsed).total_seconds() / 86400.0, 0.0)
    if age_days <= 0:
        return 1.0
    # Linear decay over 180 days
    return max(0.0, 1.0 - (age_days / 180.0))


def _context_tag_hit(tags: Set[str], priority_tags: Set[str]) -> bool:
    """
    Determine if any tag equals or relates to a priority tag.
    
    Parameters:
        tags (Set[str]): Candidate tag strings to check.
        priority_tags (Set[str]): Priority tag strings to match against.
    
    Returns:
        `true` if any tag equals a priority, starts with a priority, or contains a priority substring, `false` otherwise.
    """
    if not tags or not priority_tags:
        return False
    for tag in tags:
        for priority in priority_tags:
            if tag == priority or tag.startswith(priority) or priority in tag:
                return True
    return False


def _compute_context_bonus(
    result: Dict[str, Any],
    memory: Dict[str, Any],
    tag_terms: Set[str],
    metadata_terms: Set[str],
    context_profile: Optional[Dict[str, Any]],
) -> float:
    """
    Compute an additive context-based bonus score for a search result based on a provided context profile.
    
    Parameters:
        result (Dict[str, Any]): The search result object; used to obtain an identifier fallback when checking anchored IDs.
        memory (Dict[str, Any]): The stored memory record associated with the result; used for type and other metadata fields.
        tag_terms (Set[str]): Lowercased tag terms extracted from the memory.
        metadata_terms (Set[str]): Lowercased terms extracted from the memory's metadata.
        context_profile (Optional[Dict[str, Any]]): Profile that may include:
            - "weights": dict of contribution weights for "tag", "type", "keyword", "anchor".
            - "priority_tags": set of tags to prioritize.
            - "priority_types": set of memory types to prioritize.
            - "priority_ids": set of memory IDs to anchor to.
            - "priority_keywords": set of metadata keywords to prioritize.
            If None or empty, no bonus is applied.
    
    Returns:
        float: The accumulated context bonus (>= 0.0) computed by summing applicable weighted contributions.
    """
    if not context_profile:
        return 0.0

    weights = context_profile.get("weights") or {}
    priority_tags: Set[str] = context_profile.get("priority_tags") or set()
    priority_types: Set[str] = context_profile.get("priority_types") or set()
    priority_ids: Set[str] = context_profile.get("priority_ids") or set()
    priority_keywords: Set[str] = context_profile.get("priority_keywords") or set()

    bonus = 0.0
    if priority_tags and _context_tag_hit(tag_terms, priority_tags):
        bonus += float(weights.get("tag", 0.45))

    if priority_types:
        mem_type = memory.get("type")
        if isinstance(mem_type, str) and mem_type.strip().title() in priority_types:
            bonus += float(weights.get("type", 0.25))

    if priority_keywords and metadata_terms:
        if any(keyword in metadata_terms for keyword in priority_keywords):
            bonus += float(weights.get("keyword", 0.2))

    if priority_ids:
        memory_id = str(result.get("id") or memory.get("id") or "")
        if memory_id and memory_id in priority_ids:
            bonus += float(weights.get("anchor", 0.9))

    return bonus


def _compute_metadata_score(
    result: Dict[str, Any],
    query: str,
    tokens: List[str],
    context_profile: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute a composite metadata score for a search result and provide a per-component breakdown.
    
    Parameters:
        result (Dict[str, Any]): Search result containing fields like `match_type`, `match_score`, and an optional `memory` dict with `metadata`, `tags`, `importance`, `confidence`, `timestamp`, and `id`.
        query (str): Original query string used for exact-match checks against metadata terms.
        tokens (List[str]): Tokenized query terms used to measure tag/metadata token hits.
        context_profile (Optional[Dict[str, Any]]): Optional profile that can contribute a context-based bonus. Expected keys (optional) include `weights` (mapping of component names to weight floats), `priority_tags` (set of tags), `priority_types` (sequence of memory types), `priority_ids` (sequence of memory ids), and `priority_keywords` (set of keywords).
    
    Returns:
        Tuple[float, Dict[str, float]]: A tuple where the first element is the final aggregated score and the second element is a dictionary of individual component scores with keys: `"vector"`, `"keyword"`, `"tag"`, `"importance"`, `"confidence"`, `"recency"`, `"exact"`, and `"context"`.
    """
    memory = result.get("memory", {})
    metadata = _parse_metadata_field(memory.get("metadata")) if memory else {}
    metadata_terms = _collect_metadata_terms(metadata) if isinstance(metadata, dict) else set()

    tags = memory.get("tags") or []
    tag_terms = {str(tag).lower() for tag in tags if isinstance(tag, str)}

    token_hits = 0
    for token in tokens:
        if token in tag_terms or token in metadata_terms:
            token_hits += 1

    exact_match = 0.0
    normalized_query = query.lower().strip()
    if normalized_query and normalized_query in metadata_terms:
        exact_match = 1.0

    importance = memory.get("importance")
    importance_score = float(importance) if isinstance(importance, (int, float)) else 0.0

    confidence = memory.get("confidence")
    confidence_score = float(confidence) if isinstance(confidence, (int, float)) else 0.0

    recency_score = _compute_recency_score(memory.get("timestamp"))

    tag_score = token_hits / max(len(tokens), 1) if tokens else 0.0

    vector_component = result.get("match_score", 0.0) if result.get("match_type") == "vector" else 0.0
    keyword_component = result.get("match_score", 0.0) if result.get("match_type") in {"keyword", "trending"} else 0.0

    context_bonus = _compute_context_bonus(result, memory, tag_terms, metadata_terms, context_profile)

    final = (
        SEARCH_WEIGHT_VECTOR * vector_component
        + SEARCH_WEIGHT_KEYWORD * keyword_component
        + SEARCH_WEIGHT_TAG * tag_score
        + SEARCH_WEIGHT_IMPORTANCE * importance_score
        + SEARCH_WEIGHT_CONFIDENCE * confidence_score
        + SEARCH_WEIGHT_RECENCY * recency_score
        + SEARCH_WEIGHT_EXACT * exact_match
        + context_bonus
    )

    components = {
        "vector": vector_component,
        "keyword": keyword_component,
        "tag": tag_score,
        "importance": importance_score,
        "confidence": confidence_score,
        "recency": recency_score,
        "exact": exact_match,
        "context": context_bonus,
    }

    return final, components
