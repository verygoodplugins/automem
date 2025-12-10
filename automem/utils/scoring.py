from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from automem.config import (
    SEARCH_WEIGHT_CONFIDENCE,
    SEARCH_WEIGHT_EXACT,
    SEARCH_WEIGHT_IMPORTANCE,
    SEARCH_WEIGHT_KEYWORD,
    SEARCH_WEIGHT_RECENCY,
    SEARCH_WEIGHT_RELATION,
    SEARCH_WEIGHT_TAG,
    SEARCH_WEIGHT_VECTOR,
)
from automem.utils.time import _parse_iso_datetime


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

    vector_component = (
        result.get("match_score", 0.0) if result.get("match_type") == "vector" else 0.0
    )
    keyword_component = (
        result.get("match_score", 0.0)
        if result.get("match_type") in {"keyword", "trending"}
        else 0.0
    )

    relation_component = 0.0
    if result.get("match_type") == "relation":
        relation_component = float(
            result.get("relation_score", result.get("match_score", 0.0)) or 0.0
        )
    elif "relation_score" in result:
        relation_component = float(result.get("relation_score") or 0.0)

    context_bonus = _compute_context_bonus(
        result, memory, tag_terms, metadata_terms, context_profile
    )

    final = (
        SEARCH_WEIGHT_VECTOR * vector_component
        + SEARCH_WEIGHT_KEYWORD * keyword_component
        + SEARCH_WEIGHT_RELATION * relation_component
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
        "relation": relation_component,
        "tag": tag_score,
        "importance": importance_score,
        "confidence": confidence_score,
        "recency": recency_score,
        "exact": exact_match,
        "context": context_bonus,
    }

    return final, components
