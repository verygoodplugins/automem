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


def _compute_metadata_score(
    result: Dict[str, Any],
    query: str,
    tokens: List[str],
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

    vector_component = result.get("match_score", 0.0) if result.get("match_type") == "vector" else 0.0
    keyword_component = result.get("match_score", 0.0) if result.get("match_type") in {"keyword", "trending"} else 0.0

    final = (
        SEARCH_WEIGHT_VECTOR * vector_component
        + SEARCH_WEIGHT_KEYWORD * keyword_component
        + SEARCH_WEIGHT_TAG * tag_score
        + SEARCH_WEIGHT_IMPORTANCE * importance_score
        + SEARCH_WEIGHT_CONFIDENCE * confidence_score
        + SEARCH_WEIGHT_RECENCY * recency_score
        + SEARCH_WEIGHT_EXACT * exact_match
    )

    components = {
        "vector": vector_component,
        "keyword": keyword_component,
        "tag": tag_score,
        "importance": importance_score,
        "confidence": confidence_score,
        "recency": recency_score,
        "exact": exact_match,
    }

    return final, components

