from __future__ import annotations

import re
from typing import Any, List, Optional, Set


def _normalize_tag_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        if not raw.strip():
            return []
        return [part.strip() for part in raw.split(",") if part.strip()]
    if isinstance(raw, (list, tuple, set)):
        tags: List[str] = []
        for item in raw:
            if isinstance(item, str) and item.strip():
                tags.append(item.strip())
        return tags
    return []


def _expand_tag_prefixes(tag: str) -> List[str]:
    """Expand a tag into all prefixes using ':' as the canonical delimiter."""
    parts = re.split(r"[:/]", tag)
    prefixes: List[str] = []
    accumulator: List[str] = []
    for part in parts:
        if not part:
            continue
        accumulator.append(part)
        prefixes.append(":".join(accumulator))
    return prefixes


def _compute_tag_prefixes(tags: List[str]) -> List[str]:
    """Compute unique, lowercased tag prefixes for fast prefix filtering."""
    seen: Set[str] = set()
    prefixes: List[str] = []
    for tag in tags or []:
        normalized = (tag or "").strip().lower()
        if not normalized:
            continue
        for prefix in _expand_tag_prefixes(normalized):
            if prefix not in seen:
                seen.add(prefix)
                prefixes.append(prefix)
    return prefixes


def _prepare_tag_filters(tag_filters: Optional[List[str]]) -> List[str]:
    """Normalize incoming tag filters for matching and persistence."""
    return [
        tag.strip().lower()
        for tag in (tag_filters or [])
        if isinstance(tag, str) and tag.strip()
    ]

