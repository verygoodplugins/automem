from __future__ import annotations

from typing import Iterable, List, Optional

from qdrant_client import models as qdrant_models

from automem.utils.tags import _prepare_tag_filters


def _build_qdrant_tag_filter(
    tags: Optional[List[str]],
    mode: str = "any",
    match: str = "exact",
    excluded_types: Optional[Iterable[str]] = None,
):
    """Build a Qdrant filter for tag constraints, supporting mode/match semantics.

    Extracted for reuse by Qdrant interactions.
    """
    normalized_tags = _prepare_tag_filters(tags)
    normalized_excluded_types = [
        str(memory_type).strip()
        for memory_type in (excluded_types or [])
        if str(memory_type).strip()
    ]
    if not normalized_tags and not normalized_excluded_types:
        return None

    target_key = "tag_prefixes" if match == "prefix" else "tags"
    normalized_mode = "all" if mode == "all" else "any"
    must_conditions = []

    if normalized_mode == "any":
        if normalized_tags:
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key=target_key,
                    match=qdrant_models.MatchAny(any=normalized_tags),
                )
            )
    else:
        must_conditions.extend(
            qdrant_models.FieldCondition(
                key=target_key,
                match=qdrant_models.MatchValue(value=tag),
            )
            for tag in normalized_tags
        )

    must_not_conditions = (
        [
            qdrant_models.FieldCondition(
                key="type",
                match=qdrant_models.MatchAny(any=normalized_excluded_types),
            )
        ]
        if normalized_excluded_types
        else []
    )

    return qdrant_models.Filter(must=must_conditions, must_not=must_not_conditions)
