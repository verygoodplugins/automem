from __future__ import annotations


def _build_graph_tag_predicate(tag_mode: str, tag_match: str) -> str:
    """Construct a Cypher predicate for tag filtering with mode/match semantics.

    Mirrors the implementation in app.py.
    """
    normalized_mode = "all" if tag_mode == "all" else "any"
    normalized_match = "prefix" if tag_match == "prefix" else "exact"
    tags_expr = "[tag IN coalesce(m.tags, []) | toLower(tag)]"

    if normalized_match == "exact":
        if normalized_mode == "all":
            return f"ALL(req IN $tag_filters WHERE req IN {tags_expr})"
        return f"ANY(tag IN {tags_expr} WHERE tag IN $tag_filters)"

    prefixes_expr = "coalesce(m.tag_prefixes, [])"
    prefix_any = f"ANY(req IN $tag_filters WHERE req IN {prefixes_expr})"
    prefix_all = f"ALL(req IN $tag_filters WHERE req IN {prefixes_expr})"
    # Legacy nodes have no prefix index yet. Match complete hierarchy segments,
    # with either delimiter, rather than arbitrary string prefixes across hyphens.
    legacy_tag = "replace(tag, '/', ':')"
    legacy_match = f"({legacy_tag} = req OR {legacy_tag} STARTS WITH (req + ':'))"
    fallback_any = f"ANY(req IN $tag_filters WHERE ANY(tag IN {tags_expr} WHERE {legacy_match}))"
    fallback_all = f"ALL(req IN $tag_filters WHERE ANY(tag IN {tags_expr} WHERE {legacy_match}))"

    if normalized_mode == "all":
        return (
            f"((size({prefixes_expr}) > 0 AND {prefix_all}) "
            f"OR (size({prefixes_expr}) = 0 AND {fallback_all}))"
        )

    return (
        f"((size({prefixes_expr}) > 0 AND {prefix_any}) "
        f"OR (size({prefixes_expr}) = 0 AND {fallback_any}))"
    )
