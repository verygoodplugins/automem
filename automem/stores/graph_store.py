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
    fallback_any = (
        f"ANY(req IN $tag_filters WHERE ANY(tag IN {tags_expr} WHERE tag STARTS WITH req))"
    )
    fallback_all = (
        f"ALL(req IN $tag_filters WHERE ANY(tag IN {tags_expr} WHERE tag STARTS WITH req))"
    )

    if normalized_mode == "all":
        return (
            f"((size({prefixes_expr}) > 0 AND {prefix_all}) "
            f"OR (size({prefixes_expr}) = 0 AND {fallback_all}))"
        )

    return (
        f"((size({prefixes_expr}) > 0 AND {prefix_any}) "
        f"OR (size({prefixes_expr}) = 0 AND {fallback_any}))"
    )
