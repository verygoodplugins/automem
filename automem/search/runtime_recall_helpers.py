from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Callable, Dict, Iterable, List, Optional

from flask import abort, request
from qdrant_client import QdrantClient

from automem.config import RECALL_EXCLUDED_TYPES

_parse_iso_datetime: Optional[Callable[[Any], Optional[Any]]] = None
_prepare_tag_filters: Optional[Callable[[Optional[List[str]]], List[str]]] = None
_build_graph_tag_predicate: Optional[Callable[[str, str], str]] = None
_build_qdrant_tag_filter: Optional[Callable[..., Any]] = None
_serialize_node: Optional[Callable[[Any], Dict[str, Any]]] = None
_fetch_relations: Optional[Callable[[Any, str], List[Dict[str, Any]]]] = None
_extract_keywords: Optional[Callable[[str], List[str]]] = None
_coerce_embedding: Optional[Callable[[Any], Optional[List[float]]]] = None
_generate_real_embedding: Optional[Callable[[str], List[float]]] = None
_logger: Any = None
_collection_name: str = ""

METADATA_SEARCH_FIELDS = (
    "source",
    "source_agent",
    "source_agents",
    "repo",
    "project",
    "tool",
    "surface",
    "applies_to",
    "trigger",
    "provider",
    "model",
    "entities",
)
# Safety guard: these must never become searchable even if a future edit adds
# them to METADATA_SEARCH_FIELDS. None of them are in the whitelist today.
METADATA_SKIP_FIELDS = {
    "original_content",
    "enrichment",
    "semantic_neighbors",
    "patterns_detected",
}
METADATA_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "source": ("source",),
    "source_agent": ("source agent", "source agents"),
    "source_agents": ("source agents", "source agent"),
    "repo": ("repo", "repository"),
    "project": ("project",),
    "tool": ("tool",),
    "surface": ("surface",),
    "applies_to": ("applies to", "apply to"),
    "trigger": ("trigger",),
    "provider": ("provider",),
    "model": ("model",),
    "entities": ("entity", "entities"),
}
METADATA_QUERY_STOPWORDS = {
    "all",
    "any",
    "about",
    "by",
    "find",
    "for",
    "from",
    "in",
    "me",
    "memory",
    "memories",
    "of",
    "on",
    "please",
    "show",
    "that",
    "the",
    "to",
    "with",
}
METADATA_FIELD_TOKENS = {
    "source",
    "agent",
    "agents",
    "repo",
    "repository",
    "project",
    "tool",
    "surface",
    "applies",
    "apply",
    "trigger",
    "provider",
    "model",
    "entity",
    "entities",
    "metadata",
}
METADATA_PREFILTER_MAX_TERMS = 12
METADATA_SCAN_LIMIT_MIN = 200
METADATA_SCAN_LIMIT_MAX = 1000
MAX_METADATA_STRING_LENGTH = 96
MAX_METADATA_ARRAY_LENGTH = 12


def _normalize_tags(tags: List[Any]) -> List[str]:
    return [str(tag).strip().lower() for tag in tags if isinstance(tag, str) and str(tag).strip()]


def _parse_metadata_for_search(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _ascii_search_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _search_tokens(value: Any) -> set[str]:
    normalized = _ascii_search_text(value)
    return {token for token in re.findall(r"[a-z0-9]+", normalized) if len(token) >= 2}


def _ordered_search_tokens(value: Any) -> list[str]:
    normalized = _ascii_search_text(value)
    seen: set[str] = set()
    tokens: list[str] = []
    for token in re.findall(r"[a-z0-9]+", normalized):
        if len(token) < 2 or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def _iter_scalar_metadata_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and len(stripped) <= MAX_METADATA_STRING_LENGTH:
            yield stripped
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        yield str(value)
        return
    if isinstance(value, (list, tuple, set)):
        values = list(value)
        if len(values) > MAX_METADATA_ARRAY_LENGTH:
            return
        for item in values:
            yield from _iter_scalar_metadata_values(item)


def _iter_metadata_search_values(metadata: Dict[str, Any]) -> Iterable[tuple[str, str]]:
    for field in METADATA_SEARCH_FIELDS:
        if field in METADATA_SKIP_FIELDS or field not in metadata:
            continue
        raw = metadata.get(field)
        if field == "entities":
            if not isinstance(raw, dict):
                continue
            for category, values in raw.items():
                category_text = str(category).strip().lower()
                if not category_text:
                    continue
                # Entity people are content-derived and noisy personal names;
                # they are always excluded from sidecar matching.
                if category_text == "people":
                    continue
                if isinstance(values, dict):
                    continue
                for item in _iter_scalar_metadata_values(values):
                    yield f"entities.{category_text}", item
            continue
        if isinstance(raw, dict):
            continue
        for item in _iter_scalar_metadata_values(raw):
            yield field, item


def _metadata_prefilter_terms(query_text: str) -> list[str]:
    terms = _ordered_search_tokens(query_text)
    value_terms = [
        term
        for term in terms
        if term not in METADATA_FIELD_TOKENS and term not in METADATA_QUERY_STOPWORDS
    ]
    return value_terms[:METADATA_PREFILTER_MAX_TERMS]


def _requested_metadata_fields(query_text: str) -> set[str]:
    normalized = _ascii_search_text(query_text)
    if not normalized:
        return set()

    padded = f" {normalized} "
    requested: set[str] = set()
    phrase_fields: set[str] = set()
    for field, aliases in METADATA_FIELD_ALIASES.items():
        for alias in aliases:
            alias_text = _ascii_search_text(alias)
            if " " in alias_text and f" {alias_text} " in padded:
                requested.add(field)
                phrase_fields.add(field)

    tokens = set(normalized.split())
    for field, aliases in METADATA_FIELD_ALIASES.items():
        if field in phrase_fields:
            continue
        for alias in aliases:
            alias_text = _ascii_search_text(alias)
            if " " in alias_text:
                continue
            if alias_text in tokens:
                if alias_text == "source" and (
                    "source_agent" in requested or "source_agents" in requested
                ):
                    continue
                requested.add(field)

    return requested


def _metadata_field_requested(field: str, requested_fields: set[str]) -> bool:
    if not requested_fields:
        return True
    base = field.split(".", 1)[0]
    if base in {"source_agent", "source_agents"}:
        return bool({"source_agent", "source_agents"} & requested_fields)
    if base == "entities":
        return "entities" in requested_fields
    return base in requested_fields


def _metadata_value_has_strong_evidence(
    *,
    value_hits: set[str],
    value_tokens: set[str],
    query_value_tokens: set[str],
    exact_hit: bool,
    field_requested: bool,
    requested_fields: set[str],
) -> bool:
    if len(value_tokens) > 1 and len(value_hits) >= min(2, len(value_tokens)):
        return True

    if len(value_hits) != 1:
        return False

    hit = next(iter(value_hits))
    if field_requested and requested_fields and exact_hit and len(hit) >= 3:
        return True
    if len(hit) < 5:
        return False
    if field_requested and requested_fields:
        return True
    return exact_hit and len(query_value_tokens) <= 3


def _metadata_match_score(query_text: str, metadata: Dict[str, Any]) -> tuple[float, list[str]]:
    query_tokens = _search_tokens(query_text)
    if not query_tokens:
        return 0.0, []
    query_value_tokens = {
        token
        for token in query_tokens
        if token not in METADATA_FIELD_TOKENS
        and token not in METADATA_QUERY_STOPWORDS
        and len(token) >= 3
    }
    if not query_value_tokens:
        return 0.0, []

    requested_fields = _requested_metadata_fields(query_text)
    normalized_query = _ascii_search_text(query_text)
    matched_values: list[str] = []
    best_score = 0.0

    for field, value in _iter_metadata_search_values(metadata):
        value_text = _ascii_search_text(value)
        value_tokens = _search_tokens(value)
        if not value_text or not value_tokens:
            continue

        value_hits = query_value_tokens & value_tokens
        exact_hit = value_text in normalized_query
        if not value_hits:
            continue

        field_requested = _metadata_field_requested(field, requested_fields)
        # Generated entities are frequently content-derived; keep them out of the
        # general sidecar path unless the query explicitly asks for entity metadata.
        if field.startswith("entities.") and "entities" not in requested_fields:
            continue
        # Repo queries are high-cardinality and often share owner/suffix tokens
        # like "verygoodplugins" and "mcp"; require the value to cover the
        # requested repo terms unless the full normalized value is present.
        if field == "repo" and "repo" in requested_fields and not exact_hit:
            repo_query_tokens = query_value_tokens - value_tokens
            if repo_query_tokens:
                continue
        if requested_fields and not field_requested and len(value_hits) < 2:
            continue
        if not _metadata_value_has_strong_evidence(
            value_hits=value_hits,
            value_tokens=value_tokens,
            query_value_tokens=query_value_tokens,
            exact_hit=exact_hit,
            field_requested=field_requested,
            requested_fields=requested_fields,
        ):
            continue

        value_ratio = len(value_hits) / max(len(value_tokens), 1)
        query_ratio = len(value_hits) / max(len(query_value_tokens), 1)
        score = min(
            1.0,
            0.15
            + (0.45 * value_ratio)
            + (0.20 * query_ratio)
            + (0.15 if exact_hit else 0.0)
            + (0.20 if requested_fields and field_requested else 0.0),
        )
        if requested_fields and not field_requested:
            score *= 0.6
        if score > best_score:
            best_score = score
        matched_values.append(f"{field}: {value}")

    return best_score, matched_values


def configure_recall_helpers(
    *,
    parse_iso_datetime: Callable[[Any], Optional[Any]],
    prepare_tag_filters: Callable[[Optional[List[str]]], List[str]],
    build_graph_tag_predicate: Callable[[str, str], str],
    build_qdrant_tag_filter: Callable[[Optional[List[str]], str, str], Any],
    serialize_node: Callable[[Any], Dict[str, Any]],
    fetch_relations: Callable[[Any, str], List[Dict[str, Any]]],
    extract_keywords: Callable[[str], List[str]],
    coerce_embedding: Callable[[Any], Optional[List[float]]],
    generate_real_embedding: Callable[[str], List[float]],
    logger: Any,
    collection_name: str,
) -> None:
    global _parse_iso_datetime
    global _prepare_tag_filters
    global _build_graph_tag_predicate
    global _build_qdrant_tag_filter
    global _serialize_node
    global _fetch_relations
    global _extract_keywords
    global _coerce_embedding
    global _generate_real_embedding
    global _logger
    global _collection_name

    _parse_iso_datetime = parse_iso_datetime
    _prepare_tag_filters = prepare_tag_filters
    _build_graph_tag_predicate = build_graph_tag_predicate
    _build_qdrant_tag_filter = build_qdrant_tag_filter
    _serialize_node = serialize_node
    _fetch_relations = fetch_relations
    _extract_keywords = extract_keywords
    _coerce_embedding = coerce_embedding
    _generate_real_embedding = generate_real_embedding
    _logger = logger
    _collection_name = collection_name


def _result_passes_filters(
    result: Dict[str, Any],
    start_time: Optional[str],
    end_time: Optional[str],
    tag_filters: Optional[List[str]] = None,
    tag_mode: str = "any",
    tag_match: str = "prefix",
    exclude_tags: Optional[List[str]] = None,
) -> bool:
    parse_iso_datetime = _parse_iso_datetime
    prepare_tag_filters = _prepare_tag_filters
    if parse_iso_datetime is None or prepare_tag_filters is None:
        raise RuntimeError("recall helpers are not configured")

    memory = result.get("memory", {}) or {}

    # Universal exclusion of internal artifact types (e.g. MetaPattern cluster
    # summaries). This is the single chokepoint applied to vector, keyword, and
    # metadata candidates plus every expansion/state-filter path, so excluded
    # types can never surface in user-facing recall regardless of how they were
    # retrieved.
    if RECALL_EXCLUDED_TYPES:
        memory_type = memory.get("type")
        if memory_type and memory_type in RECALL_EXCLUDED_TYPES:
            return False

    timestamp = memory.get("timestamp")
    if start_time or end_time:
        parsed = parse_iso_datetime(timestamp) if timestamp else None
        parsed_start = parse_iso_datetime(start_time) if start_time else None
        parsed_end = parse_iso_datetime(end_time) if end_time else None
        if parsed is None:
            return False
        if parsed_start and parsed < parsed_start:
            return False
        if parsed_end and parsed > parsed_end:
            return False

    if tag_filters:
        normalized_filters = prepare_tag_filters(tag_filters)
        if normalized_filters:
            normalized_mode = "all" if tag_mode == "all" else "any"
            normalized_match = "prefix" if tag_match == "prefix" else "exact"

            tags = memory.get("tags") or []
            lowered_tags = _normalize_tags(tags)

            if normalized_match == "exact":
                tag_set = set(lowered_tags)
                if not tag_set:
                    return False
                if normalized_mode == "all":
                    if not all(filter_tag in tag_set for filter_tag in normalized_filters):
                        return False
                else:
                    if not any(filter_tag in tag_set for filter_tag in normalized_filters):
                        return False
            else:
                prefixes = memory.get("tag_prefixes") or []
                prefix_set = {
                    str(prefix).strip().lower()
                    for prefix in prefixes
                    if isinstance(prefix, str) and str(prefix).strip()
                }

                def _tags_start_with() -> bool:
                    if not lowered_tags:
                        return False
                    if normalized_mode == "all":
                        return all(
                            any(tag.startswith(filter_tag) for tag in lowered_tags)
                            for filter_tag in normalized_filters
                        )
                    return any(
                        tag.startswith(filter_tag)
                        for filter_tag in normalized_filters
                        for tag in lowered_tags
                    )

                if prefix_set:
                    if normalized_mode == "all":
                        if not all(filter_tag in prefix_set for filter_tag in normalized_filters):
                            return False
                    else:
                        if not any(filter_tag in prefix_set for filter_tag in normalized_filters):
                            return False
                else:
                    if not _tags_start_with():
                        return False

    if exclude_tags:
        normalized_exclude = prepare_tag_filters(exclude_tags)
        if normalized_exclude:
            tags = memory.get("tags") or []
            lowered_tags = _normalize_tags(tags)

            tag_set = set(lowered_tags)
            if any(exclude_tag in tag_set for exclude_tag in normalized_exclude):
                return False

            if any(
                tag.startswith(exclude_tag)
                for exclude_tag in normalized_exclude
                for tag in lowered_tags
            ):
                return False

    return True


def _format_graph_result(
    graph: Any,
    node: Any,
    score: Optional[float],
    match_type: str,
    seen_ids: set[str],
) -> Optional[Dict[str, Any]]:
    serialize_node = _serialize_node
    fetch_relations = _fetch_relations
    if serialize_node is None or fetch_relations is None:
        raise RuntimeError("recall helpers are not configured")

    data = serialize_node(node)
    memory_id = str(data.get("id")) if data.get("id") is not None else None
    if not memory_id or memory_id in seen_ids:
        return None

    seen_ids.add(memory_id)
    relations: List[Dict[str, Any]] = fetch_relations(graph, memory_id)

    numeric_score = float(score) if score is not None else 0.0
    return {
        "id": memory_id,
        "score": numeric_score,
        "match_score": numeric_score,
        "match_type": match_type,
        "source": "graph",
        "memory": data,
        "relations": relations,
    }


def _graph_trending_results(
    graph: Any,
    limit: int,
    seen_ids: set[str],
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    tag_filters: Optional[List[str]] = None,
    tag_mode: str = "any",
    tag_match: str = "prefix",
) -> List[Dict[str, Any]]:
    prepare_tag_filters = _prepare_tag_filters
    build_graph_tag_predicate = _build_graph_tag_predicate
    logger = _logger
    if prepare_tag_filters is None or build_graph_tag_predicate is None or logger is None:
        raise RuntimeError("recall helpers are not configured")

    try:
        sort_param = (
            ((request.args.get("sort") or request.args.get("order_by") or "score") or "")
            .strip()
            .lower()
        )
        order_by = {
            "time_asc": "m.timestamp ASC, m.importance DESC",
            "time_desc": "m.timestamp DESC, m.importance DESC",
            "updated_asc": "coalesce(m.updated_at, m.timestamp) ASC, m.importance DESC",
            "updated_desc": "coalesce(m.updated_at, m.timestamp) DESC, m.importance DESC",
        }.get(sort_param, "m.importance DESC, m.timestamp DESC")

        where_clauses = ["coalesce(m.archived, false) = false"]
        params: Dict[str, Any] = {"limit": limit}
        if RECALL_EXCLUDED_TYPES:
            where_clauses.append("NOT coalesce(m.type, '') IN $excluded_types")
            params["excluded_types"] = list(RECALL_EXCLUDED_TYPES)
        if start_time:
            where_clauses.append("m.timestamp >= $start_time")
            params["start_time"] = start_time
        if end_time:
            where_clauses.append("m.timestamp <= $end_time")
            params["end_time"] = end_time
        if tag_filters:
            normalized_filters = prepare_tag_filters(tag_filters)
            if normalized_filters:
                where_clauses.append(build_graph_tag_predicate(tag_mode, tag_match))
                params["tag_filters"] = normalized_filters

        query = f"""
            MATCH (m:Memory)
            WHERE {' AND '.join(where_clauses)}
            RETURN m
            ORDER BY {order_by}
            LIMIT $limit
        """
        result = graph.query(query, params)
    except Exception:
        logger.exception("Failed to load trending memories")
        return []

    trending: List[Dict[str, Any]] = []
    for row in getattr(result, "result_set", []) or []:
        record = _format_graph_result(graph, row[0], None, "trending", seen_ids)
        if record is None:
            continue
        importance = record["memory"].get("importance")
        record["score"] = float(importance) if isinstance(importance, (int, float)) else 0.0
        record["match_score"] = record["score"]
        trending.append(record)

    return trending


def _graph_keyword_search(
    graph: Any,
    query_text: str,
    limit: int,
    seen_ids: set[str],
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    tag_filters: Optional[List[str]] = None,
    tag_mode: str = "any",
    tag_match: str = "prefix",
) -> List[Dict[str, Any]]:
    extract_keywords = _extract_keywords
    prepare_tag_filters = _prepare_tag_filters
    build_graph_tag_predicate = _build_graph_tag_predicate
    logger = _logger
    if (
        extract_keywords is None
        or prepare_tag_filters is None
        or build_graph_tag_predicate is None
        or logger is None
    ):
        raise RuntimeError("recall helpers are not configured")

    normalized = query_text.strip().lower()
    if not normalized or normalized == "*":
        return _graph_trending_results(
            graph,
            limit,
            seen_ids,
            start_time,
            end_time,
            tag_filters,
            tag_mode,
            tag_match,
        )

    keywords = extract_keywords(normalized)
    phrase = normalized if len(normalized) >= 3 else ""

    try:
        base_where = ["m.content IS NOT NULL", "coalesce(m.archived, false) = false"]
        params: Dict[str, Any] = {"limit": limit}
        if RECALL_EXCLUDED_TYPES:
            base_where.append("NOT coalesce(m.type, '') IN $excluded_types")
            params["excluded_types"] = list(RECALL_EXCLUDED_TYPES)
        if start_time:
            base_where.append("m.timestamp >= $start_time")
            params["start_time"] = start_time
        if end_time:
            base_where.append("m.timestamp <= $end_time")
            params["end_time"] = end_time
        if tag_filters:
            normalized_filters = prepare_tag_filters(tag_filters)
            if normalized_filters:
                base_where.append(build_graph_tag_predicate(tag_mode, tag_match))
                params["tag_filters"] = normalized_filters

        where_clause = " AND ".join(base_where)

        if keywords:
            params.update({"keywords": keywords, "phrase": phrase})
            # Raw Cypher maximum: content (+2) and tag (+1) per keyword, plus
            # the whole-phrase bonus (+2 content, +1 tag). Used to normalize
            # scores into 0-1 so the keyword channel blends with the others.
            max_raw_score = 3 * len(keywords) + (3 if phrase else 0)
            query = f"""
                MATCH (m:Memory)
                WHERE {where_clause}
                WITH m,
                     toLower(m.content) AS content,
                     [tag IN coalesce(m.tags, []) | toLower(tag)] AS tags
                UNWIND $keywords AS kw
                WITH m, content, tags, kw,
                     CASE WHEN content CONTAINS kw THEN 2 ELSE 0 END +
                     CASE WHEN any(tag IN tags WHERE tag CONTAINS kw) THEN 1 ELSE 0 END AS kw_score
                WITH m, content, tags, SUM(kw_score) AS keyword_score
                WITH m, keyword_score +
                     CASE WHEN $phrase <> '' AND content CONTAINS $phrase THEN 2 ELSE 0 END +
                     CASE WHEN $phrase <> '' AND any(tag IN tags WHERE tag CONTAINS $phrase) THEN 1 ELSE 0 END AS score
                WHERE score > 0
                RETURN m, score
                ORDER BY score DESC, m.importance DESC, m.timestamp DESC
                LIMIT $limit
            """
            result = graph.query(query, params)
        elif phrase:
            params.update({"phrase": phrase})
            max_raw_score = 3
            query = f"""
                MATCH (m:Memory)
                WHERE {where_clause}
                WITH m,
                     toLower(m.content) AS content,
                     [tag IN coalesce(m.tags, []) | toLower(tag)] AS tags
                WITH m,
                     CASE WHEN content CONTAINS $phrase THEN 2 ELSE 0 END +
                     CASE WHEN any(tag IN tags WHERE tag CONTAINS $phrase) THEN 1 ELSE 0 END AS score
                WHERE score > 0
                RETURN m, score
                ORDER BY score DESC, m.importance DESC, m.timestamp DESC
                LIMIT $limit
            """
            result = graph.query(query, params)
        else:
            return _graph_trending_results(
                graph,
                limit,
                seen_ids,
                start_time,
                end_time,
                tag_filters,
                tag_mode,
                tag_match,
            )
    except Exception:
        logger.exception("Graph keyword search failed")
        return []

    matches: List[Dict[str, Any]] = []
    for row in getattr(result, "result_set", []) or []:
        node = row[0]
        score = row[1] if len(row) > 1 else None
        if score is not None and max_raw_score > 0:
            score = min(1.0, float(score) / max_raw_score)
        record = _format_graph_result(graph, node, score, "keyword", seen_ids)
        if record is None:
            continue
        matches.append(record)

    return matches


def _metadata_keyword_search(
    graph: Any,
    query_text: str,
    limit: int,
    seen_ids: set[str],
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    tag_filters: Optional[List[str]] = None,
    tag_mode: str = "any",
    tag_match: str = "prefix",
    exclude_tags: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    prepare_tag_filters = _prepare_tag_filters
    build_graph_tag_predicate = _build_graph_tag_predicate
    serialize_node = _serialize_node
    fetch_relations = _fetch_relations
    logger = _logger
    if (
        prepare_tag_filters is None
        or build_graph_tag_predicate is None
        or serialize_node is None
        or fetch_relations is None
        or logger is None
    ):
        raise RuntimeError("recall helpers are not configured")

    normalized = query_text.strip().lower()
    if not normalized or normalized == "*" or limit <= 0:
        return []

    keywords = _metadata_prefilter_terms(normalized)
    if not keywords:
        return []

    try:
        base_where = ["m.metadata IS NOT NULL", "coalesce(m.archived, false) = false"]
        scan_limit = min(max(limit * 25, METADATA_SCAN_LIMIT_MIN), METADATA_SCAN_LIMIT_MAX)
        params: Dict[str, Any] = {"limit": scan_limit, "keywords": keywords}
        if RECALL_EXCLUDED_TYPES:
            base_where.append("NOT coalesce(m.type, '') IN $excluded_types")
            params["excluded_types"] = list(RECALL_EXCLUDED_TYPES)
        if start_time:
            base_where.append("m.timestamp >= $start_time")
            params["start_time"] = start_time
        if end_time:
            base_where.append("m.timestamp <= $end_time")
            params["end_time"] = end_time
        if tag_filters:
            normalized_filters = prepare_tag_filters(tag_filters)
            if normalized_filters:
                base_where.append(build_graph_tag_predicate(tag_mode, tag_match))
                params["tag_filters"] = normalized_filters

        query = f"""
            MATCH (m:Memory)
            WHERE {' AND '.join(base_where)}
            WITH m, toLower(m.metadata) AS metadata_text
            UNWIND $keywords AS kw
            WITH m, metadata_text, kw,
                 CASE WHEN metadata_text CONTAINS kw THEN 1 ELSE 0 END AS kw_score
            WITH m, SUM(kw_score) AS score
            WHERE score > 0
            RETURN m, score
            ORDER BY score DESC, m.importance DESC, m.timestamp DESC
            LIMIT $limit
        """
        result = graph.query(query, params)
    except Exception:
        logger.exception("Graph metadata keyword search failed")
        return []

    candidates: List[Dict[str, Any]] = []
    for row in getattr(result, "result_set", []) or []:
        node = row[0]
        data = serialize_node(node)
        memory_id = str(data.get("id")) if data.get("id") is not None else None
        if not memory_id or memory_id in seen_ids:
            continue
        if data.get("archived"):
            continue
        metadata = _parse_metadata_for_search(data.get("metadata"))
        match_score, _ = _metadata_match_score(query_text, metadata)
        if match_score <= 0:
            continue

        record: Dict[str, Any] = {
            "id": memory_id,
            "score": match_score,
            "match_score": match_score,
            "match_type": "metadata",
            "source": "graph",
            "memory": data,
            "relations": [],
            "score_components": {"metadata": match_score},
        }
        if not _result_passes_filters(
            record,
            start_time,
            end_time,
            tag_filters,
            tag_mode,
            tag_match,
            exclude_tags,
        ):
            continue
        candidates.append(record)

    candidates.sort(
        key=lambda record: (
            float(record.get("match_score") or 0.0),
            float((record.get("memory") or {}).get("importance") or 0.0),
            str((record.get("memory") or {}).get("timestamp") or ""),
        ),
        reverse=True,
    )

    matches: List[Dict[str, Any]] = []
    for record in candidates:
        memory_id = str(record.get("id") or "")
        if not memory_id or memory_id in seen_ids:
            continue
        seen_ids.add(memory_id)
        # Relations are fetched only for kept records — candidates can number in
        # the hundreds while the trimmed result is at most `limit`.
        record["relations"] = fetch_relations(graph, memory_id)
        matches.append(record)
        if len(matches) >= limit:
            break

    return matches


def _vector_filter_only_tag_search(
    qdrant_client: Optional[QdrantClient],
    tag_filters: Optional[List[str]],
    tag_mode: str,
    tag_match: str,
    limit: int,
    seen_ids: set[str],
) -> List[Dict[str, Any]]:
    build_qdrant_tag_filter = _build_qdrant_tag_filter
    logger = _logger
    collection_name = _collection_name
    if build_qdrant_tag_filter is None or logger is None or not collection_name:
        raise RuntimeError("recall helpers are not configured")

    if qdrant_client is None or not tag_filters or limit <= 0:
        return []

    query_filter = build_qdrant_tag_filter(
        tag_filters,
        tag_mode,
        tag_match,
        excluded_types=RECALL_EXCLUDED_TYPES,
    )
    if query_filter is None:
        return []

    try:
        points, _ = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
    except Exception:
        logger.exception("Qdrant tag-only scroll failed")
        return []

    results: List[Dict[str, Any]] = []
    for point in points or []:
        memory_id = str(point.id)
        if memory_id in seen_ids:
            continue

        payload = point.payload or {}
        if payload.get("archived"):
            continue

        seen_ids.add(memory_id)
        importance = payload.get("importance")
        try:
            score = float(importance)
        except (TypeError, ValueError):
            score = 0.0

        results.append(
            {
                "id": memory_id,
                "score": score,
                "match_score": score,
                "match_type": "tag",
                "source": "qdrant",
                "memory": payload,
                "relations": [],
            }
        )

    return results


def _vector_search(
    qdrant_client: Optional[QdrantClient],
    graph: Any,
    query_text: str,
    embedding_param: Optional[str],
    limit: int,
    seen_ids: set[str],
    tag_filters: Optional[List[str]] = None,
    tag_mode: str = "any",
    tag_match: str = "prefix",
) -> List[Dict[str, Any]]:
    build_qdrant_tag_filter = _build_qdrant_tag_filter
    coerce_embedding = _coerce_embedding
    generate_real_embedding = _generate_real_embedding
    logger = _logger
    collection_name = _collection_name
    if (
        build_qdrant_tag_filter is None
        or coerce_embedding is None
        or generate_real_embedding is None
        or logger is None
        or not collection_name
    ):
        raise RuntimeError("recall helpers are not configured")

    if qdrant_client is None:
        return []

    normalized = (query_text or "").strip()
    if not embedding_param and normalized in {"", "*"}:
        return []

    embedding: Optional[List[float]] = None

    if embedding_param:
        try:
            embedding = coerce_embedding(embedding_param)
        except ValueError as exc:
            abort(400, description=str(exc))
    elif normalized:
        logger.debug("Generating embedding for query: %s", normalized)
        embedding = generate_real_embedding(normalized)

    if not embedding:
        return []

    query_filter = build_qdrant_tag_filter(
        tag_filters,
        tag_mode,
        tag_match,
        excluded_types=RECALL_EXCLUDED_TYPES,
    )

    try:
        vector_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=embedding,
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
        )
    except Exception:
        logger.exception("Qdrant search failed")
        return []

    matches: List[Dict[str, Any]] = []
    for hit in vector_results:
        memory_id = str(hit.id)
        if memory_id in seen_ids:
            continue

        payload = hit.payload or {}
        if payload.get("archived"):
            continue

        seen_ids.add(memory_id)
        score = float(hit.score) if hit.score is not None else 0.0

        matches.append(
            {
                "id": memory_id,
                "score": score,
                "match_score": score,
                "match_type": "vector",
                "source": "qdrant",
                "memory": payload,
                "relations": [],
            }
        )

    return matches


def _hydrate_vector_relations(
    graph: Any, results: List[Dict[str, Any]], logger: Any = None
) -> None:
    fetch_relations = _fetch_relations
    if graph is None or fetch_relations is None:
        return

    for result in results:
        if result.get("source") != "qdrant" or result.get("match_type") != "vector":
            continue
        if result.get("relations"):
            continue

        memory = result.get("memory") or {}
        memory_id = str(
            result.get("id") or memory.get("id") or memory.get("memory_id") or ""
        ).strip()
        if not memory_id:
            continue

        try:
            result["relations"] = fetch_relations(graph, memory_id)
        except Exception:
            if logger is not None:
                logger.exception("Failed to hydrate vector relations for %s", memory_id)
            result.setdefault("relations", [])
