from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from flask import abort, request
from qdrant_client import QdrantClient

_parse_iso_datetime: Optional[Callable[[Any], Optional[Any]]] = None
_prepare_tag_filters: Optional[Callable[[Optional[List[str]]], List[str]]] = None
_build_graph_tag_predicate: Optional[Callable[[str, str], str]] = None
_build_qdrant_tag_filter: Optional[Callable[[Optional[List[str]], str, str], Any]] = None
_serialize_node: Optional[Callable[[Any], Dict[str, Any]]] = None
_fetch_relations: Optional[Callable[[Any, str], List[Dict[str, Any]]]] = None
_extract_keywords: Optional[Callable[[str], List[str]]] = None
_coerce_embedding: Optional[Callable[[Any], Optional[List[float]]]] = None
_generate_real_embedding: Optional[Callable[[str], List[float]]] = None
_logger: Any = None
_collection_name: str = ""


def _normalize_tags(tags: List[Any]) -> List[str]:
    return [str(tag).strip().lower() for tag in tags if isinstance(tag, str) and str(tag).strip()]


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
        base_where = ["m.content IS NOT NULL"]
        params: Dict[str, Any] = {"limit": limit}
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
        record = _format_graph_result(graph, node, score, "keyword", seen_ids)
        if record is None:
            continue
        matches.append(record)

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

    query_filter = build_qdrant_tag_filter(tag_filters, tag_mode, tag_match)
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
        seen_ids.add(memory_id)

        payload = point.payload or {}
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
    fetch_relations = _fetch_relations
    logger = _logger
    collection_name = _collection_name
    if (
        build_qdrant_tag_filter is None
        or coerce_embedding is None
        or generate_real_embedding is None
        or fetch_relations is None
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

    query_filter = build_qdrant_tag_filter(tag_filters, tag_mode, tag_match)

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

        seen_ids.add(memory_id)
        payload = hit.payload or {}
        relations = fetch_relations(graph, memory_id) if graph is not None else []
        score = float(hit.score) if hit.score is not None else 0.0

        matches.append(
            {
                "id": memory_id,
                "score": score,
                "match_score": score,
                "match_type": "vector",
                "source": "qdrant",
                "memory": payload,
                "relations": relations,
            }
        )

    return matches
