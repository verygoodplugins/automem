from __future__ import annotations

from typing import Any


def recall_memories(
    *,
    request_obj: Any,
    perf_counter_fn: Any,
    parse_time_expression_fn: Any,
    normalize_timestamp_fn: Any,
    normalize_tag_list_fn: Any,
    handle_recall_fn: Any,
    get_memory_graph_fn: Any,
    get_qdrant_client_fn: Any,
    extract_keywords_fn: Any,
    compute_metadata_score_fn: Any,
    result_passes_filters_fn: Any,
    graph_keyword_search_fn: Any,
    vector_search_fn: Any,
    vector_filter_only_tag_search_fn: Any,
    recall_max_limit: int,
    logger: Any,
    allowed_relations: Any,
    recall_relation_limit: int,
    recall_expansion_limit: int,
    emit_event_fn: Any,
    utc_now_fn: Any,
    abort_fn: Any,
) -> Any:
    query_start = perf_counter_fn()
    query_text = (request_obj.args.get("query") or "").strip()
    try:
        requested_limit = int(request_obj.args.get("limit", 5))
    except (TypeError, ValueError):
        requested_limit = 5
    limit = max(1, min(requested_limit, recall_max_limit))
    time_query = request_obj.args.get("time_query") or request_obj.args.get("time")
    start_param = request_obj.args.get("start")
    end_param = request_obj.args.get("end")
    tags_param = request_obj.args.getlist("tags") or request_obj.args.get("tags")

    tag_mode = (request_obj.args.get("tag_mode") or "any").strip().lower()
    if tag_mode not in {"any", "all"}:
        tag_mode = "any"

    tag_match = (request_obj.args.get("tag_match") or "prefix").strip().lower()
    if tag_match not in {"exact", "prefix"}:
        tag_match = "prefix"

    time_start, time_end = parse_time_expression_fn(time_query)

    if start_param:
        try:
            time_start = normalize_timestamp_fn(start_param)
        except ValueError as exc:
            abort_fn(400, description=f"Invalid start time: {exc}")

    if end_param:
        try:
            time_end = normalize_timestamp_fn(end_param)
        except ValueError as exc:
            abort_fn(400, description=f"Invalid end time: {exc}")

    tag_filters = normalize_tag_list_fn(tags_param)

    response = handle_recall_fn(
        get_memory_graph_fn,
        get_qdrant_client_fn,
        normalize_tag_list_fn,
        normalize_timestamp_fn,
        parse_time_expression_fn,
        extract_keywords_fn,
        compute_metadata_score_fn,
        result_passes_filters_fn,
        graph_keyword_search_fn,
        vector_search_fn,
        vector_filter_only_tag_search_fn,
        recall_max_limit,
        logger,
        allowed_relations=allowed_relations,
        relation_limit=recall_relation_limit,
        expansion_limit_default=recall_expansion_limit,
    )

    elapsed_ms = int((perf_counter_fn() - query_start) * 1000)
    result_count = 0
    try:
        resp_data = response[0] if isinstance(response, tuple) else response
        if hasattr(resp_data, "get_json"):
            data = resp_data.get_json(silent=True) or {}
            result_count = len(data.get("memories", []))
    except Exception as e:
        logger.debug("Failed to parse response for result_count", exc_info=e)

    emit_event_fn(
        "memory.recall",
        {
            "query": query_text[:50] if query_text else "(no query)",
            "limit": limit,
            "result_count": result_count,
            "elapsed_ms": elapsed_ms,
            "tags": tag_filters[:3] if tag_filters else [],
        },
        utc_now_fn,
    )

    return response
