from __future__ import annotations

from typing import Any, Callable, Optional

from automem.api.admin import create_admin_blueprint_full
from automem.api.consolidation import create_consolidation_blueprint_full
from automem.api.enrichment import create_enrichment_blueprint
from automem.api.graph import create_graph_blueprint
from automem.api.health import create_health_blueprint
from automem.api.memory import create_memory_blueprint_full
from automem.api.recall import create_recall_blueprint
from automem.api.stream import create_stream_blueprint
from automem.api.viewer import create_viewer_blueprint, is_viewer_enabled


def register_blueprints(
    *,
    app: Any,
    get_memory_graph_fn: Callable[[], Any],
    get_qdrant_client_fn: Callable[[], Any],
    state: Any,
    graph_name: str,
    collection_name: str,
    utc_now_fn: Callable[[], str],
    require_admin_token_fn: Callable[[], None],
    enqueue_enrichment_fn: Callable[..., None],
    enrichment_max_attempts: int,
    normalize_tag_list_fn: Callable[[Any], list[str]],
    normalize_timestamp_fn: Callable[[Any], str],
    parse_time_expression_fn: Callable[[Any], tuple[Optional[str], Optional[str]]],
    extract_keywords_fn: Callable[[str], list[str]],
    compute_metadata_score_fn: Callable[..., float],
    result_passes_filters_fn: Callable[..., bool],
    graph_keyword_search_fn: Callable[..., list[dict[str, Any]]],
    vector_search_fn: Callable[..., list[dict[str, Any]]],
    vector_filter_only_tag_search_fn: Callable[..., list[dict[str, Any]]],
    recall_max_limit: int,
    logger: Any,
    allowed_relations: set[str],
    recall_relation_limit: int,
    serialize_node_fn: Callable[[Any], dict[str, Any]],
    summarize_relation_node_fn: Callable[[dict[str, Any]], dict[str, Any]],
    update_last_accessed_fn: Callable[[list[str]], None],
    jit_enrich_fn: Optional[Callable[..., Any]],
    normalize_tags_fn: Callable[[Any], list[str]],
    compute_tag_prefixes_fn: Callable[[list[str]], list[str]],
    coerce_importance_fn: Callable[[Any], float],
    coerce_embedding_fn: Callable[[Any], Optional[list[float]]],
    parse_metadata_field_fn: Callable[[Any], Any],
    generate_real_embedding_fn: Callable[[str], list[float]],
    enqueue_embedding_fn: Callable[[str, str], None],
    classify_memory_fn: Callable[[str], tuple[str, float]],
    point_struct_cls: Any,
    relationship_types: dict[str, dict[str, Any]],
    get_openai_client_fn: Callable[[], Any],
    init_openai_fn: Callable[[], None],
    effective_vector_size_fn: Callable[[], int],
    embedding_model: str,
    build_consolidator_from_config_fn: Callable[[Any, Any], Any],
    persist_consolidation_run_fn: Callable[[Any, dict[str, Any]], None],
    build_scheduler_from_graph_fn: Callable[[Any], Any],
    load_recent_runs_fn: Callable[[Any, int], list[dict[str, Any]]],
    consolidation_tick_seconds: int,
    consolidation_history_limit: int,
    require_api_token_fn: Callable[[], None],
) -> None:
    health_bp = create_health_blueprint(
        get_memory_graph_fn,
        get_qdrant_client_fn,
        state,
        graph_name,
        collection_name,
        utc_now_fn,
    )

    enrichment_bp = create_enrichment_blueprint(
        require_admin_token_fn,
        state,
        enqueue_enrichment_fn,
        enrichment_max_attempts,
    )

    recall_bp = create_recall_blueprint(
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
        allowed_relations,
        recall_relation_limit,
        serialize_node_fn,
        summarize_relation_node_fn,
        update_last_accessed_fn,
        jit_enrich_fn=jit_enrich_fn,
    )

    memory_bp = create_memory_blueprint_full(
        get_memory_graph_fn,
        get_qdrant_client_fn,
        normalize_tags_fn,
        normalize_tag_list_fn,
        compute_tag_prefixes_fn,
        coerce_importance_fn,
        coerce_embedding_fn,
        normalize_timestamp_fn,
        utc_now_fn,
        serialize_node_fn,
        parse_metadata_field_fn,
        generate_real_embedding_fn,
        enqueue_enrichment_fn,
        enqueue_embedding_fn,
        classify_memory_fn,
        point_struct_cls,
        collection_name,
        allowed_relations,
        relationship_types,
        state,
        logger,
        update_last_accessed_fn,
        get_openai_client_fn,
    )

    admin_bp = create_admin_blueprint_full(
        require_admin_token_fn,
        init_openai_fn,
        get_openai_client_fn,
        get_qdrant_client_fn,
        get_memory_graph_fn,
        point_struct_cls,
        collection_name,
        effective_vector_size_fn,
        embedding_model,
        utc_now_fn,
        logger,
    )

    consolidation_bp = create_consolidation_blueprint_full(
        get_memory_graph_fn,
        get_qdrant_client_fn,
        build_consolidator_from_config_fn,
        persist_consolidation_run_fn,
        build_scheduler_from_graph_fn,
        load_recent_runs_fn,
        state,
        consolidation_tick_seconds,
        consolidation_history_limit,
        logger,
    )

    graph_bp = create_graph_blueprint(
        get_memory_graph_fn,
        get_qdrant_client_fn,
        serialize_node_fn,
        collection_name,
        logger,
    )

    stream_bp = create_stream_blueprint(
        require_api_token=require_api_token_fn,
    )

    app.register_blueprint(health_bp)
    app.register_blueprint(enrichment_bp)
    app.register_blueprint(memory_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(recall_bp)
    app.register_blueprint(consolidation_bp)
    app.register_blueprint(graph_bp)
    app.register_blueprint(stream_bp)

    if is_viewer_enabled():
        viewer_bp = create_viewer_blueprint()
        app.register_blueprint(viewer_bp)
        logger.info("Graph Viewer compatibility route enabled at /viewer/")
