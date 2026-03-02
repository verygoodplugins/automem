from __future__ import annotations

import os
import sys
from typing import Any, Callable


def wire_recall_and_blueprints(
    *,
    module: Any,
    configure_recall_helpers_fn: Callable[..., None],
    register_blueprints_fn: Callable[..., None],
) -> None:
    configure_recall_helpers_fn(
        parse_iso_datetime=module._parse_iso_datetime,
        prepare_tag_filters=module._prepare_tag_filters,
        build_graph_tag_predicate=module._build_graph_tag_predicate,
        build_qdrant_tag_filter=module._build_qdrant_tag_filter,
        serialize_node=module._serialize_node,
        fetch_relations=module._fetch_relations,
        extract_keywords=module._extract_keywords,
        coerce_embedding=module._coerce_embedding,
        generate_real_embedding=module._generate_real_embedding,
        logger=module.logger,
        collection_name=module.COLLECTION_NAME,
    )

    register_blueprints_fn(
        app=module.app,
        get_memory_graph_fn=module.get_memory_graph,
        get_qdrant_client_fn=module.get_qdrant_client,
        state=module.state,
        graph_name=module.GRAPH_NAME,
        collection_name=module.COLLECTION_NAME,
        utc_now_fn=module.utc_now,
        require_admin_token_fn=module._require_admin_token,
        enqueue_enrichment_fn=module.enqueue_enrichment,
        enrichment_max_attempts=module.ENRICHMENT_MAX_ATTEMPTS,
        normalize_tag_list_fn=module._normalize_tag_list,
        normalize_timestamp_fn=module._normalize_timestamp,
        parse_time_expression_fn=module._parse_time_expression,
        extract_keywords_fn=module._extract_keywords,
        compute_metadata_score_fn=module._compute_metadata_score,
        result_passes_filters_fn=module._result_passes_filters,
        graph_keyword_search_fn=module._graph_keyword_search,
        vector_search_fn=module._vector_search,
        vector_filter_only_tag_search_fn=module._vector_filter_only_tag_search,
        recall_max_limit=module.RECALL_MAX_LIMIT,
        logger=module.logger,
        allowed_relations=module.ALLOWED_RELATIONS,
        recall_relation_limit=module.RECALL_RELATION_LIMIT,
        serialize_node_fn=module._serialize_node,
        summarize_relation_node_fn=module._summarize_relation_node,
        update_last_accessed_fn=module.update_last_accessed,
        jit_enrich_fn=module.jit_enrich_lightweight if module.JIT_ENRICHMENT_ENABLED else None,
        normalize_tags_fn=module._normalize_tags,
        compute_tag_prefixes_fn=module._compute_tag_prefixes,
        coerce_importance_fn=module._coerce_importance,
        coerce_embedding_fn=module._coerce_embedding,
        parse_metadata_field_fn=module._parse_metadata_field,
        generate_real_embedding_fn=module._generate_real_embedding,
        generate_real_embeddings_batch_fn=getattr(module, "_generate_real_embeddings_batch", None),
        enqueue_embedding_fn=module.enqueue_embedding,
        classify_memory_fn=lambda content: module.memory_classifier.classify(content),
        point_struct_cls=module.PointStruct,
        relationship_types=module.RELATIONSHIP_TYPES,
        get_openai_client_fn=module.get_openai_client,
        init_openai_fn=module.init_openai,
        effective_vector_size_fn=lambda: module.state.effective_vector_size,
        embedding_model=module.EMBEDDING_MODEL,
        build_consolidator_from_config_fn=module._build_consolidator_from_config,
        persist_consolidation_run_fn=module._persist_consolidation_run,
        build_scheduler_from_graph_fn=module._build_scheduler_from_graph,
        load_recent_runs_fn=module._load_recent_runs,
        consolidation_tick_seconds=module.CONSOLIDATION_TICK_SECONDS,
        consolidation_history_limit=module.CONSOLIDATION_HISTORY_LIMIT,
        require_api_token_fn=module.require_api_token,
    )


def run_default_server(*, module: Any) -> None:
    port = int(os.environ.get("PORT", "8001"))
    module.logger.info("Starting Flask API on port %s", port)
    init_steps = [
        ("init_falkordb", module.init_falkordb),
        ("init_qdrant", module.init_qdrant),
        ("init_openai", module.init_openai),
        ("init_embedding_provider", module.init_embedding_provider),
        ("init_enrichment_pipeline", module.init_enrichment_pipeline),
        ("init_embedding_pipeline", module.init_embedding_pipeline),
        ("init_consolidation_scheduler", module.init_consolidation_scheduler),
        ("init_sync_worker", module.init_sync_worker),
    ]
    failed_step = "unknown"
    try:
        for failed_step, init_fn in init_steps:
            init_fn()
    except Exception:
        module.logger.exception("Server initialization failed at step %s", failed_step)
        for cleanup_name in ("stop_sync_worker", "stop_consolidation_scheduler"):
            cleanup_fn = getattr(module, cleanup_name, None)
            if callable(cleanup_fn):
                try:
                    cleanup_fn()
                except Exception:
                    module.logger.exception("Cleanup step %s failed", cleanup_name)
        sys.exit(1)
    # Use :: for IPv6 dual-stack (Railway internal networking uses IPv6)
    module.app.run(host="::", port=port, debug=False)
