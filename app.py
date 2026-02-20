"""AutoMem Memory Service API.

Provides a small Flask API that stores memories in FalkorDB and Qdrant.
This module focuses on being resilient: it validates requests, handles
transient outages, and degrades gracefully when one of the backing services
is unavailable.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Any, Dict, List, Optional

from falkordb import FalkorDB
from flask import Flask, abort, jsonify, request
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models

try:
    from qdrant_client.http.exceptions import UnexpectedResponse
except ImportError:  # Allow tests to import without full qdrant client installed
    UnexpectedResponse = Exception  # type: ignore[misc,assignment]

try:  # Allow tests to import without full qdrant client installed
    from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams
except Exception:  # pragma: no cover - degraded import path
    try:
        from qdrant_client.http import models as _qmodels

        Distance = getattr(_qmodels, "Distance", None)
        PointStruct = getattr(_qmodels, "PointStruct", None)
        VectorParams = getattr(_qmodels, "VectorParams", None)
        PayloadSchemaType = getattr(_qmodels, "PayloadSchemaType", None)
    except Exception:
        Distance = PointStruct = VectorParams = None
        PayloadSchemaType = None

# Provide a simple PointStruct shim for tests/environments lacking qdrant models
if PointStruct is None:  # pragma: no cover - test shim

    class PointStruct:  # type: ignore[no-redef]
        def __init__(self, id: str, vector: List[float], payload: Dict[str, Any]):
            self.id = id
            self.vector = vector
            self.payload = payload


from werkzeug.exceptions import HTTPException

# Make OpenAI import optional to allow running without it
try:
    from openai import OpenAI  # type: ignore
except ImportError:
    OpenAI = None  # type: ignore

# SSE streaming for real-time observability
from automem.api.auth_helpers import extract_api_token as _extract_api_token_helper
from automem.api.auth_helpers import require_admin_token as _require_admin_token_helper
from automem.api.auth_helpers import require_api_token as _require_api_token_helper
from automem.api.runtime_bootstrap import register_blueprints as _register_blueprints_runtime
from automem.api.stream import emit_event
from automem.app_helper_bindings import create_app_helper_runtime
from automem.classification.memory_classifier import MemoryClassifier
from automem.consolidation.runtime_bindings import create_consolidation_runtime
from automem.embedding.runtime_bindings import create_embedding_runtime
from automem.embedding.runtime_helpers import coerce_embedding as _coerce_embedding_value
from automem.embedding.runtime_helpers import coerce_importance as _coerce_importance_value
from automem.embedding.runtime_helpers import (
    generate_placeholder_embedding as _generate_placeholder_embedding_value,
)
from automem.embedding.runtime_helpers import (
    generate_real_embedding as _generate_real_embedding_value,
)
from automem.embedding.runtime_helpers import (
    generate_real_embeddings_batch as _generate_real_embeddings_batch_value,
)
from automem.embedding.runtime_helpers import normalize_tags as _normalize_tags_value
from automem.enrichment.runtime_bindings import create_enrichment_runtime
from automem.enrichment.runtime_queue_bindings import create_enrichment_queue_runtime
from automem.runtime_environment import configure_logging, ensure_local_package_importable
from automem.runtime_wiring import run_default_server, wire_recall_and_blueprints
from automem.service_runtime_bindings import create_service_runtime
from automem.service_state import EnrichmentJob, EnrichmentStats, ServiceState

# Environment is loaded by automem.config

logger = configure_logging(level=logging.INFO)
ensure_local_package_importable(file_path=__file__)

app = Flask(__name__)

# Import canonical configuration constants
from automem.config import (
    ADMIN_TOKEN,
    ALLOWED_RELATIONS,
    API_TOKEN,
    CLASSIFICATION_MODEL,
    COLLECTION_NAME,
    CONSOLIDATION_ARCHIVE_THRESHOLD,
    CONSOLIDATION_CLUSTER_INTERVAL_SECONDS,
    CONSOLIDATION_CONTROL_LABEL,
    CONSOLIDATION_CONTROL_NODE_ID,
    CONSOLIDATION_CREATIVE_INTERVAL_SECONDS,
    CONSOLIDATION_DECAY_IMPORTANCE_THRESHOLD,
    CONSOLIDATION_DECAY_INTERVAL_SECONDS,
    CONSOLIDATION_DELETE_THRESHOLD,
    CONSOLIDATION_FORGET_INTERVAL_SECONDS,
    CONSOLIDATION_GRACE_PERIOD_DAYS,
    CONSOLIDATION_HISTORY_LIMIT,
    CONSOLIDATION_IMPORTANCE_PROTECTION_THRESHOLD,
    CONSOLIDATION_PROTECTED_TYPES,
    CONSOLIDATION_RUN_LABEL,
    CONSOLIDATION_TASK_FIELDS,
    CONSOLIDATION_TICK_SECONDS,
    EMBEDDING_MODEL,
    ENRICHMENT_ENABLE_SUMMARIES,
    ENRICHMENT_FAILURE_BACKOFF_SECONDS,
    ENRICHMENT_IDLE_SLEEP_SECONDS,
    ENRICHMENT_MAX_ATTEMPTS,
    ENRICHMENT_SIMILARITY_LIMIT,
    ENRICHMENT_SIMILARITY_THRESHOLD,
    ENRICHMENT_SPACY_MODEL,
    FALKORDB_PORT,
    GRAPH_NAME,
    JIT_ENRICHMENT_ENABLED,
    MEMORY_TYPES,
    RECALL_EXPANSION_LIMIT,
    RECALL_RELATION_LIMIT,
    RELATIONSHIP_TYPES,
    SEARCH_WEIGHT_CONFIDENCE,
    SEARCH_WEIGHT_EXACT,
    SEARCH_WEIGHT_IMPORTANCE,
    SEARCH_WEIGHT_KEYWORD,
    SEARCH_WEIGHT_RECENCY,
    SEARCH_WEIGHT_TAG,
    SEARCH_WEIGHT_VECTOR,
    SYNC_AUTO_REPAIR,
    SYNC_CHECK_INTERVAL_SECONDS,
    TYPE_ALIASES,
    VECTOR_SIZE,
    normalize_memory_type,
)
from automem.search.runtime_keywords import load_keyword_runtime
from automem.search.runtime_recall_helpers import (
    _graph_keyword_search,
    _result_passes_filters,
    _vector_filter_only_tag_search,
    _vector_search,
    configure_recall_helpers,
)
from automem.search.runtime_relations import fetch_relations as _fetch_relations_runtime
from automem.stores.graph_store import _build_graph_tag_predicate
from automem.stores.vector_store import _build_qdrant_tag_filter
from automem.sync.runtime_bindings import create_sync_runtime
from automem.utils.entity_extraction import (
    _slugify,
    configure_entity_extraction,
    extract_entities,
    generate_summary,
)
from automem.utils.graph import _serialize_node, _summarize_relation_node
from automem.utils.scoring import _compute_metadata_score, _parse_metadata_field
from automem.utils.tags import (
    _compute_tag_prefixes,
    _expand_tag_prefixes,
    _normalize_tag_list,
    _prepare_tag_filters,
)

# Shared utils and helpers
from automem.utils.time import (
    _normalize_timestamp,
    _parse_iso_datetime,
    _parse_time_expression,
    utc_now,
)
from automem.utils.validation import get_effective_vector_size, validate_vector_dimensions

# Embedding batching configuration
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "20"))
EMBEDDING_BATCH_TIMEOUT_SECONDS = float(os.getenv("EMBEDDING_BATCH_TIMEOUT_SECONDS", "2.0"))

"""Note: default types/relations/weights are imported from automem.config"""

SEARCH_STOPWORDS, ENTITY_STOPWORDS, ENTITY_BLOCKLIST, _extract_keywords = load_keyword_runtime()

# Search weights are imported from automem.config

# Maximum number of results returned by /recall
RECALL_MAX_LIMIT = int(os.getenv("RECALL_MAX_LIMIT", "100"))

# API tokens are imported from automem.config


configure_entity_extraction(
    search_stopwords=SEARCH_STOPWORDS,
    entity_stopwords=ENTITY_STOPWORDS,
    entity_blocklist=ENTITY_BLOCKLIST,
    spacy_model=ENRICHMENT_SPACY_MODEL,
)


state = ServiceState()


def _extract_api_token() -> Optional[str]:
    return _extract_api_token_helper(request, API_TOKEN)


def get_openai_client() -> Optional[OpenAI]:
    return state.openai_client


def _require_admin_token() -> None:
    _require_admin_token_helper(
        request_obj=request,
        admin_token=ADMIN_TOKEN,
        abort_fn=abort,
    )


@app.before_request
def require_api_token() -> None:
    _require_api_token_helper(
        request_obj=request,
        api_token=API_TOKEN,
        extract_api_token_fn=_extract_api_token,
        abort_fn=abort,
    )


_service_runtime = create_service_runtime(
    get_state_fn=lambda: state,
    logger=logger,
    openai_cls=OpenAI,
    get_env_fn=os.getenv,
    vector_size_config_fn=lambda: VECTOR_SIZE,
    embedding_model_fn=lambda: EMBEDDING_MODEL,
    falkordb_cls=FalkorDB,
    graph_name=GRAPH_NAME,
    falkordb_port=FALKORDB_PORT,
    qdrant_client_cls=QdrantClient,
    collection_name=COLLECTION_NAME,
    get_effective_vector_size_fn=get_effective_vector_size,
    vector_params_cls=VectorParams,
    distance_enum=Distance,
    payload_schema_type_enum=PayloadSchemaType,
    get_init_falkordb_fn=lambda: init_falkordb,
    get_init_qdrant_fn=lambda: init_qdrant,
)

init_openai = _service_runtime.init_openai
init_embedding_provider = _service_runtime.init_embedding_provider
init_falkordb = _service_runtime.init_falkordb
init_qdrant = _service_runtime.init_qdrant
_ensure_qdrant_collection = _service_runtime.ensure_qdrant_collection
get_memory_graph = _service_runtime.get_memory_graph
get_qdrant_client = _service_runtime.get_qdrant_client


memory_classifier = MemoryClassifier(
    normalize_memory_type=normalize_memory_type,
    ensure_openai_client=init_openai,
    get_openai_client=get_openai_client,
    classification_model=CLASSIFICATION_MODEL,
    logger=logger,
)


_enrichment_queue_runtime = create_enrichment_queue_runtime(
    get_state_fn=lambda: state,
    logger=logger,
    queue_cls=Queue,
    thread_cls=Thread,
    enrichment_job_cls=EnrichmentJob,
    get_memory_graph_fn=get_memory_graph,
    utc_now_fn=utc_now,
    enrichment_idle_sleep_seconds=ENRICHMENT_IDLE_SLEEP_SECONDS,
    enrichment_max_attempts=ENRICHMENT_MAX_ATTEMPTS,
    enrichment_failure_backoff_seconds=ENRICHMENT_FAILURE_BACKOFF_SECONDS,
    empty_exc=Empty,
    enrich_memory_fn=lambda memory_id, forced=False: enrich_memory(memory_id, forced=forced),
    emit_event_fn=emit_event,
    perf_counter_fn=time.perf_counter,
    sleep_fn=time.sleep,
)

init_enrichment_pipeline = _enrichment_queue_runtime.init_enrichment_pipeline
enqueue_enrichment = _enrichment_queue_runtime.enqueue_enrichment
update_last_accessed = _enrichment_queue_runtime.update_last_accessed
enrichment_worker = _enrichment_queue_runtime.enrichment_worker


_consolidation_runtime = create_consolidation_runtime(
    state=state,
    logger=logger,
    get_memory_graph_fn=get_memory_graph,
    get_qdrant_client_fn=get_qdrant_client,
    emit_event_fn=emit_event,
    utc_now_fn=utc_now,
    perf_counter_fn=time.perf_counter,
    parse_iso_datetime_fn=_parse_iso_datetime,
    stop_event_cls=Event,
    thread_cls=Thread,
    run_label=CONSOLIDATION_RUN_LABEL,
    control_label=CONSOLIDATION_CONTROL_LABEL,
    control_node_id=CONSOLIDATION_CONTROL_NODE_ID,
    task_fields=CONSOLIDATION_TASK_FIELDS,
    history_limit=CONSOLIDATION_HISTORY_LIMIT,
    tick_seconds=CONSOLIDATION_TICK_SECONDS,
    decay_interval_seconds=CONSOLIDATION_DECAY_INTERVAL_SECONDS,
    creative_interval_seconds=CONSOLIDATION_CREATIVE_INTERVAL_SECONDS,
    cluster_interval_seconds=CONSOLIDATION_CLUSTER_INTERVAL_SECONDS,
    forget_interval_seconds=CONSOLIDATION_FORGET_INTERVAL_SECONDS,
    delete_threshold=CONSOLIDATION_DELETE_THRESHOLD,
    archive_threshold=CONSOLIDATION_ARCHIVE_THRESHOLD,
    grace_period_days=CONSOLIDATION_GRACE_PERIOD_DAYS,
    importance_protection_threshold=CONSOLIDATION_IMPORTANCE_PROTECTION_THRESHOLD,
    protected_types=set(CONSOLIDATION_PROTECTED_TYPES),
    decay_importance_threshold=CONSOLIDATION_DECAY_IMPORTANCE_THRESHOLD,
)

_load_recent_runs = _consolidation_runtime.load_recent_runs
_persist_consolidation_run = _consolidation_runtime.persist_consolidation_run
_build_consolidator_from_config = _consolidation_runtime.build_consolidator_from_config
_build_scheduler_from_graph = _consolidation_runtime.build_scheduler_from_graph
_run_consolidation_tick = _consolidation_runtime.run_consolidation_tick
consolidation_worker = _consolidation_runtime.consolidation_worker
init_consolidation_scheduler = _consolidation_runtime.init_consolidation_scheduler


_embedding_runtime = create_embedding_runtime(
    get_state_fn=lambda: state,
    logger=logger,
    queue_cls=Queue,
    thread_cls=Thread,
    batch_size=EMBEDDING_BATCH_SIZE,
    batch_timeout_seconds=EMBEDDING_BATCH_TIMEOUT_SECONDS,
    empty_exc=Empty,
    sleep_fn=time.sleep,
    time_fn=time.time,
    get_qdrant_client_fn=get_qdrant_client,
    get_memory_graph_fn=get_memory_graph,
    collection_name=COLLECTION_NAME,
    point_struct_cls=PointStruct,
    utc_now_fn=utc_now,
    generate_real_embedding_fn=lambda content: _generate_real_embedding(content),
    generate_real_embeddings_batch_fn=lambda contents: _generate_real_embeddings_batch(contents),
)

init_embedding_pipeline = _embedding_runtime.init_embedding_pipeline
enqueue_embedding = _embedding_runtime.enqueue_embedding
embedding_worker = _embedding_runtime.embedding_worker
_process_embedding_batch = _embedding_runtime.process_embedding_batch
_store_embedding_in_qdrant = _embedding_runtime.store_embedding_in_qdrant
generate_and_store_embedding = _embedding_runtime.generate_and_store_embedding


# ---------------------------------------------------------------------------
# Background Sync Worker
# ---------------------------------------------------------------------------


_sync_runtime = create_sync_runtime(
    get_state_fn=lambda: state,
    logger=logger,
    sync_auto_repair=SYNC_AUTO_REPAIR,
    sync_check_interval_seconds=SYNC_CHECK_INTERVAL_SECONDS,
    stop_event_cls=Event,
    thread_cls=Thread,
    sleep_fn=time.sleep,
    get_memory_graph_fn=get_memory_graph,
    get_qdrant_client_fn=get_qdrant_client,
    collection_name=COLLECTION_NAME,
    utc_now_fn=utc_now,
    enqueue_embedding_fn=enqueue_embedding,
)

init_sync_worker = _sync_runtime.init_sync_worker
sync_worker = _sync_runtime.sync_worker
_run_sync_check = _sync_runtime.run_sync_check


_enrichment_runtime = create_enrichment_runtime(
    get_memory_graph_fn=get_memory_graph,
    get_qdrant_client_fn=get_qdrant_client,
    parse_metadata_field_fn=_parse_metadata_field,
    normalize_tag_list_fn=_normalize_tag_list,
    extract_entities_fn=extract_entities,
    slugify_fn=_slugify,
    compute_tag_prefixes_fn=_compute_tag_prefixes,
    classify_memory_fn=memory_classifier.classify,
    search_stopwords=SEARCH_STOPWORDS,
    enrichment_enable_summaries=ENRICHMENT_ENABLE_SUMMARIES,
    generate_summary_fn=generate_summary,
    utc_now_fn=utc_now,
    collection_name=COLLECTION_NAME,
    enrichment_similarity_limit=ENRICHMENT_SIMILARITY_LIMIT,
    enrichment_similarity_threshold=ENRICHMENT_SIMILARITY_THRESHOLD,
    unexpected_response_exc=UnexpectedResponse,
    logger=logger,
)

jit_enrich_lightweight = _enrichment_runtime.jit_enrich_lightweight
enrich_memory = _enrichment_runtime.enrich_memory
_temporal_cutoff = _enrichment_runtime.temporal_cutoff
find_temporal_relationships = _enrichment_runtime.find_temporal_relationships
detect_patterns = _enrichment_runtime.detect_patterns
link_semantic_neighbors = _enrichment_runtime.link_semantic_neighbors


@app.errorhandler(Exception)
def handle_exceptions(exc: Exception):
    """Return JSON responses for both HTTP and unexpected errors."""
    if isinstance(exc, HTTPException):
        response = {
            "status": "error",
            "code": exc.code,
            "message": exc.description or exc.name,
        }
        return jsonify(response), exc.code

    logger.exception("Unhandled error")
    response = {
        "status": "error",
        "code": 500,
        "message": "Internal server error",
    }
    return jsonify(response), 500


_app_helpers = create_app_helper_runtime(
    get_state_fn=lambda: state,
    abort_fn=abort,
    init_embedding_provider_fn=init_embedding_provider,
    get_generate_placeholder_embedding_fn=lambda: _generate_placeholder_embedding,
    normalize_tags_value_fn=_normalize_tags_value,
    coerce_importance_value_fn=_coerce_importance_value,
    coerce_embedding_value_fn=_coerce_embedding_value,
    generate_placeholder_embedding_value_fn=_generate_placeholder_embedding_value,
    generate_real_embedding_value_fn=_generate_real_embedding_value,
    generate_real_embeddings_batch_value_fn=_generate_real_embeddings_batch_value,
    fetch_relations_runtime_fn=_fetch_relations_runtime,
    relation_limit=RECALL_RELATION_LIMIT,
    serialize_node_fn=_serialize_node,
    summarize_relation_node_fn=_summarize_relation_node,
    logger=logger,
)

_normalize_tags = _app_helpers.normalize_tags
_coerce_importance = _app_helpers.coerce_importance
_coerce_embedding = _app_helpers.coerce_embedding
_generate_placeholder_embedding = _app_helpers.generate_placeholder_embedding
_generate_real_embedding = _app_helpers.generate_real_embedding
_generate_real_embeddings_batch = _app_helpers.generate_real_embeddings_batch
_fetch_relations = _app_helpers.fetch_relations


wire_recall_and_blueprints(
    module=sys.modules[__name__],
    configure_recall_helpers_fn=configure_recall_helpers,
    register_blueprints_fn=_register_blueprints_runtime,
)


if __name__ == "__main__":
    run_default_server(module=sys.modules[__name__])
