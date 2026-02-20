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
import re
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

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
from automem.classification.memory_classifier import MemoryClassifier
from automem.consolidation.runtime_bindings import create_consolidation_runtime
from automem.embedding.provider_init import init_embedding_provider as _init_embedding_provider
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
from automem.embedding.runtime_pipeline import embedding_worker as _embedding_worker_runtime
from automem.embedding.runtime_pipeline import enqueue_embedding as _enqueue_embedding_runtime
from automem.embedding.runtime_pipeline import (
    generate_and_store_embedding as _generate_and_store_embedding_runtime,
)
from automem.embedding.runtime_pipeline import (
    init_embedding_pipeline as _init_embedding_pipeline_runtime,
)
from automem.embedding.runtime_pipeline import (
    process_embedding_batch as _process_embedding_batch_runtime,
)
from automem.embedding.runtime_pipeline import (
    store_embedding_in_qdrant as _store_embedding_in_qdrant_runtime,
)
from automem.enrichment.runtime_bindings import create_enrichment_runtime
from automem.enrichment.runtime_worker import enqueue_enrichment as _enqueue_enrichment_runtime
from automem.enrichment.runtime_worker import enrichment_worker as _enrichment_worker_runtime
from automem.enrichment.runtime_worker import (
    init_enrichment_pipeline as _init_enrichment_pipeline_runtime,
)
from automem.enrichment.runtime_worker import update_last_accessed as _update_last_accessed_runtime
from automem.service_runtime import get_memory_graph as _get_memory_graph_runtime
from automem.service_runtime import get_qdrant_client as _get_qdrant_client_runtime
from automem.service_runtime import init_openai as _init_openai_runtime
from automem.service_state import EnrichmentJob, EnrichmentStats, ServiceState

# Environment is loaded by automem.config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,  # Write to stdout so Railway correctly parses log levels
)
logger = logging.getLogger("automem.api")

# Configure Flask and Werkzeug loggers to use stdout instead of stderr
# This ensures Railway correctly parses log levels instead of treating everything as "error"
for logger_name in ["werkzeug", "flask.app"]:
    framework_logger = logging.getLogger(logger_name)
    framework_logger.handlers.clear()
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    framework_logger.addHandler(stdout_handler)
    framework_logger.setLevel(logging.INFO)

# Ensure local package imports work when only app.py is copied
try:
    import automem  # type: ignore
except Exception:
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

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
from automem.search.runtime_recall_helpers import (
    _graph_keyword_search,
    _result_passes_filters,
    _vector_filter_only_tag_search,
    _vector_search,
    configure_recall_helpers,
)
from automem.search.runtime_relations import fetch_relations as _fetch_relations_runtime
from automem.stores.graph_store import _build_graph_tag_predicate
from automem.stores.runtime_clients import (
    ensure_qdrant_collection as _ensure_qdrant_collection_runtime,
)
from automem.stores.runtime_clients import init_falkordb as _init_falkordb_runtime
from automem.stores.runtime_clients import init_qdrant as _init_qdrant_runtime
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

# Keyword/NER constants come from automem.utils.text if available
SEARCH_STOPWORDS: Set[str] = set()
ENTITY_STOPWORDS: Set[str] = set()
ENTITY_BLOCKLIST: Set[str] = set()

# Search weights are imported from automem.config

# Maximum number of results returned by /recall
RECALL_MAX_LIMIT = int(os.getenv("RECALL_MAX_LIMIT", "100"))

# API tokens are imported from automem.config


try:
    from automem.utils.text import ENTITY_BLOCKLIST as _AM_ENTITY_BLOCKLIST
    from automem.utils.text import ENTITY_STOPWORDS as _AM_ENTITY_STOPWORDS
    from automem.utils.text import SEARCH_STOPWORDS as _AM_SEARCH_STOPWORDS
    from automem.utils.text import _extract_keywords as _AM_extract_keywords

    # Override local constants if package is available
    SEARCH_STOPWORDS = _AM_SEARCH_STOPWORDS
    ENTITY_STOPWORDS = _AM_ENTITY_STOPWORDS
    ENTITY_BLOCKLIST = _AM_ENTITY_BLOCKLIST
    _extract_keywords = _AM_extract_keywords
except Exception:
    # Define local fallback for keyword extraction
    def _extract_keywords(text: str) -> List[str]:
        if not text:
            return []
        words = re.findall(r"[A-Za-z0-9_\-]+", text.lower())
        keywords: List[str] = []
        seen: set[str] = set()
        for word in words:
            cleaned = word.strip("-_")
            if len(cleaned) < 3:
                continue
            if cleaned in SEARCH_STOPWORDS:
                continue
            if cleaned in seen:
                continue
            seen.add(cleaned)
            keywords.append(cleaned)
        return keywords


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


def init_openai() -> None:
    _init_openai_runtime(
        state=state,
        logger=logger,
        openai_cls=OpenAI,
        get_env_fn=os.getenv,
    )


memory_classifier = MemoryClassifier(
    normalize_memory_type=normalize_memory_type,
    ensure_openai_client=init_openai,
    get_openai_client=get_openai_client,
    classification_model=CLASSIFICATION_MODEL,
    logger=logger,
)


def init_embedding_provider() -> None:
    _init_embedding_provider(
        state=state,
        logger=logger,
        vector_size_config=VECTOR_SIZE,
        embedding_model=EMBEDDING_MODEL,
    )


def init_falkordb() -> None:
    _init_falkordb_runtime(
        state=state,
        logger=logger,
        falkordb_cls=FalkorDB,
        graph_name=GRAPH_NAME,
        falkordb_port=FALKORDB_PORT,
    )


def init_qdrant() -> None:
    _init_qdrant_runtime(
        state=state,
        logger=logger,
        qdrant_client_cls=QdrantClient,
        ensure_collection_fn=_ensure_qdrant_collection,
    )


def _ensure_qdrant_collection() -> None:
    _ensure_qdrant_collection_runtime(
        state=state,
        logger=logger,
        collection_name=COLLECTION_NAME,
        vector_size_config=VECTOR_SIZE,
        get_effective_vector_size_fn=get_effective_vector_size,
        vector_params_cls=VectorParams,
        distance_enum=Distance,
        payload_schema_type_enum=PayloadSchemaType,
    )


def get_memory_graph() -> Any:
    return _get_memory_graph_runtime(
        state=state,
        init_falkordb_fn=init_falkordb,
    )


def get_qdrant_client() -> Optional[QdrantClient]:
    return _get_qdrant_client_runtime(
        state=state,
        init_qdrant_fn=init_qdrant,
    )


def init_enrichment_pipeline() -> None:
    _init_enrichment_pipeline_runtime(
        state=state,
        logger=logger,
        queue_cls=Queue,
        thread_cls=Thread,
        worker_target=enrichment_worker,
    )


def enqueue_enrichment(memory_id: str, *, forced: bool = False, attempt: int = 0) -> None:
    _enqueue_enrichment_runtime(
        state=state,
        memory_id=memory_id,
        forced=forced,
        attempt=attempt,
        enrichment_job_cls=EnrichmentJob,
    )


# ---------------------------------------------------------------------------
# Access Tracking (updates last_accessed on recall)
# ---------------------------------------------------------------------------


def update_last_accessed(memory_ids: List[str]) -> None:
    _update_last_accessed_runtime(
        memory_ids=memory_ids,
        get_memory_graph_fn=get_memory_graph,
        utc_now_fn=utc_now,
        logger=logger,
    )


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


def enrichment_worker() -> None:
    _enrichment_worker_runtime(
        state=state,
        logger=logger,
        enrichment_idle_sleep_seconds=ENRICHMENT_IDLE_SLEEP_SECONDS,
        enrichment_max_attempts=ENRICHMENT_MAX_ATTEMPTS,
        enrichment_failure_backoff_seconds=ENRICHMENT_FAILURE_BACKOFF_SECONDS,
        empty_exc=Empty,
        enrich_memory_fn=enrich_memory,
        emit_event_fn=emit_event,
        utc_now_fn=utc_now,
        enqueue_enrichment_fn=enqueue_enrichment,
        perf_counter_fn=time.perf_counter,
        sleep_fn=time.sleep,
    )


def init_embedding_pipeline() -> None:
    _init_embedding_pipeline_runtime(
        state=state,
        logger=logger,
        queue_cls=Queue,
        thread_cls=Thread,
        worker_target=embedding_worker,
    )


def enqueue_embedding(memory_id: str, content: str) -> None:
    _enqueue_embedding_runtime(state=state, memory_id=memory_id, content=content)


def embedding_worker() -> None:
    _embedding_worker_runtime(
        state=state,
        logger=logger,
        batch_size=EMBEDDING_BATCH_SIZE,
        batch_timeout_seconds=EMBEDDING_BATCH_TIMEOUT_SECONDS,
        empty_exc=Empty,
        process_batch_fn=_process_embedding_batch,
        sleep_fn=time.sleep,
        time_fn=time.time,
    )


def _process_embedding_batch(batch: List[Tuple[str, str]]) -> None:
    _process_embedding_batch_runtime(
        state=state,
        batch=batch,
        logger=logger,
        generate_real_embeddings_batch_fn=_generate_real_embeddings_batch,
        store_embedding_in_qdrant_fn=_store_embedding_in_qdrant,
    )


def _store_embedding_in_qdrant(memory_id: str, content: str, embedding: List[float]) -> None:
    _store_embedding_in_qdrant_runtime(
        memory_id=memory_id,
        content=content,
        embedding=embedding,
        get_qdrant_client_fn=get_qdrant_client,
        get_memory_graph_fn=get_memory_graph,
        collection_name=COLLECTION_NAME,
        point_struct_cls=PointStruct,
        utc_now_fn=utc_now,
        logger=logger,
    )


def generate_and_store_embedding(memory_id: str, content: str) -> None:
    _generate_and_store_embedding_runtime(
        memory_id=memory_id,
        content=content,
        generate_real_embedding_fn=_generate_real_embedding,
        store_embedding_in_qdrant_fn=_store_embedding_in_qdrant,
    )


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


def _normalize_tags(value: Any) -> List[str]:
    try:
        return _normalize_tags_value(value)
    except ValueError as exc:
        abort(400, description=str(exc))


def _coerce_importance(value: Any) -> float:
    try:
        return _coerce_importance_value(value)
    except ValueError as exc:
        abort(400, description=str(exc))


def _coerce_embedding(value: Any) -> Optional[List[float]]:
    return _coerce_embedding_value(value, state.effective_vector_size)


def _generate_placeholder_embedding(content: str) -> List[float]:
    return _generate_placeholder_embedding_value(content, state.effective_vector_size)


def _generate_real_embedding(content: str) -> List[float]:
    return _generate_real_embedding_value(
        content,
        init_embedding_provider=init_embedding_provider,
        state=state,
        logger=logger,
        placeholder_embedding=_generate_placeholder_embedding,
    )


def _generate_real_embeddings_batch(contents: List[str]) -> List[List[float]]:
    return _generate_real_embeddings_batch_value(
        contents,
        init_embedding_provider=init_embedding_provider,
        state=state,
        logger=logger,
        placeholder_embedding=_generate_placeholder_embedding,
    )


def _fetch_relations(graph: Any, memory_id: str) -> List[Dict[str, Any]]:
    return _fetch_relations_runtime(
        graph=graph,
        memory_id=memory_id,
        relation_limit=RECALL_RELATION_LIMIT,
        serialize_node_fn=_serialize_node,
        summarize_relation_node_fn=_summarize_relation_node,
        logger=logger,
    )


configure_recall_helpers(
    parse_iso_datetime=_parse_iso_datetime,
    prepare_tag_filters=_prepare_tag_filters,
    build_graph_tag_predicate=_build_graph_tag_predicate,
    build_qdrant_tag_filter=_build_qdrant_tag_filter,
    serialize_node=_serialize_node,
    fetch_relations=_fetch_relations,
    extract_keywords=_extract_keywords,
    coerce_embedding=_coerce_embedding,
    generate_real_embedding=_generate_real_embedding,
    logger=logger,
    collection_name=COLLECTION_NAME,
)

_register_blueprints_runtime(
    app=app,
    get_memory_graph_fn=get_memory_graph,
    get_qdrant_client_fn=get_qdrant_client,
    state=state,
    graph_name=GRAPH_NAME,
    collection_name=COLLECTION_NAME,
    utc_now_fn=utc_now,
    require_admin_token_fn=_require_admin_token,
    enqueue_enrichment_fn=enqueue_enrichment,
    enrichment_max_attempts=ENRICHMENT_MAX_ATTEMPTS,
    normalize_tag_list_fn=_normalize_tag_list,
    normalize_timestamp_fn=_normalize_timestamp,
    parse_time_expression_fn=_parse_time_expression,
    extract_keywords_fn=_extract_keywords,
    compute_metadata_score_fn=_compute_metadata_score,
    result_passes_filters_fn=_result_passes_filters,
    graph_keyword_search_fn=_graph_keyword_search,
    vector_search_fn=_vector_search,
    vector_filter_only_tag_search_fn=_vector_filter_only_tag_search,
    recall_max_limit=RECALL_MAX_LIMIT,
    logger=logger,
    allowed_relations=ALLOWED_RELATIONS,
    recall_relation_limit=RECALL_RELATION_LIMIT,
    serialize_node_fn=_serialize_node,
    summarize_relation_node_fn=_summarize_relation_node,
    update_last_accessed_fn=update_last_accessed,
    jit_enrich_fn=jit_enrich_lightweight if JIT_ENRICHMENT_ENABLED else None,
    normalize_tags_fn=_normalize_tags,
    compute_tag_prefixes_fn=_compute_tag_prefixes,
    coerce_importance_fn=_coerce_importance,
    coerce_embedding_fn=_coerce_embedding,
    parse_metadata_field_fn=_parse_metadata_field,
    generate_real_embedding_fn=_generate_real_embedding,
    enqueue_embedding_fn=enqueue_embedding,
    classify_memory_fn=lambda content: memory_classifier.classify(content),
    point_struct_cls=PointStruct,
    relationship_types=RELATIONSHIP_TYPES,
    get_openai_client_fn=get_openai_client,
    init_openai_fn=init_openai,
    effective_vector_size_fn=lambda: state.effective_vector_size,
    embedding_model=EMBEDDING_MODEL,
    build_consolidator_from_config_fn=_build_consolidator_from_config,
    persist_consolidation_run_fn=_persist_consolidation_run,
    build_scheduler_from_graph_fn=_build_scheduler_from_graph,
    load_recent_runs_fn=_load_recent_runs,
    consolidation_tick_seconds=CONSOLIDATION_TICK_SECONDS,
    consolidation_history_limit=CONSOLIDATION_HISTORY_LIMIT,
    require_api_token_fn=require_api_token,
)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8001"))
    logger.info("Starting Flask API on port %s", port)
    init_falkordb()
    init_qdrant()
    init_openai()  # Still needed for memory type classification
    init_embedding_provider()  # New provider pattern for embeddings
    init_enrichment_pipeline()
    init_embedding_pipeline()
    init_consolidation_scheduler()
    init_sync_worker()
    # Use :: for IPv6 dual-stack (Railway internal networking uses IPv6)
    app.run(host="::", port=port, debug=False)
