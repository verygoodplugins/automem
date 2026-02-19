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
from flask import Blueprint, Flask, abort, jsonify, request
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

from consolidation import ConsolidationScheduler, MemoryConsolidator

# Make OpenAI import optional to allow running without it
try:
    from openai import OpenAI  # type: ignore
except ImportError:
    OpenAI = None  # type: ignore

# SSE streaming for real-time observability
from automem.analytics.runtime_helpers import analyze_memories as _analyze_memories_runtime
from automem.analytics.runtime_helpers import startup_recall as _startup_recall_runtime
from automem.api.auth_helpers import extract_api_token as _extract_api_token_helper
from automem.api.auth_helpers import require_admin_token as _require_admin_token_helper
from automem.api.auth_helpers import require_api_token as _require_api_token_helper
from automem.api.stream import create_stream_blueprint, emit_event
from automem.classification.memory_classifier import MemoryClassifier
from automem.consolidation.runtime_helpers import (
    apply_scheduler_overrides as _apply_scheduler_overrides_runtime,
)
from automem.consolidation.runtime_helpers import (
    build_consolidator_from_config as _build_consolidator_from_config_runtime,
)
from automem.consolidation.runtime_helpers import (
    load_control_record as _load_control_record_runtime,
)
from automem.consolidation.runtime_helpers import load_recent_runs as _load_recent_runs_runtime
from automem.consolidation.runtime_helpers import (
    persist_consolidation_run as _persist_consolidation_run_runtime,
)
from automem.consolidation.runtime_helpers import tasks_for_mode as _tasks_for_mode_runtime
from automem.consolidation.runtime_routes import (
    consolidate_memories as _consolidate_memories_runtime,
)
from automem.consolidation.runtime_routes import (
    consolidation_status as _consolidation_status_runtime,
)
from automem.consolidation.runtime_routes import create_association as _create_association_runtime
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
from automem.enrichment.runtime_helpers import detect_patterns as _detect_patterns_runtime
from automem.enrichment.runtime_helpers import (
    find_temporal_relationships as _find_temporal_relationships_runtime,
)
from automem.enrichment.runtime_helpers import (
    link_semantic_neighbors as _link_semantic_neighbors_runtime,
)
from automem.enrichment.runtime_helpers import temporal_cutoff as _temporal_cutoff_runtime
from automem.enrichment.runtime_orchestration import enrich_memory as _enrich_memory_runtime
from automem.enrichment.runtime_orchestration import (
    jit_enrich_lightweight as _jit_enrich_lightweight_runtime,
)
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

# Legacy blueprint placeholders for deprecated route definitions below.
# These are not registered with the app and are safe to keep until full removal.
admin_bp = Blueprint("admin_legacy", __name__)
memory_bp = Blueprint("memory_legacy", __name__)
recall_bp = Blueprint("recall_legacy", __name__)
consolidation_bp = Blueprint("consolidation_legacy", __name__)

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
from automem.search.runtime_relations import get_related_memories as _get_related_memories_runtime
from automem.stores.graph_store import _build_graph_tag_predicate
from automem.stores.runtime_clients import (
    ensure_qdrant_collection as _ensure_qdrant_collection_runtime,
)
from automem.stores.runtime_clients import init_falkordb as _init_falkordb_runtime
from automem.stores.runtime_clients import init_qdrant as _init_qdrant_runtime
from automem.stores.vector_store import _build_qdrant_tag_filter
from automem.sync.runtime_worker import init_sync_worker as _init_sync_worker_runtime
from automem.sync.runtime_worker import run_sync_check as _run_sync_check_runtime
from automem.sync.runtime_worker import sync_worker as _sync_worker_runtime
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
    """Initialize OpenAI client for memory type classification (not embeddings)."""
    if state.openai_client is not None:
        return

    # Check if OpenAI is available at all
    if OpenAI is None:
        logger.info("OpenAI package not installed (used for memory type classification)")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.info("OpenAI API key not provided (used for memory type classification)")
        return

    try:
        state.openai_client = OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized for memory type classification")
    except Exception:
        logger.exception("Failed to initialize OpenAI client")
        state.openai_client = None


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
    init_falkordb()
    return state.memory_graph


def get_qdrant_client() -> Optional[QdrantClient]:
    init_qdrant()
    return state.qdrant


def init_enrichment_pipeline() -> None:
    """Initialize the background enrichment pipeline."""
    if state.enrichment_queue is not None:
        return

    state.enrichment_queue = Queue()
    state.enrichment_thread = Thread(target=enrichment_worker, daemon=True)
    state.enrichment_thread.start()
    logger.info("Enrichment pipeline initialized")


def enqueue_enrichment(memory_id: str, *, forced: bool = False, attempt: int = 0) -> None:
    if not memory_id or state.enrichment_queue is None:
        return

    job = EnrichmentJob(memory_id=memory_id, attempt=attempt, forced=forced)

    with state.enrichment_lock:
        if not forced and (
            memory_id in state.enrichment_pending or memory_id in state.enrichment_inflight
        ):
            return

        state.enrichment_pending.add(memory_id)
        state.enrichment_queue.put(job)


# ---------------------------------------------------------------------------
# Access Tracking (updates last_accessed on recall)
# ---------------------------------------------------------------------------


def update_last_accessed(memory_ids: List[str]) -> None:
    """Update last_accessed timestamp for retrieved memories (direct, synchronous)."""
    if not memory_ids:
        return

    graph = get_memory_graph()
    if graph is None:
        return

    timestamp = utc_now()
    try:
        graph.query(
            """
            UNWIND $ids AS mid
            MATCH (m:Memory {id: mid})
            SET m.last_accessed = $ts
            """,
            {"ids": memory_ids, "ts": timestamp},
        )
        logger.debug("Updated last_accessed for %d memories", len(memory_ids))
    except Exception:
        logger.exception("Failed to update last_accessed for memories")


def _load_control_record(graph: Any) -> Dict[str, Any]:
    return _load_control_record_runtime(
        graph,
        logger=logger,
        control_label=CONSOLIDATION_CONTROL_LABEL,
        control_node_id=CONSOLIDATION_CONTROL_NODE_ID,
        task_fields=CONSOLIDATION_TASK_FIELDS,
        utc_now_fn=utc_now,
    )


def _load_recent_runs(graph: Any, limit: int) -> List[Dict[str, Any]]:
    return _load_recent_runs_runtime(
        graph,
        limit,
        logger=logger,
        run_label=CONSOLIDATION_RUN_LABEL,
    )


def _apply_scheduler_overrides(scheduler: ConsolidationScheduler) -> None:
    _apply_scheduler_overrides_runtime(
        scheduler,
        decay_interval_seconds=CONSOLIDATION_DECAY_INTERVAL_SECONDS,
        creative_interval_seconds=CONSOLIDATION_CREATIVE_INTERVAL_SECONDS,
        cluster_interval_seconds=CONSOLIDATION_CLUSTER_INTERVAL_SECONDS,
        forget_interval_seconds=CONSOLIDATION_FORGET_INTERVAL_SECONDS,
    )


def _tasks_for_mode(mode: str) -> List[str]:
    return _tasks_for_mode_runtime(mode, CONSOLIDATION_TASK_FIELDS)


def _persist_consolidation_run(graph: Any, result: Dict[str, Any]) -> None:
    _persist_consolidation_run_runtime(
        graph,
        result,
        logger=logger,
        run_label=CONSOLIDATION_RUN_LABEL,
        control_label=CONSOLIDATION_CONTROL_LABEL,
        control_node_id=CONSOLIDATION_CONTROL_NODE_ID,
        task_fields=CONSOLIDATION_TASK_FIELDS,
        history_limit=CONSOLIDATION_HISTORY_LIMIT,
        utc_now_fn=utc_now,
    )


def _build_scheduler_from_graph(graph: Any) -> Optional[ConsolidationScheduler]:
    vector_store = get_qdrant_client()
    consolidator = _build_consolidator_from_config(graph, vector_store)
    scheduler = ConsolidationScheduler(consolidator)
    _apply_scheduler_overrides(scheduler)

    control = _load_control_record(graph)
    for task, field in CONSOLIDATION_TASK_FIELDS.items():
        iso_value = control.get(field)
        last_run = _parse_iso_datetime(iso_value)
        if last_run and task in scheduler.schedules:
            scheduler.schedules[task]["last_run"] = last_run

    return scheduler


def _build_consolidator_from_config(graph: Any, vector_store: Any) -> MemoryConsolidator:
    return _build_consolidator_from_config_runtime(
        graph,
        vector_store,
        memory_consolidator_cls=MemoryConsolidator,
        delete_threshold=CONSOLIDATION_DELETE_THRESHOLD,
        archive_threshold=CONSOLIDATION_ARCHIVE_THRESHOLD,
        grace_period_days=CONSOLIDATION_GRACE_PERIOD_DAYS,
        importance_protection_threshold=CONSOLIDATION_IMPORTANCE_PROTECTION_THRESHOLD,
        protected_types=set(CONSOLIDATION_PROTECTED_TYPES),
    )


def _run_consolidation_tick() -> None:
    graph = get_memory_graph()
    if graph is None:
        return

    scheduler = _build_scheduler_from_graph(graph)
    if scheduler is None:
        return

    try:
        tick_start = time.perf_counter()
        results = scheduler.run_scheduled_tasks(
            decay_threshold=CONSOLIDATION_DECAY_IMPORTANCE_THRESHOLD
        )
        for result in results:
            _persist_consolidation_run(graph, result)

            # Emit SSE event for real-time monitoring
            task_type = result.get("mode", "unknown")
            steps = result.get("steps", {})
            affected_count = 0

            # Count affected memories from each step
            if "decay" in steps:
                affected_count += steps["decay"].get("updated", 0)
            if "creative" in steps:
                affected_count += steps["creative"].get("created", 0)
            if "cluster" in steps:
                affected_count += steps["cluster"].get("meta_memories_created", 0)
            if "forget" in steps:
                affected_count += steps["forget"].get("archived", 0)
                affected_count += steps["forget"].get("deleted", 0)

            elapsed_ms = int((time.perf_counter() - tick_start) * 1000)
            next_runs = scheduler.get_next_runs()

            emit_event(
                "consolidation.run",
                {
                    "task_type": task_type,
                    "affected_count": affected_count,
                    "elapsed_ms": elapsed_ms,
                    "success": result.get("success", False),
                    "next_scheduled": next_runs.get(task_type, "unknown"),
                    "steps": list(steps.keys()),
                },
                utc_now,
            )
    except Exception:
        logger.exception("Consolidation scheduler tick failed")


def consolidation_worker() -> None:
    """Background loop that triggers consolidation tasks."""
    logger.info("Consolidation scheduler thread started")
    while state.consolidation_stop_event and not state.consolidation_stop_event.wait(
        CONSOLIDATION_TICK_SECONDS
    ):
        _run_consolidation_tick()


def init_consolidation_scheduler() -> None:
    """Ensure the background consolidation scheduler is running."""
    if state.consolidation_thread and state.consolidation_thread.is_alive():
        return

    stop_event = Event()
    state.consolidation_stop_event = stop_event
    state.consolidation_thread = Thread(
        target=consolidation_worker,
        daemon=True,
        name="consolidation-scheduler",
    )
    state.consolidation_thread.start()
    # Kick off an initial tick so schedules are populated quickly.
    _run_consolidation_tick()
    logger.info("Consolidation scheduler initialized")


def enrichment_worker() -> None:
    """Background worker that processes memories for enrichment."""
    while True:
        try:
            if state.enrichment_queue is None:
                time.sleep(ENRICHMENT_IDLE_SLEEP_SECONDS)
                continue

            try:
                job: EnrichmentJob = state.enrichment_queue.get(
                    timeout=ENRICHMENT_IDLE_SLEEP_SECONDS
                )
            except Empty:
                continue

            with state.enrichment_lock:
                state.enrichment_pending.discard(job.memory_id)
                state.enrichment_inflight.add(job.memory_id)

            enrich_start = time.perf_counter()
            emit_event(
                "enrichment.start",
                {
                    "memory_id": job.memory_id,
                    "attempt": job.attempt + 1,
                },
                utc_now,
            )

            try:
                processed = enrich_memory(job.memory_id, forced=job.forced)
                state.enrichment_stats.record_success(job.memory_id)
                elapsed_ms = int((time.perf_counter() - enrich_start) * 1000)
                emit_event(
                    "enrichment.complete",
                    {
                        "memory_id": job.memory_id,
                        "success": True,
                        "elapsed_ms": elapsed_ms,
                        "skipped": not processed,
                    },
                    utc_now,
                )
                if not processed:
                    logger.debug("Enrichment skipped for %s (already processed)", job.memory_id)
            except Exception as exc:  # pragma: no cover - background thread
                state.enrichment_stats.record_failure(str(exc))
                elapsed_ms = int((time.perf_counter() - enrich_start) * 1000)
                emit_event(
                    "enrichment.failed",
                    {
                        "memory_id": job.memory_id,
                        "error": str(exc)[:100],
                        "attempt": job.attempt + 1,
                        "elapsed_ms": elapsed_ms,
                        "will_retry": job.attempt + 1 < ENRICHMENT_MAX_ATTEMPTS,
                    },
                    utc_now,
                )
                logger.exception("Failed to enrich memory %s", job.memory_id)
                if job.attempt + 1 < ENRICHMENT_MAX_ATTEMPTS:
                    time.sleep(ENRICHMENT_FAILURE_BACKOFF_SECONDS)
                    enqueue_enrichment(job.memory_id, forced=job.forced, attempt=job.attempt + 1)
                else:
                    logger.error(
                        "Giving up on enrichment for %s after %s attempts",
                        job.memory_id,
                        job.attempt + 1,
                    )
            finally:
                with state.enrichment_lock:
                    state.enrichment_inflight.discard(job.memory_id)
                state.enrichment_queue.task_done()
        except Exception:  # pragma: no cover - defensive catch-all
            logger.exception("Error in enrichment worker loop")
            time.sleep(ENRICHMENT_FAILURE_BACKOFF_SECONDS)


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


def init_sync_worker() -> None:
    _init_sync_worker_runtime(
        state=state,
        logger=logger,
        sync_auto_repair=SYNC_AUTO_REPAIR,
        sync_check_interval_seconds=SYNC_CHECK_INTERVAL_SECONDS,
        stop_event_cls=Event,
        thread_cls=Thread,
        worker_target=sync_worker,
    )


def sync_worker() -> None:
    _sync_worker_runtime(
        state=state,
        logger=logger,
        sync_check_interval_seconds=SYNC_CHECK_INTERVAL_SECONDS,
        run_sync_check_fn=_run_sync_check,
        sleep_fn=time.sleep,
    )


def _run_sync_check() -> None:
    _run_sync_check_runtime(
        state=state,
        logger=logger,
        get_memory_graph_fn=get_memory_graph,
        get_qdrant_client_fn=get_qdrant_client,
        collection_name=COLLECTION_NAME,
        utc_now_fn=utc_now,
        enqueue_embedding_fn=enqueue_embedding,
    )


def jit_enrich_lightweight(memory_id: str, properties: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return _jit_enrich_lightweight_runtime(
        memory_id=memory_id,
        properties=properties,
        get_memory_graph_fn=get_memory_graph,
        get_qdrant_client_fn=get_qdrant_client,
        parse_metadata_field_fn=_parse_metadata_field,
        normalize_tag_list_fn=_normalize_tag_list,
        extract_entities_fn=extract_entities,
        slugify_fn=_slugify,
        compute_tag_prefixes_fn=_compute_tag_prefixes,
        enrichment_enable_summaries=ENRICHMENT_ENABLE_SUMMARIES,
        generate_summary_fn=generate_summary,
        utc_now_fn=utc_now,
        collection_name=COLLECTION_NAME,
        logger=logger,
    )


def enrich_memory(memory_id: str, *, forced: bool = False) -> bool:
    return _enrich_memory_runtime(
        memory_id=memory_id,
        forced=forced,
        get_memory_graph_fn=get_memory_graph,
        get_qdrant_client_fn=get_qdrant_client,
        parse_metadata_field_fn=_parse_metadata_field,
        normalize_tag_list_fn=_normalize_tag_list,
        extract_entities_fn=extract_entities,
        slugify_fn=_slugify,
        compute_tag_prefixes_fn=_compute_tag_prefixes,
        find_temporal_relationships_fn=find_temporal_relationships,
        detect_patterns_fn=detect_patterns,
        link_semantic_neighbors_fn=link_semantic_neighbors,
        enrichment_enable_summaries=ENRICHMENT_ENABLE_SUMMARIES,
        generate_summary_fn=generate_summary,
        utc_now_fn=utc_now,
        collection_name=COLLECTION_NAME,
        unexpected_response_exc=UnexpectedResponse,
        logger=logger,
    )


def _temporal_cutoff() -> str:
    return _temporal_cutoff_runtime()


def find_temporal_relationships(graph: Any, memory_id: str, limit: int = 5) -> int:
    return _find_temporal_relationships_runtime(
        graph=graph,
        memory_id=memory_id,
        limit=limit,
        cutoff_fn=_temporal_cutoff,
        utc_now_fn=utc_now,
        logger=logger,
    )


def detect_patterns(graph: Any, memory_id: str, content: str) -> List[Dict[str, Any]]:
    return _detect_patterns_runtime(
        graph=graph,
        memory_id=memory_id,
        content=content,
        classify_fn=memory_classifier.classify,
        search_stopwords=SEARCH_STOPWORDS,
        utc_now_fn=utc_now,
        logger=logger,
    )


def link_semantic_neighbors(graph: Any, memory_id: str) -> List[Tuple[str, float]]:
    return _link_semantic_neighbors_runtime(
        graph=graph,
        memory_id=memory_id,
        get_qdrant_client_fn=get_qdrant_client,
        collection_name=COLLECTION_NAME,
        similarity_limit=ENRICHMENT_SIMILARITY_LIMIT,
        similarity_threshold=ENRICHMENT_SIMILARITY_THRESHOLD,
        utc_now_fn=utc_now,
        logger=logger,
    )


# Legacy route implementations retained for reference only.
# NOTE: These are bound to unregistered "*_legacy" blueprints and are not active.
# Active endpoints live in automem/api/* blueprints registered above.


@admin_bp.route("/admin/reembed", methods=["POST"])
def admin_reembed() -> Any:
    """Legacy admin handler; route now provided by automem.api.admin blueprint."""
    abort(410, description="/admin/reembed moved to blueprint")


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


@memory_bp.route("/memory", methods=["POST"])
def store_memory() -> Any:
    query_start = time.perf_counter()
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        abort(400, description="JSON body is required")

    content = (payload.get("content") or "").strip()
    if not content:
        abort(400, description="'content' is required")

    tags = _normalize_tags(payload.get("tags"))
    tags_lower = [t.strip().lower() for t in tags if isinstance(t, str) and t.strip()]
    tag_prefixes = _compute_tag_prefixes(tags_lower)
    importance = _coerce_importance(payload.get("importance"))
    memory_id = payload.get("id") or str(uuid.uuid4())

    metadata_raw = payload.get("metadata")
    if metadata_raw is None:
        metadata: Dict[str, Any] = {}
    elif isinstance(metadata_raw, dict):
        metadata = metadata_raw
    else:
        abort(400, description="'metadata' must be an object")
    metadata_json = json.dumps(metadata, default=str)

    # Accept explicit type/confidence or classify automatically
    raw_type = payload.get("type")
    type_confidence = payload.get("confidence")

    if raw_type:
        # Normalize type (handles aliases and case variations)
        memory_type, was_normalized = normalize_memory_type(raw_type)

        # Empty string means unknown type that couldn't be mapped
        if not memory_type:
            valid_types = sorted(MEMORY_TYPES)
            alias_examples = ", ".join(f"'{k}'" for k in list(TYPE_ALIASES.keys())[:5])
            abort(
                400,
                description=(
                    f"Invalid memory type '{raw_type}'. "
                    f"Must be one of: {', '.join(valid_types)}, "
                    f"or aliases like {alias_examples}..."
                ),
            )

        if was_normalized and memory_type != raw_type:
            logger.debug("Normalized type '%s' -> '%s'", raw_type, memory_type)

        # Use provided confidence or default
        if type_confidence is None:
            type_confidence = 0.9  # High confidence for explicit types
        else:
            type_confidence = _coerce_importance(type_confidence)
    else:
        # Auto-classify if no type provided
        memory_type, type_confidence = memory_classifier.classify(content)

    # Handle temporal validity fields
    t_valid = payload.get("t_valid")
    t_invalid = payload.get("t_invalid")
    if t_valid:
        try:
            t_valid = _normalize_timestamp(t_valid)
        except ValueError as exc:
            abort(400, description=f"Invalid t_valid: {exc}")
    if t_invalid:
        try:
            t_invalid = _normalize_timestamp(t_invalid)
        except ValueError as exc:
            abort(400, description=f"Invalid t_invalid: {exc}")

    try:
        embedding = _coerce_embedding(payload.get("embedding"))
    except ValueError as exc:
        abort(400, description=str(exc))

    graph = get_memory_graph()
    if graph is None:
        abort(503, description="FalkorDB is unavailable")

    created_at = payload.get("timestamp")
    if created_at:
        try:
            created_at = _normalize_timestamp(created_at)
        except ValueError as exc:
            abort(400, description=str(exc))
    else:
        created_at = utc_now()

    updated_at = payload.get("updated_at")
    if updated_at:
        try:
            updated_at = _normalize_timestamp(updated_at)
        except ValueError as exc:
            abort(400, description=f"Invalid updated_at: {exc}")
    else:
        updated_at = created_at

    last_accessed = payload.get("last_accessed")
    if last_accessed:
        try:
            last_accessed = _normalize_timestamp(last_accessed)
        except ValueError as exc:
            abort(400, description=f"Invalid last_accessed: {exc}")
    else:
        last_accessed = updated_at

    try:
        graph.query(
            """
            MERGE (m:Memory {id: $id})
            SET m.content = $content,
                m.timestamp = $timestamp,
                m.importance = $importance,
                m.tags = $tags,
                m.tag_prefixes = $tag_prefixes,
                m.type = $type,
                m.confidence = $confidence,
                m.t_valid = $t_valid,
                m.t_invalid = $t_invalid,
                m.updated_at = $updated_at,
                m.last_accessed = $last_accessed,
                m.metadata = $metadata,
                m.processed = false
            RETURN m
            """,
            {
                "id": memory_id,
                "content": content,
                "timestamp": created_at,
                "importance": importance,
                "tags": tags,
                "tag_prefixes": tag_prefixes,
                "type": memory_type,
                "confidence": type_confidence,
                "t_valid": t_valid or created_at,
                "t_invalid": t_invalid,
                "updated_at": updated_at,
                "last_accessed": last_accessed,
                "metadata": metadata_json,
            },
        )
    except Exception:  # pragma: no cover - log full stack trace in production
        logger.exception("Failed to persist memory in FalkorDB")
        abort(500, description="Failed to store memory in FalkorDB")

    # Queue for enrichment
    enqueue_enrichment(memory_id)

    # Queue for async embedding generation (if no embedding provided)
    embedding_status = "skipped"
    qdrant_client = get_qdrant_client()

    if embedding is not None:
        # Sync path: User provided embedding, store immediately
        embedding_status = "provided"
        qdrant_result = None
        if qdrant_client is not None:
            try:
                qdrant_client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[
                        PointStruct(
                            id=memory_id,
                            vector=embedding,
                            payload={
                                "content": content,
                                "tags": tags,
                                "tag_prefixes": tag_prefixes,
                                "importance": importance,
                                "timestamp": created_at,
                                "type": memory_type,
                                "confidence": type_confidence,
                                "updated_at": updated_at,
                                "last_accessed": last_accessed,
                                "metadata": metadata,
                            },
                        )
                    ],
                )
                qdrant_result = "stored"
            except Exception:  # pragma: no cover - log full stack trace in production
                logger.exception("Qdrant upsert failed")
                qdrant_result = "failed"
    elif qdrant_client is not None:
        # Async path: Queue embedding generation
        enqueue_embedding(memory_id, content)
        embedding_status = "queued"
        qdrant_result = "queued"
    else:
        qdrant_result = "unconfigured"

    response = {
        "status": "success",
        "memory_id": memory_id,
        "stored_at": created_at,
        "type": memory_type,
        "confidence": type_confidence,
        "qdrant": qdrant_result,
        "embedding_status": embedding_status,
        "enrichment": "queued" if state.enrichment_queue else "disabled",
        "metadata": metadata,
        "timestamp": created_at,
        "updated_at": updated_at,
        "last_accessed": last_accessed,
        "query_time_ms": round((time.perf_counter() - query_start) * 1000, 2),
    }

    # Structured logging for performance analysis
    logger.info(
        "memory_stored",
        extra={
            "memory_id": memory_id,
            "type": memory_type,
            "importance": importance,
            "tags_count": len(tags),
            "content_length": len(content),
            "latency_ms": response["query_time_ms"],
            "embedding_status": embedding_status,
            "qdrant_status": qdrant_result,
            "enrichment_queued": bool(state.enrichment_queue),
        },
    )

    # Emit SSE event for real-time monitoring
    emit_event(
        "memory.store",
        {
            "id": memory_id,
            "content_preview": content[:100] + "..." if len(content) > 100 else content,
            "type": memory_type,
            "importance": importance,
            "tags": tags[:5],
            "size_bytes": len(content),
            "elapsed_ms": int(response["query_time_ms"]),
        },
        utc_now,
    )

    return jsonify(response), 201


@memory_bp.route("/memory/<memory_id>", methods=["PATCH"])
def update_memory(memory_id: str) -> Any:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        abort(400, description="JSON body is required")

    graph = get_memory_graph()
    if graph is None:
        abort(503, description="FalkorDB is unavailable")

    result = graph.query("MATCH (m:Memory {id: $id}) RETURN m", {"id": memory_id})
    if not getattr(result, "result_set", None):
        abort(404, description="Memory not found")

    current_node = result.result_set[0][0]
    current = _serialize_node(current_node)

    new_content = payload.get("content", current.get("content"))
    tags = _normalize_tag_list(payload.get("tags", current.get("tags")))
    tags_lower = [t.strip().lower() for t in tags if isinstance(t, str) and t.strip()]
    tag_prefixes = _compute_tag_prefixes(tags_lower)
    importance = payload.get("importance", current.get("importance"))
    memory_type = payload.get("type", current.get("type"))
    confidence = payload.get("confidence", current.get("confidence"))
    timestamp = payload.get("timestamp", current.get("timestamp"))
    metadata_raw = payload.get("metadata", _parse_metadata_field(current.get("metadata")))
    updated_at = payload.get("updated_at", current.get("updated_at", utc_now()))
    last_accessed = payload.get("last_accessed", current.get("last_accessed"))

    if metadata_raw is None:
        metadata: Dict[str, Any] = {}
    elif isinstance(metadata_raw, dict):
        metadata = metadata_raw
    else:
        abort(400, description="'metadata' must be an object")
    metadata_json = json.dumps(metadata, default=str)

    if timestamp:
        try:
            timestamp = _normalize_timestamp(timestamp)
        except ValueError as exc:
            abort(400, description=f"Invalid timestamp: {exc}")

    if updated_at:
        try:
            updated_at = _normalize_timestamp(updated_at)
        except ValueError as exc:
            abort(400, description=f"Invalid updated_at: {exc}")

    if last_accessed:
        try:
            last_accessed = _normalize_timestamp(last_accessed)
        except ValueError as exc:
            abort(400, description=f"Invalid last_accessed: {exc}")

    update_query = """
        MATCH (m:Memory {id: $id})
        SET m.content = $content,
            m.tags = $tags,
            m.tag_prefixes = $tag_prefixes,
            m.importance = $importance,
            m.type = $type,
            m.confidence = $confidence,
            m.timestamp = $timestamp,
            m.metadata = $metadata,
            m.updated_at = $updated_at,
            m.last_accessed = $last_accessed
        RETURN m
    """

    graph.query(
        update_query,
        {
            "id": memory_id,
            "content": new_content,
            "tags": tags,
            "tag_prefixes": tag_prefixes,
            "importance": importance,
            "type": memory_type,
            "confidence": confidence,
            "timestamp": timestamp,
            "metadata": metadata_json,
            "updated_at": updated_at,
            "last_accessed": last_accessed,
        },
    )

    qdrant_client = get_qdrant_client()
    vector = None
    if qdrant_client is not None:
        if new_content != current.get("content"):
            vector = _generate_real_embedding(new_content)
        else:
            try:
                existing = qdrant_client.retrieve(
                    collection_name=COLLECTION_NAME,
                    ids=[memory_id],
                    with_vectors=True,
                )
                if existing:
                    vector = existing[0].vector
            except Exception:
                logger.exception("Failed to retrieve existing vector; regenerating")
                vector = _generate_real_embedding(new_content)

        if vector is not None:
            payload = {
                "content": new_content,
                "tags": tags,
                "tag_prefixes": tag_prefixes,
                "importance": importance,
                "timestamp": timestamp,
                "type": memory_type,
                "confidence": confidence,
                "updated_at": updated_at,
                "last_accessed": last_accessed,
                "metadata": metadata,
            }
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=[PointStruct(id=memory_id, vector=vector, payload=payload)],
            )

    return jsonify({"status": "success", "memory_id": memory_id})


@memory_bp.route("/memory/<memory_id>", methods=["DELETE"])
def delete_memory(memory_id: str) -> Any:
    graph = get_memory_graph()
    if graph is None:
        abort(503, description="FalkorDB is unavailable")

    result = graph.query("MATCH (m:Memory {id: $id}) RETURN m", {"id": memory_id})
    if not getattr(result, "result_set", None):
        abort(404, description="Memory not found")

    graph.query("MATCH (m:Memory {id: $id}) DETACH DELETE m", {"id": memory_id})

    qdrant_client = get_qdrant_client()
    if qdrant_client is not None:
        try:
            if "qdrant_models" in globals() and qdrant_models is not None:
                selector = qdrant_models.PointIdsList(points=[memory_id])
            else:
                selector = {"points": [memory_id]}
            qdrant_client.delete(collection_name=COLLECTION_NAME, points_selector=selector)
        except Exception:
            logger.exception("Failed to delete vector for memory %s", memory_id)

    return jsonify({"status": "success", "memory_id": memory_id})


@memory_bp.route("/memory/by-tag", methods=["GET"])
def memories_by_tag() -> Any:
    raw_tags = request.args.getlist("tags") or request.args.get("tags")
    tags = _normalize_tag_list(raw_tags)
    if not tags:
        abort(400, description="'tags' query parameter is required")

    limit = max(1, min(int(request.args.get("limit", 20)), 200))

    graph = get_memory_graph()
    if graph is None:
        abort(503, description="FalkorDB is unavailable")

    params = {
        "tags": [tag.lower() for tag in tags],
        "limit": limit,
    }

    query = """
        MATCH (m:Memory)
        WHERE ANY(tag IN coalesce(m.tags, []) WHERE toLower(tag) IN $tags)
        RETURN m
        ORDER BY m.importance DESC, m.timestamp DESC
        LIMIT $limit
    """

    try:
        result = graph.query(query, params)
    except Exception:
        logger.exception("Tag search failed")
        abort(500, description="Failed to search by tag")

    memories: List[Dict[str, Any]] = []
    for row in getattr(result, "result_set", []) or []:
        data = _serialize_node(row[0])
        data["metadata"] = _parse_metadata_field(data.get("metadata"))
        memories.append(data)

    return jsonify(
        {"status": "success", "tags": tags, "count": len(memories), "memories": memories}
    )


@recall_bp.route("/recall", methods=["GET"])
def recall_memories() -> Any:
    query_start = time.perf_counter()
    query_text = (request.args.get("query") or "").strip()
    try:
        requested_limit = int(request.args.get("limit", 5))
    except (TypeError, ValueError):
        requested_limit = 5
    limit = max(1, min(requested_limit, RECALL_MAX_LIMIT))
    embedding_param = request.args.get("embedding")
    time_query = request.args.get("time_query") or request.args.get("time")
    start_param = request.args.get("start")
    end_param = request.args.get("end")
    tags_param = request.args.getlist("tags") or request.args.get("tags")

    tag_mode = (request.args.get("tag_mode") or "any").strip().lower()
    if tag_mode not in {"any", "all"}:
        tag_mode = "any"

    tag_match = (request.args.get("tag_match") or "prefix").strip().lower()
    if tag_match not in {"exact", "prefix"}:
        tag_match = "prefix"

    time_start, time_end = _parse_time_expression(time_query)

    start_time = time_start
    end_time = time_end

    if start_param:
        try:
            start_time = _normalize_timestamp(start_param)
        except ValueError as exc:
            abort(400, description=f"Invalid start time: {exc}")

    if end_param:
        try:
            end_time = _normalize_timestamp(end_param)
        except ValueError as exc:
            abort(400, description=f"Invalid end time: {exc}")

    tag_filters = _normalize_tag_list(tags_param)

    seen_ids: set[str] = set()
    graph = get_memory_graph()
    qdrant_client = get_qdrant_client()

    results: List[Dict[str, Any]] = []
    vector_matches: List[Dict[str, Any]] = []
    # Delegate implementation to recall blueprint module (kept for backward-compatibility)
    from automem.api.recall import handle_recall  # local import to avoid cycles

    response = handle_recall(
        get_memory_graph,
        get_qdrant_client,
        _normalize_tag_list,
        _normalize_timestamp,
        _parse_time_expression,
        _extract_keywords,
        _compute_metadata_score,
        _result_passes_filters,
        _graph_keyword_search,
        _vector_search,
        _vector_filter_only_tag_search,
        RECALL_MAX_LIMIT,
        logger,
        allowed_relations=ALLOWED_RELATIONS,
        relation_limit=RECALL_RELATION_LIMIT,
        expansion_limit_default=RECALL_EXPANSION_LIMIT,
    )

    # Emit SSE event for real-time monitoring
    elapsed_ms = int((time.perf_counter() - query_start) * 1000)
    result_count = 0
    try:
        # Response is either a tuple (response, status) or Response object
        resp_data = response[0] if isinstance(response, tuple) else response
        if hasattr(resp_data, "get_json"):
            data = resp_data.get_json(silent=True) or {}
            result_count = len(data.get("memories", []))
    except Exception as e:
        logger.debug("Failed to parse response for result_count", exc_info=e)

    emit_event(
        "memory.recall",
        {
            "query": query_text[:50] if query_text else "(no query)",
            "limit": limit,
            "result_count": result_count,
            "elapsed_ms": elapsed_ms,
            "tags": tag_filters[:3] if tag_filters else [],
        },
        utc_now,
    )

    return response


@memory_bp.route("/associate", methods=["POST"])
def create_association() -> Any:
    return _create_association_runtime(
        request_obj=request,
        coerce_importance_fn=_coerce_importance,
        get_memory_graph_fn=get_memory_graph,
        allowed_relations=ALLOWED_RELATIONS,
        relationship_types=RELATIONSHIP_TYPES,
        utc_now_fn=utc_now,
        abort_fn=abort,
        jsonify_fn=jsonify,
        logger=logger,
    )


@consolidation_bp.route("/consolidate", methods=["POST"])
def consolidate_memories() -> Any:
    return _consolidate_memories_runtime(
        request_obj=request,
        get_memory_graph_fn=get_memory_graph,
        init_consolidation_scheduler_fn=init_consolidation_scheduler,
        get_qdrant_client_fn=get_qdrant_client,
        memory_consolidator_cls=MemoryConsolidator,
        persist_consolidation_run_fn=_persist_consolidation_run,
        abort_fn=abort,
        jsonify_fn=jsonify,
        logger=logger,
    )


@consolidation_bp.route("/consolidate/status", methods=["GET"])
def consolidation_status() -> Any:
    return _consolidation_status_runtime(
        get_memory_graph_fn=get_memory_graph,
        init_consolidation_scheduler_fn=init_consolidation_scheduler,
        build_scheduler_from_graph_fn=_build_scheduler_from_graph,
        load_recent_runs_fn=_load_recent_runs,
        consolidation_history_limit=CONSOLIDATION_HISTORY_LIMIT,
        consolidation_tick_seconds=CONSOLIDATION_TICK_SECONDS,
        state=state,
        abort_fn=abort,
        jsonify_fn=jsonify,
        logger=logger,
    )


@recall_bp.route("/startup-recall", methods=["GET"])
def startup_recall() -> Any:
    return _startup_recall_runtime(
        get_memory_graph_fn=get_memory_graph,
        jsonify_fn=jsonify,
        abort_fn=abort,
        logger=logger,
    )


@recall_bp.route("/analyze", methods=["GET"])
def analyze_memories() -> Any:
    return _analyze_memories_runtime(
        get_memory_graph_fn=get_memory_graph,
        extract_entities_fn=extract_entities,
        utc_now_fn=utc_now,
        perf_counter_fn=time.perf_counter,
        jsonify_fn=jsonify,
        abort_fn=abort,
        logger=logger,
    )


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


@recall_bp.route("/memories/<memory_id>/related", methods=["GET"])
def get_related_memories(memory_id: str) -> Any:
    return _get_related_memories_runtime(
        memory_id=memory_id,
        request_args=request.args,
        get_memory_graph_fn=get_memory_graph,
        allowed_relations=ALLOWED_RELATIONS,
        relation_limit=RECALL_RELATION_LIMIT,
        serialize_node_fn=_serialize_node,
        logger=logger,
        abort_fn=abort,
        jsonify_fn=jsonify,
    )


from automem.api.admin import create_admin_blueprint_full
from automem.api.consolidation import create_consolidation_blueprint_full
from automem.api.enrichment import create_enrichment_blueprint
from automem.api.graph import create_graph_blueprint

# Register blueprints after all routes are defined
from automem.api.health import create_health_blueprint
from automem.api.memory import create_memory_blueprint_full
from automem.api.recall import create_recall_blueprint

health_bp = create_health_blueprint(
    get_memory_graph,
    get_qdrant_client,
    state,
    GRAPH_NAME,
    COLLECTION_NAME,
    utc_now,
)

enrichment_bp = create_enrichment_blueprint(
    _require_admin_token,
    state,
    enqueue_enrichment,
    ENRICHMENT_MAX_ATTEMPTS,
)

recall_bp = create_recall_blueprint(
    get_memory_graph,
    get_qdrant_client,
    _normalize_tag_list,
    _normalize_timestamp,
    _parse_time_expression,
    _extract_keywords,
    _compute_metadata_score,
    _result_passes_filters,
    _graph_keyword_search,
    _vector_search,
    _vector_filter_only_tag_search,
    RECALL_MAX_LIMIT,
    logger,
    ALLOWED_RELATIONS,
    RECALL_RELATION_LIMIT,
    _serialize_node,
    _summarize_relation_node,
    update_last_accessed,
    jit_enrich_fn=jit_enrich_lightweight if JIT_ENRICHMENT_ENABLED else None,
)

memory_bp = create_memory_blueprint_full(
    get_memory_graph,
    get_qdrant_client,
    _normalize_tags,
    _normalize_tag_list,
    _compute_tag_prefixes,
    _coerce_importance,
    _coerce_embedding,
    _normalize_timestamp,
    utc_now,
    _serialize_node,
    _parse_metadata_field,
    _generate_real_embedding,
    enqueue_enrichment,
    enqueue_embedding,
    lambda content: memory_classifier.classify(content),
    PointStruct,
    COLLECTION_NAME,
    ALLOWED_RELATIONS,
    RELATIONSHIP_TYPES,
    state,
    logger,
    update_last_accessed,
    get_openai_client,
)

admin_bp = create_admin_blueprint_full(
    _require_admin_token,
    init_openai,
    get_openai_client,
    get_qdrant_client,
    get_memory_graph,
    PointStruct,
    COLLECTION_NAME,
    lambda: state.effective_vector_size,  # Use runtime-detected dimension
    EMBEDDING_MODEL,
    utc_now,
    logger,
)

consolidation_bp = create_consolidation_blueprint_full(
    get_memory_graph,
    get_qdrant_client,
    _build_consolidator_from_config,
    _persist_consolidation_run,
    _build_scheduler_from_graph,
    _load_recent_runs,
    state,
    CONSOLIDATION_TICK_SECONDS,
    CONSOLIDATION_HISTORY_LIMIT,
    logger,
)

graph_bp = create_graph_blueprint(
    get_memory_graph,
    get_qdrant_client,
    _serialize_node,
    COLLECTION_NAME,
    logger,
)

stream_bp = create_stream_blueprint(
    require_api_token=require_api_token,
)

app.register_blueprint(health_bp)
app.register_blueprint(enrichment_bp)
app.register_blueprint(memory_bp)
app.register_blueprint(admin_bp)
app.register_blueprint(recall_bp)
app.register_blueprint(consolidation_bp)
app.register_blueprint(graph_bp)
app.register_blueprint(stream_bp)


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
