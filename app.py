"""AutoMem Memory Service API.

Provides a small Flask API that stores memories in FalkorDB and Qdrant.
This module focuses on being resilient: it validates requests, handles
transient outages, and degrades gracefully when one of the backing services
is unavailable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from threading import Thread, Event
from queue import Queue
import time

from dotenv import load_dotenv
from flask import Flask, abort, jsonify, request
from falkordb import FalkorDB
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from werkzeug.exceptions import HTTPException
from openai import OpenAI
from consolidation import MemoryConsolidator, ConsolidationScheduler

# Load environment variables before configuring the application.
load_dotenv()
load_dotenv(Path.home() / ".config" / "automem" / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("automem.api")

app = Flask(__name__)

# Configuration constants
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "memories")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE") or os.getenv("QDRANT_VECTOR_SIZE", "768"))
GRAPH_NAME = os.getenv("FALKORDB_GRAPH", "memories")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))

# Consolidation scheduling defaults (seconds unless noted)
CONSOLIDATION_TICK_SECONDS = int(os.getenv("CONSOLIDATION_TICK_SECONDS", "60"))
CONSOLIDATION_DECAY_INTERVAL_SECONDS = int(os.getenv("CONSOLIDATION_DECAY_INTERVAL_SECONDS", "300"))
CONSOLIDATION_CREATIVE_INTERVAL_SECONDS = int(os.getenv("CONSOLIDATION_CREATIVE_INTERVAL_SECONDS", str(3600)))
CONSOLIDATION_CLUSTER_INTERVAL_SECONDS = int(os.getenv("CONSOLIDATION_CLUSTER_INTERVAL_SECONDS", str(21600)))
CONSOLIDATION_FORGET_INTERVAL_SECONDS = int(os.getenv("CONSOLIDATION_FORGET_INTERVAL_SECONDS", str(86400)))
CONSOLIDATION_DECAY_IMPORTANCE_THRESHOLD = float(os.getenv("CONSOLIDATION_DECAY_IMPORTANCE_THRESHOLD", "0.7"))
CONSOLIDATION_HISTORY_LIMIT = int(os.getenv("CONSOLIDATION_HISTORY_LIMIT", "20"))
CONSOLIDATION_CONTROL_LABEL = "ConsolidationControl"
CONSOLIDATION_RUN_LABEL = "ConsolidationRun"
CONSOLIDATION_CONTROL_NODE_ID = os.getenv("CONSOLIDATION_CONTROL_NODE_ID", "global")
CONSOLIDATION_TASK_FIELDS = {
    "decay": "decay_last_run",
    "creative": "creative_last_run",
    "cluster": "cluster_last_run",
    "forget": "forget_last_run",
    "full": "full_last_run",
}

# Memory types for classification
MEMORY_TYPES = {
    "Decision", "Pattern", "Preference", "Style",
    "Habit", "Insight", "Context", "Memory"  # Memory is the default base type
}

# Enhanced relationship types with their properties
RELATIONSHIP_TYPES = {
    # Original relationships
    "RELATES_TO": {"description": "General relationship"},
    "LEADS_TO": {"description": "Causal relationship"},
    "OCCURRED_BEFORE": {"description": "Temporal relationship"},

    # New PKG relationships
    "PREFERS_OVER": {"description": "Preference relationship", "properties": ["context", "strength", "reason"]},
    "EXEMPLIFIES": {"description": "Pattern example", "properties": ["pattern_type", "confidence"]},
    "CONTRADICTS": {"description": "Conflicting information", "properties": ["resolution", "reason"]},
    "REINFORCES": {"description": "Strengthens pattern", "properties": ["strength", "observations"]},
    "INVALIDATED_BY": {"description": "Superseded information", "properties": ["reason", "timestamp"]},
    "EVOLVED_INTO": {"description": "Evolution of knowledge", "properties": ["confidence", "reason"]},
    "DERIVED_FROM": {"description": "Derived knowledge", "properties": ["transformation", "confidence"]},
    "PART_OF": {"description": "Hierarchical relationship", "properties": ["role", "context"]},
}

ALLOWED_RELATIONS = set(RELATIONSHIP_TYPES.keys())


def utc_now() -> str:
    """Return an ISO formatted UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse ISO strings that may end with Z into aware datetimes."""
    if not value:
        return None

    candidate = value.strip()
    if not candidate:
        return None

    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"

    try:
        return datetime.fromisoformat(candidate)
    except ValueError:
        return None


def _normalize_timestamp(raw: Any) -> str:
    """Validate and normalise an incoming timestamp string to UTC ISO format."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Timestamp must be a non-empty ISO formatted string")

    candidate = raw.strip()
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:  # pragma: no cover - validation path
        raise ValueError("Invalid ISO timestamp") from exc

    return parsed.astimezone(timezone.utc).isoformat()


class MemoryClassifier:
    """Classifies memories into specific types based on content patterns."""

    PATTERNS = {
        "Decision": [
            r"decided to", r"chose (\w+) over", r"going with", r"picked",
            r"selected", r"will use", r"choosing", r"opted for"
        ],
        "Pattern": [
            r"usually", r"typically", r"tend to", r"pattern i noticed",
            r"often", r"frequently", r"regularly", r"consistently"
        ],
        "Preference": [
            r"prefer", r"like.*better", r"favorite", r"always use",
            r"rather than", r"instead of", r"favor"
        ],
        "Style": [
            r"wrote.*in.*style", r"communicated", r"responded to",
            r"formatted as", r"using.*tone", r"expressed as"
        ],
        "Habit": [
            r"always", r"every time", r"habitually", r"routine",
            r"daily", r"weekly", r"monthly"
        ],
        "Insight": [
            r"realized", r"discovered", r"learned that", r"understood",
            r"figured out", r"insight", r"revelation"
        ],
        "Context": [
            r"during", r"while working on", r"in the context of",
            r"when", r"at the time", r"situation was"
        ],
    }

    def classify(self, content: str) -> tuple[str, float]:
        """
        Classify memory type and return confidence score.
        Returns: (type, confidence)
        """
        content_lower = content.lower()

        for memory_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    # Start with base confidence based on pattern match
                    confidence = 0.6

                    # Boost confidence for multiple pattern matches
                    matches = sum(1 for p in patterns if re.search(p, content_lower))
                    if matches > 1:
                        confidence = min(0.95, confidence + (matches * 0.1))

                    return memory_type, confidence

        # Default to base Memory type with lower confidence
        return "Memory", 0.3


memory_classifier = MemoryClassifier()


def extract_entities(content: str) -> Dict[str, List[str]]:
    """Extract entities from memory content."""
    entities = {
        "tools": [],
        "projects": [],
        "people": [],
        "concepts": [],
    }

    # Simple pattern-based extraction (can be enhanced with NER)
    tool_patterns = [
        r"(?:use|using|deploy|deployed|with|via)\s+([A-Z][a-zA-Z]+(?:DB|JS|\.js|\.py)?)",
        r"([A-Z][a-zA-Z]+)\s+(?:vs|versus|over|instead of)",
    ]

    for pattern in tool_patterns:
        matches = re.findall(pattern, content)
        entities["tools"].extend(matches)

    # Extract project names (words in quotes or capitalized phrases)
    project_patterns = [
        r'"([^"]+)"',  # Quoted strings
        r'`([^`]+)`',  # Backtick strings
        r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b',  # Capitalized phrases
    ]

    for pattern in project_patterns[:2]:  # Only quotes and backticks for projects
        matches = re.findall(pattern, content)
        entities["projects"].extend(matches)

    return entities


@dataclass
class ServiceState:
    falkordb: Optional[FalkorDB] = None
    memory_graph: Any = None
    qdrant: Optional[QdrantClient] = None
    openai_client: Optional[OpenAI] = None
    enrichment_queue: Optional[Queue] = None
    enrichment_thread: Optional[Thread] = None
    consolidation_thread: Optional[Thread] = None
    consolidation_stop_event: Optional[Event] = None


state = ServiceState()


def init_openai() -> None:
    """Initialize OpenAI client for embedding generation."""
    if state.openai_client is not None:
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OpenAI API key not provided; embeddings will be placeholders")
        return

    try:
        state.openai_client = OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized for embedding generation")
    except Exception:
        logger.exception("Failed to initialize OpenAI client")
        state.openai_client = None


def init_falkordb() -> None:
    """Initialize FalkorDB connection if not already connected."""
    if state.memory_graph is not None:
        return

    host = (
        os.getenv("FALKORDB_HOST")
        or os.getenv("RAILWAY_PRIVATE_DOMAIN")
        or os.getenv("RAILWAY_PUBLIC_DOMAIN")
        or "localhost"
    )

    try:
        logger.info("Connecting to FalkorDB at %s:%s", host, FALKORDB_PORT)
        state.falkordb = FalkorDB(host=host, port=FALKORDB_PORT)
        state.memory_graph = state.falkordb.select_graph(GRAPH_NAME)
        logger.info("FalkorDB connection established")
    except Exception:  # pragma: no cover - log full stack trace in production
        logger.exception("Failed to initialize FalkorDB connection")
        state.falkordb = None
        state.memory_graph = None


def init_qdrant() -> None:
    """Initialize Qdrant connection and ensure the collection exists."""
    if state.qdrant is not None:
        return

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url:
        logger.info("Qdrant URL not provided; skipping client initialization")
        return

    try:
        logger.info("Connecting to Qdrant at %s", url)
        state.qdrant = QdrantClient(url=url, api_key=api_key)
        _ensure_qdrant_collection()
        logger.info("Qdrant connection established")
    except Exception:  # pragma: no cover - log full stack trace in production
        logger.exception("Failed to initialize Qdrant client")
        state.qdrant = None


def _ensure_qdrant_collection() -> None:
    """Create the Qdrant collection if it does not already exist."""
    if state.qdrant is None:
        return

    try:
        collections = state.qdrant.get_collections()
        existing = {collection.name for collection in collections.collections}
        if COLLECTION_NAME not in existing:
            logger.info("Creating Qdrant collection '%s'", COLLECTION_NAME)
            state.qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
    except Exception:  # pragma: no cover - log full stack trace in production
        logger.exception("Failed to ensure Qdrant collection; disabling client")
        state.qdrant = None


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


def _load_control_record(graph: Any) -> Dict[str, Any]:
    """Fetch or create the consolidation control node."""
    try:
        result = graph.query(
            f"""
            MERGE (c:{CONSOLIDATION_CONTROL_LABEL} {{id: $id}})
            RETURN c
            """,
            {"id": CONSOLIDATION_CONTROL_NODE_ID},
        )
    except Exception:
        logger.exception("Failed to load consolidation control record")
        return {}

    if not getattr(result, "result_set", None):
        return {}

    node = result.result_set[0][0]
    properties = getattr(node, "properties", None)
    if isinstance(properties, dict):
        return dict(properties)
    if isinstance(node, dict):
        return dict(node)
    return {}


def _load_recent_runs(graph: Any, limit: int) -> List[Dict[str, Any]]:
    """Return recent consolidation run records."""
    try:
        result = graph.query(
            f"""
            MATCH (r:{CONSOLIDATION_RUN_LABEL})
            RETURN r
            ORDER BY r.started_at DESC
            LIMIT $limit
            """,
            {"limit": limit},
        )
    except Exception:
        logger.exception("Failed to load consolidation history")
        return []

    runs: List[Dict[str, Any]] = []
    for row in getattr(result, "result_set", []) or []:
        node = row[0]
        properties = getattr(node, "properties", None)
        if isinstance(properties, dict):
            runs.append(dict(properties))
        elif isinstance(node, dict):
            runs.append(dict(node))
    return runs


def _apply_scheduler_overrides(scheduler: ConsolidationScheduler) -> None:
    """Override default scheduler intervals using configuration."""
    overrides = {
        'decay': timedelta(seconds=CONSOLIDATION_DECAY_INTERVAL_SECONDS),
        'creative': timedelta(seconds=CONSOLIDATION_CREATIVE_INTERVAL_SECONDS),
        'cluster': timedelta(seconds=CONSOLIDATION_CLUSTER_INTERVAL_SECONDS),
        'forget': timedelta(seconds=CONSOLIDATION_FORGET_INTERVAL_SECONDS),
    }

    for task, interval in overrides.items():
        if task in scheduler.schedules:
            scheduler.schedules[task]['interval'] = interval


def _tasks_for_mode(mode: str) -> List[str]:
    """Map a consolidation mode to its task identifiers."""
    if mode == 'full':
        return ['decay', 'creative', 'cluster', 'forget', 'full']
    if mode in CONSOLIDATION_TASK_FIELDS:
        return [mode]
    return [mode]


def _persist_consolidation_run(graph: Any, result: Dict[str, Any]) -> None:
    """Record consolidation outcomes and update scheduler metadata."""
    mode = result.get('mode', 'unknown')
    completed_at = result.get('completed_at') or utc_now()
    started_at = result.get('started_at') or completed_at
    success = bool(result.get('success'))
    dry_run = bool(result.get('dry_run'))

    try:
        graph.query(
            f"""
            CREATE (r:{CONSOLIDATION_RUN_LABEL} {{
                id: $id,
                mode: $mode,
                task: $task,
                success: $success,
                dry_run: $dry_run,
                started_at: $started_at,
                completed_at: $completed_at,
                result: $result
            }})
            """,
            {
                "id": uuid.uuid4().hex,
                "mode": mode,
                "task": mode,
                "success": success,
                "dry_run": dry_run,
                "started_at": started_at,
                "completed_at": completed_at,
                "result": json.dumps(result, default=str),
            },
        )
    except Exception:
        logger.exception("Failed to record consolidation run history")

    for task in _tasks_for_mode(mode):
        field = CONSOLIDATION_TASK_FIELDS.get(task)
        if not field:
            continue
        try:
            graph.query(
                f"""
                MERGE (c:{CONSOLIDATION_CONTROL_LABEL} {{id: $id}})
                SET c.{field} = $timestamp
                """,
                {
                    "id": CONSOLIDATION_CONTROL_NODE_ID,
                    "timestamp": completed_at,
                },
            )
        except Exception:
            logger.exception("Failed to update consolidation control for task %s", task)

    try:
        graph.query(
            f"""
            MATCH (r:{CONSOLIDATION_RUN_LABEL})
            WITH r ORDER BY r.started_at DESC
            SKIP $keep
            DELETE r
            """,
            {"keep": CONSOLIDATION_HISTORY_LIMIT},
        )
    except Exception:
        logger.exception("Failed to prune consolidation history")


def _build_scheduler_from_graph(graph: Any) -> Optional[ConsolidationScheduler]:
    vector_store = get_qdrant_client()
    consolidator = MemoryConsolidator(graph, vector_store)
    scheduler = ConsolidationScheduler(consolidator)
    _apply_scheduler_overrides(scheduler)

    control = _load_control_record(graph)
    for task, field in CONSOLIDATION_TASK_FIELDS.items():
        iso_value = control.get(field)
        last_run = _parse_iso_datetime(iso_value)
        if last_run and task in scheduler.schedules:
            scheduler.schedules[task]['last_run'] = last_run

    return scheduler


def _run_consolidation_tick() -> None:
    graph = get_memory_graph()
    if graph is None:
        return

    scheduler = _build_scheduler_from_graph(graph)
    if scheduler is None:
        return

    try:
        results = scheduler.run_scheduled_tasks(
            decay_threshold=CONSOLIDATION_DECAY_IMPORTANCE_THRESHOLD
        )
        for result in results:
            _persist_consolidation_run(graph, result)
    except Exception:
        logger.exception("Consolidation scheduler tick failed")


def consolidation_worker() -> None:
    """Background loop that triggers consolidation tasks."""
    logger.info("Consolidation scheduler thread started")
    while state.consolidation_stop_event and not state.consolidation_stop_event.wait(CONSOLIDATION_TICK_SECONDS):
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
            if state.enrichment_queue and not state.enrichment_queue.empty():
                memory_id = state.enrichment_queue.get(timeout=1)
                enrich_memory(memory_id)
                state.enrichment_queue.task_done()
            else:
                time.sleep(2)  # Sleep when queue is empty
        except Exception:
            logger.exception("Error in enrichment worker")
            time.sleep(5)  # Back off on error


def enrich_memory(memory_id: str) -> None:
    """Enrich a memory with relationships, patterns, and entity extraction."""
    graph = get_memory_graph()
    if graph is None:
        return

    try:
        # Get the memory
        result = graph.query(
            "MATCH (m:Memory {id: $id}) RETURN m",
            {"id": memory_id}
        )

        if not result.result_set:
            return

        memory = result.result_set[0][0]
        content = memory.properties.get("content", "")

        # Extract entities (for future use)
        extract_entities(content)

        # Find related memories (temporal proximity)
        find_temporal_relationships(graph, memory_id)

        # Detect patterns
        detect_patterns(graph, memory_id, content)

        # Mark as processed
        graph.query(
            "MATCH (m:Memory {id: $id}) SET m.processed = true",
            {"id": memory_id}
        )

        logger.debug("Enriched memory %s", memory_id)

    except Exception:
        logger.exception("Failed to enrich memory %s", memory_id)


def find_temporal_relationships(graph: Any, memory_id: str) -> None:
    """Find and create temporal relationships with recent memories."""
    try:
        # Find memories from recent time window
        # Note: FalkorDB doesn't support datetime() function, use string comparison
        result = graph.query(
            """
            MATCH (m1:Memory {id: $id})
            MATCH (m2:Memory)
            WHERE m2.id <> $id
                AND m2.timestamp IS NOT NULL
                AND m1.timestamp IS NOT NULL
                AND m2.timestamp < m1.timestamp
            RETURN m2.id, m2.content
            ORDER BY m2.timestamp DESC
            LIMIT 5
            """,
            {"id": memory_id}
        )

        for related_id, related_content in result.result_set:
            # Create PRECEDED_BY relationship
            graph.query(
                """
                MATCH (m1:Memory {id: $id1})
                MATCH (m2:Memory {id: $id2})
                MERGE (m1)-[r:PRECEDED_BY {
                    created_at: $timestamp
                }]->(m2)
                """,
                {"id1": memory_id, "id2": related_id, "timestamp": utc_now()}
            )

    except Exception:
        logger.exception("Failed to find temporal relationships")


def detect_patterns(graph: Any, memory_id: str, content: str) -> None:
    """Detect if this memory exemplifies or creates patterns."""
    try:
        # Look for similar memories to detect patterns
        memory_type, confidence = memory_classifier.classify(content)

        # Search for similar memories of the same type
        result = graph.query(
            """
            MATCH (m:Memory)
            WHERE m.type = $type
                AND m.id <> $id
                AND m.confidence > 0.5
            RETURN m.id, m.content
            LIMIT 10
            """,
            {"type": memory_type, "id": memory_id}
        )

        similar_count = len(result.result_set)

        if similar_count >= 3:
            # We might have a pattern!
            pattern_id = f"pattern-{memory_type}-{uuid.uuid4().hex[:8]}"

            # Create or strengthen pattern
            graph.query(
                """
                MERGE (p:Pattern {type: $type})
                ON CREATE SET
                    p.id = $pattern_id,
                    p.content = $description,
                    p.confidence = $initial_confidence,
                    p.observations = 1,
                    p.created_at = $timestamp
                ON MATCH SET
                    p.confidence = CASE
                        WHEN p.confidence < 0.95 THEN p.confidence + 0.05
                        ELSE 0.95
                    END,
                    p.observations = p.observations + 1
                """,
                {
                    "type": memory_type,
                    "pattern_id": pattern_id,
                    "description": f"Pattern of {memory_type}",
                    "initial_confidence": 0.3,
                    "timestamp": utc_now()
                }
            )

            # Create EXEMPLIFIES relationship
            graph.query(
                """
                MATCH (m:Memory {id: $memory_id})
                MATCH (p:Pattern {type: $type})
                MERGE (m)-[r:EXEMPLIFIES {
                    confidence: $confidence,
                    created_at: $timestamp
                }]->(p)
                """,
                {
                    "type": memory_type,
                    "memory_id": memory_id,
                    "confidence": confidence,
                    "timestamp": utc_now()
                }
            )

    except Exception:
        logger.exception("Failed to detect patterns")


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


@app.route("/health", methods=["GET"])
def health() -> Any:
    graph_available = get_memory_graph() is not None
    qdrant_available = get_qdrant_client() is not None

    status = "healthy" if graph_available and qdrant_available else "degraded"
    health_data = {
        "status": status,
        "falkordb": "connected" if graph_available else "disconnected",
        "qdrant": "connected" if qdrant_available else "disconnected",
        "timestamp": utc_now(),
        "graph": GRAPH_NAME,
    }
    return jsonify(health_data)


@app.route("/memory", methods=["POST"])
def store_memory() -> Any:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        abort(400, description="JSON body is required")

    content = (payload.get("content") or "").strip()
    if not content:
        abort(400, description="'content' is required")

    tags = _normalize_tags(payload.get("tags"))
    importance = _coerce_importance(payload.get("importance"))
    memory_id = payload.get("id") or str(uuid.uuid4())

    metadata_raw = payload.get("metadata")
    if metadata_raw is None:
        metadata: Dict[str, Any] = {}
    elif isinstance(metadata_raw, dict):
        metadata = metadata_raw
    else:
        abort(400, description="'metadata' must be an object")

    # Classify the memory type
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
    qdrant_client = get_qdrant_client()
    embedding_status = "skipped"

    if embedding is None and qdrant_client is not None:
        # Generate real embedding using OpenAI API or fallback
        embedding = _generate_real_embedding(content)
        embedding_status = "generated" if state.openai_client else "placeholder"
    elif embedding is not None:
        embedding_status = "provided"

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
                "type": memory_type,
                "confidence": type_confidence,
                "t_valid": t_valid or created_at,
                "t_invalid": t_invalid,
                "updated_at": updated_at,
                "last_accessed": last_accessed,
                "metadata": metadata,
            },
        )
    except Exception:  # pragma: no cover - log full stack trace in production
        logger.exception("Failed to persist memory in FalkorDB")
        abort(500, description="Failed to store memory in FalkorDB")

    # Queue for enrichment
    if state.enrichment_queue is not None:
        state.enrichment_queue.put(memory_id)

    qdrant_result: Optional[str] = None
    if qdrant_client is not None and embedding is not None:
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

    response = {
        "status": "success",
        "memory_id": memory_id,
        "stored_at": created_at,
        "type": memory_type,
        "confidence": type_confidence,
        "qdrant": qdrant_result or "unconfigured",
        "embedding_status": embedding_status,
        "enrichment": "queued" if state.enrichment_queue else "disabled",
        "metadata": metadata,
        "timestamp": created_at,
        "updated_at": updated_at,
        "last_accessed": last_accessed,
    }
    return jsonify(response), 201


@app.route("/recall", methods=["GET"])
def recall_memories() -> Any:
    query_text = (request.args.get("query") or "").strip()
    limit = max(1, min(int(request.args.get("limit", 5)), 50))
    embedding_param = request.args.get("embedding")

    results: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    # Try vector search first
    qdrant_client = get_qdrant_client()
    if qdrant_client is not None:
        embedding = None

        # If no embedding provided but we have query text, generate one
        if not embedding_param and query_text:
            logger.debug("Generating embedding for query: %s", query_text)
            embedding = _generate_real_embedding(query_text)
        elif embedding_param:
            try:
                embedding = _coerce_embedding(embedding_param)
            except ValueError as exc:
                abort(400, description=str(exc))

        if embedding:
            try:
                vector_results = qdrant_client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=embedding,
                    limit=limit,
                )
                graph = get_memory_graph()
                for hit in vector_results:
                    memory_id = str(hit.id)
                    seen_ids.add(memory_id)
                    base = {
                        "id": memory_id,
                        "score": hit.score,
                        "source": "qdrant",
                        "memory": hit.payload,
                        "relations": [],
                    }
                    if graph is not None:
                        relations = _fetch_relations(graph, memory_id)
                        if relations:
                            base["relations"] = relations
                    results.append(base)
                logger.debug("Vector search returned %d results", len(vector_results))
            except Exception:  # pragma: no cover - log full stack trace in production
                logger.exception("Qdrant search failed")

    # Also do keyword search if we have query text
    graph = get_memory_graph()
    if graph is not None and query_text:
        try:
            # First try exact phrase match
            text_results = graph.query(
                """
                MATCH (m:Memory)
                WHERE toLower(m.content) CONTAINS toLower($query)
                RETURN m
                ORDER BY m.importance DESC
                LIMIT $limit
                """,
                {"query": query_text, "limit": limit},
            )

            # If no exact phrase matches, try word-by-word matching
            if len(text_results.result_set) == 0:
                # Split query into words for better matching
                query_words = query_text.lower().split()
                if len(query_words) > 1:
                    logger.debug(
                        "No exact match, trying word-by-word for: %s", query_words
                    )
                    # Create a Cypher query that matches all words
                    conditions = " AND ".join(
                        [
                            f"toLower(m.content) CONTAINS '{word}'"
                            for word in query_words
                        ]
                    )
                    text_results = graph.query(
                        f"""
                        MATCH (m:Memory)
                        WHERE {conditions}
                        RETURN m
                        ORDER BY m.importance DESC
                        LIMIT $limit
                        """,
                        {"limit": limit},
                    )

            for row in text_results.result_set:
                node = row[0]
                data = _serialize_node(node)
                memory_id = str(data.get("id"))
                if memory_id in seen_ids:
                    continue
                results.append(
                    {
                        "id": memory_id,
                        "score": None,
                        "source": "graph",
                        "memory": data,
                        "relations": _fetch_relations(graph, memory_id),
                    }
                )
            logger.debug(
                "Graph search returned %d new results",
                len([r for r in results if r["source"] == "graph"]),
            )
        except Exception:  # pragma: no cover - log full stack trace in production
            logger.exception("Graph text search failed")

    # Sort results by score (vector results first, then graph results)
    results.sort(key=lambda x: (x["score"] is None, -x["score"] if x["score"] else 0))

    return jsonify({"status": "success", "results": results, "count": len(results)})


@app.route("/associate", methods=["POST"])
def create_association() -> Any:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        abort(400, description="JSON body is required")

    memory1_id = (payload.get("memory1_id") or "").strip()
    memory2_id = (payload.get("memory2_id") or "").strip()
    relation_type = (payload.get("type") or "RELATES_TO").upper()
    strength = _coerce_importance(payload.get("strength", 0.5))

    if not memory1_id or not memory2_id:
        abort(400, description="'memory1_id' and 'memory2_id' are required")
    if memory1_id == memory2_id:
        abort(400, description="Cannot associate a memory with itself")
    if relation_type not in ALLOWED_RELATIONS:
        abort(
            400, description=f"Relation type must be one of {sorted(ALLOWED_RELATIONS)}"
        )

    graph = get_memory_graph()
    if graph is None:
        abort(503, description="FalkorDB is unavailable")

    timestamp = utc_now()

    # Build relationship properties based on type
    relationship_props = {
        "strength": strength,
        "updated_at": timestamp,
    }

    # Add type-specific properties if provided
    relation_config = RELATIONSHIP_TYPES.get(relation_type, {})
    if "properties" in relation_config:
        for prop in relation_config["properties"]:
            if prop in payload:
                relationship_props[prop] = payload[prop]

    # Build the SET clause dynamically
    set_clauses = [f"r.{key} = ${key}" for key in relationship_props]
    set_clause = ", ".join(set_clauses)

    try:
        result = graph.query(
            f"""
            MATCH (m1:Memory {{id: $id1}})
            MATCH (m2:Memory {{id: $id2}})
            MERGE (m1)-[r:{relation_type}]->(m2)
            SET {set_clause}
            RETURN r
            """,
            {
                "id1": memory1_id,
                "id2": memory2_id,
                **relationship_props,
            },
        )
    except Exception:  # pragma: no cover - log full stack trace in production
        logger.exception("Failed to create association")
        abort(500, description="Failed to create association")

    if not result.result_set:
        abort(404, description="One or both memories do not exist")

    response = {
        "status": "success",
        "message": f"Association created between {memory1_id} and {memory2_id}",
        "relation_type": relation_type,
        "strength": strength,
    }

    # Add additional properties to response
    for prop in relation_config.get("properties", []):
        if prop in relationship_props:
            response[prop] = relationship_props[prop]

    return jsonify(response), 201


@app.route("/consolidate", methods=["POST"])
def consolidate_memories() -> Any:
    """Run memory consolidation."""
    data = request.get_json() or {}
    mode = data.get('mode', 'full')
    dry_run = data.get('dry_run', True)

    graph = get_memory_graph()
    if graph is None:
        abort(503, description="FalkorDB is unavailable")

    init_consolidation_scheduler()

    try:
        vector_store = get_qdrant_client()
        consolidator = MemoryConsolidator(graph, vector_store)
        results = consolidator.consolidate(mode=mode, dry_run=dry_run)

        if not dry_run:
            _persist_consolidation_run(graph, results)

        return jsonify({
            "status": "success",
            "consolidation": results
        }), 200
    except Exception as e:
        logger.error(f"Consolidation failed: {e}")
        return jsonify({
            "error": "Consolidation failed",
            "details": str(e)
        }), 500


@app.route("/consolidate/status", methods=["GET"])
def consolidation_status() -> Any:
    """Get consolidation scheduler status."""
    graph = get_memory_graph()
    if graph is None:
        abort(503, description="FalkorDB is unavailable")

    try:
        init_consolidation_scheduler()
        scheduler = _build_scheduler_from_graph(graph)
        history = _load_recent_runs(graph, CONSOLIDATION_HISTORY_LIMIT)
        next_runs = scheduler.get_next_runs() if scheduler else {}

        return jsonify({
            "status": "success",
            "next_runs": next_runs,
            "history": history,
            "thread_alive": bool(state.consolidation_thread and state.consolidation_thread.is_alive()),
            "tick_seconds": CONSOLIDATION_TICK_SECONDS,
        }), 200
    except Exception as e:
        logger.error(f"Failed to get consolidation status: {e}")
        return jsonify({
            "error": "Failed to get status",
            "details": str(e)
        }), 500


@app.route("/startup-recall", methods=["GET"])
def startup_recall() -> Any:
    """Recall critical lessons at session startup."""
    graph = get_memory_graph()
    if graph is None:
        abort(503, description="FalkorDB is unavailable")

    try:
        # Search for critical lessons and system rules
        lesson_query = """
            MATCH (m:Memory)
            WHERE 'critical' IN m.tags OR 'lesson' IN m.tags OR 'ai-assistant' IN m.tags
            RETURN m.id as id, m.content as content, m.tags as tags,
                   m.importance as importance, m.type as type, m.metadata as metadata
            ORDER BY m.importance DESC
            LIMIT 10
        """

        lesson_results = graph.query(lesson_query)
        lessons = []

        if lesson_results.result_set:
            for row in lesson_results.result_set:
                lessons.append({
                    'id': row[0],
                    'content': row[1],
                    'tags': row[2] if row[2] else [],
                    'importance': row[3] if row[3] else 0.5,
                    'type': row[4] if row[4] else 'Memory',
                    'metadata': json.loads(row[5]) if row[5] else {}
                })

        # Get system rules
        system_query = """
            MATCH (m:Memory)
            WHERE 'system' IN m.tags OR 'memory-recall' IN m.tags
            RETURN m.id as id, m.content as content, m.tags as tags
            LIMIT 5
        """

        system_results = graph.query(system_query)
        system_rules = []

        if system_results.result_set:
            for row in system_results.result_set:
                system_rules.append({
                    'id': row[0],
                    'content': row[1],
                    'tags': row[2] if row[2] else []
                })

        response = {
            'status': 'success',
            'critical_lessons': lessons,
            'system_rules': system_rules,
            'lesson_count': len(lessons),
            'has_critical': any(l.get('importance', 0) >= 0.9 for l in lessons),
            'summary': f"Recalled {len(lessons)} lesson(s) and {len(system_rules)} system rule(s)"
        }

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Startup recall failed: {e}")
        return jsonify({
            "error": "Startup recall failed",
            "details": str(e)
        }), 500


@app.route("/analyze", methods=["GET"])
def analyze_memories() -> Any:
    """Analyze memory patterns, preferences, and insights."""
    graph = get_memory_graph()
    if graph is None:
        abort(503, description="FalkorDB is unavailable")

    analytics = {
        "memory_types": {},
        "patterns": [],
        "preferences": [],
        "temporal_insights": {},
        "entity_frequency": {},
        "confidence_distribution": {},
    }

    try:
        # Analyze memory type distribution
        type_result = graph.query(
            """
            MATCH (m:Memory)
            WHERE m.type IS NOT NULL
            RETURN m.type, COUNT(m) as count, AVG(m.confidence) as avg_confidence
            ORDER BY count DESC
            """
        )

        for mem_type, count, avg_conf in type_result.result_set:
            analytics["memory_types"][mem_type] = {
                "count": count,
                "average_confidence": round(avg_conf, 3) if avg_conf else 0,
            }

        # Find patterns with high confidence
        pattern_result = graph.query(
            """
            MATCH (p:Pattern)
            WHERE p.confidence > 0.6
            RETURN p.type, p.content, p.confidence, p.observations
            ORDER BY p.confidence DESC
            LIMIT 10
            """
        )

        for p_type, content, confidence, observations in pattern_result.result_set:
            analytics["patterns"].append({
                "type": p_type,
                "description": content,
                "confidence": round(confidence, 3) if confidence else 0,
                "observations": observations or 0,
            })

        # Find preferences (PREFERS_OVER relationships)
        pref_result = graph.query(
            """
            MATCH (m1:Memory)-[r:PREFERS_OVER]->(m2:Memory)
            RETURN m1.content, m2.content, r.context, r.strength
            ORDER BY r.strength DESC
            LIMIT 10
            """
        )

        for preferred, over, context, strength in pref_result.result_set:
            analytics["preferences"].append({
                "prefers": preferred,
                "over": over,
                "context": context,
                "strength": round(strength, 3) if strength else 0,
            })

        # Temporal insights - simplified for FalkorDB compatibility
        try:
            temporal_result = graph.query(
                """
                MATCH (m:Memory)
                WHERE m.timestamp IS NOT NULL
                RETURN m.timestamp, m.importance
                LIMIT 100
                """
            )

            # Process temporal data in Python
            from collections import defaultdict
            hour_data = defaultdict(lambda: {"count": 0, "total_importance": 0})

            for timestamp, importance in temporal_result.result_set:
                if timestamp and len(timestamp) > 13:
                    # Extract hour from timestamp string
                    hour_str = timestamp[11:13]
                    if hour_str.isdigit():
                        hour = int(hour_str)
                        hour_data[hour]["count"] += 1
                        hour_data[hour]["total_importance"] += importance or 0.5

            # Calculate averages
            for hour, data in hour_data.items():
                if data["count"] > 0:
                    analytics["temporal_insights"][f"hour_{hour:02d}"] = {
                        "count": data["count"],
                        "avg_importance": round(data["total_importance"] / data["count"], 3)
                    }
        except Exception:
            # Skip temporal insights if query fails
            pass

        # Confidence distribution
        conf_result = graph.query(
            """
            MATCH (m:Memory)
            WHERE m.confidence IS NOT NULL
            RETURN
                CASE
                    WHEN m.confidence < 0.3 THEN 'low'
                    WHEN m.confidence < 0.7 THEN 'medium'
                    ELSE 'high'
                END as level,
                COUNT(m) as count
            """
        )

        for level, count in conf_result.result_set:
            analytics["confidence_distribution"][level] = count

        # Entity extraction insights (top mentioned tools/projects)
        entity_result = graph.query(
            """
            MATCH (m:Memory)
            WHERE m.content IS NOT NULL
            RETURN m.content
            LIMIT 100
            """
        )

        entity_counts: Dict[str, Dict[str, int]] = {
            "tools": {},
            "projects": {},
        }

        for (content,) in entity_result.result_set:
            entities = extract_entities(content)
            for tool in entities.get("tools", []):
                entity_counts["tools"][tool] = entity_counts["tools"].get(tool, 0) + 1
            for project in entities.get("projects", []):
                entity_counts["projects"][project] = entity_counts["projects"].get(project, 0) + 1

        # Top 5 most mentioned
        analytics["entity_frequency"]["tools"] = sorted(
            entity_counts["tools"].items(), key=lambda x: x[1], reverse=True
        )[:5]
        analytics["entity_frequency"]["projects"] = sorted(
            entity_counts["projects"].items(), key=lambda x: x[1], reverse=True
        )[:5]

    except Exception:
        logger.exception("Failed to generate analytics")
        abort(500, description="Failed to generate analytics")

    return jsonify({
        "status": "success",
        "analytics": analytics,
        "generated_at": utc_now(),
    })


def _normalize_tags(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(tag, str) for tag in value):
        return value
    abort(400, description="'tags' must be a list of strings or a single string")


def _coerce_importance(value: Any) -> float:
    if value is None:
        return 0.5
    try:
        score = float(value)
    except (TypeError, ValueError):
        abort(400, description="'importance' must be a number")
    if score < 0 or score > 1:
        abort(400, description="'importance' must be between 0 and 1")
    return score


def _coerce_embedding(value: Any) -> Optional[List[float]]:
    if value is None or value == "":
        return None
    vector: List[Any]
    if isinstance(value, list):
        vector = value
    elif isinstance(value, str):
        vector = [part.strip() for part in value.split(",") if part.strip()]
    else:
        raise ValueError(
            "Embedding must be a list of floats or a comma-separated string"
        )

    if len(vector) != VECTOR_SIZE:
        raise ValueError(f"Embedding must contain exactly {VECTOR_SIZE} values")

    try:
        return [float(component) for component in vector]
    except ValueError as exc:
        raise ValueError("Embedding must contain numeric values") from exc


def _generate_placeholder_embedding(content: str) -> List[float]:
    """Generate a deterministic embedding vector from the content."""
    digest = hashlib.sha256(content.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = random.Random(seed)
    return [rng.random() for _ in range(VECTOR_SIZE)]


def _generate_real_embedding(content: str) -> List[float]:
    """Generate a real semantic embedding using OpenAI's API."""
    init_openai()

    if state.openai_client is None:
        logger.debug("OpenAI client not available, falling back to placeholder")
        return _generate_placeholder_embedding(content)

    try:
        # Use the smaller, cheaper model for embeddings
        response = state.openai_client.embeddings.create(
            input=content,
            model="text-embedding-3-small",
            dimensions=VECTOR_SIZE,  # OpenAI allows specifying dimensions
        )
        embedding = response.data[0].embedding
        logger.debug(
            "Generated OpenAI embedding for content (length: %d)", len(content)
        )
        return embedding
    except Exception as e:
        logger.warning("Failed to generate OpenAI embedding: %s", str(e))
        return _generate_placeholder_embedding(content)


def _serialize_node(node: Any) -> Dict[str, Any]:
    properties = getattr(node, "properties", None)
    if isinstance(properties, dict):
        return dict(properties)
    # When FalkorDB returns plain dictionaries already
    if isinstance(node, dict):
        return dict(node)
    return {"value": node}


def _fetch_relations(graph: Any, memory_id: str) -> List[Dict[str, Any]]:
    try:
        records = graph.query(
            """
            MATCH (m:Memory {id: $id})-[r]->(related:Memory)
            RETURN type(r) as relation_type, r.strength as strength, related
            """,
            {"id": memory_id},
        )
    except Exception:  # pragma: no cover - log full stack trace in production
        logger.exception("Failed to fetch relations for memory %s", memory_id)
        return []

    connections: List[Dict[str, Any]] = []
    for relation_type, strength, related in records.result_set:
        connections.append(
            {
                "type": relation_type,
                "strength": strength,
                "memory": _serialize_node(related),
            }
        )
    return connections


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8001"))
    logger.info("Starting Flask API on port %s", port)
    init_falkordb()
    init_qdrant()
    init_openai()
    init_enrichment_pipeline()
    init_consolidation_scheduler()
    app.run(host="0.0.0.0", port=port, debug=False)
