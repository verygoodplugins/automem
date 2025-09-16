"""AutoMem Memory Service API.

Provides a small Flask API that stores memories in FalkorDB and Qdrant.
This module focuses on being resilient: it validates requests, handles
transient outages, and degrades gracefully when one of the backing services
is unavailable.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from threading import Thread
from queue import Queue
import time

import numpy as np
from dotenv import load_dotenv
from flask import Flask, abort, jsonify, request
from falkordb import FalkorDB
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from werkzeug.exceptions import HTTPException
from openai import OpenAI

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
    rng = np.random.default_rng(seed)
    return rng.random(VECTOR_SIZE).tolist()


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
    app.run(host="0.0.0.0", port=port, debug=False)
