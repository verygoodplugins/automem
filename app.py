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
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from flask import Flask, abort, jsonify, request
from falkordb import FalkorDB
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from werkzeug.exceptions import HTTPException

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
ALLOWED_RELATIONS = {"RELATES_TO", "LEADS_TO", "OCCURRED_BEFORE"}
GRAPH_NAME = os.getenv("FALKORDB_GRAPH", "memories")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))


def utc_now() -> str:
    """Return an ISO formatted UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ServiceState:
    falkordb: Optional[FalkorDB] = None
    memory_graph: Any = None
    qdrant: Optional[QdrantClient] = None


state = ServiceState()


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

    try:
        embedding = _coerce_embedding(payload.get("embedding"))
    except ValueError as exc:
        abort(400, description=str(exc))
    qdrant_client = get_qdrant_client()
    embedding_status = "skipped"

    if embedding is None and qdrant_client is not None:
        embedding = _generate_placeholder_embedding(content)
        embedding_status = "placeholder"
    elif embedding is not None:
        embedding_status = "provided"

    graph = get_memory_graph()
    if graph is None:
        abort(503, description="FalkorDB is unavailable")

    created_at = utc_now()
    try:
        graph.query(
            """
            MERGE (m:Memory {id: $id})
            SET m.content = $content,
                m.timestamp = $timestamp,
                m.importance = $importance,
                m.tags = $tags
            RETURN m
            """,
            {
                "id": memory_id,
                "content": content,
                "timestamp": created_at,
                "importance": importance,
                "tags": tags,
            },
        )
    except Exception:  # pragma: no cover - log full stack trace in production
        logger.exception("Failed to persist memory in FalkorDB")
        abort(500, description="Failed to store memory in FalkorDB")

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
        "qdrant": qdrant_result or "unconfigured",
        "embedding_status": embedding_status,
    }
    return jsonify(response), 201


@app.route("/recall", methods=["GET"])
def recall_memories() -> Any:
    query_text = (request.args.get("query") or "").strip()
    limit = max(1, min(int(request.args.get("limit", 5)), 50))
    embedding_param = request.args.get("embedding")

    results: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    qdrant_client = get_qdrant_client()
    if qdrant_client is not None and embedding_param:
        try:
            embedding = _coerce_embedding(embedding_param)
        except ValueError as exc:
            abort(400, description=str(exc))
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
        except Exception:  # pragma: no cover - log full stack trace in production
            logger.exception("Qdrant search failed")

    graph = get_memory_graph()
    if graph is not None and query_text:
        try:
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
        except Exception:  # pragma: no cover - log full stack trace in production
            logger.exception("Graph text search failed")

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
        abort(400, description=f"Relation type must be one of {sorted(ALLOWED_RELATIONS)}")

    graph = get_memory_graph()
    if graph is None:
        abort(503, description="FalkorDB is unavailable")

    timestamp = utc_now()
    try:
        result = graph.query(
            f"""
            MATCH (m1:Memory {{id: $id1}})
            MATCH (m2:Memory {{id: $id2}})
            MERGE (m1)-[r:{relation_type}]->(m2)
            SET r.strength = $strength,
                r.updated_at = $updated_at
            RETURN r
            """,
            {
                "id1": memory1_id,
                "id2": memory2_id,
                "strength": strength,
                "updated_at": timestamp,
            },
        )
    except Exception:  # pragma: no cover - log full stack trace in production
        logger.exception("Failed to create association")
        abort(500, description="Failed to create association")

    if not result.result_set:
        abort(404, description="One or both memories do not exist")

    return (
        jsonify(
            {
                "status": "success",
                "message": f"Association created between {memory1_id} and {memory2_id}",
                "relation_type": relation_type,
                "strength": strength,
            }
        ),
        201,
    )


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
        raise ValueError("Embedding must be a list of floats or a comma-separated string")

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
    app.run(host="0.0.0.0", port=port, debug=False)
