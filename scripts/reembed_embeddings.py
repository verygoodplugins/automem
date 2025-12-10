#!/usr/bin/env python3
"""Re-embed existing memories and upsert vectors into Qdrant.

Usage:
    python scripts/reembed_embeddings.py [--batch-size 32] [--limit 0]

Environment:
    EMBEDDING_MODEL: OpenAI embedding model (default: text-embedding-3-large)
    VECTOR_SIZE: Embedding dimension (default: 3072; set to your current collection size before migrating)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv
from falkordb import FalkorDB
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from automem.config import EMBEDDING_MODEL, VECTOR_SIZE

logger = logging.getLogger("reembed")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout
)


def load_environment() -> None:
    load_dotenv()
    load_dotenv(Path.home() / ".config" / "automem" / ".env")


def get_graph() -> Any:
    host = (
        os.getenv("FALKORDB_HOST")
        or os.getenv("RAILWAY_PRIVATE_DOMAIN")
        or os.getenv("RAILWAY_PUBLIC_DOMAIN")
        or "localhost"
    )
    port = int(os.getenv("FALKORDB_PORT", "6379"))
    password = os.getenv("FALKORDB_PASSWORD")

    logger.info(
        "Connecting to FalkorDB at %s:%s (auth: %s)", host, port, "yes" if password else "no"
    )

    if password:
        db = FalkorDB(host=host, port=port, password=password, username="default")
    else:
        db = FalkorDB(host=host, port=port)

    graph_name = os.getenv("FALKORDB_GRAPH", "memories")
    logger.info("Using graph '%s'", graph_name)
    return db.select_graph(graph_name)


def get_qdrant_client() -> Optional[QdrantClient]:
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url:
        logger.error("QDRANT_URL is not configured; aborting re-embedding")
        return None
    logger.info("Connecting to Qdrant at %s", url)
    return QdrantClient(url=url, api_key=api_key)


def fetch_memories(graph: Any, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    query = """
        MATCH (m:Memory)
        RETURN m.id AS id,
               m.content AS content,
               m.tags AS tags,
               m.importance AS importance,
               m.timestamp AS timestamp,
               m.type AS type,
               m.confidence AS confidence,
               m.metadata AS metadata,
               m.updated_at AS updated_at,
               m.last_accessed AS last_accessed
        ORDER BY m.timestamp
    """
    params: Dict[str, Any] = {}
    if limit is not None and limit > 0:
        query += " LIMIT $limit"
        params["limit"] = limit

    result = graph.query(query, params)
    rows = getattr(result, "result_set", result)
    memories: List[Dict[str, Any]] = []
    for row in rows or []:
        memories.append(
            {
                "id": row[0],
                "content": row[1],
                "tags": row[2] or [],
                "importance": row[3] if row[3] is not None else 0.5,
                "timestamp": row[4],
                "type": row[5] or "Memory",
                "confidence": row[6] if row[6] is not None else 0.3,
                "metadata": row[7],
                "updated_at": row[8],
                "last_accessed": row[9],
            }
        )
    logger.info("Loaded %d memories from FalkorDB", len(memories))
    return memories


def parse_metadata(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            decoded = json.loads(raw)
            if isinstance(decoded, dict):
                return decoded
        except json.JSONDecodeError:
            logger.debug("Failed to parse metadata JSON for value: %s", raw)
    return {}


def chunked(iterable: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for idx in range(0, len(iterable), size):
        yield iterable[idx : idx + size]


def reembed_memories(memories: List[Dict[str, Any]], batch_size: int) -> None:
    client = OpenAI()
    qdrant = get_qdrant_client()
    if qdrant is None:
        raise SystemExit(1)

    collection = os.getenv("QDRANT_COLLECTION", "memories")
    vector_size = VECTOR_SIZE
    embedding_model = EMBEDDING_MODEL

    logger.info("Using embedding model: %s (dimension: %d)", embedding_model, vector_size)

    total = len(memories)
    processed = 0

    for batch in chunked(memories, batch_size):
        texts = [m["content"] or "" for m in batch]
        logger.info("Embedding batch %d-%d of %d", processed + 1, processed + len(batch), total)
        response = client.embeddings.create(
            model=embedding_model,
            input=texts,
            dimensions=vector_size,
        )
        points: List[PointStruct] = []
        for mem, data in zip(batch, response.data):
            vector = data.embedding
            payload = {
                "content": mem["content"],
                "tags": mem["tags"],
                "importance": mem["importance"],
                "timestamp": mem["timestamp"],
                "type": mem["type"],
                "confidence": mem["confidence"],
                "updated_at": mem["updated_at"],
                "last_accessed": mem["last_accessed"],
                "metadata": parse_metadata(mem["metadata"]),
            }
            points.append(PointStruct(id=mem["id"], vector=vector, payload=payload))
        qdrant.upsert(collection_name=collection, points=points)
        processed += len(batch)
    logger.info("Re-embedded %d memories", processed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-embed memories into Qdrant")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument(
        "--limit", type=int, default=0, help="Optional limit of memories to process"
    )
    args = parser.parse_args()

    load_environment()
    graph = get_graph()
    limit = args.limit if args.limit > 0 else None
    memories = fetch_memories(graph, limit=limit)
    if not memories:
        logger.info("No memories found")
        return
    reembed_memories(memories, batch_size=max(1, args.batch_size))


if __name__ == "__main__":
    main()
