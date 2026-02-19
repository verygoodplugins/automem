from __future__ import annotations

import re
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Set, Tuple


def temporal_cutoff() -> str:
    """Return an ISO timestamp 7 days ago to bound temporal queries."""
    return (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()


def find_temporal_relationships(
    *,
    graph: Any,
    memory_id: str,
    limit: int,
    cutoff_fn: Callable[[], str],
    utc_now_fn: Callable[[], str],
    logger: Any,
) -> int:
    """Find and create temporal relationships with recent memories."""
    created = 0
    try:
        result = graph.query(
            """
            MATCH (m1:Memory {id: $id})
            WITH m1, m1.timestamp AS ts
            WHERE ts IS NOT NULL
            MATCH (m2:Memory)
            WHERE m2.id <> $id
                AND m2.timestamp IS NOT NULL
                AND m2.timestamp < ts
                AND m2.timestamp > $cutoff
            RETURN m2.id
            ORDER BY m2.timestamp DESC
            LIMIT $limit
            """,
            {"id": memory_id, "limit": limit, "cutoff": cutoff_fn()},
            timeout=5000,
        )

        timestamp = utc_now_fn()
        for (related_id,) in result.result_set:
            if not related_id:
                continue
            graph.query(
                """
                MATCH (m1:Memory {id: $id1})
                MATCH (m2:Memory {id: $id2})
                MERGE (m1)-[r:PRECEDED_BY]->(m2)
                SET r.updated_at = $timestamp,
                    r.count = COALESCE(r.count, 0) + 1
                """,
                {"id1": memory_id, "id2": related_id, "timestamp": timestamp},
            )
            created += 1
    except Exception:
        logger.exception("Failed to find temporal relationships")

    return created


def detect_patterns(
    *,
    graph: Any,
    memory_id: str,
    content: str,
    classify_fn: Callable[[str], Tuple[str, float]],
    search_stopwords: Set[str],
    utc_now_fn: Callable[[], str],
    logger: Any,
) -> List[Dict[str, Any]]:
    """Detect if this memory exemplifies or creates patterns."""
    detected: List[Dict[str, Any]] = []

    try:
        memory_type, confidence = classify_fn(content)
        result = graph.query(
            """
            MATCH (m:Memory)
            WHERE m.type = $type
                AND m.id <> $id
                AND m.confidence > 0.5
            RETURN m.id, m.content
            LIMIT 10
            """,
            {"type": memory_type, "id": memory_id},
        )

        similar_texts = [content]
        similar_texts.extend(row[1] for row in result.result_set if len(row) > 1)
        similar_count = len(result.result_set)

        if similar_count >= 3:
            tokens = Counter()
            for text in similar_texts:
                for token in re.findall(r"[a-zA-Z]{4,}", (text or "").lower()):
                    if token in search_stopwords:
                        continue
                    tokens[token] += 1

            top_terms = [term for term, _ in tokens.most_common(5)]
            pattern_id = f"pattern-{memory_type}-{uuid.uuid4().hex[:8]}"
            description = f"Pattern across {similar_count + 1} {memory_type} memories" + (
                f" highlighting {', '.join(top_terms)}" if top_terms else ""
            )

            graph.query(
                """
                MERGE (p:Pattern {type: $type})
                ON CREATE SET
                    p.id = $pattern_id,
                    p.content = $description,
                    p.confidence = $initial_confidence,
                    p.observations = 1,
                    p.key_terms = $key_terms,
                    p.created_at = $timestamp
                ON MATCH SET
                    p.confidence = CASE
                        WHEN p.confidence < 0.95 THEN p.confidence + 0.05
                        ELSE 0.95
                    END,
                    p.observations = p.observations + 1,
                    p.key_terms = $key_terms,
                    p.updated_at = $timestamp
                """,
                {
                    "type": memory_type,
                    "pattern_id": pattern_id,
                    "description": description,
                    "initial_confidence": 0.35,
                    "key_terms": top_terms,
                    "timestamp": utc_now_fn(),
                },
            )

            graph.query(
                """
                MATCH (m:Memory {id: $memory_id})
                MATCH (p:Pattern {type: $type})
                MERGE (m)-[r:EXEMPLIFIES]->(p)
                SET r.confidence = $confidence,
                    r.updated_at = $timestamp
                """,
                {
                    "type": memory_type,
                    "memory_id": memory_id,
                    "confidence": confidence,
                    "timestamp": utc_now_fn(),
                },
            )

            detected.append(
                {
                    "type": memory_type,
                    "similar_memories": similar_count,
                    "key_terms": top_terms,
                }
            )
    except Exception:
        logger.exception("Failed to detect patterns")

    return detected


def link_semantic_neighbors(
    *,
    graph: Any,
    memory_id: str,
    get_qdrant_client_fn: Callable[[], Any],
    collection_name: str,
    similarity_limit: int,
    similarity_threshold: float,
    utc_now_fn: Callable[[], str],
    logger: Any,
) -> List[Tuple[str, float]]:
    client = get_qdrant_client_fn()
    if client is None:
        return []

    try:
        points = client.retrieve(
            collection_name=collection_name,
            ids=[memory_id],
            with_vectors=True,
            with_payload=False,
        )
    except Exception:
        logger.exception("Failed to fetch vector for memory %s", memory_id)
        return []

    if not points or getattr(points[0], "vector", None) is None:
        return []

    query_vector = points[0].vector

    try:
        neighbors = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=similarity_limit + 1,
            with_payload=False,
        )
    except Exception:
        logger.exception("Semantic neighbor search failed for %s", memory_id)
        return []

    created: List[Tuple[str, float]] = []
    timestamp = utc_now_fn()

    for neighbour in neighbors:
        neighbour_id = str(neighbour.id)
        if neighbour_id == memory_id:
            continue

        score = float(neighbour.score or 0.0)
        if score < similarity_threshold:
            continue

        params = {
            "id1": memory_id,
            "id2": neighbour_id,
            "score": score,
            "timestamp": timestamp,
        }

        graph.query(
            """
            MATCH (a:Memory {id: $id1})
            MATCH (b:Memory {id: $id2})
            MERGE (a)-[r:SIMILAR_TO]->(b)
            SET r.score = $score,
                r.updated_at = $timestamp
            """,
            params,
        )

        graph.query(
            """
            MATCH (a:Memory {id: $id1})
            MATCH (b:Memory {id: $id2})
            MERGE (b)-[r:SIMILAR_TO]->(a)
            SET r.score = $score,
                r.updated_at = $timestamp
            """,
            params,
        )

        created.append((neighbour_id, score))

    return created
