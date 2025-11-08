#!/usr/bin/env python3
"""
Memory Consolidation Engine - Dream-inspired memory processing

Implements biological memory consolidation patterns:
- Exponential decay for aging memories
- Creative association discovery during "REM-like" processing
- Semantic clustering for memory compression
- Controlled forgetting with archival
"""

import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional, Protocol, Sequence, Set
from uuid import uuid4

try:  # pragma: no cover - optional dependency in tests
    from qdrant_client.http import models as qdrant_models
except ImportError:  # pragma: no cover - degraded mode when qdrant is absent
    qdrant_models = None

from automem.utils.time import _parse_iso_datetime

logger = logging.getLogger(__name__)


class GraphLike(Protocol):
    """Protocol describing the single method we rely on from FalkorDB graphs."""

    def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        ...


class VectorStoreProtocol(Protocol):
    """Minimal protocol for the vector store client used by the consolidator."""

    def delete(self, collection_name: str, points_selector: Any) -> Any:  # pragma: no cover - protocol definition
        ...


@dataclass
class MemoryRow:
    """Normalized representation of a memory row returned by the graph."""

    id: str
    content: str
    timestamp: Optional[str] = None
    importance: Optional[float] = None
    type: Optional[str] = None
    embeddings: Optional[Any] = None
    confidence: Optional[float] = None
    last_accessed: Optional[str] = None


def _load_embedding(raw: Any) -> Optional[List[float]]:
    """Convert stored embedding payloads into a list of floats."""

    if raw is None:
        return None

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("Invalid embedding JSON payload")
            return None

    if isinstance(raw, (list, tuple)):
        try:
            return [float(v) for v in raw]
        except (TypeError, ValueError):
            return None

    return None


def _cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors."""

    if len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = math.sqrt(sum(a * a for a in vec1))
    norm_b = math.sqrt(sum(b * b for b in vec2))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = dot_product / (norm_a * norm_b)
    return max(-1.0, min(1.0, similarity))


class MemoryConsolidator:
    """
    Consolidates memories through dream-inspired processes.

    Biological inspiration:
    - Sleep consolidates memories by strengthening important connections
    - Dreams create novel associations between disparate memories
    - Forgetting is controlled and serves learning
    """

    def __init__(self, graph: GraphLike, vector_store: Optional[VectorStoreProtocol] = None):
        self.graph = graph
        self.vector_store = vector_store
        self._graph_id = id(graph)  # Unique ID for cache invalidation

        # Decay parameters (tunable)
        self.base_decay_rate = 0.1  # Daily decay rate
        self.reinforcement_bonus = 0.2  # Strength added when accessed
        self.relationship_preservation = 0.3  # Extra weight for connected memories

        # Clustering parameters
        self.min_cluster_size = 3
        self.similarity_threshold = 0.75

        # Forgetting thresholds
        self.archive_threshold = 0.2  # Archive below this relevance
        self.delete_threshold = 0.05  # Delete below this (very old, unused)

    def _query_graph(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Sequence[Any]]:
        """Execute graph query and return the raw result set from FalkorDB."""

        params = params or {}
        try:
            result = self.graph.query(query, params)
        except Exception as exc:  # pragma: no cover - surface errors to caller
            logger.exception("Graph query failed: %s", exc)
            raise

        rows = getattr(result, "result_set", result)
        return list(rows or [])

    @lru_cache(maxsize=10000)
    def _get_relationship_count_cached_impl(self, memory_id: str, hour_key: int) -> int:
        """
        Implementation of relationship count query with caching.
        
        The hour_key parameter causes cache invalidation every hour,
        balancing freshness with performance (~80% query reduction).
        """
        relationship_query = """
            MATCH (m:Memory {id: $id})-[r]-(other:Memory)
            RETURN COUNT(DISTINCT r) as rel_count
        """
        rel_result = self._query_graph(relationship_query, {"id": memory_id})
        if rel_result and len(rel_result[0]) > 0 and rel_result[0][0] is not None:
            return int(rel_result[0][0])
        return 0
    
    def _get_relationship_count(self, memory_id: str) -> int:
        """Get relationship count for a memory with hourly cache invalidation."""
        hour_key = int(time.time() / 3600)  # Changes every hour
        try:
            return self._get_relationship_count_cached_impl(memory_id, hour_key)
        except Exception:
            logger.exception("Failed to get relationship count for %s", memory_id)
            return 0

    def calculate_relevance_score(
        self,
        memory: Dict[str, Any],
        current_time: datetime = None
    ) -> float:
        """
        Calculate relevance score using exponential decay.

        Factors:
        - Time decay (exponential)
        - Access frequency (reinforcement)
        - Relationship density (connections preserve memories)
        - Importance (explicit user signal)
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # Parse timestamps
        created_at = _parse_iso_datetime(memory.get('timestamp')) or current_time
        last_accessed = (
            _parse_iso_datetime(memory.get('last_accessed'))
            or _parse_iso_datetime(memory.get('timestamp'))
            or current_time
        )

        # Calculate age-based decay (hours-scale resolution keeps frequent runs meaningful)
        age_days = max(0.0, (current_time - created_at).total_seconds() / 86400)
        decay_factor = math.exp(-self.base_decay_rate * age_days)

        # Calculate access-based reinforcement using the same finer-grained clock
        access_recency_days = max(0.0, (current_time - last_accessed).total_seconds() / 86400)
        access_factor = 1.0 if access_recency_days < 1 else math.exp(-0.05 * access_recency_days)

        # Get relationship count for this memory (with caching for performance)
        rel_count = float(self._get_relationship_count(memory['id']))
        relationship_factor = 1.0 + (self.relationship_preservation * math.log1p(max(rel_count, 0)))

        # Importance factor (user-defined priority)
        importance = float(memory.get('importance', 0.5) or 0.0)

        # Confidence factor (well-classified memories are preserved)
        confidence = float(memory.get('confidence', 0.5) or 0.0)

        # Combined relevance score
        relevance = (
            decay_factor *
            (0.3 + 0.3 * access_factor) *  # Access contributes 30%
            relationship_factor *
            (0.5 + importance) *  # Importance scales from 0.5 to 1.5
            (0.7 + 0.3 * confidence)  # Confidence adds up to 30%
        )

        return min(1.0, relevance)  # Cap at 1.0

    def discover_creative_associations(
        self,
        sample_size: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Discover non-obvious connections between memories.

        Like REM sleep, randomly activates disparate memories
        and looks for hidden patterns or connections.
        """
        associations: List[Dict[str, Any]] = []

        sample_query = """
            MATCH (m:Memory)
            WHERE m.relevance_score > 0.3
            RETURN m.id as id, m.content as content, m.type as type,
                   m.embeddings as embeddings, m.timestamp as timestamp
            ORDER BY rand()
            LIMIT $limit
        """

        sample_rows = self._query_graph(sample_query, {"limit": sample_size})
        if len(sample_rows) < 2:
            return associations

        memories: List[MemoryRow] = []
        for row in sample_rows:
            timestamp = row[4] if len(row) > 4 else None
            memories.append(
                MemoryRow(
                    id=str(row[0]),
                    content=row[1] or "",
                    type=row[2] or "Memory",
                    embeddings=_load_embedding(row[3]),
                    timestamp=timestamp,
                )
            )

        for idx, mem1 in enumerate(memories):
            for mem2 in memories[idx + 1 :]:
                existing_check = """
                    MATCH (m1:Memory {id: $id1})-[r]-(m2:Memory {id: $id2})
                    RETURN COUNT(r) as count
                """
                existing_rows = self._query_graph(
                    existing_check,
                    {"id1": mem1.id, "id2": mem2.id},
                )
                if existing_rows and existing_rows[0] and existing_rows[0][0]:
                    continue

                similarity = 0.0
                if mem1.embeddings and mem2.embeddings:
                    try:
                        similarity = _cosine_similarity(mem1.embeddings, mem2.embeddings)
                        if math.isnan(similarity):
                            similarity = 0.0
                    except Exception:
                        logger.debug(
                            "Unable to compute similarity between %s and %s",
                            mem1.id,
                            mem2.id,
                        )
                        similarity = 0.0

                connection_type: Optional[str] = None
                confidence = 0.0

                if mem1.type == "Decision" and mem2.type == "Decision":
                    if similarity < 0.3:
                        connection_type = "CONTRASTS_WITH"
                        confidence = 0.6
                elif {mem1.type, mem2.type} == {"Insight", "Pattern"} and similarity > 0.5:
                    connection_type = "EXPLAINS"
                    confidence = 0.7
                elif similarity > 0.7 and (mem1.type or "") != (mem2.type or ""):
                    connection_type = "SHARES_THEME"
                    confidence = min(1.0, similarity)
                else:
                    t1 = _parse_iso_datetime(mem1.timestamp)
                    t2 = _parse_iso_datetime(mem2.timestamp)
                    if t1 and t2:
                        if abs((t1 - t2).days) < 7 and similarity < 0.4:
                            connection_type = "PARALLEL_CONTEXT"
                            confidence = 0.5

                if connection_type:
                    associations.append(
                        {
                            "memory1_id": mem1.id,
                            "memory2_id": mem2.id,
                            "type": connection_type,
                            "confidence": confidence,
                            "similarity": similarity,
                            "discovered_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )

        return associations

    def cluster_similar_memories(self) -> List[Dict[str, Any]]:
        """
        Cluster highly similar memories for potential compression.

        Uses DBSCAN clustering on embeddings to find groups of
        semantically similar memories that could be consolidated.
        """
        clusters = []

        # Get memories with embeddings
        embedding_query = """
            MATCH (m:Memory)
            WHERE m.embeddings IS NOT NULL
                AND m.relevance_score > 0.3
            RETURN m.id as id, m.content as content,
                   m.embeddings as embeddings, m.type as type
        """

        result = self._query_graph(embedding_query, {})
        if len(result) < self.min_cluster_size:
            return clusters

        # Extract embeddings
        memories: List[MemoryRow] = []
        embeddings: List[List[float]] = []
        for row in result:
            embedding = _load_embedding(row[2] if len(row) > 2 else None)
            if embedding is None:
                continue

            memories.append(
                MemoryRow(
                    id=str(row[0]),
                    content=row[1] or "",
                    embeddings=embedding,
                    type=row[3] if len(row) > 3 else "Memory",
                )
            )
            embeddings.append(embedding)

        if len(embeddings) < self.min_cluster_size:
            return clusters

        adjacency: Dict[int, Set[int]] = {idx: set() for idx in range(len(memories))}
        for i in range(len(memories)):
            for j in range(i + 1, len(memories)):
                similarity = _cosine_similarity(embeddings[i], embeddings[j])
                if similarity >= self.similarity_threshold:
                    adjacency[i].add(j)
                    adjacency[j].add(i)

        components: List[List[MemoryRow]] = []
        visited: Set[int] = set()

        for idx in range(len(memories)):
            if idx in visited:
                continue

            stack = [idx]
            component: List[int] = []

            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.append(current)
                stack.extend(adjacency[current] - visited)

            if len(component) >= self.min_cluster_size:
                components.append([memories[i] for i in sorted(component)])

        for cluster_mems in components:
            # Calculate cluster theme
            types = [m.type or 'Memory' for m in cluster_mems]
            dominant_type = max(set(types), key=types.count)

            # Find temporal range
            timestamps = []
            for m in cluster_mems:
                parsed = _parse_iso_datetime(m.timestamp)
                if parsed:
                    timestamps.append(parsed)

            if timestamps:
                time_span_days = (max(timestamps) - min(timestamps)).days
            else:
                time_span_days = 0

            clusters.append({
                'cluster_id': str(uuid4()),
                'memory_ids': [m.id for m in cluster_mems],
                'size': len(cluster_mems),
                'dominant_type': dominant_type,
                'time_span_days': time_span_days,
                'sample_content': cluster_mems[0].content[:100],
                'created_at': datetime.now(timezone.utc).isoformat()
            })

        return clusters

    def apply_controlled_forgetting(
        self,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Archive or delete low-relevance memories.

        Returns statistics about what was archived/deleted.
        """
        stats = {
            'examined': 0,
            'archived': [],
            'deleted': [],
            'preserved': 0
        }

        # Get all memories with scores
        all_memories_query = """
            MATCH (m:Memory)
            RETURN m.id as id, m.content as content,
                   m.relevance_score as score, m.timestamp as timestamp,
                   m.type as type, m.importance as importance,
                   m.last_accessed as last_accessed
        """

        result = self._query_graph(all_memories_query, {})
        current_time = datetime.now(timezone.utc)

        for row in result:
            stats['examined'] += 1

            # Result row order: id, content, score, timestamp, type, importance, last_accessed
            memory = {
                'id': row[0],
                'content': row[1],
                'relevance_score': row[2],
                'timestamp': row[3],
                'type': row[4],
                'importance': row[5],
                'last_accessed': row[6] if len(row) > 6 else None
            }

            # Calculate current relevance
            relevance = self.calculate_relevance_score(memory, current_time)

            # Determine fate
            if relevance < self.delete_threshold:
                stats['deleted'].append({
                    'id': memory['id'],
                    'content_preview': memory['content'][:50],
                    'relevance': relevance,
                    'type': memory.get('type', 'Memory')
                })

                if not dry_run:
                    # Delete from graph
                    delete_query = """
                        MATCH (m:Memory {id: $id})
                        DETACH DELETE m
                    """
                    self._query_graph(delete_query, {"id": memory['id']})

                    # Delete from vector store if present
                    if self.vector_store:
                        try:
                            if qdrant_models is not None:
                                selector = qdrant_models.PointIdsList(points=[memory['id']])
                            else:  # Fallback for dummy clients during tests
                                selector = {"points": [memory['id']]}

                            self.vector_store.delete(
                                collection_name="memories",
                                points_selector=selector,
                            )
                        except Exception:
                            logger.exception("Vector store deletion failed for %s", memory['id'])

            elif relevance < self.archive_threshold:
                stats['archived'].append({
                    'id': memory['id'],
                    'content_preview': memory['content'][:50],
                    'relevance': relevance,
                    'type': memory.get('type', 'Memory')
                })

                if not dry_run:
                    # Mark as archived (keep in graph but flag it)
                    archive_query = """
                        MATCH (m:Memory {id: $id})
                        SET m.archived = true,
                            m.archived_at = $archived_at,
                            m.relevance_score = $score
                    """
                    self._query_graph(archive_query, {
                        "id": memory['id'],
                        "archived_at": current_time.isoformat(),
                        "score": relevance
                    })
            else:
                stats['preserved'] += 1

                if not dry_run:
                    # Update relevance score
                    update_query = """
                        MATCH (m:Memory {id: $id})
                        SET m.relevance_score = $score
                    """
                    self._query_graph(update_query, {
                        "id": memory['id'],
                        "score": relevance
                    })

        return stats

    def consolidate(
        self,
        mode: str = 'full',
        dry_run: bool = True,
        decay_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run full consolidation cycle.

        Modes:
        - 'full': All consolidation steps
        - 'decay': Just update relevance scores
        - 'creative': Just discover associations
        - 'cluster': Just find clusters
        - 'forget': Just archive/delete
        """
        results = {
            'mode': mode,
            'dry_run': dry_run,
            'started_at': datetime.now(timezone.utc).isoformat(),
            'steps': {}
        }

        try:
            # Step 1: Update relevance scores (decay)
            if mode in ['full', 'decay']:
                logger.info("Applying exponential decay to memories...")
                threshold = decay_threshold if mode == 'decay' else None
                decay_stats = self._apply_decay(importance_threshold=threshold)
                results['steps']['decay'] = decay_stats

            # Step 2: Discover creative associations
            if mode in ['full', 'creative']:
                logger.info("Discovering creative associations...")
                associations = self.discover_creative_associations(sample_size=30)

                # Create the discovered relationships
                created = 0
                for assoc in associations:
                    if not dry_run:
                        create_query = """
                            MATCH (m1:Memory {id: $id1})
                            MATCH (m2:Memory {id: $id2})
                            CREATE (m1)-[r:DISCOVERED {
                                type: $type,
                                confidence: $confidence,
                                similarity: $similarity,
                                discovered_at: $discovered_at
                            }]->(m2)
                        """
                        try:
                            self.graph.query(create_query, {
                                "id1": assoc['memory1_id'],
                                "id2": assoc['memory2_id'],
                                "type": assoc['type'],
                                "confidence": assoc['confidence'],
                                "similarity": assoc['similarity'],
                                "discovered_at": assoc['discovered_at']
                            })
                            created += 1
                        except:
                            pass

                results['steps']['creative'] = {
                    'discovered': len(associations),
                    'created': created,
                    'sample_associations': associations[:3] if associations else []
                }

            # Step 3: Cluster similar memories
            if mode in ['full', 'cluster']:
                logger.info("Clustering similar memories...")
                clusters = self.cluster_similar_memories()

                # Create cluster meta-memories if significant
                meta_created = 0
                for cluster in clusters:
                    if cluster['size'] >= 5 and not dry_run:
                        # Create a meta-memory representing the cluster
                        meta_content = f"Meta-pattern: {cluster['dominant_type']} cluster with {cluster['size']} memories over {cluster['time_span_days']} days. Theme: {cluster['sample_content']}"

                        meta_query = """
                            CREATE (m:Memory:MetaMemory {
                                id: $id,
                                content: $content,
                                type: 'MetaPattern',
                                confidence: 0.8,
                                cluster_size: $size,
                                timestamp: $timestamp,
                                relevance_score: 0.9
                            })
                        """

                        self.graph.query(meta_query, {
                            "id": cluster['cluster_id'],
                            "content": meta_content,
                            "size": cluster['size'],
                            "timestamp": cluster['created_at']
                        })

                        # Link meta-memory to cluster members
                        for mem_id in cluster['memory_ids']:
                            link_query = """
                                MATCH (meta:MetaMemory {id: $meta_id})
                                MATCH (m:Memory {id: $mem_id})
                                CREATE (meta)-[:SUMMARIZES]->(m)
                            """
                            try:
                                self.graph.query(link_query, {
                                    "meta_id": cluster['cluster_id'],
                                    "mem_id": mem_id
                                })
                            except:
                                pass

                        meta_created += 1

                results['steps']['cluster'] = {
                    'clusters_found': len(clusters),
                    'meta_memories_created': meta_created,
                    'sample_clusters': clusters[:2] if clusters else []
                }

            # Step 4: Controlled forgetting
            if mode in ['full', 'forget']:
                logger.info("Applying controlled forgetting...")
                forget_stats = self.apply_controlled_forgetting(dry_run=dry_run)
                results['steps']['forget'] = forget_stats

            results['completed_at'] = datetime.now(timezone.utc).isoformat()
            results['success'] = True

        except Exception as e:
            logger.error(f"Consolidation error: {e}")
            results['error'] = str(e)
            results['success'] = False

        return results

    def _apply_decay(self, importance_threshold: Optional[float] = None) -> Dict[str, Any]:
        """Apply decay to all memories and return statistics."""
        stats = {
            'processed': 0,
            'avg_relevance_before': 0,
            'avg_relevance_after': 0,
            'distribution': {
                'high': 0,      # > 0.7
                'medium': 0,    # 0.3 - 0.7
                'low': 0,       # 0.1 - 0.3
                'archive': 0    # < 0.1
            }
        }

        filters = ["(m.archived IS NULL OR m.archived = false)"]
        params: Dict[str, Any] = {}

        if importance_threshold is not None:
            filters.append("m.importance IS NOT NULL AND m.importance >= $importance_threshold")
            params["importance_threshold"] = float(importance_threshold)

        where_clause = " AND ".join(filters)
        all_query = f"""
            MATCH (m:Memory)
            WHERE {where_clause}
            RETURN m.id as id, m.content as content,
                   m.timestamp as timestamp, m.importance as importance,
                   m.last_accessed as last_accessed,
                   m.relevance_score as old_score
        """

        memories = self._query_graph(all_query, params)
        if not memories:
            return stats

        total_before = 0
        total_after = 0

        for row in memories:
            stats['processed'] += 1

            # Result set contains lists where each row is a list of values
            # Order matches the RETURN clause: id, content, timestamp, importance, last_accessed, old_score
            memory = {
                'id': row[0] if len(row) > 0 else None,
                'content': row[1] if len(row) > 1 else None,
                'timestamp': row[2] if len(row) > 2 else None,
                'importance': row[3] if len(row) > 3 else None,
                'last_accessed': row[4] if len(row) > 4 else None,
                'old_score': row[5] if len(row) > 5 else None
            }

            # Previous score
            old_score = float(memory.get('old_score', 0.5)) if memory.get('old_score') else 0.5
            total_before += old_score

            # Calculate new score
            new_score = self.calculate_relevance_score(memory)
            total_after += new_score

            # Categorize
            if new_score > 0.7:
                stats['distribution']['high'] += 1
            elif new_score > 0.3:
                stats['distribution']['medium'] += 1
            elif new_score > 0.1:
                stats['distribution']['low'] += 1
            else:
                stats['distribution']['archive'] += 1

            # Update in graph
            update_query = """
                MATCH (m:Memory {id: $id})
                SET m.relevance_score = $score
            """
            self._query_graph(update_query, {
                "id": memory['id'],
                "score": new_score
            })

        if stats['processed'] > 0:
            stats['avg_relevance_before'] = total_before / stats['processed']
            stats['avg_relevance_after'] = total_after / stats['processed']

        return stats


class ConsolidationScheduler:
    """
    Schedules and manages periodic consolidation runs.

    Different consolidation frequencies for different operations:
    - Decay: Daily (quick, just updates scores)
    - Creative: Weekly (finds new associations)
    - Clustering: Monthly (reorganizes knowledge)
    - Forgetting: Quarterly (permanent changes)
    """

    def __init__(self, consolidator: MemoryConsolidator):
        self.consolidator = consolidator
        self.schedules = {
            'decay': {'interval': timedelta(days=1), 'last_run': None},
            'creative': {'interval': timedelta(days=7), 'last_run': None},
            'cluster': {'interval': timedelta(days=30), 'last_run': None},
            'forget': {'interval': timedelta(days=90), 'last_run': None}
        }
        self.history = []

    def should_run(self, task_type: str) -> bool:
        """Check if a task should run based on schedule."""
        if task_type not in self.schedules:
            return False

        schedule = self.schedules[task_type]
        if schedule['last_run'] is None:
            return True

        time_since = datetime.now(timezone.utc) - schedule['last_run']
        return time_since >= schedule['interval']

    def run_scheduled_tasks(self, force: Optional[str] = None, decay_threshold: Optional[float] = None) -> List[Dict]:
        """
        Run scheduled consolidation tasks.

        Args:
            force: Force run a specific task type regardless of schedule
            decay_threshold: Optional importance filter for decay runs
        """
        results = []

        tasks_to_run = []
        if force and force in self.schedules:
            tasks_to_run = [force]
        else:
            tasks_to_run = [t for t in self.schedules if self.should_run(t)]

        for task_type in tasks_to_run:
            logger.info(f"Running scheduled {task_type} consolidation...")

            # Run with appropriate mode
            if task_type == 'decay':
                result = self.consolidator.consolidate(
                    mode=task_type,
                    dry_run=False,
                    decay_threshold=decay_threshold,
                )
            else:
                result = self.consolidator.consolidate(
                    mode=task_type,
                    dry_run=False
                )

            # Update schedule
            self.schedules[task_type]['last_run'] = datetime.now(timezone.utc)

            # Record in history
            self.history.append({
                'task': task_type,
                'run_at': datetime.now(timezone.utc).isoformat(),
                'result': result
            })

            results.append(result)

        return results

    def get_next_runs(self) -> Dict[str, str]:
        """Get when each task will next run."""
        next_runs = {}

        for task_type, schedule in self.schedules.items():
            if schedule['last_run'] is None:
                next_runs[task_type] = "Due now"
            else:
                next_run = schedule['last_run'] + schedule['interval']
                if next_run <= datetime.now(timezone.utc):
                    next_runs[task_type] = "Due now"
                else:
                    time_until = next_run - datetime.now(timezone.utc)
                    days = time_until.days
                    hours = time_until.seconds // 3600

                    if days > 0:
                        next_runs[task_type] = f"In {days} day(s)"
                    else:
                        next_runs[task_type] = f"In {hours} hour(s)"

        return next_runs
