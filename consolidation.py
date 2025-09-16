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
import random
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

import numpy as np
from flask import current_app
from scipy.spatial.distance import cosine
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)


class MemoryConsolidator:
    """
    Consolidates memories through dream-inspired processes.

    Biological inspiration:
    - Sleep consolidates memories by strengthening important connections
    - Dreams create novel associations between disparate memories
    - Forgetting is controlled and serves learning
    """

    def __init__(self, graph: Any, vector_store: Any = None):
        self.graph = graph
        self.vector_store = vector_store

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

    def _query_graph(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute graph query and return result set."""
        result = self.graph.query(query, params or {})
        # Handle FalkorDB QueryResult object
        if hasattr(result, 'result_set'):
            return result.result_set
        return result if result else []

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
        timestamp_str = memory.get('timestamp') or current_time.isoformat()
        created_at = datetime.fromisoformat(
            timestamp_str.replace('Z', '+00:00') if timestamp_str else current_time.isoformat()
        )

        last_accessed_str = memory.get('last_accessed') or memory.get('timestamp') or current_time.isoformat()
        last_accessed = datetime.fromisoformat(
            last_accessed_str.replace('Z', '+00:00') if last_accessed_str else current_time.isoformat()
        )

        # Calculate age-based decay
        age_days = (current_time - created_at).days
        decay_factor = math.exp(-self.base_decay_rate * age_days)

        # Calculate access-based reinforcement
        access_recency_days = (current_time - last_accessed).days
        access_factor = 1.0 if access_recency_days < 1 else math.exp(-0.05 * access_recency_days)

        # Get relationship count for this memory
        relationship_query = """
            MATCH (m:Memory {id: $id})-[r]-(other:Memory)
            RETURN COUNT(DISTINCT r) as rel_count
        """
        result = self._query_graph(relationship_query, {"id": memory['id']})
        rel_count = result[0][0] if result else 0  # First column of first row
        relationship_factor = 1.0 + (self.relationship_preservation * math.log1p(rel_count))

        # Importance factor (user-defined priority)
        importance = float(memory.get('importance', 0.5))

        # Confidence factor (well-classified memories are preserved)
        confidence = float(memory.get('confidence', 0.5))

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
        associations = []

        # Sample random memories for creative processing
        sample_query = """
            MATCH (m:Memory)
            WHERE m.relevance_score > 0.3
            RETURN m.id as id, m.content as content, m.type as type,
                   m.embeddings as embeddings, m.timestamp as timestamp
            ORDER BY rand()
            LIMIT $limit
        """

        sample_result = self._query_graph(sample_query, {"limit": sample_size})
        if len(sample_result) < 2:
            return associations

        # Convert rows to dicts for easier access
        memories = []
        for row in sample_result:
            # Result row order: id, content, type, embeddings, timestamp
            memories.append({
                'id': row[0],
                'content': row[1],
                'type': row[2],
                'embeddings': row[3],
                'timestamp': row[4] if len(row) > 4 else None
            })

        # Look for creative connections
        for i, mem1 in enumerate(memories):
            for mem2 in memories[i+1:]:
                # Skip if already connected
                existing_check = """
                    MATCH (m1:Memory {id: $id1})-[r]-(m2:Memory {id: $id2})
                    RETURN COUNT(r) as count
                """
                existing = self._query_graph(
                    existing_check,
                    {"id1": mem1['id'], "id2": mem2['id']}
                )
                if existing and existing[0][0] > 0:  # First column of first row
                    continue

                # Calculate semantic similarity if embeddings exist
                similarity = 0
                if mem1.get('embeddings') and mem2.get('embeddings'):
                    try:
                        emb1 = json.loads(mem1['embeddings']) if isinstance(mem1['embeddings'], str) else mem1['embeddings']
                        emb2 = json.loads(mem2['embeddings']) if isinstance(mem2['embeddings'], str) else mem2['embeddings']
                        similarity = 1 - cosine(emb1, emb2)
                    except:
                        pass

                # Look for conceptual bridges
                connection_type = None
                confidence = 0

                # Pattern: Opposite decisions/preferences might be interesting
                if mem1.get('type') == 'Decision' and mem2.get('type') == 'Decision':
                    if similarity < 0.3:  # Low similarity = potentially opposite
                        connection_type = 'CONTRASTS_WITH'
                        confidence = 0.6

                # Pattern: Insights often explain patterns
                elif 'Insight' in [mem1.get('type'), mem2.get('type')] and \
                     'Pattern' in [mem1.get('type'), mem2.get('type')]:
                    if similarity > 0.5:
                        connection_type = 'EXPLAINS'
                        confidence = 0.7

                # Pattern: Similar contexts suggest common themes
                elif similarity > 0.7 and mem1.get('type') != mem2.get('type'):
                    connection_type = 'SHARES_THEME'
                    confidence = similarity

                # Pattern: Temporal proximity with different domains
                try:
                    t1 = datetime.fromisoformat(mem1.get('timestamp', '').replace('Z', '+00:00'))
                    t2 = datetime.fromisoformat(mem2.get('timestamp', '').replace('Z', '+00:00'))
                    time_diff_days = abs((t1 - t2).days)

                    if time_diff_days < 7 and similarity < 0.4:
                        # Different topics discussed in same week - might be parallel concerns
                        connection_type = 'PARALLEL_CONTEXT'
                        confidence = 0.5
                except:
                    pass

                if connection_type:
                    associations.append({
                        'memory1_id': mem1['id'],
                        'memory2_id': mem2['id'],
                        'type': connection_type,
                        'confidence': confidence,
                        'similarity': similarity,
                        'discovered_at': datetime.now(timezone.utc).isoformat()
                    })

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
        memories = []
        embeddings = []
        for row in result:
            # Result row order: id, content, embeddings, type
            mem = {
                'id': row[0],
                'content': row[1],
                'embeddings': row[2],
                'type': row[3]
            }
            try:
                emb = json.loads(mem['embeddings']) if isinstance(mem['embeddings'], str) else mem['embeddings']
                embeddings.append(emb)
                memories.append(mem)
            except:
                continue

        if len(embeddings) < self.min_cluster_size:
            return clusters

        # Perform clustering
        X = np.array(embeddings)
        clustering = DBSCAN(
            eps=(1 - self.similarity_threshold),  # Convert similarity to distance
            min_samples=self.min_cluster_size,
            metric='cosine'
        ).fit(X)

        # Group memories by cluster
        cluster_groups = defaultdict(list)
        for idx, label in enumerate(clustering.labels_):
            if label != -1:  # Skip noise points
                cluster_groups[label].append(memories[idx])

        # Create cluster summaries
        for cluster_id, cluster_mems in cluster_groups.items():
            # Calculate cluster theme
            types = [m.get('type', 'Memory') for m in cluster_mems]
            dominant_type = max(set(types), key=types.count)

            # Find temporal range
            timestamps = []
            for m in cluster_mems:
                try:
                    timestamps.append(
                        datetime.fromisoformat(m.get('timestamp', '').replace('Z', '+00:00'))
                    )
                except:
                    pass

            if timestamps:
                time_span_days = (max(timestamps) - min(timestamps)).days
            else:
                time_span_days = 0

            clusters.append({
                'cluster_id': str(uuid4()),
                'memory_ids': [m['id'] for m in cluster_mems],
                'size': len(cluster_mems),
                'dominant_type': dominant_type,
                'time_span_days': time_span_days,
                'sample_content': cluster_mems[0]['content'][:100],
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
                   m.type as type, m.importance as importance
        """

        result = self._query_graph(all_memories_query, {})
        current_time = datetime.now(timezone.utc)

        for row in result:
            stats['examined'] += 1

            # Result row order: id, content, score, timestamp, type, importance
            memory = {
                'id': row[0],
                'content': row[1],
                'relevance_score': row[2],
                'timestamp': row[3],
                'type': row[4],
                'importance': row[5]
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
                            self.vector_store.delete(
                                collection_name="memories",
                                points_selector={"filter": {"match": {"id": memory['id']}}}
                            )
                        except:
                            pass

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
        dry_run: bool = True
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
                decay_stats = self._apply_decay()
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

    def _apply_decay(self) -> Dict[str, Any]:
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

        # Get all memories
        all_query = """
            MATCH (m:Memory)
            WHERE m.archived IS NULL OR m.archived = false
            RETURN m.id as id, m.content as content,
                   m.timestamp as timestamp, m.importance as importance,
                   m.last_accessed as last_accessed,
                   m.relevance_score as old_score
        """

        memories = self._query_graph(all_query, {})
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

    def run_scheduled_tasks(self, force: Optional[str] = None) -> List[Dict]:
        """
        Run scheduled consolidation tasks.

        Args:
            force: Force run a specific task type regardless of schedule
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
            result = self.consolidator.consolidate(
                mode=task_type,
                dry_run=False  # Actually perform the operations
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