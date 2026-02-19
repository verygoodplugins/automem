from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List


class FakeResult:
    def __init__(self, rows: List[List[Any]]) -> None:
        self.result_set = rows


class FakeNode:
    def __init__(self, properties: Dict[str, Any]) -> None:
        self.properties = properties


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _returns_whole_memory_node(query: str) -> bool:
    # Match `RETURN m` but not `RETURN m.id`, `RETURN m.content`, etc.
    return re.search(r"\bRETURN\s+m\b(?![\w.])", query) is not None


class FakeGraph:
    """Shared fake FalkorDB graph used across unit tests.

    This merges behaviors that previously lived in DummyGraph, MockGraph,
    and multiple FakeGraph variants.
    """

    def __init__(self, *, seed_enrichment_fixture: bool = False) -> None:
        self.queries: List[tuple[str, Dict[str, Any]]] = []

        # Generic memory storage + associations
        self.memories: Dict[str, Dict[str, Any]] = {}
        self.nodes: set[str] = set()
        self.relationships: List[Dict[str, Any]] = []

        # Enrichment tracking
        self.temporal_calls: List[Dict[str, Any]] = []
        self.pattern_calls: List[Dict[str, Any]] = []
        self.exemplifies_calls: List[Dict[str, Any]] = []
        self.update_calls: List[Dict[str, Any]] = []
        self.temporal_related_ids: List[str] = []
        self.pattern_source_rows: List[List[Any]] = []

        # Consolidation fixtures + tracking
        self.relationship_counts: Dict[str, int] = {}
        self.sample_rows: List[List[Any]] = []
        self.existing_pairs: set[frozenset[str]] = set()
        self.cluster_rows: List[List[Any]] = []
        self.decay_rows: List[List[Any]] = []
        self.forgetting_rows: List[List[Any]] = []
        self.deleted: List[str] = []
        self.archived: List[tuple[str, float]] = []
        self.updated_scores: List[tuple[str, float]] = []

        if seed_enrichment_fixture:
            self.memories["mem-1"] = {
                "id": "mem-1",
                "content": 'Met with Alice about SuperWhisper deployment on project "Launchpad".',
                "tags": ["meeting"],
                "metadata": {},
                "processed": False,
                "summary": None,
                "importance": 0.5,
                "confidence": 0.7,
                "type": "Memory",
                "timestamp": _utc_now(),
            }
            self.temporal_related_ids = ["mem-older"]
            self.pattern_source_rows = [
                ["mem-a", "Pattern insight about automation"],
                ["mem-b", "Another automation pattern emerges"],
                ["mem-c", "Automation habit noted"],
            ]

    def query(self, query: str, params: Dict[str, Any] | None = None, **kwargs: Any) -> FakeResult:
        del kwargs
        params = params or {}
        self.queries.append((query, params))

        # Consolidation engine query patterns
        if "COUNT(DISTINCT r)" in query:
            memory_id = params.get("id")
            return FakeResult([[self.relationship_counts.get(memory_id, 0)]])

        if "RETURN COUNT(r) as count" in query and "$id1" in query:
            key = frozenset((params["id1"], params["id2"]))
            return FakeResult([[1 if key in self.existing_pairs else 0]])

        if "ORDER BY rand()" in query and "LIMIT $limit" in query:
            limit = params.get("limit")
            rows = self.sample_rows if limit is None else self.sample_rows[:limit]
            return FakeResult(rows)

        if "WHERE m.embeddings IS NOT NULL" in query:
            return FakeResult(self.cluster_rows)

        if "m.relevance_score as old_score" in query:
            return FakeResult(self.decay_rows)

        if "m.relevance_score as score" in query and "m.last_accessed as last_accessed" in query:
            return FakeResult(self.forgetting_rows)

        if "SET m.archived = true" in query:
            memory_id = str(params.get("id") or "")
            score = float(params.get("score") or 0.0)
            self.archived.append((memory_id, score))
            return FakeResult([])

        if "SET m.relevance_score = $score" in query:
            memory_id = str(params.get("id") or "")
            score = float(params.get("score") or 0.0)
            self.updated_scores.append((memory_id, score))
            memory = self.memories.get(memory_id)
            if memory is not None:
                memory["relevance_score"] = score
            return FakeResult([])

        # JIT canonical enrichment state check
        if "MATCH (m:Memory {id: $id}) RETURN m.enriched, m.processed" in query:
            memory_id = params.get("id")
            memory = self.memories.get(memory_id)
            if memory is None:
                return FakeResult([])
            return FakeResult([[bool(memory.get("enriched")), bool(memory.get("processed"))]])

        # Enrichment patterns
        if "RETURN m2.id" in query and "PRECEDED_BY" not in query:
            return FakeResult([[memory_id] for memory_id in self.temporal_related_ids])

        if "MERGE (m1)-[r:PRECEDED_BY]" in query:
            self.temporal_calls.append(dict(params))
            return FakeResult([])

        if (
            "MATCH (m:Memory)" in query
            and "m.type = $type" in query
            and "RETURN m.id, m.content" in query
        ):
            return FakeResult(self.pattern_source_rows)

        if "MERGE (p:Pattern" in query:
            self.pattern_calls.append(dict(params))
            return FakeResult([])

        if "MERGE (m)-[r:EXEMPLIFIES]" in query:
            self.exemplifies_calls.append(dict(params))
            return FakeResult([])

        if "SET m.metadata" in query and "m.enriched = true" in query:
            self.update_calls.append(dict(params))
            memory_id = str(params.get("id") or "")
            memory = self.memories.get(memory_id)
            if memory is not None:
                memory.update(
                    {
                        "metadata": params.get("metadata", memory.get("metadata", {})),
                        "tags": params.get("tags", memory.get("tags", [])),
                        "tag_prefixes": params.get("tag_prefixes", memory.get("tag_prefixes", [])),
                        "summary": params.get("summary", memory.get("summary")),
                        "enriched": True,
                        "processed": True,
                        "enriched_at": params.get("enriched_at", memory.get("enriched_at")),
                    }
                )
            return FakeResult([])

        # Memory create/upsert
        if "MERGE (m:Memory {id:" in query or "CREATE (m:Memory {id:" in query:
            memory_id = str(params["id"])
            self.nodes.add(memory_id)
            existing = self.memories.get(memory_id, {})
            self.memories[memory_id] = {
                "id": memory_id,
                "content": params.get("content", existing.get("content", "")),
                "tags": params.get("tags", existing.get("tags", [])),
                "tag_prefixes": params.get("tag_prefixes", existing.get("tag_prefixes", [])),
                "importance": params.get("importance", existing.get("importance", 0.5)),
                "type": params.get("type", existing.get("type", "Memory")),
                "timestamp": params.get("timestamp", existing.get("timestamp", _utc_now())),
                "metadata": params.get("metadata", existing.get("metadata", "{}")),
                "updated_at": params.get("updated_at", existing.get("updated_at")),
                "last_accessed": params.get("last_accessed", existing.get("last_accessed")),
                "confidence": params.get("confidence", existing.get("confidence", 1.0)),
                "summary": params.get("summary", existing.get("summary")),
                "processed": params.get("processed", existing.get("processed", False)),
                "enriched": params.get("enriched", existing.get("enriched", False)),
                "enriched_at": params.get("enriched_at", existing.get("enriched_at")),
            }
            return FakeResult([[FakeNode(self.memories[memory_id])]])

        # Memory update
        if "MATCH (m:Memory {id:" in query and "SET m.content" in query:
            memory_id = str(params["id"])
            memory = self.memories.get(memory_id)
            if memory is None:
                return FakeResult([])
            memory.update(
                {
                    "content": params.get("content", memory.get("content")),
                    "tags": params.get("tags", memory.get("tags")),
                    "tag_prefixes": params.get("tag_prefixes", memory.get("tag_prefixes")),
                    "importance": params.get("importance", memory.get("importance")),
                    "metadata": params.get("metadata", memory.get("metadata")),
                    "type": params.get("type", memory.get("type")),
                    "confidence": params.get("confidence", memory.get("confidence")),
                    "updated_at": params.get("updated_at", _utc_now()),
                }
            )
            return FakeResult([[FakeNode(memory)]])

        # Memory retrieval by id
        if (
            "MATCH (m:Memory {id:" in query
            and _returns_whole_memory_node(query)
            and "WHERE" not in query
        ):
            memory_id = params.get("id") or params.get("id1") or params.get("id2")
            memory = self.memories.get(str(memory_id)) if memory_id else None
            if memory is None:
                return FakeResult([])
            return FakeResult([[FakeNode(memory)]])

        # Delete by id (used by API and consolidation)
        if "MATCH (m:Memory {id:" in query and "DELETE m" in query:
            memory_id = str(params.get("id") or "")
            if memory_id:
                self.deleted.append(memory_id)
            if memory_id in self.memories:
                del self.memories[memory_id]
                return FakeResult([["deleted"]])
            return FakeResult([])

        # Search by exact tag pattern
        if "MATCH (m:Memory)" in query and "$tag IN m.tags" in query:
            tag = str(params.get("tag") or "").strip().lower()
            results = []
            for memory in self.memories.values():
                tags = [str(value).strip().lower() for value in (memory.get("tags") or [])]
                if tag and tag in tags:
                    results.append([FakeNode(memory)])
            return FakeResult(results)

        # Search by tags list (/memory/by-tag)
        if "MATCH (m:Memory)" in query and "toLower(tag) IN $tags" in query:
            tags = {
                str(tag).strip().lower()
                for tag in (params.get("tags") or [])
                if isinstance(tag, str) and tag.strip()
            }
            results = []
            for memory in self.memories.values():
                memory_tags = [
                    str(tag).strip().lower()
                    for tag in (memory.get("tags") or [])
                    if isinstance(tag, str) and tag.strip()
                ]
                if any(tag in tags for tag in memory_tags):
                    results.append(memory)

            results.sort(
                key=lambda memory: (
                    float(memory.get("importance") or 0.0),
                    str(memory.get("timestamp") or ""),
                ),
                reverse=True,
            )

            limit = int(params.get("limit") or len(results))
            return FakeResult([[FakeNode(memory)] for memory in results[:limit]])

        # Bulk fetch for reembed
        if "MATCH (m:Memory)" in query and "RETURN m.id, m.content" in query:
            rows: List[List[Any]] = []
            for memory_id, memory in self.memories.items():
                if memory.get("content"):
                    rows.append([memory_id, memory.get("content", "")])
            return FakeResult(rows)

        if (
            "MATCH (m:Memory)" in query
            and "RETURN m.id AS id" in query
            and "m.content AS content" in query
        ):
            rows = []
            for memory_id, memory in self.memories.items():
                if memory.get("content"):
                    rows.append(
                        [
                            memory_id,
                            memory.get("content", ""),
                            memory.get("tags", []),
                            memory.get("importance", 0.5),
                            memory.get("timestamp"),
                            memory.get("type", "Context"),
                            memory.get("confidence", 0.6),
                            memory.get("metadata", "{}"),
                            memory.get("updated_at"),
                            memory.get("last_accessed"),
                        ]
                    )
            return FakeResult(rows)

        # Startup recall query patterns
        if "WHERE 'critical' IN m.tags OR 'lesson' IN m.tags OR 'ai-assistant' IN m.tags" in query:
            rows = []
            for memory in self.memories.values():
                tags = [
                    str(tag).strip().lower()
                    for tag in (memory.get("tags") or [])
                    if isinstance(tag, str) and tag.strip()
                ]
                if any(tag in {"critical", "lesson", "ai-assistant"} for tag in tags):
                    metadata = memory.get("metadata", "{}")
                    if isinstance(metadata, dict):
                        metadata = json.dumps(metadata)
                    rows.append(
                        [
                            memory.get("id"),
                            memory.get("content"),
                            memory.get("tags", []),
                            memory.get("importance", 0.5),
                            memory.get("type", "Context"),
                            metadata,
                        ]
                    )
            rows.sort(key=lambda row: float(row[3] or 0.0), reverse=True)
            return FakeResult(rows[:10])

        if "WHERE 'system' IN m.tags OR 'memory-recall' IN m.tags" in query:
            rows = []
            for memory in self.memories.values():
                tags = [
                    str(tag).strip().lower()
                    for tag in (memory.get("tags") or [])
                    if isinstance(tag, str) and tag.strip()
                ]
                if any(tag in {"system", "memory-recall"} for tag in tags):
                    rows.append([memory.get("id"), memory.get("content"), memory.get("tags", [])])
            return FakeResult(rows[:5])

        # Association creation
        if (
            "MATCH (m1:Memory" in query
            and "MATCH (m2:Memory" in query
            and "MERGE (m1)-[r:" in query
        ):
            memory1_id = str(params.get("id1") or "")
            memory2_id = str(params.get("id2") or "")
            relation_type = "RELATES_TO"
            match = re.search(r"MERGE \(m1\)-\[r:([A-Z_]+)\]->\(m2\)", query)
            if match:
                relation_type = match.group(1)
            self.relationships.append(
                {
                    "id1": memory1_id,
                    "id2": memory2_id,
                    "type": relation_type,
                    "strength": float(params.get("strength") or 0.5),
                    "context": params.get("context"),
                    "reason": params.get("reason"),
                }
            )
            return FakeResult([["created"]])

        # Recall-style graph query over memories
        if (
            "MATCH (m:Memory)" in query
            and _returns_whole_memory_node(query)
            and "ORDER BY" in query
        ):
            results = list(self.memories.values())

            tag_filters = [
                str(tag).strip().lower()
                for tag in (params.get("tag_filters") or [])
                if isinstance(tag, str) and tag.strip()
            ]
            exact_tags = [
                str(tag).strip().lower()
                for tag in (params.get("tags") or [])
                if isinstance(tag, str) and tag.strip()
            ]

            if tag_filters:
                filtered = []
                for memory in results:
                    memory_tags = [
                        str(tag).strip().lower()
                        for tag in (memory.get("tags") or [])
                        if isinstance(tag, str) and tag.strip()
                    ]
                    if any(
                        any(tag.startswith(prefix) for tag in memory_tags) for prefix in tag_filters
                    ):
                        filtered.append(memory)
                results = filtered
            elif exact_tags:
                filtered = []
                for memory in results:
                    memory_tags = [
                        str(tag).strip().lower()
                        for tag in (memory.get("tags") or [])
                        if isinstance(tag, str) and tag.strip()
                    ]
                    if any(tag in memory_tags for tag in exact_tags):
                        filtered.append(memory)
                results = filtered

            def _timestamp_key(memory: Dict[str, Any]) -> str:
                if "coalesce(m.updated_at, m.timestamp)" in query:
                    return str(memory.get("updated_at") or memory.get("timestamp") or "")
                return str(memory.get("timestamp") or "")

            def _importance(memory: Dict[str, Any]) -> float:
                try:
                    return float(memory.get("importance") or 0.0)
                except (TypeError, ValueError):
                    return 0.0

            if "ORDER BY m.timestamp ASC" in query:
                results.sort(key=lambda memory: (_timestamp_key(memory), -_importance(memory)))
            elif "ORDER BY m.timestamp DESC" in query:
                results.sort(
                    key=lambda memory: (_timestamp_key(memory), _importance(memory)), reverse=True
                )
            elif "ORDER BY coalesce(m.updated_at, m.timestamp) ASC" in query:
                results.sort(key=lambda memory: (_timestamp_key(memory), -_importance(memory)))
            elif "ORDER BY coalesce(m.updated_at, m.timestamp) DESC" in query:
                results.sort(
                    key=lambda memory: (_timestamp_key(memory), _importance(memory)), reverse=True
                )
            else:
                results.sort(
                    key=lambda memory: (_importance(memory), _timestamp_key(memory)), reverse=True
                )

            limit = int(params.get("limit") or len(results))
            return FakeResult([[FakeNode(memory)] for memory in results[:limit]])

        # Analytics query patterns
        if "MATCH (m:Memory)" in query and "RETURN m.type, COUNT(m)" in query:
            type_counts: Dict[str, Dict[str, float]] = {}
            for memory in self.memories.values():
                memory_type = str(memory.get("type") or "Memory")
                current = type_counts.setdefault(memory_type, {"count": 0, "total": 0.0})
                current["count"] += 1
                current["total"] += float(memory.get("confidence") or 0.5)
            rows = []
            for memory_type, data in type_counts.items():
                count = int(data["count"])
                avg = data["total"] / count if count else 0.0
                rows.append([memory_type, count, avg])
            return FakeResult(rows)

        if "MATCH (p:Pattern)" in query:
            return FakeResult([])

        if "MATCH (m1:Memory)-[r:PREFERS_OVER]" in query:
            rows = []
            for rel in self.relationships:
                if rel.get("type") != "PREFERS_OVER":
                    continue
                memory1 = self.memories.get(str(rel.get("id1") or ""), {})
                memory2 = self.memories.get(str(rel.get("id2") or ""), {})
                rows.append(
                    [
                        memory1.get("content", ""),
                        memory2.get("content", ""),
                        rel.get("context"),
                        rel.get("strength", 0.5),
                    ]
                )
            rows.sort(key=lambda row: float(row[3] or 0.0), reverse=True)
            return FakeResult(rows[:10])

        if "RETURN m.timestamp, m.importance" in query:
            rows = []
            for memory in self.memories.values():
                if memory.get("timestamp") is not None:
                    rows.append([memory.get("timestamp"), memory.get("importance", 0.5)])
            return FakeResult(rows[:100])

        if "WHEN m.confidence" in query:
            buckets = {"low": 0, "medium": 0, "high": 0}
            for memory in self.memories.values():
                try:
                    confidence = float(memory.get("confidence") or 0.0)
                except (TypeError, ValueError):
                    confidence = 0.0
                if confidence < 0.3:
                    buckets["low"] += 1
                elif confidence < 0.7:
                    buckets["medium"] += 1
                else:
                    buckets["high"] += 1
            rows = [[level, count] for level, count in buckets.items() if count > 0]
            if not rows:
                rows = [["medium", 0]]
            return FakeResult(rows)

        if (
            "MATCH (m:Memory)" in query
            and "WHERE m.confidence IS NOT NULL" in query
            and "RETURN m.confidence" in query
        ):
            rows = []
            for memory in self.memories.values():
                confidence = memory.get("confidence")
                if confidence is None:
                    continue
                rows.append([confidence])
            return FakeResult(rows[:500])

        if (
            "MATCH (m:Memory)" in query
            and "WHERE m.metadata IS NOT NULL" in query
            and "RETURN m.metadata" in query
        ):
            rows = []
            for memory in self.memories.values():
                metadata = memory.get("metadata")
                if metadata is None:
                    continue
                rows.append([metadata])
            return FakeResult(rows[:200])

        if "MATCH (m:Memory)" in query and "RETURN m.content" in query:
            rows = [[memory.get("content", "")] for memory in list(self.memories.values())[:100]]
            return FakeResult(rows)

        if "MATCH (m:Memory {id:" in query and "RETURN type" in query:
            return FakeResult([])

        # Default: no rows
        return FakeResult([])
