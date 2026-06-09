"""Tests for the Entity Identity Synthesis feature.

Covers:
- Entity node creation from tags (migration)
- Dedup candidate detection (string similarity, co-occurrence)
- Merge operation
- Identity synthesis (mocked LLM)
- Recall with entity injection
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from tests.support.fake_graph import FakeGraph, FakeNode, FakeResult

# ---------------------------------------------------------------------------
# Helpers: Entity-aware FakeGraph extension
# ---------------------------------------------------------------------------


class EntityFakeGraph(FakeGraph):
    """FakeGraph with Entity node support for testing."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.entity_edges: List[Dict[str, str]] = []  # {"entity_id": ..., "memory_id": ...}

    def query(self, query: str, params: Dict[str, Any] | None = None, **kwargs: Any) -> FakeResult:
        params = params or {}
        self.queries.append((query, params))

        # MERGE Entity node
        if "MERGE (e:Entity {id: $id})" in query:
            eid = params["id"]
            if eid not in self.entities:
                self.entities[eid] = {
                    "id": eid,
                    "slug": params.get("slug", ""),
                    "category": params.get("category", ""),
                    "name": params.get("name", ""),
                    "aliases": params.get("aliases", []),
                    "identity": params.get("identity"),
                    "identity_version": params.get("identity_version", 0),
                    "identity_updated_at": params.get("identity_updated_at"),
                    "identity_source_count": params.get("identity_source_count", 0),
                    "merged_into": None,
                    "created_at": params.get("created_at"),
                    "last_referenced_at": params.get("last_referenced_at"),
                }
            else:
                # ON MATCH SET updates
                ent = self.entities[eid]
                if "identity_source_count" in params:
                    ent["identity_source_count"] = params["identity_source_count"]
                if "last_referenced_at" in params:
                    ent["last_referenced_at"] = params["last_referenced_at"]
                if "identity" in params:
                    ent["identity"] = params["identity"]
                if "identity_version" in params:
                    ent["identity_version"] = params["identity_version"]
            return FakeResult([])

        # MERGE REFERENCED_IN edge (single or batched via UNWIND)
        if "MERGE (e)-[:REFERENCED_IN]->(m)" in query:
            entity_id = params.get("entity_id") or params.get("canonical_id")
            # Batched: UNWIND $mem_ids
            mem_ids = params.get("mem_ids")
            if entity_id and mem_ids:
                for mid in mem_ids:
                    edge = {"entity_id": entity_id, "memory_id": mid}
                    if edge not in self.entity_edges:
                        self.entity_edges.append(edge)
                return FakeResult([])
            mem_id = params.get("mem_id")
            if entity_id and mem_id:
                # Avoid duplicate edges
                edge = {"entity_id": entity_id, "memory_id": mem_id}
                if edge not in self.entity_edges:
                    self.entity_edges.append(edge)
            return FakeResult([])

        # Batch entity-memory edge query for dedup
        if (
            "MATCH (e:Entity)-[:REFERENCED_IN]->(m:Memory)" in query
            and "RETURN e.id, m.id" in query
        ):
            rows = []
            for edge in self.entity_edges:
                ent = self.entities.get(edge["entity_id"])
                if ent and not ent.get("merged_into"):
                    rows.append([edge["entity_id"], edge["memory_id"]])
            return FakeResult(rows)

        # Entity query: MATCH (e:Entity) ... RETURN e.id, e.slug, ...
        if "MATCH (e:Entity)" in query and "e.merged_into IS NULL" in query:
            # Batch slug lookup (recall entity injection)
            if "e.slug IN $slugs" in query:
                slugs = params.get("slugs", [])
                rows = []
                for ent in self.entities.values():
                    if ent.get("merged_into"):
                        continue
                    if ent.get("identity") is None and "e.identity IS NOT NULL" in query:
                        continue
                    matched = ent["slug"] in slugs or any(
                        a in slugs for a in (ent.get("aliases") or [])
                    )
                    if matched:
                        rows.append(
                            [
                                ent["id"],
                                ent["slug"],
                                ent["category"],
                                ent["name"],
                                ent.get("aliases", []),
                                ent.get("identity"),
                                ent.get("identity_source_count", 0),
                                ent.get("identity_updated_at"),
                            ]
                        )
                return FakeResult(rows)

            # Single slug lookup
            if "e.slug = $slug" in query or "$slug IN e.aliases" in query:
                slug = params.get("slug", "")
                for ent in self.entities.values():
                    if ent.get("merged_into"):
                        continue
                    if ent["slug"] == slug or slug in (ent.get("aliases") or []):
                        # Compute ref_count for this entity
                        ref_count = sum(1 for e in self.entity_edges if e["entity_id"] == ent["id"])
                        return FakeResult(
                            [
                                [
                                    ent["id"],
                                    ent["slug"],
                                    ent["category"],
                                    ent["name"],
                                    ent.get("aliases", []),
                                    ent.get("identity"),
                                    ent.get("identity_version", 0),
                                    ent.get("identity_updated_at"),
                                    ent.get("identity_source_count", 0),
                                    ref_count,
                                    ent.get("created_at"),
                                    ent.get("last_referenced_at"),
                                ]
                            ]
                        )
                return FakeResult([])

            if "RETURN e.id, e.slug, e.category, e.aliases" in query:
                rows = []
                for ent in self.entities.values():
                    if ent.get("merged_into"):
                        continue
                    rows.append(
                        [
                            ent["id"],
                            ent["slug"],
                            ent["category"],
                            ent.get("aliases", []),
                        ]
                    )
                return FakeResult(rows)

            if "RETURN e.id, e.identity_source_count, e.identity" in query:
                rows = []
                for ent in self.entities.values():
                    if ent.get("merged_into"):
                        continue
                    ref_count = sum(1 for e in self.entity_edges if e["entity_id"] == ent["id"])
                    rows.append(
                        [
                            ent["id"],
                            ent.get("identity_source_count", 0),
                            ent.get("identity"),
                            ref_count,
                        ]
                    )
                return FakeResult(rows)

            # Generic list (with ref_count)
            rows = []
            for ent in self.entities.values():
                if ent.get("merged_into"):
                    continue
                ref_count = sum(1 for e in self.entity_edges if e["entity_id"] == ent["id"])
                rows.append(
                    [
                        ent["id"],
                        ent["slug"],
                        ent["category"],
                        ent["name"],
                        ent.get("aliases", []),
                        ent.get("identity"),
                        ent.get("identity_version", 0),
                        ent.get("identity_updated_at"),
                        ent.get("identity_source_count", 0),
                        ref_count,
                        ent.get("created_at"),
                        ent.get("last_referenced_at"),
                    ]
                )
            return FakeResult(rows)

        # Entity by ID
        if "MATCH (e:Entity {id: $id})" in query:
            eid = params.get("id")
            ent = self.entities.get(eid)
            if not ent:
                return FakeResult([])

            if "REFERENCED_IN" in query:
                # Get linked memories
                mem_ids = [e["memory_id"] for e in self.entity_edges if e["entity_id"] == eid]
                rows = []
                for mid in mem_ids:
                    mem = self.memories.get(mid)
                    if mem:
                        row = [
                            mem.get("id"),
                            mem.get("content", ""),
                            mem.get("importance", 0.5),
                            mem.get("timestamp"),
                            mem.get("type"),
                        ]
                        if "m.t_valid" in query:
                            row.extend(
                                [
                                    mem.get("t_valid"),
                                    mem.get("t_invalid"),
                                    mem.get("archived"),
                                ]
                            )
                        rows.append(row)
                return FakeResult(rows)

            if "RETURN e.name, e.category, e.identity, e.identity_version" in query:
                return FakeResult(
                    [
                        [
                            ent["name"],
                            ent["category"],
                            ent.get("identity"),
                            ent.get("identity_version", 0),
                        ]
                    ]
                )

            if "RETURN e.slug, e.aliases" in query:
                return FakeResult([[ent["slug"], ent.get("aliases", [])]])

            if "SET e.identity" in query:
                ent["identity"] = params.get("identity")
                ent["identity_version"] = params.get("version", 0)
                ent["identity_updated_at"] = params.get("now")
                ent["identity_source_count"] = params.get("source_count", 0)
                return FakeResult([])

            if "SET e.merged_into" in query:
                ent["merged_into"] = params.get("canonical_id")
                return FakeResult([])

            if "SET e.aliases" in query:
                # New behavior: full replacement (deduplicated in Python)
                if "aliases" in params:
                    ent["aliases"] = params["aliases"]
                else:
                    new_aliases = params.get("new_aliases", [])
                    ent["aliases"] = (ent.get("aliases") or []) + new_aliases
                return FakeResult([])

            if "RETURN e.aliases" in query:
                return FakeResult([[ent.get("aliases", [])]])

            return FakeResult([[FakeNode(ent)]])

        if "UNWIND $ids AS source_id" in query and "related.archived" in query:
            source_ids = {str(memory_id) for memory_id in (params.get("ids") or [])}
            requested_types = {str(rel_type) for rel_type in (params.get("types") or [])}
            rows = []
            for rel in self.relationships:
                if str(rel.get("id1") or "") not in source_ids:
                    continue
                if requested_types and str(rel.get("type") or "") not in requested_types:
                    continue
                related = self.memories.get(str(rel.get("id2") or ""))
                if related is None:
                    continue
                rows.append(
                    [
                        rel.get("id1"),
                        related.get("archived"),
                        related.get("t_valid"),
                        related.get("t_invalid"),
                    ]
                )
            return FakeResult(rows)

        # Memory tags query for migration
        if "MATCH (m:Memory)" in query and "RETURN m.id, m.tags" in query:
            rows = []
            for mem in self.memories.values():
                rows.append([mem["id"], mem.get("tags", [])])
            return FakeResult(rows)

        return super().query(query, params, **kwargs)


# ---------------------------------------------------------------------------
# Test: Entity node creation from tags (migration logic)
# ---------------------------------------------------------------------------


class TestEntityMigration:
    """Test the migration script logic."""

    def test_collect_entity_tags(self) -> None:
        """Entity tags are correctly grouped by tag."""
        from scripts.migrate_entity_nodes import ENTITY_TAG_RE, collect_entity_tags

        graph = EntityFakeGraph()
        graph.memories["m1"] = {
            "id": "m1",
            "content": "Alice is great",
            "tags": ["entity:people:alice-smith", "entity:organizations:acme-corp"],
            "importance": 0.8,
        }
        graph.memories["m2"] = {
            "id": "m2",
            "content": "Alice likes music",
            "tags": ["entity:people:alice-smith", "music"],
            "importance": 0.6,
        }

        result = collect_entity_tags(graph)
        assert "entity:people:alice-smith" in result
        assert set(result["entity:people:alice-smith"]) == {"m1", "m2"}
        assert "entity:organizations:acme-corp" in result
        assert result["entity:organizations:acme-corp"] == ["m1"]

    def test_slug_to_name(self) -> None:
        from scripts.migrate_entity_nodes import slug_to_name

        assert slug_to_name("alice-smith") == "Alice Smith"
        assert slug_to_name("acme-corp") == "Acme Corp"
        assert slug_to_name("cool-startup") == "Cool Startup"

    def test_migration_creates_entities(self) -> None:
        """Migration creates Entity nodes and edges."""
        from scripts.migrate_entity_nodes import run_migration

        graph = EntityFakeGraph()
        graph.memories["m1"] = {
            "id": "m1",
            "content": "Alice at Acme Corp",
            "tags": ["entity:people:alice-smith"],
            "importance": 0.8,
        }
        entity_map = {"entity:people:alice-smith": ["m1"]}

        run_migration(graph, entity_map, dry_run=False)

        assert "entity:people:alice-smith" in graph.entities
        ent = graph.entities["entity:people:alice-smith"]
        assert ent["slug"] == "alice-smith"
        assert ent["category"] == "people"
        assert ent["name"] == "Alice Smith"
        assert len(graph.entity_edges) == 1

    def test_migration_dry_run(self) -> None:
        """Dry run doesn't create anything."""
        from scripts.migrate_entity_nodes import run_migration

        graph = EntityFakeGraph()
        entity_map = {"entity:people:alice-smith": ["m1"]}

        run_migration(graph, entity_map, dry_run=True)
        assert len(graph.entities) == 0
        assert len(graph.entity_edges) == 0

    def test_migration_dry_run_reports_rejected_historical_tags(self) -> None:
        """Dry run audits low-quality historical tags before creating Entity nodes."""
        from scripts.migrate_entity_nodes import run_migration

        graph = EntityFakeGraph()
        entity_map = {
            "entity:people:completed": ["m1"],
            "entity:people:alex-beck-s": ["m2"],
            "entity:people:alex": ["m3"],
        }

        stats = run_migration(graph, entity_map, dry_run=True)

        assert stats["accepted_entities"] == 1
        assert stats["accepted"][0]["id"] == "entity:people:alex-beck"
        assert stats["accepted"][0]["source_tags"] == ["entity:people:alex-beck-s"]
        assert stats["rejected_entities"] == [
            {
                "tag": "entity:people:alex",
                "category": "people",
                "slug": "alex",
                "reason": "migration_low_confidence",
                "references": 1,
            },
            {
                "tag": "entity:people:completed",
                "category": "people",
                "slug": "completed",
                "reason": "low_signal_people_slug",
                "references": 1,
            },
        ]
        assert len(graph.entities) == 0
        assert len(graph.entity_edges) == 0

    def test_migration_omits_structural_review_only_entities(self) -> None:
        """Migration uses structural review gates instead of corpus-specific blockers."""
        from scripts.migrate_entity_nodes import run_migration

        graph = EntityFakeGraph()
        entity_map = {
            "entity:tools:vectorstorex": ["m1"],
            "entity:concepts:source-hygiene": ["m2"],
            "entity:tools:atlas-db": ["m3"],
            "entity:people:alice-smith": ["m4"],
            "entity:tools:three-nights": ["m5"],
            "entity:tools:tracking-codes": ["m6"],
        }

        stats = run_migration(graph, entity_map, dry_run=False)

        assert set(graph.entities) == {
            "entity:people:alice-smith",
            "entity:tools:atlas-db",
        }
        assert {item["tag"] for item in stats["rejected_entities"]} == {
            "entity:concepts:source-hygiene",
            "entity:tools:vectorstorex",
            "entity:tools:three-nights",
            "entity:tools:tracking-codes",
        }
        assert {item["reason"] for item in stats["rejected_entities"]} == {
            "migration_review_low_signal_phrase",
            "migration_review_single_token_nonperson",
        }

    def test_migration_has_no_corpus_specific_blocker_table(self) -> None:
        """Public migration readiness must be rule-based, not a failed-run token list."""
        import scripts.migrate_entity_nodes as migration

        assert not hasattr(migration, "MIGRATION_READINESS_BLOCKERS")

    def test_entity_audit_summary_counts_reasons_and_samples(self) -> None:
        """Summary audit returns compact counts and samples instead of full arrays."""
        from automem.api.entity import _audit_memory_entity_tags

        graph = EntityFakeGraph()
        graph.memories["m1"] = {
            "id": "m1",
            "tags": [
                "entity:tools:qdrant",
                "entity:tools:them",
                "entity:concepts:400ms",
            ],
        }
        graph.memories["m2"] = {
            "id": "m2",
            "tags": [
                "entity:people:alex-beck-s",
                "entity:tools:qdrant",
                "entity:tools:ud83d-udea7-active-projects",
            ],
        }

        audit = _audit_memory_entity_tags(graph, summary=True, limit=2)

        assert audit["counts"]["accepted_entities"] == 2
        assert audit["counts"]["rejected_entities"] == 3
        assert audit["counts_by_reason"]["generic_entity_slug"] == 1
        assert audit["counts_by_reason"]["duration_or_count_slug"] == 1
        assert audit["counts_by_reason"]["unicode_escape_slug"] == 1
        assert audit["counts_by_category"]["tools"]["accepted"] == 1
        assert audit["counts_by_category"]["tools"]["rejected"] == 2
        assert audit["accepted_sample"][0]["id"] == "entity:people:alex-beck"
        assert len(audit["accepted_sample"]) <= 2
        assert len(audit["rejected_sample"]) <= 2
        assert "accepted" not in audit
        assert "rejected_entities" not in audit

    def test_entity_audit_filters_reason_and_category(self) -> None:
        """Full audit can page/filter accepted and rejected rows."""
        from automem.api.entity import _audit_memory_entity_tags

        graph = EntityFakeGraph()
        graph.memories["m1"] = {
            "id": "m1",
            "tags": [
                "entity:tools:qdrant",
                "entity:tools:them",
                "entity:concepts:400ms",
            ],
        }

        audit = _audit_memory_entity_tags(
            graph,
            summary=False,
            category="tools",
            reason="generic_entity_slug",
            limit=10,
            offset=0,
        )

        assert audit["accepted"] == []
        assert [item["tag"] for item in audit["rejected_entities"]] == ["entity:tools:them"]


# ---------------------------------------------------------------------------
# Test: Entity API
# ---------------------------------------------------------------------------


class TestEntityAPI:
    @pytest.mark.parametrize("limit", ["0", "-1"])
    def test_list_entities_rejects_non_positive_limit(self, limit: str) -> None:
        from automem.api.entity import create_entity_blueprint

        graph = EntityFakeGraph()
        app = Flask(__name__)
        app.register_blueprint(create_entity_blueprint(lambda: graph, MagicMock()))
        client = app.test_client()

        response = client.get(f"/entities?limit={limit}")

        assert response.status_code == 400


# ---------------------------------------------------------------------------
# Test: Entity dedup
# ---------------------------------------------------------------------------


class TestEntityDedup:
    """Test entity deduplication logic."""

    def test_levenshtein_distance(self) -> None:
        from automem.consolidation.entity_dedup import _levenshtein

        assert _levenshtein("kitten", "sitting") == 3
        assert _levenshtein("abc", "abc") == 0
        assert _levenshtein("", "abc") == 3

    def test_slug_similarity_exact(self) -> None:
        from automem.consolidation.entity_dedup import _slug_similarity

        assert _slug_similarity("alice", "alice") == 1.0

    def test_slug_similarity_substring(self) -> None:
        from automem.consolidation.entity_dedup import _slug_similarity

        sim = _slug_similarity("alice", "alice-smith")
        assert sim > 0.2  # alice is substring

    def test_memory_overlap(self) -> None:
        from automem.consolidation.entity_dedup import _memory_overlap

        a = {"m1", "m2", "m3"}
        b = {"m1", "m2", "m3", "m4", "m5"}
        assert _memory_overlap(a, b) == 1.0  # all of a are in b

        c = {"m1", "m6"}
        assert _memory_overlap(c, b) == 0.5  # 1 of 2

    def test_find_candidates_substring_overlap(self) -> None:
        """Substring + high overlap → auto-merge candidate."""
        from automem.consolidation.entity_dedup import find_merge_candidates

        graph = EntityFakeGraph()
        # Create two entities with substring relationship and shared memories
        graph.entities["entity:people:alice"] = {
            "id": "entity:people:alice",
            "slug": "alice",
            "category": "people",
            "aliases": [],
            "merged_into": None,
        }
        graph.entities["entity:people:alice-smith"] = {
            "id": "entity:people:alice-smith",
            "slug": "alice-smith",
            "category": "people",
            "aliases": [],
            "merged_into": None,
        }

        # alice references m1, m2, m3
        for mid in ["m1", "m2", "m3"]:
            graph.entity_edges.append({"entity_id": "entity:people:alice", "memory_id": mid})
            graph.memories[mid] = {"id": mid, "content": f"Memory {mid}", "tags": []}

        # alice-smith references m1, m2, m3, m4, m5
        for mid in ["m1", "m2", "m3", "m4", "m5"]:
            graph.entity_edges.append({"entity_id": "entity:people:alice-smith", "memory_id": mid})
            if mid not in graph.memories:
                graph.memories[mid] = {
                    "id": mid,
                    "content": f"Memory {mid}",
                    "tags": [],
                }

        auto_merge, _review = find_merge_candidates(graph)
        assert len(auto_merge) >= 1
        # The canonical should be the longer slug
        assert auto_merge[0].canonical_id == "entity:people:alice-smith"
        assert auto_merge[0].alias_id == "entity:people:alice"

    def test_plain_first_name_with_multiple_targets_requires_review(self) -> None:
        """Ambiguous generic first names are not auto-merged."""
        from automem.consolidation.entity_dedup import find_merge_candidates

        graph = EntityFakeGraph()
        for eid, slug in [
            ("entity:people:alex", "alex"),
            ("entity:people:alex-beck", "alex-beck"),
            ("entity:people:alex-panagis", "alex-panagis"),
        ]:
            graph.entities[eid] = {
                "id": eid,
                "slug": slug,
                "category": "people",
                "aliases": [],
                "merged_into": None,
            }

        for mid in ["m1", "m2"]:
            graph.memories[mid] = {"id": mid, "content": f"Memory {mid}", "tags": []}
            graph.entity_edges.append({"entity_id": "entity:people:alex", "memory_id": mid})
            graph.entity_edges.append({"entity_id": "entity:people:alex-beck", "memory_id": mid})
            graph.entity_edges.append({"entity_id": "entity:people:alex-panagis", "memory_id": mid})

        auto_merge, review = find_merge_candidates(graph)

        assert all(candidate.alias_id != "entity:people:alex" for candidate in auto_merge)
        assert any(candidate.alias_id == "entity:people:alex" for candidate in review)

    def test_different_categories_not_merged(self) -> None:
        """Entities in different categories are not candidates."""
        from automem.consolidation.entity_dedup import find_merge_candidates

        graph = EntityFakeGraph()
        graph.entities["entity:people:python"] = {
            "id": "entity:people:python",
            "slug": "python",
            "category": "people",
            "aliases": [],
            "merged_into": None,
        }
        graph.entities["entity:tools:python"] = {
            "id": "entity:tools:python",
            "slug": "python",
            "category": "tools",
            "aliases": [],
            "merged_into": None,
        }
        auto_merge, _review = find_merge_candidates(graph)
        assert len(auto_merge) == 0
        assert len(_review) == 0

    def test_merge_entities(self) -> None:
        """Merge moves edges and updates aliases."""
        from automem.consolidation.entity_dedup import merge_entities

        graph = EntityFakeGraph()
        graph.entities["entity:people:alice-smith"] = {
            "id": "entity:people:alice-smith",
            "slug": "alice-smith",
            "category": "people",
            "aliases": [],
            "merged_into": None,
        }
        graph.entities["entity:people:alice"] = {
            "id": "entity:people:alice",
            "slug": "alice",
            "category": "people",
            "aliases": [],
            "merged_into": None,
        }
        graph.memories["m1"] = {"id": "m1", "content": "test", "tags": []}
        graph.entity_edges.append({"entity_id": "entity:people:alice", "memory_id": "m1"})

        result = merge_entities(graph, "entity:people:alice-smith", "entity:people:alice")

        assert result.canonical_id == "entity:people:alice-smith"
        assert result.alias_slug == "alice"
        assert result.edges_moved == 1
        assert graph.entities["entity:people:alice"]["merged_into"] == "entity:people:alice-smith"
        assert "alice" in graph.entities["entity:people:alice-smith"]["aliases"]

    def test_merge_entities_deletes_only_copied_edges(self) -> None:
        """Alias edge cleanup is scoped to the edges copied in this merge."""
        from automem.consolidation.entity_dedup import merge_entities

        graph = EntityFakeGraph()
        graph.entities["entity:people:alice-smith"] = {
            "id": "entity:people:alice-smith",
            "slug": "alice-smith",
            "category": "people",
            "aliases": [],
            "merged_into": None,
        }
        graph.entities["entity:people:alice"] = {
            "id": "entity:people:alice",
            "slug": "alice",
            "category": "people",
            "aliases": [],
            "merged_into": None,
        }
        graph.memories["m1"] = {"id": "m1", "content": "test", "tags": []}
        graph.entity_edges.append({"entity_id": "entity:people:alice", "memory_id": "m1"})

        merge_entities(graph, "entity:people:alice-smith", "entity:people:alice")

        delete_queries = [(query, params) for query, params in graph.queries if "DELETE r" in query]
        assert delete_queries, "merge should remove copied alias REFERENCED_IN edges"
        delete_query, delete_params = delete_queries[-1]
        assert "m.id IN $mem_ids" in delete_query
        assert delete_params["mem_ids"] == ["m1"]


# ---------------------------------------------------------------------------
# Test: Identity synthesis
# ---------------------------------------------------------------------------


class TestIdentitySynthesis:
    """Test identity synthesis with mocked LLM."""

    def test_synthesize_identity(self) -> None:
        """Identity synthesis calls LLM and stores result."""
        from automem.consolidation.identity_synthesis import synthesize_identity

        graph = EntityFakeGraph()
        graph.entities["entity:people:alice-smith"] = {
            "id": "entity:people:alice-smith",
            "slug": "alice-smith",
            "category": "people",
            "name": "Alice Smith",
            "aliases": [],
            "identity": None,
            "identity_version": 0,
            "identity_updated_at": None,
            "identity_source_count": 3,
            "merged_into": None,
        }
        graph.memories["m1"] = {
            "id": "m1",
            "content": "Alice works at Acme Corp as a manager.",
            "importance": 0.9,
            "timestamp": "2026-01-15T12:00:00Z",
            "type": "Context",
            "tags": ["entity:people:alice-smith"],
        }
        graph.entity_edges.append({"entity_id": "entity:people:alice-smith", "memory_id": "m1"})

        # Mock OpenAI client
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Alice Smith is a manager at Acme Corp.")
                )
            ]
        )

        result = synthesize_identity(
            graph, "entity:people:alice-smith", mock_client, model="test-model"
        )

        assert result == "Alice Smith is a manager at Acme Corp."
        assert graph.entities["entity:people:alice-smith"]["identity"] == result
        assert graph.entities["entity:people:alice-smith"]["identity_version"] == 1
        mock_client.chat.completions.create.assert_called_once()

    def test_synthesize_no_memories(self) -> None:
        """Returns None when entity has no linked memories."""
        from automem.consolidation.identity_synthesis import synthesize_identity

        graph = EntityFakeGraph()
        graph.entities["entity:people:unknown"] = {
            "id": "entity:people:unknown",
            "slug": "unknown",
            "category": "people",
            "name": "Unknown",
            "aliases": [],
            "identity": None,
            "identity_version": 0,
            "identity_updated_at": None,
            "identity_source_count": 0,
            "merged_into": None,
        }

        mock_client = MagicMock()
        result = synthesize_identity(graph, "entity:people:unknown", mock_client)

        assert result is None
        mock_client.chat.completions.create.assert_not_called()

    def test_run_identity_consolidation(self) -> None:
        """Full identity consolidation run with dedup + synthesis."""
        from automem.consolidation.identity_synthesis import run_identity_consolidation

        graph = EntityFakeGraph()
        graph.entities["entity:people:alice-smith"] = {
            "id": "entity:people:alice-smith",
            "slug": "alice-smith",
            "category": "people",
            "name": "Alice Smith",
            "aliases": [],
            "identity": None,
            "identity_version": 0,
            "identity_updated_at": None,
            "identity_source_count": 3,
            "merged_into": None,
        }
        graph.memories["m1"] = {
            "id": "m1",
            "content": "Alice works at Acme Corp.",
            "importance": 0.9,
            "timestamp": "2026-01-15T12:00:00Z",
            "type": "Context",
            "tags": ["entity:people:alice-smith"],
        }
        graph.entity_edges.append({"entity_id": "entity:people:alice-smith", "memory_id": "m1"})

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="Alice is a manager at Acme Corp."))
            ]
        )

        result = run_identity_consolidation(graph, mock_client, min_references=1)

        assert result["identities_synthesized"] == 1
        assert result["entities_examined"] >= 1

    def test_dry_run_no_llm_calls(self) -> None:
        """Dry run counts but doesn't call LLM."""
        from automem.consolidation.identity_synthesis import run_identity_consolidation

        graph = EntityFakeGraph()
        graph.entities["entity:people:alice-smith"] = {
            "id": "entity:people:alice-smith",
            "slug": "alice-smith",
            "category": "people",
            "name": "Alice Smith",
            "aliases": [],
            "identity": None,
            "identity_version": 0,
            "identity_updated_at": None,
            "identity_source_count": 3,
            "merged_into": None,
        }
        # Add an edge so ref_count > 0
        graph.memories["m1"] = {
            "id": "m1",
            "content": "test",
            "tags": [],
            "importance": 0.5,
        }
        graph.entity_edges.append({"entity_id": "entity:people:alice-smith", "memory_id": "m1"})

        mock_client = MagicMock()
        result = run_identity_consolidation(graph, mock_client, dry_run=True)

        assert result["identities_synthesized"] == 1  # counted but not actually called
        mock_client.chat.completions.create.assert_not_called()

    def test_skip_unchanged_entities(self) -> None:
        """Entities with existing identity and matching source count are skipped."""
        from automem.consolidation.identity_synthesis import run_identity_consolidation

        graph = EntityFakeGraph()
        graph.entities["entity:people:alice-smith"] = {
            "id": "entity:people:alice-smith",
            "slug": "alice-smith",
            "category": "people",
            "name": "Alice Smith",
            "aliases": [],
            "identity": "Alice is a manager at Acme Corp.",
            "identity_version": 1,
            "identity_updated_at": "2026-01-15T12:00:00Z",
            "identity_source_count": 1,  # matches actual edge count
            "merged_into": None,
        }
        graph.memories["m1"] = {
            "id": "m1",
            "content": "Alice works at Acme Corp.",
            "importance": 0.9,
            "timestamp": "2026-01-15T12:00:00Z",
            "type": "Context",
            "tags": ["entity:people:alice-smith"],
        }
        graph.entity_edges.append({"entity_id": "entity:people:alice-smith", "memory_id": "m1"})

        mock_client = MagicMock()
        result = run_identity_consolidation(graph, mock_client, min_references=1)

        # Should skip — identity exists and source count matches
        assert result["identities_synthesized"] == 0
        mock_client.chat.completions.create.assert_not_called()

    def test_synthesize_identity_uses_only_current_memories(self) -> None:
        """Expired memories do not affect identity prompt or source count."""
        from automem.consolidation.identity_synthesis import synthesize_identity

        now = datetime.now(timezone.utc)
        graph = EntityFakeGraph()
        graph.entities["entity:people:alice-smith"] = {
            "id": "entity:people:alice-smith",
            "slug": "alice-smith",
            "category": "people",
            "name": "Alice Smith",
            "aliases": [],
            "identity": None,
            "identity_version": 0,
            "identity_updated_at": None,
            "identity_source_count": 0,
            "merged_into": None,
        }
        graph.memories["active"] = {
            "id": "active",
            "content": "Alice is the current finance lead at Acme.",
            "importance": 0.9,
            "timestamp": now.isoformat(),
            "type": "Context",
            "tags": ["entity:people:alice-smith"],
        }
        graph.memories["expired"] = {
            "id": "expired",
            "content": "Alice was previously described as a designer.",
            "importance": 1.0,
            "timestamp": (now - timedelta(days=90)).isoformat(),
            "t_invalid": (now - timedelta(days=1)).isoformat(),
            "type": "Context",
            "tags": ["entity:people:alice-smith"],
        }
        graph.entity_edges.append({"entity_id": "entity:people:alice-smith", "memory_id": "active"})
        graph.entity_edges.append(
            {"entity_id": "entity:people:alice-smith", "memory_id": "expired"}
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Alice is finance lead."))]
        )

        synthesize_identity(graph, "entity:people:alice-smith", mock_client, model="test-model")

        prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        assert "current finance lead" in prompt
        assert "previously described as a designer" not in prompt
        assert graph.entities["entity:people:alice-smith"]["identity_source_count"] == 1


# ---------------------------------------------------------------------------
# Test: Recall with entity injection
# ---------------------------------------------------------------------------


class TestRecallEntityInjection:
    """Test entity identity injection in recall response."""

    def test_extract_query_entities(self) -> None:
        """Query entity extraction picks up capitalized names."""
        from automem.api.recall import _extract_query_entities

        entities = _extract_query_entities("Tell me about Alice and her work at Acme Corp")
        # "Alice" is first word (skipped), but "Acme" or "Corp" should be extracted
        assert "Acme" in entities or "Corp" in entities

    def test_extract_entities_from_results(self) -> None:
        """Entity tags in results are extracted."""
        from automem.api.recall import _extract_entities_from_results

        results = [
            {
                "memory": {
                    "tags": ["entity:people:alice-smith", "work"],
                    "metadata": {},
                }
            }
        ]
        entities = _extract_entities_from_results(results)
        assert "alice smith" in entities or "alice-smith" in {e.replace(" ", "-") for e in entities}

    def test_entity_identity_payload_includes_slug(self) -> None:
        """Recall entity identity serialization includes canonical slug."""
        from automem.api.recall import _serialize_entity_identity_row

        payload = _serialize_entity_identity_row(
            [
                "entity:people:alice-smith",
                "alice-smith",
                "people",
                "Alice Smith",
                ["alice"],
                "Alice Smith is a finance lead.",
                4,
                "2026-01-15T12:00:00Z",
            ]
        )

        assert payload["id"] == "entity:people:alice-smith"
        assert payload["slug"] == "alice-smith"
