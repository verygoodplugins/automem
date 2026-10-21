"""Deterministic edits interleaved with extraction and atomic persistence."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from automem.enrichment import runtime_helpers
from automem.enrichment.runtime_orchestration import enrich_memory, jit_enrich_lightweight
from automem.utils.tags import _compute_tag_prefixes, _normalize_tag_list
from tests.support.fake_graph import FakeGraph, FakeNode, FakeResult


class InterleavingGraph(FakeGraph):
    def __init__(self):
        super().__init__()
        self.before_commit = None
        self.memories["memory"] = {
            "id": "memory",
            "content": "Notes from Project Launchpad",
            "tags": ["old"],
            "metadata": json.dumps({"source": "old"}),
            "importance": 0.5,
            "type": "Memory",
            "processed": False,
        }

    def query(self, query, params=None, **kwargs):
        if "SET m.metadata" in query and self.before_commit is not None:
            callback, self.before_commit = self.before_commit, None
            callback()
        if query == "MATCH (m:Memory {id: $id}) RETURN m":
            memory = self.memories.get(params["id"])
            return FakeResult([[FakeNode(copy.deepcopy(memory))]]) if memory else FakeResult([])
        return super().query(query, params, **kwargs)


def perform_enrichment(
    graph,
    *,
    jit=False,
    during_extraction=lambda: None,
    qdrant=None,
    real_helpers=False,
    during_summary=lambda: None,
):
    def extract(content):
        during_extraction()
        return {"projects": ["Launchpad"]}

    def summary(content, previous):
        during_summary()
        return "Generated summary"

    kwargs = dict(
        memory_id="memory",
        get_memory_graph_fn=lambda: graph,
        get_qdrant_client_fn=lambda: qdrant,
        parse_metadata_field_fn=lambda raw: json.loads(raw) if isinstance(raw, str) else raw,
        normalize_tag_list_fn=_normalize_tag_list,
        extract_entities_fn=extract,
        slugify_fn=lambda value: value.lower(),
        compute_tag_prefixes_fn=_compute_tag_prefixes,
        enrichment_enable_summaries=True,
        generate_summary_fn=summary,
        utc_now_fn=lambda: "2026-10-20T00:00:00Z",
        collection_name="test",
        logger=MagicMock(),
    )
    if jit:
        return jit_enrich_lightweight(properties=copy.deepcopy(graph.memories["memory"]), **kwargs)
    helpers = (
        dict(
            find_temporal_relationships_fn=lambda graph, mid: runtime_helpers.find_temporal_relationships(
                graph=graph,
                memory_id=mid,
                limit=5,
                cutoff_fn=lambda: "2020-01-01",
                utc_now_fn=lambda: "2026-10-20",
                logger=MagicMock(),
            ),
            detect_patterns_fn=lambda graph, mid, content: runtime_helpers.detect_patterns(
                graph=graph,
                memory_id=mid,
                content=content,
                classify_fn=lambda content: ("Memory", 0.8),
                search_stopwords=set(),
                utc_now_fn=lambda: "2026-10-20",
                logger=MagicMock(),
            ),
            link_semantic_neighbors_fn=lambda *args: [],
        )
        if real_helpers
        else dict(
            find_temporal_relationships_fn=lambda *args: 0,
            detect_patterns_fn=lambda *args: [],
            link_semantic_neighbors_fn=lambda *args: [],
        )
    )
    return enrich_memory(
        forced=False,
        **helpers,
        unexpected_response_exc=RuntimeError,
        **kwargs,
    )


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("phase", ["extraction", "commit"])
def test_concurrent_patch_fields_survive_enrichment(jit, phase):
    graph = InterleavingGraph()
    patched = {
        "tags": ["new", "updated"],
        "metadata": json.dumps({"status": "patched", "nested": {"keep": 1}}),
        "importance": 0.9,
        "type": "Decision",
        "confidence": 0.95,
        "t_valid": "2026-10-20T01:00:00Z",
        "t_invalid": "2026-10-21T01:00:00Z",
        "summary": "Caller summary",
        "updated_at": "caller timestamp",
    }
    edit = lambda: graph.memories["memory"].update(copy.deepcopy(patched))
    if phase == "commit":
        graph.before_commit = edit
    result = perform_enrichment(
        graph, jit=jit, during_extraction=edit if phase == "extraction" else lambda: None
    )
    assert result
    memory = graph.memories["memory"]
    assert memory["tags"] == ["new", "updated", "entity:projects:launchpad"]
    metadata = json.loads(memory["metadata"])
    assert metadata["status"] == "patched"
    assert metadata["nested"] == {"keep": 1}
    assert "source" not in metadata
    for key, value in patched.items():
        if key not in {"tags", "metadata"}:
            assert memory[key] == value
    assert metadata["entities"]["projects"] == ["Launchpad"]
    if jit:
        assert result["importance"] == 0.9
        assert result["tags"] == memory["tags"]


@pytest.mark.parametrize("jit", [False, True])
def test_content_edit_does_not_commit_stale_extraction(jit):
    graph = InterleavingGraph()
    edit = lambda: graph.memories["memory"].update(
        content="Entirely new source", metadata='{"status":"new"}'
    )
    if jit:
        assert perform_enrichment(graph, jit=True, during_extraction=edit) is None
    else:
        with pytest.raises(RuntimeError, match="changed during enrichment"):
            perform_enrichment(graph, during_extraction=edit)
    assert graph.memories["memory"]["content"] == "Entirely new source"
    assert graph.memories["memory"]["tags"] == ["old"]
    assert json.loads(graph.memories["memory"]["metadata"]) == {"status": "new"}
    assert not graph.memories["memory"].get("processed")


@pytest.mark.parametrize("jit", [False, True])
def test_deleted_memory_is_not_resurrected_by_enrichment(jit):
    graph = InterleavingGraph()
    result = perform_enrichment(
        graph, jit=jit, during_extraction=lambda: graph.memories.pop("memory")
    )
    assert not result
    assert "memory" not in graph.memories


def test_jit_returns_latest_untouched_fields_after_concurrent_importance_edit():
    graph = InterleavingGraph()
    graph.before_commit = lambda: graph.memories["memory"].update(importance=0.9, type="Decision")
    updated = perform_enrichment(graph, jit=True)
    assert updated["importance"] == 0.9
    assert updated["type"] == "Decision"


class InterleavingVector:
    def __init__(self, *, content_changed=False):
        self.content_changed = content_changed
        self.payload = {
            "content": "Notes from Project Launchpad",
            "tags": ["old"],
            "metadata": {"source": "old"},
        }
        self.first = True

    def set_payload(self, *, points, payload, key=None, **kwargs):
        # A PATCH reaches Qdrant after enrichment committed to the graph.
        if self.first:
            self.first = False
            self.payload = {
                "content": (
                    "Updated source" if self.content_changed else "Notes from Project Launchpad"
                ),
                "tags": ["new"],
                "metadata": {"status": "patched"},
            }
        if not isinstance(points, list):
            for condition in points.must_not or []:
                if any(tag not in condition.match.except_values for tag in self.payload["tags"]):
                    return
            conditions = points.must
            for condition in conditions:
                if getattr(condition, "key", None) is not None:
                    if condition.values_count is not None:
                        if len(self.payload["tags"]) != condition.values_count.gte:
                            return
                    else:
                        value = self.payload.get(condition.key)
                        if isinstance(value, list):
                            if condition.match.value not in value:
                                return
                        elif condition.match.value != value:
                            return
        target = self.payload
        for part in key.split(".") if key else []:
            target = target.setdefault(part, {})
        target.update(copy.deepcopy(payload))


@pytest.mark.parametrize("jit", [False, True])
def test_vector_sync_cannot_overwrite_later_patch_tags_or_metadata(jit):
    graph = InterleavingGraph()
    vector = InterleavingVector()
    assert perform_enrichment(graph, jit=jit, qdrant=vector)
    assert vector.payload["tags"] == ["new"]
    assert vector.payload["metadata"]["status"] == "patched"
    assert "source" not in vector.payload["metadata"]
    assert vector.payload["metadata"]["entities"]["projects"] == ["Launchpad"]


@pytest.mark.parametrize("jit", [False, True])
def test_vector_sync_keeps_entity_tags_when_source_tags_are_current(jit):
    graph = InterleavingGraph()
    vector = InterleavingVector()
    vector.first = False
    assert perform_enrichment(graph, jit=jit, qdrant=vector)
    assert vector.payload["tags"] == ["old", "entity:projects:launchpad"]
    assert vector.payload["metadata"]["source"] == "old"


@pytest.mark.parametrize("jit", [False, True])
def test_vector_sync_preserves_caller_fields_after_late_content_patch(jit):
    graph = InterleavingGraph()
    vector = InterleavingVector(content_changed=True)
    assert perform_enrichment(graph, jit=jit, qdrant=vector)
    assert vector.payload["content"] == "Updated source"
    assert vector.payload["tags"] == ["new"]
    assert vector.payload["metadata"]["status"] == "patched"


@pytest.mark.parametrize("jit", [False, True])
def test_duplicate_source_tags_cannot_hide_new_caller_tag(jit):
    graph = InterleavingGraph()
    graph.memories["memory"]["tags"] = ["old", "old"]
    vector = InterleavingVector()
    vector.first = False
    vector.payload["tags"] = ["old", "new"]
    assert perform_enrichment(graph, jit=jit, qdrant=vector)
    assert vector.payload["tags"] == ["old", "new"]


def test_rejected_content_guard_has_no_relationship_side_effects():
    graph = InterleavingGraph()
    original = copy.deepcopy(graph.memories["memory"])
    graph.temporal_related_ids = ["older"]
    graph.pattern_source_rows = [[f"other-{i}", "Related project content"] for i in range(3)]

    def edit():
        graph.memories["memory"]["content"] = "Updated source"

    with pytest.raises(RuntimeError, match="changed during enrichment"):
        perform_enrichment(graph, real_helpers=True, during_summary=edit)
    assert graph.temporal_calls == []
    assert graph.pattern_calls == []
    assert graph.exemplifies_calls == []
    assert graph.memories["memory"] == {**original, "content": "Updated source"}


def test_relationships_and_derived_fields_share_one_atomic_commit():
    class RecordingGraph(InterleavingGraph):
        def __init__(self):
            super().__init__()
            self.write_snapshots = []

        def query(self, query, params=None, **kwargs):
            if (
                "SET m.metadata" in query
                or "MERGE (m1)-[r:PRECEDED_BY]" in query
                or "MERGE (p:Pattern" in query
            ):
                self.write_snapshots.append((query, copy.deepcopy(self.memories["memory"])))
            return super().query(query, params, **kwargs)

    graph = RecordingGraph()
    graph.temporal_related_ids = ["older"]
    graph.pattern_source_rows = [[f"other-{i}", "Related content"] for i in range(3)]
    original = copy.deepcopy(graph.memories["memory"])
    assert perform_enrichment(graph, real_helpers=True)
    assert len(graph.write_snapshots) == 1
    query, snapshot = graph.write_snapshots[0]
    assert snapshot == original
    assert "MERGE (p:Pattern" in query
    assert "MERGE (m1)-[r:PRECEDED_BY]" in query
    assert "SET m.metadata" in query
    assert "m.processed = true" in query


@pytest.mark.parametrize("phase", ["planning", "commit"])
def test_timestamp_edit_rejects_obsolete_temporal_plan(phase):
    graph = InterleavingGraph()
    graph.memories["memory"]["timestamp"] = "2026-10-20T01:00:00Z"
    graph.temporal_related_ids = ["older"]
    original = copy.deepcopy(graph.memories["memory"])

    def edit():
        graph.memories["memory"]["timestamp"] = "2026-10-19T01:00:00Z"

    if phase == "commit":
        graph.before_commit = edit
    with pytest.raises(RuntimeError, match="timestamp changed during enrichment"):
        perform_enrichment(
            graph, real_helpers=True, during_summary=edit if phase == "planning" else lambda: None
        )
    assert graph.temporal_calls == []
    assert graph.memories["memory"] == {**original, "timestamp": "2026-10-19T01:00:00Z"}


def test_atomic_commit_failure_does_not_mark_memory_processed():
    class FailingGraph(InterleavingGraph):
        fail = True

        def query(self, query, params=None, **kwargs):
            if self.fail and params and any(key.startswith("enrichment_") for key in params):
                self.fail = False
                raise ConnectionError("planned relationship write failed")
            return super().query(query, params, **kwargs)

    graph = FailingGraph()
    graph.temporal_related_ids = ["older"]
    with pytest.raises(ConnectionError, match="planned relationship write failed"):
        perform_enrichment(graph, real_helpers=True)
    assert graph.memories["memory"]["processed"] is False
    assert graph.memories["memory"].get("enriched", False) is False
    assert perform_enrichment(graph, real_helpers=True)
    assert graph.memories["memory"]["processed"] is True
    assert graph.memories["memory"]["enriched"] is True


def test_late_content_change_suppresses_effects_and_remains_eligible():
    class EditingGraph(InterleavingGraph):
        edit = True

        def query(self, query, params=None, **kwargs):
            if self.edit and params and any(key.startswith("enrichment_") for key in params):
                self.edit = False
                self.memories["memory"]["content"] = "Caller edit after CAS"
            return super().query(query, params, **kwargs)

    graph = EditingGraph()
    graph.temporal_related_ids = ["older"]
    with pytest.raises(RuntimeError, match="changed during enrichment"):
        perform_enrichment(graph, real_helpers=True)
    assert graph.temporal_calls == []
    assert graph.pattern_calls == []
    assert graph.memories["memory"]["processed"] is False
    assert graph.memories["memory"].get("enriched", False) is False
