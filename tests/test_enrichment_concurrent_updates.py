"""Deterministic edits interleaved with extraction and atomic persistence."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

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


def perform_enrichment(graph, *, jit=False, during_extraction=lambda: None, qdrant=None):
    def extract(content):
        during_extraction()
        return {"projects": ["Launchpad"]}

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
        generate_summary_fn=lambda content, previous: "Generated summary",
        utc_now_fn=lambda: "2026-10-20T00:00:00Z",
        collection_name="test",
        logger=MagicMock(),
    )
    if jit:
        return jit_enrich_lightweight(properties=copy.deepcopy(graph.memories["memory"]), **kwargs)
    return enrich_memory(
        forced=False,
        find_temporal_relationships_fn=lambda *args: 0,
        detect_patterns_fn=lambda *args: [],
        link_semantic_neighbors_fn=lambda *args: [],
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
