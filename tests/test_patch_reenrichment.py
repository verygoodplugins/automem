"""A content PATCH clears enrichment-derived fields and re-queues enrichment once."""

import copy
import json
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import app
from automem.enrichment.runtime_orchestration import EnrichmentConflictError, enrich_memory
from automem.stores.vector_payload import build_qdrant_payload, content_fingerprint
from automem.utils.entity_extraction import _slugify
from automem.utils.tags import _compute_tag_prefixes, _normalize_tag_list
from tests.test_api_endpoints import (  # noqa: F401
    MockQdrantClient,
    auth_headers,
    client,
    mock_state,
)

MEMORY_ID = "00000000-0000-0000-0000-0000000002a7"
OLD_CONTENT = "Notes from Project Launchpad"
NEW_CONTENT = "Notes from Project Orbit"
ENRICHED_METADATA = {
    "source": "user",
    "entities": {"projects": ["Launchpad"]},
    "enrichment": {"last_run": "2026-11-01T00:00:00Z", "content_hash": "sha256:old"},
}
ENRICHED_TAGS = ["work", "entity:projects:launchpad"]


class GuardedQdrant(MockQdrantClient):
    """Mock Qdrant that applies the HasId / content_hash / IsEmpty / tag filters."""

    def __init__(self):
        super().__init__()
        self.before_set_payload = None

    def _matches(self, payload, flt):
        for condition in flt.must or []:
            if getattr(condition, "has_id", None) is not None:
                continue
            if getattr(condition, "is_empty", None) is not None:
                if payload.get(condition.is_empty.key):
                    return False
            elif getattr(condition, "values_count", None) is not None:
                if len(payload.get(condition.key) or []) != condition.values_count.gte:
                    return False
            else:
                value = payload.get(condition.key)
                if isinstance(value, list):
                    if condition.match.value not in value:
                        return False
                elif value != condition.match.value:
                    return False
        for condition in flt.must_not or []:
            if any(tag not in condition.match.except_values for tag in payload.get("tags", [])):
                return False
        return True

    def set_payload(self, collection_name, payload, points, key=None, wait=False, **kwargs):
        if self.before_set_payload is not None:
            callback, self.before_set_payload = self.before_set_payload, None
            callback()
        for point_id, point in self.points.items():
            has_id = next(
                (c.has_id for c in points.must if getattr(c, "has_id", None) is not None), []
            )
            if point_id not in has_id or not self._matches(point["payload"], points):
                continue
            target = point["payload"]
            for part in key.split(".") if key else []:
                target = target.setdefault(part, {})
            target.update(copy.deepcopy(payload))

    def retrieve(self, collection_name, ids, with_payload=True, with_vectors=False):
        return [
            SimpleNamespace(
                id=point_id,
                payload=copy.deepcopy(self.points[point_id]["payload"]),
                vector=self.points[point_id]["vector"],
            )
            for point_id in ids
            if point_id in self.points
        ]


def _seed(mock_state, *, enriched=True):
    metadata = ENRICHED_METADATA if enriched else {"source": "user"}
    tags = ENRICHED_TAGS if enriched else ["work"]
    mock_state.memory_graph.memories[MEMORY_ID] = {
        "id": MEMORY_ID,
        "content": OLD_CONTENT,
        "tags": list(tags),
        "tag_prefixes": _compute_tag_prefixes(list(tags)),
        "metadata": json.dumps(metadata),
        "importance": 0.5,
        "type": "Memory",
        "timestamp": "2026-11-01T00:00:00+00:00",
        "summary": "Notes from Project Launchpad" if enriched else None,
        "enriched": enriched,
        "enriched_at": "2026-11-01T00:00:00Z" if enriched else None,
        "processed": enriched,
    }
    qdrant = GuardedQdrant()
    qdrant.points[MEMORY_ID] = {
        "vector": [0.1] * 768,
        "payload": build_qdrant_payload(
            {"content": OLD_CONTENT, "tags": list(tags), "metadata": copy.deepcopy(metadata)}
        ),
    }
    mock_state.qdrant = qdrant
    mock_state.enrichment_queue = Queue()
    return qdrant


def _queued_ids(state):
    return [job.memory_id for job in list(state.enrichment_queue.queue)]


def _run_enrichment(state, *, during_extraction=lambda: None):
    def extract(content):
        during_extraction()
        return {"projects": ["Launchpad"]} if "Launchpad" in content else {"projects": ["Orbit"]}

    return enrich_memory(
        memory_id=MEMORY_ID,
        forced=False,
        get_memory_graph_fn=lambda: state.memory_graph,
        get_qdrant_client_fn=lambda: state.qdrant,
        parse_metadata_field_fn=lambda raw: json.loads(raw) if isinstance(raw, str) else raw,
        normalize_tag_list_fn=_normalize_tag_list,
        extract_entities_fn=extract,
        slugify_fn=_slugify,
        compute_tag_prefixes_fn=_compute_tag_prefixes,
        find_temporal_relationships_fn=lambda *args: 0,
        detect_patterns_fn=lambda *args: [],
        link_semantic_neighbors_fn=lambda *args: [],
        enrichment_enable_summaries=True,
        generate_summary_fn=lambda content, previous: content,
        utc_now_fn=lambda: "2026-11-19T00:00:00Z",
        collection_name="memories",
        unexpected_response_exc=RuntimeError,
        logger=MagicMock(),
    )


def test_content_patch_clears_derived_fields_and_requeues(client, mock_state, auth_headers):
    qdrant = _seed(mock_state)

    response = client.patch(
        f"/memory/{MEMORY_ID}", headers=auth_headers, json={"content": NEW_CONTENT}
    )

    assert response.status_code == 200, response.json
    memory = mock_state.memory_graph.memories[MEMORY_ID]
    assert memory["content"] == NEW_CONTENT
    assert json.loads(memory["metadata"]) == {"source": "user"}
    assert memory["tags"] == ["work"]
    assert memory["tag_prefixes"] == ["work"]
    assert memory["summary"] is None
    assert memory["enriched"] is False
    assert memory["enriched_at"] is None
    assert memory["processed"] is False

    payload = qdrant.points[MEMORY_ID]["payload"]
    assert payload["content_hash"] == content_fingerprint(NEW_CONTENT)
    assert payload["metadata"] == {"source": "user"}
    assert payload["tags"] == ["work"]

    assert _queued_ids(mock_state) == [MEMORY_ID]
    assert mock_state.enrichment_pending == {MEMORY_ID}


def test_content_patch_keeps_explicit_caller_tags(client, mock_state, auth_headers):
    _seed(mock_state)

    response = client.patch(
        f"/memory/{MEMORY_ID}",
        headers=auth_headers,
        json={"content": NEW_CONTENT, "tags": ["work", "entity:projects:launchpad"]},
    )

    assert response.status_code == 200, response.json
    memory = mock_state.memory_graph.memories[MEMORY_ID]
    assert memory["tags"] == ["work", "entity:projects:launchpad"]
    assert "entities" not in json.loads(memory["metadata"])
    assert _queued_ids(mock_state) == [MEMORY_ID]


@pytest.mark.parametrize(
    "body",
    [
        {"metadata": {**ENRICHED_METADATA, "status": "reviewed"}},
        {"tags": ENRICHED_TAGS + ["reviewed"]},
        {"importance": 0.9},
        {"content": OLD_CONTENT, "importance": 0.9},
    ],
    ids=["metadata", "tags", "importance", "same-content"],
)
def test_non_content_patch_leaves_enrichment_untouched(client, mock_state, auth_headers, body):
    qdrant = _seed(mock_state)
    before = copy.deepcopy(mock_state.memory_graph.memories[MEMORY_ID])

    response = client.patch(f"/memory/{MEMORY_ID}", headers=auth_headers, json=body)

    assert response.status_code == 200, response.json
    memory = mock_state.memory_graph.memories[MEMORY_ID]
    for field in ("summary", "enriched", "enriched_at", "processed"):
        assert memory[field] == before[field]
    metadata = json.loads(memory["metadata"])
    assert metadata["entities"] == ENRICHED_METADATA["entities"]
    assert metadata["enrichment"] == ENRICHED_METADATA["enrichment"]
    assert "entity:projects:launchpad" in memory["tags"]
    payload = qdrant.points[MEMORY_ID]["payload"]
    assert payload["metadata"]["entities"] == ENRICHED_METADATA["entities"]
    assert payload["content_hash"] == content_fingerprint(OLD_CONTENT)
    assert _queued_ids(mock_state) == []
    assert mock_state.enrichment_pending == set()


@pytest.mark.parametrize("phase", ["extraction", "vector_sync"])
def test_inflight_enrichment_finishing_after_patch_writes_no_stale_entities(
    client, mock_state, auth_headers, phase
):
    # The worker picked up the job for the old text before the PATCH arrived.
    qdrant = _seed(mock_state, enriched=False)
    mock_state.enrichment_inflight.add(MEMORY_ID)

    def patch():
        response = client.patch(
            f"/memory/{MEMORY_ID}", headers=auth_headers, json={"content": NEW_CONTENT}
        )
        assert response.status_code == 200, response.json

    if phase == "extraction":
        # PATCH lands before the graph commit: the content compare-and-set rejects it.
        with pytest.raises(EnrichmentConflictError):
            _run_enrichment(mock_state, during_extraction=patch)
    else:
        # PATCH lands after the graph commit, before the Qdrant sync (#270 window).
        qdrant.before_set_payload = patch
        _run_enrichment(mock_state)

    memory = mock_state.memory_graph.memories[MEMORY_ID]
    assert memory["content"] == NEW_CONTENT
    assert "entities" not in json.loads(memory["metadata"])
    assert "entity:projects:launchpad" not in memory["tags"]
    assert memory["processed"] is False

    payload = qdrant.points[MEMORY_ID]["payload"]
    assert payload["content_hash"] == content_fingerprint(NEW_CONTENT)
    assert "entities" not in payload["metadata"]
    assert "enrichment" not in payload["metadata"]
    assert "entity:projects:launchpad" not in payload["tags"]

    # The in-flight job did not absorb the re-queue: one fresh job waits behind it.
    assert _queued_ids(mock_state) == [MEMORY_ID]

    # That job enriches the new text and its writes land on the new point.
    mock_state.enrichment_inflight.discard(MEMORY_ID)
    assert _run_enrichment(mock_state)
    memory = mock_state.memory_graph.memories[MEMORY_ID]
    assert json.loads(memory["metadata"])["entities"] == {"projects": ["Orbit"]}
    assert qdrant.points[MEMORY_ID]["payload"]["metadata"]["entities"] == {"projects": ["Orbit"]}


def test_requeue_happens_once_per_patch(client, mock_state, auth_headers, monkeypatch):
    from automem.enrichment import runtime_queue_bindings

    _seed(mock_state)
    real_runtime = runtime_queue_bindings._enqueue_enrichment_runtime
    calls = []

    def counting_runtime(**kwargs):
        calls.append((kwargs["memory_id"], kwargs["attempt"], kwargs["supersede_inflight"]))
        real_runtime(**kwargs)

    monkeypatch.setattr(runtime_queue_bindings, "_enqueue_enrichment_runtime", counting_runtime)

    response = client.patch(
        f"/memory/{MEMORY_ID}", headers=auth_headers, json={"content": NEW_CONTENT}
    )
    assert response.status_code == 200, response.json
    assert calls == [(MEMORY_ID, 0, True)]
    assert _queued_ids(mock_state) == [MEMORY_ID]

    # A metadata-only PATCH while the job waits adds no call and no job.
    response = client.patch(
        f"/memory/{MEMORY_ID}", headers=auth_headers, json={"metadata": {"source": "x"}}
    )
    assert response.status_code == 200, response.json
    assert len(calls) == 1

    # A second content PATCH asks once more, and the waiting job absorbs it.
    response = client.patch(
        f"/memory/{MEMORY_ID}", headers=auth_headers, json={"content": NEW_CONTENT + " v3"}
    )
    assert response.status_code == 200, response.json
    assert calls[1:] == [(MEMORY_ID, 0, True)]

    # The retry of a stale in-flight run goes through the same queue and is absorbed too.
    app.enqueue_enrichment(MEMORY_ID, attempt=1)
    assert _queued_ids(mock_state) == [MEMORY_ID]
