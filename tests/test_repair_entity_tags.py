from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "lab" / "repair_entity_tags.py"


def load_repair_module() -> Any:
    spec = importlib.util.spec_from_file_location("repair_entity_tags", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CapturingGraph:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def query(self, query: str, params: dict[str, Any] | None = None) -> Any:
        self.calls.append((query, params or {}))
        return type("Result", (), {"result_set": []})()


class SleepingGraph(CapturingGraph):
    def __init__(self, sleep_seconds: float) -> None:
        super().__init__()
        self.sleep_seconds = sleep_seconds

    def query(self, query: str, params: dict[str, Any] | None = None) -> Any:
        import time

        self.calls.append((query, params or {}))
        time.sleep(self.sleep_seconds)
        return type("Result", (), {"result_set": []})()


class MemoryScanGraph:
    def __init__(self, rows_by_id: dict[str, list[Any]]) -> None:
        self.rows_by_id = rows_by_id
        self.queries: list[str] = []
        self.params: list[dict[str, Any]] = []

    def query(self, query: str, params: dict[str, Any] | None = None) -> Any:
        self.queries.append(query)
        self.params.append(params or {})
        params = params or {}
        if "WHERE m.id IN $ids" in query:
            ids = list(params.get("ids") or [])
            # Return graph rows in a different order to prove the iterator preserves
            # deterministic caller order after fetching the batch.
            rows = [self.rows_by_id[memory_id] for memory_id in reversed(ids)]
            return type("Result", (), {"result_set": rows})()
        if "RETURN m.id" in query:
            after = str(params.get("after") or "")
            limit = int(params.get("limit") or len(self.rows_by_id))
            ids = sorted(memory_id for memory_id in self.rows_by_id if memory_id > after)
            rows = [[memory_id] for memory_id in ids[:limit]]
            return type("Result", (), {"result_set": rows})()
        rows = []
        return type("Result", (), {"result_set": rows})()


class SplittingMemoryScanGraph(MemoryScanGraph):
    def query(self, query: str, params: dict[str, Any] | None = None) -> Any:
        params = params or {}
        if "WHERE m.id IN $ids" in query and len(params.get("ids") or []) > 1:
            self.queries.append(query)
            self.params.append(params)
            raise RuntimeError("Query timed out")
        return super().query(query, params)


class TimeoutOnFullMemoryIdScanGraph(MemoryScanGraph):
    def query(self, query: str, params: dict[str, Any] | None = None) -> Any:
        params = params or {}
        if "RETURN m.id" in query and not params:
            self.queries.append(query)
            self.params.append(params)
            raise RuntimeError("Query timed out")
        return super().query(query, params)


class CapturingQdrant:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def set_payload(self, **kwargs: Any) -> None:
        self.payloads.append(kwargs)


class FlakyQdrant:
    def __init__(self, failures_by_id: dict[str, int]) -> None:
        self.failures_by_id = dict(failures_by_id)
        self.attempts_by_id: dict[str, int] = {}

    def set_payload(self, **kwargs: Any) -> None:
        memory_id = (kwargs.get("points") or [""])[0]
        self.attempts_by_id[memory_id] = self.attempts_by_id.get(memory_id, 0) + 1
        if self.attempts_by_id[memory_id] <= self.failures_by_id.get(memory_id, 0):
            raise RuntimeError(f"transient failure for {memory_id}")


class SleepingQdrant(CapturingQdrant):
    def __init__(self, sleep_seconds: float) -> None:
        super().__init__()
        self.sleep_seconds = sleep_seconds

    def set_payload(self, **kwargs: Any) -> None:
        import time

        time.sleep(self.sleep_seconds)
        super().set_payload(**kwargs)


class RetrievingQdrant:
    def __init__(self, payloads: dict[str, dict[str, Any]]) -> None:
        self.payloads = payloads
        self.calls: list[dict[str, Any]] = []

    def retrieve(self, **kwargs: Any) -> list[Any]:
        self.calls.append(kwargs)
        ids = kwargs.get("ids") or []
        return [
            type("Point", (), {"id": memory_id, "payload": self.payloads[memory_id]})()
            for memory_id in ids
            if memory_id in self.payloads
        ]


class SplittingRetrievingQdrant(RetrievingQdrant):
    def retrieve(self, **kwargs: Any) -> list[Any]:
        self.calls.append(kwargs)
        if len(kwargs.get("ids") or []) > 1:
            raise RuntimeError("timed out")
        return [
            type("Point", (), {"id": memory_id, "payload": self.payloads[memory_id]})()
            for memory_id in (kwargs.get("ids") or [])
            if memory_id in self.payloads
        ]


class DisconnectingRetrievingQdrant(RetrievingQdrant):
    def __init__(self, payloads: dict[str, dict[str, Any]]) -> None:
        super().__init__(payloads)
        self.disconnected = False

    def retrieve(self, **kwargs: Any) -> list[Any]:
        self.calls.append(kwargs)
        if not self.disconnected:
            self.disconnected = True
            raise RuntimeError("Server disconnected without sending a response")
        return [
            type("Point", (), {"id": memory_id, "payload": self.payloads[memory_id]})()
            for memory_id in (kwargs.get("ids") or [])
            if memory_id in self.payloads
        ]


def test_attach_qdrant_payloads_fetches_payloads_in_batches() -> None:
    repair = load_repair_module()
    rows = [
        {"id": "m1", "tags": ["automem"], "tag_prefixes": [], "metadata": {}, "content": ""},
        {"id": "m2", "tags": ["bugfix"], "tag_prefixes": [], "metadata": {}, "content": ""},
    ]
    qdrant = RetrievingQdrant(
        {
            "m1": {"tags": ["automem"], "tag_prefixes": ["automem"], "metadata": {}},
            "m2": {"tags": ["bugfix"], "metadata": {}},
        }
    )

    enriched = repair.attach_qdrant_payloads(
        rows,
        qdrant_client=qdrant,
        qdrant_collection="memories",
        batch_size=1,
    )

    assert enriched[0]["qdrant_payload"]["tag_prefixes"] == ["automem"]
    assert enriched[1]["qdrant_payload"]["tag_prefixes"] == []
    assert len(qdrant.calls) == 2
    assert qdrant.calls[0]["with_payload"] is True
    assert qdrant.calls[0]["with_vectors"] is False


def test_attach_qdrant_payloads_splits_timed_out_batches() -> None:
    repair = load_repair_module()
    rows = [
        {"id": "m1", "tags": ["one"], "tag_prefixes": [], "metadata": {}, "content": ""},
        {"id": "m2", "tags": ["two"], "tag_prefixes": [], "metadata": {}, "content": ""},
        {"id": "m3", "tags": ["three"], "tag_prefixes": [], "metadata": {}, "content": ""},
    ]
    qdrant = SplittingRetrievingQdrant(
        {
            "m1": {"tags": ["one"]},
            "m2": {"tags": ["two"]},
            "m3": {"tags": ["three"]},
        }
    )

    enriched = repair.attach_qdrant_payloads(
        rows,
        qdrant_client=qdrant,
        qdrant_collection="memories",
        batch_size=3,
        batch_retries=0,
    )

    assert [row["qdrant_payload"]["tags"] for row in enriched] == [["one"], ["two"], ["three"]]
    retrieve_ids = [call["ids"] for call in qdrant.calls]
    assert ["m1", "m2", "m3"] in retrieve_ids
    assert ["m1"] in retrieve_ids
    assert ["m2"] in retrieve_ids
    assert ["m3"] in retrieve_ids


def test_attach_qdrant_payloads_retries_transient_disconnects() -> None:
    repair = load_repair_module()
    rows = [
        {"id": "m1", "tags": ["one"], "tag_prefixes": [], "metadata": {}, "content": ""},
    ]
    qdrant = DisconnectingRetrievingQdrant({"m1": {"tags": ["one"]}})

    enriched = repair.attach_qdrant_payloads(
        rows,
        qdrant_client=qdrant,
        qdrant_collection="memories",
        batch_size=1,
        batch_retries=1,
        batch_retry_delay_seconds=0,
    )

    assert enriched[0]["qdrant_payload"]["tags"] == ["one"]
    assert len(qdrant.calls) == 2


def test_iter_memory_rows_uses_python_sorted_ids_for_deterministic_batches() -> None:
    repair = load_repair_module()
    graph = MemoryScanGraph(
        {
            "m2": ["m2", ["bugfix"], [], {}, "two"],
            "m1": ["m1", ["automem"], [], {}, "one"],
        }
    )

    rows = list(repair.iter_memory_rows(graph, batch_size=2))

    assert [row["id"] for row in rows] == ["m1", "m2"]
    assert graph.queries
    assert all("SKIP" not in query for query in graph.queries)
    assert "ORDER BY m.id" in graph.queries[0]
    assert graph.params[0] == {"after": "", "limit": 4}
    assert graph.params[1]["ids"] == ["m1", "m2"]


def test_iter_memory_rows_pages_memory_id_scan_without_full_graph_query() -> None:
    repair = load_repair_module()
    graph = TimeoutOnFullMemoryIdScanGraph(
        {
            "m3": ["m3", ["three"], [], {}, "three"],
            "m2": ["m2", ["two"], [], {}, "two"],
            "m1": ["m1", ["one"], [], {}, "one"],
        }
    )

    rows = list(repair.iter_memory_rows(graph, batch_size=1, id_page_size=2))

    assert [row["id"] for row in rows] == ["m1", "m2", "m3"]
    id_page_params = [
        params for query, params in zip(graph.queries, graph.params) if "ORDER BY m.id" in query
    ]
    assert id_page_params == [
        {"after": "", "limit": 2},
        {"after": "m2", "limit": 2},
    ]


def test_iter_memory_rows_splits_timed_out_property_batches() -> None:
    repair = load_repair_module()
    graph = SplittingMemoryScanGraph(
        {
            "m3": ["m3", ["three"], [], {}, "three"],
            "m2": ["m2", ["two"], [], {}, "two"],
            "m1": ["m1", ["one"], [], {}, "one"],
        }
    )

    rows = list(repair.iter_memory_rows(graph, batch_size=3, batch_retries=0))

    assert [row["id"] for row in rows] == ["m1", "m2", "m3"]
    fetch_params = [params for params in graph.params if "ids" in params]
    assert {"ids": ["m1", "m2", "m3"]} in fetch_params
    assert {"ids": ["m1"]} in fetch_params
    assert {"ids": ["m2"]} in fetch_params
    assert {"ids": ["m3"]} in fetch_params


def test_plan_sync_only_preserves_entity_tags_and_repairs_payload_shape() -> None:
    repair = load_repair_module()
    rows = [
        {
            "id": "m1",
            "tags": ["automem", "entity:people:completed", "entity:people:alex-beck-s"],
            "tag_prefixes": ["stale"],
            "metadata": json.dumps({"entities": {"tools": ["Stale Tool"]}, "source": "unit"}),
            "content": "Decision with Alex Beck.",
        }
    ]

    result = repair.build_repair_plan(rows, mode="sync-only")
    item = result.items[0]

    assert item.repaired_tags == rows[0]["tags"]
    assert item.repaired_metadata == {
        "source": "unit",
        "entities": {"people": ["Completed", "Alex Beck S"]},
    }
    assert result.rejected_tags == []
    assert result.canonicalized_tags == []
    assert result.summary()["mode"] == "sync-only"
    assert result.summary()["tag_changes"] == 0
    assert result.summary()["tag_prefix_changes"] == 1
    assert result.summary()["metadata_entity_changes"] == 1
    assert result.summary()["entity_tags_removed"] == 0
    assert result.summary()["entity_tags_added"] == 0
    assert result.summary()["bare_tags_removed"] == 0
    assert result.summary()["bare_tags_added"] == 0
    assert result.summary()["memories_with_bare_tag_changes"] == 0


def test_plan_sync_only_updates_missing_prefixes_without_tag_changes() -> None:
    repair = load_repair_module()
    rows = [
        {
            "id": "m1",
            "tags": ["automem", "bugfix", "entity:tools:qdrant"],
            "tag_prefixes": [],
            "metadata": {
                "entities": {"tools": ["Qdrant"]},
                "source": "unit",
            },
            "content": "Qdrant bugfix note.",
        }
    ]

    result = repair.build_repair_plan(rows, mode="sync-only")

    assert len(result.items) == 1
    item = result.items[0]
    assert item.tag_changed is False
    assert item.metadata_changed is False
    assert item.tag_prefix_changed is True
    assert "entity:tools:qdrant" in item.repaired_tag_prefixes


def test_plan_sync_only_includes_qdrant_only_payload_drift() -> None:
    repair = load_repair_module()
    prefixes = repair._compute_tag_prefixes(["automem", "bugfix", "entity:tools:qdrant"])
    rows = [
        {
            "id": "m1",
            "tags": ["automem", "bugfix", "entity:tools:qdrant"],
            "tag_prefixes": prefixes,
            "metadata": {
                "entities": {"tools": ["Qdrant"]},
                "source": "unit",
            },
            "qdrant_payload": {
                "tags": ["automem", "bugfix", "entity:tools:qdrant"],
                "metadata": {
                    "entities": {"tools": ["Qdrant"]},
                    "source": "unit",
                },
            },
            "content": "Qdrant bugfix note.",
        }
    ]

    result = repair.build_repair_plan(rows, mode="sync-only")

    assert len(result.items) == 1
    item = result.items[0]
    assert item.tag_changed is False
    assert item.tag_prefix_changed is False
    assert item.metadata_changed is False
    assert item.qdrant_payload_changed is True
    assert result.summary()["qdrant_payload_changes"] == 1


def test_plan_preserves_metadata_entity_casing_for_unchanged_tags() -> None:
    repair = load_repair_module()
    rows = [
        {
            "id": "m1",
            "tags": [
                "entity:tools:widget-suite",
                "entity:tools:scriptkit",
                "entity:organizations:example-commerce",
            ],
            "tag_prefixes": [],
            "metadata": {
                "entities": {
                    "tools": ["Widget Suite", "ScriptKit"],
                    "organizations": ["Example Commerce"],
                }
            },
            "content": "Widget Suite and ScriptKit integrate with Example Commerce.",
        }
    ]

    result = repair.build_repair_plan(rows, mode="sync-only")
    item = result.items[0]

    assert item.repaired_metadata["entities"] == {
        "tools": ["Widget Suite", "ScriptKit"],
        "organizations": ["Example Commerce"],
    }


def test_plan_reject_only_removes_garbage_without_canonicalizing_suffixes() -> None:
    repair = load_repair_module()
    rows = [
        {
            "id": "m1",
            "tags": ["automem", "entity:people:completed", "entity:people:alex-beck-s"],
            "tag_prefixes": [],
            "metadata": {},
            "content": "Decision with Alex Beck.",
        }
    ]

    result = repair.build_repair_plan(rows, mode="reject-only")
    item = result.items[0]

    assert "entity:people:completed" not in item.repaired_tags
    assert "entity:people:alex-beck-s" in item.repaired_tags
    assert "entity:people:alex-beck" not in item.repaired_tags
    assert result.canonicalized_tags == []
    assert result.summary()["mode"] == "reject-only"
    assert result.summary()["tag_changes"] == 1
    assert result.summary()["entity_tags_removed"] == 1
    assert result.summary()["entity_tags_added"] == 0
    assert result.summary()["bare_tags_removed"] == 0
    assert result.summary()["bare_tags_added"] == 0
    assert result.summary()["memories_with_bare_tag_changes"] == 0


def test_plan_canonicalize_safe_rejects_garbage_and_preserves_non_entity_tags(
    tmp_path: Path,
) -> None:
    repair = load_repair_module()
    rows = [
        {
            "id": "m1",
            "tags": [
                "automem",
                "decision",
                "entity:people:completed",
                "entity:people:alex-beck-s",
                "entity:people:jordan-lee",
            ],
            "tag_prefixes": [],
            "metadata": json.dumps(
                {
                    "entities": {
                        "people": ["Completed", "Alex Beck S", "Jordan Lee"],
                        "tools": ["Stale Tool"],
                    },
                    "source": "unit",
                }
            ),
            "content": "Decision with Alex Beck and Jordan Lee.",
        }
    ]

    result = repair.build_repair_plan(rows, mode="canonicalize-safe")
    assert len(result.items) == 1
    item = result.items[0]

    assert item.original_tags[:2] == ["automem", "decision"]
    assert item.repaired_tags[:2] == ["automem", "decision"]
    assert "entity:people:completed" not in item.repaired_tags
    assert "entity:people:alex-beck" in item.repaired_tags
    assert "entity:people:jordan-lee" in item.repaired_tags
    assert item.repaired_metadata["source"] == "unit"
    assert item.repaired_metadata["entities"] == {"people": ["Alex Beck", "Jordan Lee"]}

    repair.write_audit_artifacts(tmp_path, result)
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "plan.jsonl").exists()
    assert (tmp_path / "rollback.jsonl").exists()
    rejected_lines = (tmp_path / "rejected-tags.csv").read_text().splitlines()
    assert len(rejected_lines) == 2
    assert "entity:people:completed" in rejected_lines[1]
    assert "alex-beck-s" in (tmp_path / "canonicalized-tags.csv").read_text()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["entity_tags_removed"] == 2
    assert summary["entity_tags_added"] == 1
    assert summary["bare_tags_removed"] == 0
    assert summary["bare_tags_added"] == 0
    assert summary["memories_with_bare_tag_changes"] == 0


def test_bare_tag_mutations_are_detected_as_hard_failures() -> None:
    repair = load_repair_module()
    item = repair.RepairPlanItem(
        memory_id="m1",
        original_tags=["automem", "entity:people:alex"],
        repaired_tags=["decision", "entity:people:alex"],
        original_tag_prefixes=[],
        repaired_tag_prefixes=[],
        original_metadata={},
        repaired_metadata={},
        actions=[],
    )
    result = repair.RepairPlanResult(
        mode="canonicalize-safe",
        items=[item],
        processed_count=1,
        unchanged_count=0,
        rejected_tags=[],
        canonicalized_tags=[],
        ambiguous_people=[],
    )

    summary = result.summary()
    assert summary["bare_tags_removed"] == 1
    assert summary["bare_tags_added"] == 1
    assert summary["memories_with_bare_tag_changes"] == 1
    assert repair.has_bare_tag_mutations(summary) is True


def test_plan_suppresses_ambiguous_single_name_people_tags() -> None:
    repair = load_repair_module()
    rows = [
        {
            "id": "generic",
            "tags": ["entity:people:alex"],
            "tag_prefixes": [],
            "metadata": {},
            "content": "Talked with Alex.",
        },
        {
            "id": "beck",
            "tags": ["entity:people:alex-beck"],
            "tag_prefixes": [],
            "metadata": {},
            "content": "Talked with Alex Beck.",
        },
        {
            "id": "panagis",
            "tags": ["entity:people:alex-panagis"],
            "tag_prefixes": [],
            "metadata": {},
            "content": "Talked with Alex Panagis.",
        },
    ]

    result = repair.build_repair_plan(rows, mode="canonicalize-safe")
    generic = next(item for item in result.items if item.memory_id == "generic")

    assert "entity:people:alex" not in generic.repaired_tags
    assert result.ambiguous_people[0]["tag"] == "entity:people:alex"
    assert result.ambiguous_people[0]["candidates"] == [
        "entity:people:alex-beck",
        "entity:people:alex-panagis",
    ]


def test_plan_replaces_single_name_when_safe_target_is_mentioned_in_same_memory() -> None:
    repair = load_repair_module()
    rows = [
        {
            "id": "generic",
            "tags": ["entity:people:alex"],
            "tag_prefixes": [],
            "metadata": {},
            "content": "Talked with Alex Beck.",
        },
        {
            "id": "beck",
            "tags": ["entity:people:alex-beck"],
            "tag_prefixes": [],
            "metadata": {},
            "content": "Talked with Alex Beck.",
        },
    ]

    result = repair.build_repair_plan(rows, mode="canonicalize-safe")
    generic = next(item for item in result.items if item.memory_id == "generic")

    assert generic.repaired_tags == ["entity:people:alex-beck"]
    assert result.ambiguous_people == []


def test_plan_suppresses_single_name_when_only_target_is_not_in_same_memory() -> None:
    repair = load_repair_module()
    rows = [
        {
            "id": "generic",
            "tags": ["entity:people:alex"],
            "tag_prefixes": [],
            "metadata": {},
            "content": "Talked with Alex.",
        },
        {
            "id": "beck",
            "tags": ["entity:people:alex-beck"],
            "tag_prefixes": [],
            "metadata": {},
            "content": "Talked with Alex Beck.",
        },
    ]

    result = repair.build_repair_plan(rows, mode="canonicalize-safe")
    generic = next(item for item in result.items if item.memory_id == "generic")

    assert "entity:people:alex-beck" not in generic.repaired_tags
    assert "entity:people:alex" not in generic.repaired_tags
    assert result.canonicalized_tags == []
    assert result.ambiguous_people[0]["reason"] == "unsafe_single_name_people_target"
    assert result.ambiguous_people[0]["candidates"] == ["entity:people:alex-beck"]


def test_plan_suppresses_single_name_people_unsafe_structural_targets() -> None:
    repair = load_repair_module()
    examples = [
        ("docker", "docker-compose", "Docker Compose is local service tooling."),
        ("phase", "phase-five", "Phase Five was a project milestone."),
        ("sora", "sora-2", "Sora 2 was a model release."),
        ("config", "config-file", "Config file setup changed."),
    ]
    rows = []
    for single, target, content in examples:
        rows.append(
            {
                "id": f"generic-{single}",
                "tags": [f"entity:people:{single}"],
                "tag_prefixes": [],
                "metadata": {},
                "content": content,
            }
        )
        rows.append(
            {
                "id": f"target-{target}",
                "tags": [f"entity:people:{target}"],
                "tag_prefixes": [],
                "metadata": {},
                "content": content,
            }
        )

    result = repair.build_repair_plan(rows, mode="canonicalize-safe")

    assert result.canonicalized_tags == []
    for single, target, _content in examples:
        generic = next(item for item in result.items if item.memory_id == f"generic-{single}")
        assert f"entity:people:{target}" not in generic.repaired_tags
        assert f"entity:people:{single}" not in generic.repaired_tags
    reviewed_or_rejected = {
        event["tag"] for event in [*result.ambiguous_people, *result.rejected_tags]
    }
    assert reviewed_or_rejected >= {
        f"entity:people:{single}" for single, _target, _content in examples
    }


def test_plan_rejects_non_name_people_suffix_variants_instead_of_canonicalizing() -> None:
    repair = load_repair_module()
    bad_tags = [
        "entity:people:recreated-claude-code-s",
        "entity:people:sora-2-s",
        "entity:people:config-file-s",
        "entity:people:phase-five-s",
        "entity:people:alex-beck-extra-s",
    ]
    rows = [
        {
            "id": "m1",
            "tags": ["topic", *bad_tags, "entity:people:alex-beck-s"],
            "tag_prefixes": [],
            "metadata": {},
            "content": "Talked with Alex Beck about noisy generated tags.",
        }
    ]

    result = repair.build_repair_plan(rows, mode="canonicalize-safe")
    item = result.items[0]

    assert "entity:people:alex-beck" in item.repaired_tags
    assert all(tag not in item.repaired_tags for tag in bad_tags)
    assert not any(event["original_tag"] in bad_tags for event in result.canonicalized_tags)
    assert {event["tag"] for event in result.rejected_tags} >= set(bad_tags)


def test_plan_does_not_canonicalize_single_name_to_tool_like_people_target() -> None:
    repair = load_repair_module()
    rows = [
        {
            "id": "generic",
            "tags": ["entity:people:docker"],
            "tag_prefixes": [],
            "metadata": {},
            "content": "Used Docker for local services.",
        },
        {
            "id": "compose",
            "tags": ["entity:people:docker-compose"],
            "tag_prefixes": [],
            "metadata": {},
            "content": "Used Docker Compose as tooling.",
        },
    ]

    result = repair.build_repair_plan(rows, mode="canonicalize-safe")
    generic = next(item for item in result.items if item.memory_id == "generic")

    assert "entity:people:docker-compose" not in generic.repaired_tags
    assert "entity:people:docker" not in generic.repaired_tags
    assert any(event["tag"] == "entity:people:docker" for event in result.rejected_tags)


def test_apply_plan_updates_graph_and_qdrant_payload_without_vectors() -> None:
    repair = load_repair_module()
    rows = [
        {
            "id": "m1",
            "tags": ["topic", "entity:people:completed", "entity:people:alex-beck-a"],
            "tag_prefixes": [],
            "metadata": {},
            "content": "Talked with Alex Beck.",
        }
    ]
    result = repair.build_repair_plan(rows, mode="canonicalize-safe")
    graph = CapturingGraph()
    qdrant = CapturingQdrant()

    stats = repair.apply_repair_plan(
        graph,
        result.items,
        qdrant_client=qdrant,
        qdrant_collection="memories",
        batch_size=100,
    )

    assert stats["graph_updates"] == 1
    assert stats["qdrant_updates"] == 1
    assert stats["tag_updates"] == 1
    assert stats["tag_prefix_updates"] == 1
    assert stats["metadata_updates"] == 1
    graph_rows = graph.calls[0][1]["rows"]
    assert graph_rows[0]["tags"] == ["topic", "entity:people:alex-beck"]
    assert json.loads(graph_rows[0]["metadata"]) == {"entities": {"people": ["Alex Beck"]}}
    assert qdrant.payloads[0]["points"] == ["m1"]
    assert "vector" not in qdrant.payloads[0]["payload"]
    assert qdrant.payloads[0]["payload"]["tags"] == ["topic", "entity:people:alex-beck"]


def test_apply_plan_skips_graph_update_for_qdrant_only_payload_drift() -> None:
    repair = load_repair_module()
    tags = ["automem", "entity:tools:qdrant"]
    prefixes = repair._compute_tag_prefixes(tags)
    metadata = {"entities": {"tools": ["Qdrant"]}}
    item = repair.RepairPlanItem(
        memory_id="m1",
        original_tags=tags,
        repaired_tags=tags,
        original_tag_prefixes=prefixes,
        repaired_tag_prefixes=prefixes,
        original_metadata=metadata,
        repaired_metadata=metadata,
        actions=[{"action": "sync_qdrant_payload", "memory_id": "m1"}],
        original_qdrant_payload={"tags": tags, "tag_prefixes": [], "metadata": metadata},
        repaired_qdrant_payload={
            "tags": tags,
            "tag_prefixes": prefixes,
            "metadata": metadata,
        },
    )
    graph = CapturingGraph()
    qdrant = CapturingQdrant()

    stats = repair.apply_repair_plan(
        graph,
        [item],
        qdrant_client=qdrant,
        qdrant_collection="memories",
    )

    assert stats["graph_updates"] == 0
    assert graph.calls == []
    assert stats["qdrant_updates"] == 1
    assert qdrant.payloads[0]["points"] == ["m1"]


def test_apply_plan_retries_transient_qdrant_payload_failures() -> None:
    repair = load_repair_module()
    tags = ["automem", "entity:tools:qdrant"]
    prefixes = repair._compute_tag_prefixes(tags)
    metadata = {"entities": {"tools": ["Qdrant"]}}
    item = repair.RepairPlanItem(
        memory_id="m1",
        original_tags=tags,
        repaired_tags=tags,
        original_tag_prefixes=prefixes,
        repaired_tag_prefixes=prefixes,
        original_metadata=metadata,
        repaired_metadata=metadata,
        actions=[{"action": "sync_qdrant_payload", "memory_id": "m1"}],
        original_qdrant_payload={"tags": tags, "tag_prefixes": [], "metadata": metadata},
        repaired_qdrant_payload={
            "tags": tags,
            "tag_prefixes": prefixes,
            "metadata": metadata,
        },
    )
    qdrant = FlakyQdrant({"m1": 2})

    stats = repair.apply_repair_plan(
        CapturingGraph(),
        [item],
        qdrant_client=qdrant,
        qdrant_collection="memories",
        qdrant_retries=2,
        qdrant_retry_delay_seconds=0,
    )

    assert stats["qdrant_updates"] == 1
    assert stats["qdrant_failures"] == 0
    assert stats["qdrant_failure_details"] == []
    assert qdrant.attempts_by_id["m1"] == 3


def test_apply_plan_reports_qdrant_payload_failure_details() -> None:
    repair = load_repair_module()
    tags = ["automem", "entity:tools:qdrant"]
    prefixes = repair._compute_tag_prefixes(tags)
    metadata = {"entities": {"tools": ["Qdrant"]}}
    item = repair.RepairPlanItem(
        memory_id="m1",
        original_tags=tags,
        repaired_tags=tags,
        original_tag_prefixes=prefixes,
        repaired_tag_prefixes=prefixes,
        original_metadata=metadata,
        repaired_metadata=metadata,
        actions=[{"action": "sync_qdrant_payload", "memory_id": "m1"}],
        original_qdrant_payload={"tags": tags, "tag_prefixes": [], "metadata": metadata},
        repaired_qdrant_payload={
            "tags": tags,
            "tag_prefixes": prefixes,
            "metadata": metadata,
        },
    )
    qdrant = FlakyQdrant({"m1": 3})

    stats = repair.apply_repair_plan(
        CapturingGraph(),
        [item],
        qdrant_client=qdrant,
        qdrant_collection="memories",
        qdrant_retries=1,
        qdrant_retry_delay_seconds=0,
    )

    assert stats["qdrant_updates"] == 0
    assert stats["qdrant_failures"] == 1
    assert stats["qdrant_failure_details"] == [
        {"memory_id": "m1", "error": "transient failure for m1"}
    ]


def test_apply_plan_times_out_stuck_qdrant_payload_update() -> None:
    repair = load_repair_module()
    tags = ["automem", "entity:tools:qdrant"]
    prefixes = repair._compute_tag_prefixes(tags)
    metadata = {"entities": {"tools": ["Qdrant"]}}
    item = repair.RepairPlanItem(
        memory_id="m1",
        original_tags=tags,
        repaired_tags=tags,
        original_tag_prefixes=prefixes,
        repaired_tag_prefixes=prefixes,
        original_metadata=metadata,
        repaired_metadata=metadata,
        actions=[{"action": "sync_qdrant_payload", "memory_id": "m1"}],
        original_qdrant_payload={"tags": tags, "tag_prefixes": [], "metadata": metadata},
        repaired_qdrant_payload={
            "tags": tags,
            "tag_prefixes": prefixes,
            "metadata": metadata,
        },
    )
    qdrant = SleepingQdrant(0.2)

    stats = repair.apply_repair_plan(
        CapturingGraph(),
        [item],
        qdrant_client=qdrant,
        qdrant_collection="memories",
        qdrant_retries=0,
        qdrant_timeout_seconds=0.01,
    )

    assert stats["qdrant_updates"] == 0
    assert stats["qdrant_failures"] == 1
    assert "timed out" in stats["qdrant_failure_details"][0]["error"]


def test_apply_plan_times_out_stuck_graph_update_and_skips_qdrant() -> None:
    repair = load_repair_module()
    rows = [
        {
            "id": "m1",
            "tags": ["topic", "entity:people:completed"],
            "tag_prefixes": [],
            "metadata": {},
            "content": "No durable person here.",
        }
    ]
    result = repair.build_repair_plan(rows, mode="canonicalize-safe")
    graph = SleepingGraph(0.2)
    qdrant = CapturingQdrant()

    stats = repair.apply_repair_plan(
        graph,
        result.items,
        qdrant_client=qdrant,
        qdrant_collection="memories",
        graph_timeout_seconds=0.01,
    )

    assert stats["graph_updates"] == 0
    assert stats["graph_failures"] == 1
    assert "timed out" in stats["graph_failure_details"][0]["error"]
    assert qdrant.payloads == []
