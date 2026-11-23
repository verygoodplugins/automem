"""Safety and restart contracts for the original text index backfill."""

import json
from types import SimpleNamespace

import pytest

from scripts import backfill_original_search as backfill


class FakeGraph:
    def __init__(self, rows, before_update=None, missing_ids=0, duplicate_ids=0):
        self.missing_ids = missing_ids
        self.duplicate_ids = duplicate_ids
        self.rows = {row["id"]: dict(row) for row in rows}
        self.before_update = before_update
        self.writes = 0

    def query(self, query, params=None):
        params = params or {}
        if query == backfill.MISSING_ID_QUERY:
            return SimpleNamespace(result_set=[[self.missing_ids]])
        if query == backfill.DUPLICATE_ID_QUERY:
            return SimpleNamespace(result_set=[[self.duplicate_ids]])
        if "ORDER BY m.id" in query:
            rows = [r for key, r in sorted(self.rows.items()) if key > params["after"]]
            return SimpleNamespace(result_set=[self.as_row(r) for r in rows[: params["limit"]]])
        if "SET m.original_search_text" in query:
            self.writes += 1
            if self.before_update:
                self.before_update(self.rows[params["id"]])
                self.before_update = None
            row = self.rows.get(params["id"])
            if not row or any(
                row.get(field) != params[name]
                for field, name in [
                    ("content", "expected_content"),
                    ("metadata", "expected_metadata"),
                    ("original_search_text", "expected_indexed"),
                ]
            ):
                return SimpleNamespace(result_set=[])
            row["original_search_text"] = params["search_text"]
            row["labels"] = sorted(set(row["labels"]) | {"MemoryOriginalText"})
            return SimpleNamespace(result_set=[[row["id"]]])
        if "RETURN m.id, m.content, m.metadata" in query:
            row = self.rows.get(params["id"])
            return SimpleNamespace(result_set=[self.as_row(row)] if row else [])
        raise AssertionError(query)

    @staticmethod
    def as_row(row):
        return [
            row["id"],
            row.get("content"),
            row.get("metadata"),
            row.get("original_search_text"),
            row.get("labels", ["Memory"]),
        ]


def row(id, content="summary", metadata=None, indexed=None, labels=None):
    return {
        "id": id,
        "content": content,
        "metadata": metadata,
        "original_search_text": indexed,
        "labels": labels or ["Memory"],
    }


def legacy(original="long original"):
    return json.dumps({"was_summarized": True, "original_content": original})


def test_dry_run_is_default_and_never_writes():
    graph = FakeGraph([row("a", "short"), row("b", metadata=legacy())])
    summary = backfill.run(graph, batch_size=1)
    assert graph.writes == 0
    assert summary["dry_run"] is True
    assert summary["would_apply"] == 2
    assert summary["unverified_legacy"] == 1
    assert graph.rows["b"]["original_search_text"] is None


def test_apply_indexes_content_conservatively_and_is_idempotent():
    graph = FakeGraph([row("a", "short"), row("b", metadata=legacy())])
    first = backfill.run(graph, apply=True, batch_size=1)
    assert first["applied"] == 2
    assert graph.rows["b"]["original_search_text"] == "summary"
    assert "MemoryOriginalText" in graph.rows["b"]["labels"]
    before = graph.writes
    second = backfill.run(graph, apply=True)
    assert second["applied"] == 0
    assert graph.writes == before


def test_explicit_legacy_opt_in_and_existing_writer_index():
    graph = FakeGraph(
        [
            row("a", metadata=legacy()),
            row(
                "b",
                "safe",
                metadata=legacy("older"),
                indexed="newer",
                labels=["Memory", "MemoryOriginalText"],
            ),
        ]
    )
    result = backfill.run(graph, apply=True, assume_legacy_original_current=True)
    assert graph.rows["a"]["original_search_text"] == "long original"
    assert graph.rows["b"]["original_search_text"] == "newer"
    assert result["unverified_legacy"] == 2
    assert result["preserved_existing"] == 1


def test_opt_in_upgrades_a_prior_safe_content_fallback():
    graph = FakeGraph(
        [row("a", metadata=legacy(), indexed="summary", labels=["Memory", "MemoryOriginalText"])]
    )
    result = backfill.run(graph, apply=True, assume_legacy_original_current=True)
    assert result["applied"] == 1
    assert graph.rows["a"]["original_search_text"] == "long original"


@pytest.mark.parametrize(
    "field,new",
    [
        ("content", "concurrent replacement"),
        ("metadata", '{"new":true}'),
        ("original_search_text", "new writer text"),
    ],
)
def test_cas_rejects_concurrent_change(field, new):
    graph = FakeGraph([row("a")], before_update=lambda r: r.__setitem__(field, new))
    result = backfill.run(graph, apply=True)
    assert result["applied"] == 0
    assert result["conflicts"] == 1
    assert result["conflict_ids"] == ["a"]
    assert graph.rows["a"]["original_search_text"] != "summary"


def test_indexed_but_missing_label_is_repaired_without_replacing_original():
    graph = FakeGraph([row("a", indexed="full original")])
    summary = backfill.run(graph, apply=True)
    assert summary["applied"] == 1
    assert graph.rows["a"]["original_search_text"] == "full original"
    assert "MemoryOriginalText" in graph.rows["a"]["labels"]


def test_malformed_and_missing_are_reported_without_writes():
    graph = FakeGraph(
        [
            row("a", metadata="{broken"),
            row("b", content=None),
            row("c", metadata=json.dumps({"was_summarized": True})),
        ]
    )
    result = backfill.run(graph, apply=True)
    assert result["malformed"] == 2
    assert result["known_original_missing"] == 1
    assert result["applied"] == 1


def test_resume_after_id_and_limit():
    graph = FakeGraph([row("a"), row("b"), row("c")])
    result = backfill.run(graph, batch_size=1, after_id="a", limit=1, apply=True)
    assert result["scanned"] == 1
    assert result["last_id"] == "b"
    assert graph.rows["a"]["original_search_text"] is None
    assert graph.rows["b"]["original_search_text"] == "summary"
    assert graph.rows["c"]["original_search_text"] is None


def test_remote_requires_explicit_override():
    assert backfill.local_host("127.0.0.1")
    assert backfill.local_host("::1")
    assert backfill.local_host("localhost")
    assert not backfill.local_host("falkordb")
    assert not backfill.local_host("localhost.example.com")


def test_cli_refuses_nonlocal_and_apply_verify_combination():
    with pytest.raises(SystemExit):
        backfill.main(["--host", "remote.example"])
    with pytest.raises(SystemExit):
        backfill.main(["--host", "127.0.0.1", "--apply", "--verify"])


def test_verify_is_read_only_and_fails_coverage_at_cli_semantics():
    graph = FakeGraph(
        [row("a"), row("b", indexed="summary", labels=["Memory", "MemoryOriginalText"])]
    )
    summary = backfill.run(graph, verify_only=True)
    assert summary["missing_index"] == 1
    assert summary["missing_label"] == 1
    assert summary["indexed"] == 1
    assert graph.writes == 0


def test_empty_or_invalid_index_fields_are_not_mistaken_for_verified_coverage():
    graph = FakeGraph([row("a", indexed=""), row("b", indexed=123)])
    summary = backfill.run(graph, verify_only=True)
    assert summary["missing_index"] == 1
    assert summary["malformed"] == 1


@pytest.mark.integration
@pytest.mark.skipif(
    __import__("os").environ.get("RUN_ORIGINAL_INDEX_DOCKER_TESTS") != "1",
    reason="requires explicitly enabled disposable Docker stack",
)
def test_disposable_falkordb_cas_label_and_native_fulltext(monkeypatch):
    import importlib
    import subprocess
    import sys
    import time
    import uuid

    # tests/conftest.py installs a no-op module. Explicit integration must use
    # the real installed client, then restore the stub after this test.
    if not hasattr(sys.modules.get("falkordb"), "__path__"):
        monkeypatch.delitem(sys.modules, "falkordb", raising=False)
    FalkorDB = importlib.import_module("falkordb").FalkorDB

    name = "original-index-backfill-test-" + uuid.uuid4().hex[:12]
    started = False
    try:
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--rm",
                "--name",
                name,
                "--network",
                "bridge",
                "-p",
                "127.0.0.1::6379",
                "--tmpfs",
                "/data",
                "falkordb/falkordb:latest",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        started = True
        binding = (
            subprocess.run(
                ["docker", "port", name, "6379/tcp"], check=True, capture_output=True, text=True
            )
            .stdout.strip()
            .splitlines()[0]
        )
        port = int(binding.rsplit(":", 1)[1])
        for attempt in range(60):
            try:
                graph = FalkorDB(host="127.0.0.1", port=port).select_graph("backfill_test")
                graph.query("RETURN 1")
                break
            except Exception:
                if attempt == 59:
                    raise
                time.sleep(0.25)
        metadata = json.dumps(
            {"was_summarized": True, "original_content": "orchidmarker previous text"}
        )
        graph.query(
            "CREATE (:Memory {id: $id, content: $content, metadata: $metadata})",
            {"id": "a", "content": "summary text", "metadata": metadata},
        )
        graph.query(
            "CREATE (:Memory {id: $id, content: $content})", {"id": "b", "content": "ordinary"}
        )
        dry = backfill.run(graph)
        assert dry["would_apply"] == 2 and dry["unverified_legacy"] == 1, dry
        untouched = graph.query("MATCH (m:Memory) RETURN m.original_search_text").result_set
        assert all(row[0] is None for row in untouched)
        applied = backfill.run(graph, apply=True, batch_size=1)
        assert applied["applied"] == 2 and applied["verification_mismatches"] == 0
        assert backfill.run(graph, apply=True)["applied"] == 0
        graph.query(
            "CALL db.idx.fulltext.createNodeIndex('MemoryOriginalText', 'original_search_text')"
        )
        upgraded = backfill.run(graph, apply=True, assume_legacy_original_current=True)
        assert upgraded["applied"] == 1
        for attempt in range(40):
            result = graph.query(
                "CALL db.idx.fulltext.queryNodes('MemoryOriginalText', 'orchidmarker') "
                "YIELD node RETURN node.id"
            ).result_set
            if ["a"] in result:
                break
            time.sleep(0.25)
        assert ["a"] in result
        assert backfill.run(graph, verify_only=True)["missing_label"] == 0
        params = {
            "id": "a",
            "expected_content": "stale",
            "expected_metadata": metadata,
            "expected_indexed": "orchidmarker previous text",
            "search_text": "unsafe",
        }
        assert graph.query(backfill.CAS_QUERY, params).result_set == []
        # Duplicate IDs are never safe for keyset traversal or single-row CAS.
        graph.query(
            "CREATE (:Memory {id: $id, content: $content, metadata: $metadata})",
            {"id": "a", "content": "summary text", "metadata": metadata},
        )
        duplicate_scan = backfill.run(graph, apply=True)
        assert duplicate_scan["duplicate_id_groups"] == 1
        assert duplicate_scan["applied"] == 0
        params["expected_content"] = "summary text"
        assert graph.query(backfill.CAS_QUERY, params).result_set == []
    finally:
        if started:
            subprocess.run(["docker", "rm", "-f", name], check=True, capture_output=True, text=True)


def test_duplicate_id_preflight_aborts_without_any_writes():
    graph = FakeGraph([row("a")], duplicate_ids=1)
    summary = backfill.run(graph, apply=True)
    assert summary["duplicate_id_groups"] == 1
    assert summary["scanned"] == 0
    assert graph.writes == 0


def test_missing_ids_are_reported_even_when_not_keyset_scannable():
    graph = FakeGraph([row("a")], missing_ids=1)
    summary = backfill.run(graph)
    assert summary["missing_ids"] == 1
    assert summary["scanned"] == 1
