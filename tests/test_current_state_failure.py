"""Current-state verification fails closed if the graph cannot verify replacements."""

import importlib
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from flask import Flask

import automem.search.runtime_recall_helpers as helpers
from automem.api.recall import _active_replacements_for_memories, handle_recall
from automem.utils.text import _extract_keywords
from tests.support.fake_graph import FakeGraph


class FailingGraph:
    def __init__(self, *, original=False, fail_on_hop=1):
        self.original = original
        self.fail_on_hop = fail_on_hop
        self.state_attempts = 0

    def query(self, query, params):
        if "queryNodes('MemoryOriginalText'" in query and self.original:
            if params["offset"]:
                return SimpleNamespace(result_set=[])
            return SimpleNamespace(
                result_set=[
                    [
                        SimpleNamespace(
                            properties={
                                "id": "old",
                                "content": "obsolete private value",
                                "original_search_text": "needle",
                                "tags": ["scope"],
                            }
                        ),
                        1.0,
                    ]
                ]
            )
        if "source_id" in query and "related" in query:
            self.state_attempts += 1
            if self.state_attempts < self.fail_on_hop:
                return SimpleNamespace(
                    result_set=[
                        [
                            "old",
                            "INVALIDATED_BY",
                            0.9,
                            None,
                            SimpleNamespace(properties={"id": "mid", "content": "outdated"}),
                        ]
                    ]
                )
            raise RuntimeError("Query timed out: sensitive backend detail")
        return SimpleNamespace(result_set=[])


def _client(monkeypatch, *, original=False, fail_on_hop=1):
    graph = FailingGraph(original=original, fail_on_hop=fail_on_hop)
    monkeypatch.setattr(helpers, "_serialize_node", lambda node: dict(node.properties))
    monkeypatch.setattr(helpers, "_fetch_relations", lambda *_args: [])
    monkeypatch.setattr(helpers, "_prepare_tag_filters", lambda tags: tags or [])
    monkeypatch.setattr(helpers, "_build_graph_tag_predicate", lambda *_args: "true")
    monkeypatch.setattr(helpers, "_parse_iso_datetime", lambda _value: None)
    monkeypatch.setattr(helpers, "_logger", Mock())
    monkeypatch.setattr(helpers, "_result_passes_filters", lambda *_args: True)
    app = Flask(__name__)

    @app.get("/recall")
    def recall():
        matches = (
            []
            if original
            else [
                {
                    "id": "old",
                    "memory": {"id": "old", "content": "obsolete private value"},
                    "match_type": "vector",
                    "match_score": 0.9,
                    "score": 0.9,
                }
            ]
        )
        return handle_recall(
            get_memory_graph=lambda: graph,
            get_qdrant_client=lambda: object(),
            normalize_tag_list=lambda value: value if isinstance(value, list) else [],
            normalize_timestamp=lambda value: value,
            parse_time_expression=lambda _value: (None, None),
            extract_keywords=_extract_keywords,
            compute_metadata_score=lambda result, *_args: (float(result.get("match_score", 0)), {}),
            result_passes_filters=lambda *_args: True,
            graph_keyword_search=lambda *_args, **_kwargs: [],
            vector_search=lambda *_args, **_kwargs: matches,
            vector_filter_only_tag_search=lambda *_args, **_kwargs: [],
            original_text_search=helpers._original_text_search if original else None,
            recall_max_limit=50,
            logger=Mock(),
        )

    return app.test_client(), graph


def test_vector_result_never_leaks_when_current_state_lookup_fails(monkeypatch):
    client, graph = _client(monkeypatch)
    response = client.get("/recall?query=needle&limit=1&adaptive_floor=false")
    assert response.status_code == 503
    assert graph.state_attempts == 1
    assert b"obsolete private value" not in response.data
    assert b"sensitive backend detail" not in response.data


def test_original_text_candidate_does_not_swallow_current_state_failure(monkeypatch):
    client, graph = _client(monkeypatch, original=True)
    response = client.get("/recall?query=needle&limit=1&adaptive_floor=false")
    assert response.status_code == 503
    assert graph.state_attempts == 1
    assert b"obsolete private value" not in response.data
    assert b"sensitive backend detail" not in response.data


def test_chain_verification_failure_does_not_expose_intermediate_replacement(monkeypatch):
    client, graph = _client(monkeypatch, fail_on_hop=2)
    response = client.get("/recall?query=needle&limit=1&adaptive_floor=false")
    assert response.status_code == 503
    assert graph.state_attempts == 2
    assert b"obsolete private value" not in response.data
    assert b"outdated" not in response.data


def test_history_mode_does_not_require_current_state_lookup(monkeypatch):
    client, graph = _client(monkeypatch)
    response = client.get("/recall?query=needle&limit=1&state_mode=history&adaptive_floor=false")
    assert response.status_code == 200
    assert graph.state_attempts == 0
    assert response.get_json()["results"][0]["memory"]["content"] == "obsolete private value"


def test_index_independent_query_follows_chain_and_ignores_missing_ids():
    graph = FakeGraph()
    for memory_id in ("old", "mid", "current"):
        graph.memories[memory_id] = {"id": memory_id, "content": memory_id}
    graph.relationships = [
        {"id1": "old", "id2": "mid", "type": "INVALIDATED_BY"},
        {"id1": "mid", "id2": "current", "type": "EVOLVED_INTO"},
    ]
    result = _active_replacements_for_memories(
        graph, ["old", "old", "missing"], datetime.now(timezone.utc), Mock()
    )
    assert result["old"]["id"] == "current"
    assert "missing" not in result
    replacement_queries = [query for query, _params in graph.queries if "RETURN source_id" in query]
    assert replacement_queries
    assert all("UNWIND" not in query and "m.id IN $ids" in query for query in replacement_queries)


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_ORIGINAL_INDEX_DOCKER_TESTS") != "1",
    reason="requires explicitly enabled disposable Docker stack",
)
def test_real_falkordb_replacement_query_scans_once_and_preserves_chain(monkeypatch):
    # The regular test harness stubs this module; integration uses the real client.
    if not hasattr(sys.modules.get("falkordb"), "__path__"):
        monkeypatch.delitem(sys.modules, "falkordb", raising=False)
    falkordb = importlib.import_module("falkordb")
    name = "original-state-test-" + uuid.uuid4().hex[:12]
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
                graph = falkordb.FalkorDB(host="127.0.0.1", port=port).select_graph("state_test")
                graph.query("RETURN 1")
                break
            except Exception:
                if attempt == 59:
                    raise
                time.sleep(0.25)
        graph.query(
            "UNWIND range(0,5881) AS i "
            "CREATE (:Memory {id:'id-'+toString(i),timestamp:'2022-01-01T00:00:00Z'})"
        )
        graph.query(
            "CREATE (:Memory {id:'expired',t_invalid:'2020-01-01T00:00:00Z'}),"
            "(:Memory {id:'mid'}),(:Memory {id:'head'})"
        )
        graph.query(
            "MATCH (m:Memory),(r:Memory) WHERE m.id='id-1' AND r.id='expired' "
            "CREATE (m)-[:INVALIDATED_BY {updated_at:'2024',strength:1.0}]->(r)"
        )
        graph.query(
            "MATCH (m:Memory),(r:Memory) WHERE m.id='id-1' AND r.id='mid' "
            "CREATE (m)-[:INVALIDATED_BY {updated_at:'2023',strength:0.8,kind:'correction'}]->(r)"
        )
        graph.query(
            "MATCH (m:Memory),(r:Memory) WHERE m.id='mid' AND r.id='head' "
            "CREATE (m)-[:EVOLVED_INTO {updated_at:'2022'}]->(r)"
        )

        class RecordingGraph:
            def __init__(self, inner):
                self.inner = inner
                self.calls = []

            def query(self, query, params):
                self.calls.append((query, params))
                return self.inner.query(query, params)

        recorded = RecordingGraph(graph)
        result = _active_replacements_for_memories(
            recorded, ["id-1", "id-1", "missing"], datetime.now(timezone.utc), Mock()
        )
        assert result["id-1"]["id"] == "head"
        assert result["id-1"]["_state_relation"] == {
            "type": "INVALIDATED_BY",
            "strength": 0.8,
            "kind": "correction",
        }
        assert "missing" not in result
        query, params = recorded.calls[0]
        assert params["ids"] == ["id-1", "missing"]
        plan = str(graph.explain(query, params))
        assert "Node By Label Scan" in plan
        assert "Unwind" not in plan
    finally:
        if started:
            subprocess.run(["docker", "rm", "-f", name], check=True, capture_output=True, text=True)
