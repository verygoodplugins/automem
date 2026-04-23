import json

from tests.benchmarks.backends import AutoMemBackend, SearchRequest
from tests.benchmarks.longmemeval.evaluator import LongMemEvalScorer
from tests.benchmarks.test_locomo import LoCoMoConfig, LoCoMoEvaluator


class _Response:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.ok = 200 <= status_code < 300

    def json(self):
        return self._payload


def test_automem_backend_search_flattens_results(monkeypatch):
    backend = AutoMemBackend(base_url="http://example.test", api_token="token")

    def fake_get(*args, **kwargs):
        return _Response(
            payload={
                "results": [
                    {
                        "memory": {
                            "id": "mem-1",
                            "content": "User likes dark mode",
                            "metadata": {"session_id": "sess-1"},
                            "tags": ["conversation:demo"],
                        },
                        "score": 0.9,
                        "match_type": "semantic",
                    }
                ]
            }
        )

    monkeypatch.setattr("tests.benchmarks.backends.requests.get", fake_get)

    results = backend.search(
        SearchRequest(query="dark mode", scope_id="demo", tags=["conversation:demo"])
    )

    assert len(results) == 1
    assert results[0].id == "mem-1"
    assert results[0].metadata["session_id"] == "sess-1"
    assert results[0].score == 0.9


def test_automem_backend_cleanup_prefers_tracked_ids(monkeypatch):
    backend = AutoMemBackend(base_url="http://example.test", api_token="token")
    backend.remember_created_id("scope-1", "mem-1")
    backend.remember_created_id("scope-1", "mem-2")

    deleted_ids = []

    def fake_delete(*args, **kwargs):
        deleted_ids.append(args[0].rsplit("/", 1)[-1])
        return _Response(status_code=204)

    def fail_get(*args, **kwargs):
        raise AssertionError("cleanup should not query recall when tracked ids exist")

    monkeypatch.setattr("tests.benchmarks.backends.requests.delete", fake_delete)
    monkeypatch.setattr("tests.benchmarks.backends.requests.get", fail_get)

    deleted = backend.cleanup_scope("scope-1")

    assert deleted == 2
    assert sorted(deleted_ids) == ["mem-1", "mem-2"]
    assert "scope-1" not in backend.created_ids_by_scope


def test_automem_backend_search_uses_physical_scope_tag(monkeypatch):
    backend = AutoMemBackend(
        base_url="http://example.test", api_token="token", scope_prefix="bench"
    )
    seen = {}

    def fake_get(*args, **kwargs):
        seen["params"] = kwargs["params"]
        return _Response(payload={"results": []})

    monkeypatch.setattr("tests.benchmarks.backends.requests.get", fake_get)

    backend.search(
        SearchRequest(
            query="degree",
            scope_id="e47becba",
            tags=["longmemeval:e47becba"],
            tag_mode="all",
            tag_match="exact",
        )
    )

    assert seen["params"]["tags"] == ["bench:e47becba", "longmemeval:e47becba"]
    assert seen["params"]["tag_mode"] == "all"
    assert seen["params"]["tag_match"] == "exact"


def test_automem_backend_search_emulates_any_tag_mode_with_scope(monkeypatch):
    backend = AutoMemBackend(
        base_url="http://example.test", api_token="token", scope_prefix="bench"
    )
    seen_params = []

    def fake_get(*args, **kwargs):
        params = kwargs["params"]
        seen_params.append(params)
        tag = params["tags"][1]
        if tag == "tag-a":
            return _Response(
                payload={
                    "results": [
                        {
                            "memory": {
                                "id": "mem-1",
                                "content": "first",
                                "metadata": {},
                                "tags": [],
                            },
                            "score": 0.7,
                            "match_type": "semantic",
                        },
                        {
                            "memory": {
                                "id": "mem-2",
                                "content": "second",
                                "metadata": {},
                                "tags": [],
                            },
                            "score": 0.4,
                            "match_type": "semantic",
                        },
                    ]
                }
            )
        return _Response(
            payload={
                "results": [
                    {
                        "memory": {
                            "id": "mem-1",
                            "content": "first",
                            "metadata": {},
                            "tags": [],
                        },
                        "score": 0.9,
                        "match_type": "semantic",
                    },
                    {
                        "memory": {
                            "id": "mem-3",
                            "content": "third",
                            "metadata": {},
                            "tags": [],
                        },
                        "score": 0.8,
                        "match_type": "semantic",
                    },
                ]
            }
        )

    monkeypatch.setattr("tests.benchmarks.backends.requests.get", fake_get)

    results = backend.search(
        SearchRequest(
            query="degree",
            scope_id="e47becba",
            tags=["tag-a", "tag-b"],
            tag_mode="any",
            tag_match="exact",
            limit=2,
        )
    )

    assert [params["tags"] for params in seen_params] == [
        ["bench:e47becba", "tag-a"],
        ["bench:e47becba", "tag-b"],
    ]
    assert all(params["tag_mode"] == "all" for params in seen_params)
    assert [record.id for record in results] == ["mem-1", "mem-3"]
    assert [record.score for record in results] == [0.9, 0.8]


def test_automem_backend_ingest_pauses_between_batches(monkeypatch):
    backend = AutoMemBackend(base_url="http://example.test", api_token="token")
    memories = [
        {"content": "one", "_benchmark_id": "bench-1"},
        {"content": "two", "_benchmark_id": "bench-2"},
    ]
    sleep_calls = []

    monkeypatch.setattr(backend, "_supports_batch", lambda: True)
    monkeypatch.setattr("tests.benchmarks.backends.time.sleep", sleep_calls.append)

    def fake_post(*args, **kwargs):
        payload = kwargs["json"]["memories"]
        memory_id = payload[0]["content"]
        return _Response(status_code=201, payload={"memory_ids": [f"mem-{memory_id}"]})

    monkeypatch.setattr("tests.benchmarks.backends.requests.post", fake_post)

    memory_map = backend.ingest_memories(
        memories,
        scope_id="scope-1",
        batch_size=1,
        pause_between_batches=0.25,
    )

    assert memory_map == {"bench-1": "mem-one", "bench-2": "mem-two"}
    assert sleep_calls == [0.25]


class _LoCoMoBackend:
    name = "automem"

    def __init__(self, scope_prefix="initial") -> None:
        self.scope_prefix = scope_prefix
        self.cleaned = []

    def health_check(self):
        return True

    def cleanup_scope(self, scope_id: str):
        self.cleaned.append(scope_id)
        return 1


def test_locomo_cleanup_uses_sample_ids(monkeypatch, tmp_path):
    data_file = tmp_path / "locomo.json"
    data_file.write_text("[]")
    backend = _LoCoMoBackend()

    monkeypatch.setattr(
        "tests.benchmarks.test_locomo.create_backend", lambda *args, **kwargs: backend
    )

    evaluator = LoCoMoEvaluator(LoCoMoConfig(data_file=str(data_file)))

    assert evaluator.cleanup_test_data(["sample-a", "sample-b"]) is True
    assert backend.cleaned == ["sample-a", "sample-b"]


def test_locomo_eval_only_reuses_scope_prefix_from_manifest(monkeypatch, tmp_path):
    data_file = tmp_path / "locomo.json"
    data_file.write_text(json.dumps([]))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "scope_prefix": "locomo-automem-fixed",
                "conversations": {},
            }
        )
    )
    backend = _LoCoMoBackend(scope_prefix="locomo-automem-initial")

    monkeypatch.setattr(
        "tests.benchmarks.test_locomo.create_backend", lambda *args, **kwargs: backend
    )

    evaluator = LoCoMoEvaluator(LoCoMoConfig(data_file=str(data_file)))
    evaluator.run_benchmark(eval_only=True)

    assert backend.scope_prefix == "locomo-automem-fixed"


def test_longmemeval_scorer_tracks_retrieval_metrics():
    scorer = LongMemEvalScorer()
    scorer.add_result(
        {
            "question_type": "single-session-user",
            "is_correct": True,
            "is_abstention": False,
            "recall_hit_at_5": True,
        }
    )
    scorer.add_result(
        {
            "question_type": "single-session-user",
            "is_correct": False,
            "is_abstention": False,
            "recall_hit_at_5": False,
        }
    )
    scorer.add_result(
        {
            "question_type": "multi-session",
            "is_correct": True,
            "is_abstention": False,
            "recall_hit_at_5": True,
        }
    )

    scores = scorer.compute_scores()

    assert scores["overall"]["accuracy"] == 2 / 3
    assert scores["retrieval"]["recall_any_at_5"] == 2 / 3
    assert scores["retrieval_by_type"]["single-session-user"]["recall_any_at_5"] == 0.5
    assert scores["retrieval_by_type"]["multi-session"]["recall_any_at_5"] == 1.0
