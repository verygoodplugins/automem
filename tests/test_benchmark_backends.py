from tests.benchmarks.backends import AutoMemBackend, SearchRequest
from tests.benchmarks.longmemeval.evaluator import LongMemEvalScorer


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
            query="degree", scope_id="e47becba", tags=["longmemeval:e47becba"], tag_match="exact"
        )
    )

    assert seen["params"]["tags"] == ["bench:e47becba", "longmemeval:e47becba"]
    assert seen["params"]["tag_mode"] == "all"
    assert seen["params"]["tag_match"] == "exact"


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
