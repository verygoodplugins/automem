import lab_corpus as c


class FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_recall_serializes_params():
    captured = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        return FakeResp({"results": []})

    c.recall(
        "http://x",
        {"Authorization": "Bearer t"},
        "hello",
        limit=10,
        expand_relations=True,
        current_only=False,
        recency_bias="auto",
        http_get=fake_get,
    )
    assert captured["url"].endswith("/recall")
    assert captured["params"]["query"] == "hello"
    assert captured["params"]["limit"] == 10
    assert captured["params"]["expand_relations"] == "true"
    assert captured["params"]["current_only"] == "false"
    assert captured["params"]["recency_bias"] == "auto"


def test_recall_omits_recency_bias_when_none():
    captured = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["params"] = params
        return FakeResp({"results": []})

    c.recall("http://x", {}, "q", http_get=fake_get)
    assert "recency_bias" not in captured["params"]
    assert captured["params"]["current_only"] == "true"


def test_extract_ids_handles_nested_and_flat():
    payload = {
        "results": [
            {"memory": {"id": "a", "content": "x"}},
            {"id": "b", "content": "y"},
        ]
    }
    assert c.extract_ids(payload) == ["a", "b"]


def test_make_distractor_memories_are_aged_low_importance_and_tagged():
    payloads = c.make_distractor_memories(3, age_days=200, importance=0.05)
    assert len(payloads) == 3
    for p in payloads:
        assert p["tags"] == ["lab-distractor"]
        assert p["importance"] == 0.05
        assert p["timestamp"] == p["last_accessed"]
        assert p["metadata"]["lab_distractor"] is True
        assert p["timestamp"].endswith("Z")


def test_inject_distractors_returns_created_ids():
    posted = []

    def fake_post(url, json=None, headers=None, timeout=None):
        posted.append(json)
        idx = len(posted)
        return FakeResp({"memory_id": f"id-{idx}"})

    payloads = c.make_distractor_memories(2)
    ids = c.inject_distractors("http://x", {}, payloads, http_post=fake_post)
    assert ids == ["id-1", "id-2"]
    assert len(posted) == 2


def test_run_consolidation_sends_dry_run_false_in_order():
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(json)
        return FakeResp({"mode": json["mode"], "steps": {}})

    out = c.run_consolidation("http://x", {}, http_post=fake_post)
    sent_modes = [call["mode"] for call in calls]
    assert sent_modes == ["decay", "creative", "cluster", "forget"]
    assert all(call["dry_run"] is False for call in calls)
    assert set(out.keys()) == {"decay", "creative", "cluster", "forget"}


def test_run_consolidation_respects_explicit_modes():
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(json)
        return FakeResp({})

    c.run_consolidation("http://x", {}, modes=["forget"], http_post=fake_post)
    assert [call["mode"] for call in calls] == ["forget"]
