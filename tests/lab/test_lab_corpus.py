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
