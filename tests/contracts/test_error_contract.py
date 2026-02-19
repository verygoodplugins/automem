from __future__ import annotations

from typing import Any

import pytest

import app
from tests.support.fake_graph import FakeGraph


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch: pytest.MonkeyPatch):
    state = app.ServiceState()
    state.memory_graph = FakeGraph()

    monkeypatch.setattr(app, "state", state)
    monkeypatch.setattr(app, "init_falkordb", lambda: None)
    monkeypatch.setattr(app, "init_qdrant", lambda: None)
    monkeypatch.setattr(app, "init_openai", lambda: None)
    monkeypatch.setattr(app, "API_TOKEN", "test-token")
    monkeypatch.setattr(app, "ADMIN_TOKEN", "test-admin-token")

    original_testing = app.app.config.get("TESTING")
    original_propagate = app.app.config.get("PROPAGATE_EXCEPTIONS")
    app.app.config["TESTING"] = True
    app.app.config["PROPAGATE_EXCEPTIONS"] = False

    yield

    app.app.config["TESTING"] = original_testing
    app.app.config["PROPAGATE_EXCEPTIONS"] = original_propagate


@pytest.fixture
def client():
    with app.app.test_client() as client:
        yield client


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-token"}


def _assert_error_envelope(
    response: Any, expected_code: int, expected_substring: str | None = None
) -> None:
    assert response.status_code == expected_code

    body = response.get_json()
    assert isinstance(body, dict)
    assert body.get("status") == "error"
    assert body.get("code") == expected_code
    assert isinstance(body.get("message"), str)
    assert body.get("message")

    if expected_substring:
        assert expected_substring.lower() in str(body.get("message", "")).lower()


def test_error_envelope_for_validation_error(client, auth_headers) -> None:
    response = client.post("/memory", json={}, headers=auth_headers)
    _assert_error_envelope(response, 400, "content")


def test_error_envelope_for_not_found_error(client, auth_headers) -> None:
    response = client.get("/does-not-exist", headers=auth_headers)
    _assert_error_envelope(response, 404)


def test_error_envelope_for_unhandled_error(
    client, auth_headers, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise(_content: str) -> tuple[str, float]:
        raise RuntimeError("boom")

    monkeypatch.setattr(app.memory_classifier, "classify", _raise)

    response = client.post("/memory", json={"content": "trigger exception"}, headers=auth_headers)
    _assert_error_envelope(response, 500, "internal server error")
