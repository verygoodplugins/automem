"""Admin CORS and HTTP error propagation through the registered Flask app."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_api_endpoints import admin_headers, auth_headers, client, mock_state  # noqa: F401


@pytest.mark.parametrize("path", ["/admin/sync", "/backup"])
@pytest.mark.parametrize("header", ["X-Admin-Token", "X-Admin-Api-Key"])
def test_admin_headers_allowed_by_browser_preflight(client, path, header):
    response = client.options(
        path,
        headers={
            "Origin": "https://admin.example.com",
            "Access-Control-Request-Method": "GET" if path == "/backup" else "POST",
            "Access-Control-Request-Headers": f"Authorization, {header}",
        },
    )
    assert response.status_code in (200, 204)
    allowed = response.headers.get("Access-Control-Allow-Headers", "").lower()
    assert header.lower() in allowed
    assert "authorization" in allowed


def test_admin_preflight_does_not_bypass_real_auth(client, mock_state, auth_headers):
    assert client.post("/admin/sync", headers=auth_headers).status_code == 401
    assert client.get("/backup", headers=auth_headers).status_code == 401


def test_cors_preserves_configured_origin_restriction():
    # CORS origins are fixed when app.py is imported; isolate that startup rather
    # than reloading the shared test application's registered blueprints.
    script = """
import json
import app
with app.app.test_client() as client:
    result = {}
    for origin in ("https://allowed.example.com", "https://denied.example.com"):
        response = client.options("/admin/sync", headers={
            "Origin": origin, "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "X-Admin-Token",
        })
        result[origin] = response.headers.get("Access-Control-Allow-Origin")
    print(json.dumps(result))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "VIEWER_ALLOWED_ORIGINS": "https://allowed.example.com"},
        capture_output=True,
        text=True,
        check=True,
    )
    origins = json.loads(result.stdout.strip().splitlines()[-1])
    assert origins == {
        "https://allowed.example.com": "https://allowed.example.com",
        "https://denied.example.com": None,
    }


@pytest.mark.parametrize("query", ["limit=abc", "offset=-x"])
def test_audit_bad_pagination_uses_standard_400(client, mock_state, admin_headers, query):
    response = client.get(f"/entities/audit?{query}", headers=admin_headers)
    assert response.status_code == 400
    assert response.json == {
        "status": "error",
        "code": 400,
        "message": "Invalid limit/offset parameter",
    }


def test_audit_valid_pagination_and_backend_failure(client, mock_state, admin_headers, monkeypatch):
    response = client.get("/entities/audit?summary=true&limit=999&offset=-1", headers=admin_headers)
    assert response.status_code == 200, response.json
    assert response.json["limit"] == 500
    assert response.json["offset"] == 0

    def fail(*args, **kwargs):
        raise RuntimeError("graph unavailable")

    monkeypatch.setattr(mock_state.memory_graph, "query", fail)
    response = client.get("/entities/audit?summary=true", headers=admin_headers)
    assert response.status_code == 500
    assert response.json["error"] == "Failed to audit entities"


@pytest.mark.parametrize("header", ["X-Admin-Token", "X-Admin-Api-Key"])
def test_admin_headers_still_require_the_correct_token(client, mock_state, auth_headers, header):
    path = "/entities/audit?summary=true"
    assert client.get(path, headers={**auth_headers, header: "wrong-token"}).status_code == 401
    response = client.get(path, headers={**auth_headers, header: "test-admin-token"})
    assert response.status_code == 200
