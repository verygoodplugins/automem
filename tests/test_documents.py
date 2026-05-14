"""Tests for the ``/documents`` blueprint.

Uses a self-contained Flask app wired only with the documents blueprint and a
FakeBucketStore — independent of the bigger ``app.py`` bootstrap so we're
testing the blueprint contract directly. The bigger integration test suite
covers end-to-end wiring.
"""

from __future__ import annotations

import io
import json
import logging
from types import SimpleNamespace
from typing import List

import pytest
from flask import Flask

from automem.api.documents import create_documents_blueprint
from tests.support.fake_bucket import FakeBucketStore
from tests.support.fake_graph import FakeGraph


def _normalize_tags(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [t.strip() for t in value.split(",") if t.strip()]
    if isinstance(value, list):
        return [str(t).strip() for t in value if str(t).strip()]
    return []


def _compute_tag_prefixes(tags_lower: List[str]) -> List[str]:
    prefixes = []
    for tag in tags_lower:
        for i in range(1, len(tag) + 1):
            prefixes.append(tag[:i])
    return prefixes


def _coerce_importance(value) -> float:
    if value is None or value == "":
        return 0.5
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, f))


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------ fixtures
@pytest.fixture
def fake_bucket() -> FakeBucketStore:
    return FakeBucketStore(bucket="test-docs")


@pytest.fixture
def fake_graph() -> FakeGraph:
    return FakeGraph()


@pytest.fixture
def flask_app(fake_bucket: FakeBucketStore, fake_graph: FakeGraph):
    """Fresh Flask app with only the documents blueprint wired to the fakes."""
    app = Flask(__name__)
    state = SimpleNamespace(
        enrichment_queue=SimpleNamespace(qsize=lambda: 0),
        memory_graph=fake_graph,
    )

    bp = create_documents_blueprint(
        bucket_store=fake_bucket,
        get_memory_graph_fn=lambda: fake_graph,
        get_qdrant_client_fn=lambda: None,
        normalize_tags_fn=_normalize_tags,
        compute_tag_prefixes_fn=_compute_tag_prefixes,
        coerce_importance_fn=_coerce_importance,
        enqueue_enrichment_fn=lambda *_: None,
        enqueue_embedding_fn=lambda *_: None,
        collection_name="memories",
        utc_now_fn=_utc_now,
        state=state,
        qdrant_models_obj=None,
        max_bytes=10 * 1024 * 1024,
        presigned_expires=300,
    )
    app.register_blueprint(bp)
    app.logger.setLevel(logging.CRITICAL)
    return app


@pytest.fixture
def client(flask_app):
    return flask_app.test_client()


@pytest.fixture
def noop_bucket_app(fake_graph: FakeGraph):
    """App with bucket_store=None, for the unconfigured path."""
    app = Flask(__name__)
    state = SimpleNamespace(
        enrichment_queue=SimpleNamespace(qsize=lambda: 0),
        memory_graph=fake_graph,
    )
    bp = create_documents_blueprint(
        bucket_store=None,
        get_memory_graph_fn=lambda: fake_graph,
        get_qdrant_client_fn=lambda: None,
        normalize_tags_fn=_normalize_tags,
        compute_tag_prefixes_fn=_compute_tag_prefixes,
        coerce_importance_fn=_coerce_importance,
        enqueue_enrichment_fn=lambda *_: None,
        enqueue_embedding_fn=lambda *_: None,
        collection_name="memories",
        utc_now_fn=_utc_now,
        state=state,
        qdrant_models_obj=None,
        max_bytes=10 * 1024 * 1024,
        presigned_expires=300,
    )
    app.register_blueprint(bp)
    app.logger.setLevel(logging.CRITICAL)
    return app.test_client()


# ------------------------------------------------------------------- helpers
def _multipart(
    *,
    title="Test Doc",
    summary="A test document containing example content.",
    data=b"hello world",
    filename="test.txt",
    mime="text/plain",
    **extra,
):
    form = {"file": (io.BytesIO(data), filename, mime)}
    if title is not None:
        form["title"] = title
    if summary is not None:
        form["summary"] = summary
    form.update(extra)
    return form


# -------------------------------------------------------------- tests: gate
def test_upload_without_title_returns_422(client, fake_bucket):
    response = client.post(
        "/documents",
        data=_multipart(title=""),
        content_type="multipart/form-data",
    )
    assert response.status_code == 422
    assert fake_bucket.upload_calls == 0


def test_upload_without_summary_returns_422(client, fake_bucket):
    response = client.post(
        "/documents",
        data=_multipart(summary=""),
        content_type="multipart/form-data",
    )
    assert response.status_code == 422
    assert fake_bucket.upload_calls == 0


def test_upload_without_file_returns_400(client, fake_bucket):
    response = client.post(
        "/documents",
        data={"title": "t", "summary": "s"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert fake_bucket.upload_calls == 0


def test_upload_title_too_long_returns_400(client, fake_bucket):
    response = client.post(
        "/documents",
        data=_multipart(title="x" * 301),
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert fake_bucket.upload_calls == 0


# ------------------------------------------------------- tests: happy path
def test_upload_creates_memory_and_bucket_object(client, fake_bucket, fake_graph):
    response = client.post(
        "/documents",
        data=_multipart(
            title="Q2 Budget Report",
            summary="Quarterly budget numbers for marketing.",
            data=b"PDF pretend bytes",
            filename="q2.pdf",
            mime="application/pdf",
            tags=json.dumps(["finance", "q2"]),
            importance="0.8",
        ),
        content_type="multipart/form-data",
    )
    assert response.status_code == 201, response.get_data(as_text=True)
    body = response.get_json()

    assert body["status"] == "success"
    assert body["type"] == "Document"
    assert body["title"] == "Q2 Budget Report"
    assert body["document"]["mime"] == "application/pdf"
    assert body["document"]["size"] == len(b"PDF pretend bytes")
    assert body["document"]["sha256"]
    assert body["download_url"].startswith("https://fake.bucket.invalid/")
    tags_lower = [t.lower() for t in body["tags"]]
    assert "document" in tags_lower  # auto-added
    assert "finance" in tags_lower

    assert fake_bucket.upload_calls == 1
    key = body["document"]["bucket_key"]
    assert key in fake_bucket.objects
    assert fake_bucket.objects[key]["data"] == b"PDF pretend bytes"
    assert fake_bucket.objects[key]["mime"] == "application/pdf"

    # Memory node created in graph with type=Document
    mem = fake_graph.memories.get(body["memory_id"])
    assert mem is not None
    assert mem["type"] == "Document"
    assert mem["content"].startswith("Q2 Budget Report")


def test_upload_uses_provided_memory_id(client, fake_bucket):
    custom_id = "11111111-1111-1111-1111-111111111111"
    response = client.post(
        "/documents",
        data=_multipart(memory_id=custom_id),
        content_type="multipart/form-data",
    )
    assert response.status_code == 201
    assert response.get_json()["memory_id"] == custom_id


def test_upload_rejects_invalid_memory_id(client):
    response = client.post(
        "/documents",
        data=_multipart(memory_id="not-a-uuid"),
        content_type="multipart/form-data",
    )
    assert response.status_code == 400


# ----------------------------------------------------- tests: download URL
def test_download_returns_presigned_url(client, fake_bucket):
    upload = client.post(
        "/documents",
        data=_multipart(),
        content_type="multipart/form-data",
    )
    memory_id = upload.get_json()["memory_id"]

    response = client.get(f"/documents/{memory_id}/download")
    assert response.status_code == 200
    body = response.get_json()
    assert body["download_url"].startswith("https://fake.bucket.invalid/")
    assert "expires_in=" in body["download_url"]
    assert body["mime"] == "text/plain"
    assert fake_bucket.presign_calls >= 2  # upload returned one, download one more


def test_download_missing_doc_returns_404(client):
    missing = "22222222-2222-2222-2222-222222222222"
    response = client.get(f"/documents/{missing}/download")
    assert response.status_code == 404


def test_download_invalid_uuid_returns_400(client):
    response = client.get("/documents/not-a-uuid/download")
    assert response.status_code == 400


# -------------------------------------------------------------- tests: delete
def test_delete_removes_memory_and_bucket(client, fake_bucket, fake_graph):
    upload = client.post(
        "/documents",
        data=_multipart(),
        content_type="multipart/form-data",
    )
    memory_id = upload.get_json()["memory_id"]
    key = upload.get_json()["document"]["bucket_key"]
    assert key in fake_bucket.objects

    response = client.delete(f"/documents/{memory_id}")
    assert response.status_code == 200
    body = response.get_json()
    assert body["graph"] == "deleted"
    assert body["bucket"] == "deleted"
    assert key not in fake_bucket.objects
    assert fake_bucket.delete_calls == 1
    assert memory_id not in fake_graph.memories


def test_delete_missing_doc_returns_404(client):
    response = client.delete("/documents/33333333-3333-3333-3333-333333333333")
    assert response.status_code == 404


# --------------------------------------------------- tests: unconfigured
def test_upload_returns_503_when_bucket_unconfigured(noop_bucket_app):
    response = noop_bucket_app.post(
        "/documents",
        data=_multipart(),
        content_type="multipart/form-data",
    )
    assert response.status_code == 503
    text = response.get_data(as_text=True)
    assert "S3" in text or "bucket" in text.lower()


def test_download_returns_503_when_bucket_unconfigured(noop_bucket_app):
    response = noop_bucket_app.get("/documents/11111111-1111-1111-1111-111111111111/download")
    assert response.status_code == 503
