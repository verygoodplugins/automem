"""Inspect emitted log text, rather than only LogRecord/caplog attributes."""

import io
import json
import logging
import sys

import pytest

from automem.runtime_environment import configure_logging
from tests.test_api_endpoints import auth_headers, client, mock_state  # noqa: F401


@pytest.fixture
def log_output(monkeypatch):
    stream = io.StringIO()
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    monkeypatch.setattr(root, "level", root.level)
    for name in ("werkzeug", "flask.app", "automem.api"):
        logger = logging.getLogger(name)
        monkeypatch.setattr(logger, "handlers", [])
        monkeypatch.setattr(logger, "level", logger.level)
        monkeypatch.setattr(logger, "propagate", True)
    monkeypatch.setattr(sys, "stdout", stream)
    logger = configure_logging()
    return logger, stream


def test_structured_metrics_appear_in_emitted_log(log_output):
    logger, stream = log_output
    logger.info("recall_complete", extra={"latency_ms": 1.25, "results": 2, "vector_enabled": True})
    emitted = stream.getvalue().strip()
    assert "recall_complete" in emitted
    fields = json.loads(emitted.rsplit(" | ", 1)[-1])
    assert fields == {"latency_ms": 1.25, "results": 2, "vector_enabled": True}


def test_plain_logs_tracebacks_and_unserializable_extras(log_output):
    logger, stream = log_output
    logger.info("plain message %s", "ok")
    assert stream.getvalue().strip().endswith("plain message ok")
    circular = []
    circular.append(circular)
    logger.info(
        "extra values", extra={"unsupported": object(), "circular": circular, "lines": "a\nb"}
    )
    assert "extra values" in stream.getvalue()
    assert "circular" in stream.getvalue()
    assert "a\\nb" in stream.getvalue()
    try:
        raise ValueError("expected failure")
    except ValueError:
        logger.exception("operation failed", extra={"attempt": 2})
    assert "Traceback" in stream.getvalue()
    assert "ValueError: expected failure" in stream.getvalue()
    assert '"attempt": 2' in stream.getvalue()


def test_repeated_configuration_keeps_one_output_per_record(log_output, monkeypatch):
    _, stream = log_output
    monkeypatch.setattr(sys, "stdout", stream)
    logger = configure_logging()
    logger.info("once only", extra={"count": 3})
    logging.getLogger("werkzeug").info("framework once", extra={"status": 200})
    assert stream.getvalue().count("once only") == 1
    assert stream.getvalue().count("framework once") == 1
    assert '"count": 3' in stream.getvalue()
    assert '"status": 200' in stream.getvalue()


def test_recall_log_has_metrics_without_query_text(log_output, client, mock_state, auth_headers):
    mock_state.qdrant = None
    response = client.get("/recall?query=PRIVATE_QUERY_65d138e", headers=auth_headers)
    assert response.status_code == 200, response.json
    emitted = log_output[1].getvalue()
    assert "recall_complete" in emitted
    assert '"latency_ms":' in emitted
    assert "PRIVATE_QUERY_65d138e" not in emitted


def test_preconfigured_root_handler_receives_structured_fields(log_output, monkeypatch):
    logger, _ = log_output
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    monkeypatch.setattr(logging.getLogger(), "handlers", [handler])
    configure_logging()
    logger.info("existing handler", extra={"results": 4})
    assert '"results": 4' in stream.getvalue()


@pytest.mark.parametrize("status", [200, 400, 404, 500])
def test_werkzeug_access_logs_redact_query_preserving_request_details(log_output, status):
    from werkzeug.serving import WSGIRequestHandler

    # Use Werkzeug's own access-log argument construction. Flask's test client
    # does not exercise this boundary, which is used by `python app.py`.
    handler = object.__new__(WSGIRequestHandler)
    handler.path = "/recall?query=PRIVATE_ACCESS_SECRET&api_key=PRIVATE_TOKEN"
    handler.command = "GET"
    handler.request_version = "HTTP/1.1"
    handler.address_string = lambda: "127.0.0.1"
    handler.log_date_time_string = lambda: "20/Oct/2026 20:00:00"
    handler.log_request(status, 123)
    emitted = log_output[1].getvalue()
    assert "PRIVATE_ACCESS_SECRET" not in emitted
    assert "PRIVATE_TOKEN" not in emitted
    assert "GET /recall?" in emitted
    assert "HTTP/1.1" in emitted
    assert f'" {status} 123' in emitted


def test_werkzeug_non_query_access_and_framework_messages_are_preserved(log_output):
    from werkzeug.serving import WSGIRequestHandler

    handler = object.__new__(WSGIRequestHandler)
    handler.path = "/health"
    handler.command = "HEAD"
    handler.request_version = "HTTP/1.1"
    handler.address_string = lambda: "127.0.0.1"
    handler.log_date_time_string = lambda: "20/Oct/2026 20:00:00"
    handler.log_request(200, 0)
    logging.getLogger("werkzeug").warning("unrelated framework warning: %s", "details?")
    emitted = log_output[1].getvalue()
    assert '"HEAD /health HTTP/1.1" 200 0' in emitted
    assert "unrelated framework warning: details?" in emitted


def test_debug_vector_search_does_not_log_query(log_output, monkeypatch):
    from types import SimpleNamespace

    from automem.search import runtime_recall_helpers as helpers

    logger, stream = log_output
    monkeypatch.setattr(logger, "level", logging.DEBUG)
    monkeypatch.setattr(helpers, "_logger", logger)
    monkeypatch.setattr(helpers, "_build_qdrant_tag_filter", lambda *args, **kwargs: None)
    monkeypatch.setattr(helpers, "_coerce_embedding", lambda value: value)
    monkeypatch.setattr(helpers, "_collection_name", "test-memories")
    monkeypatch.setattr(helpers, "_generate_real_embedding", lambda _: [0.1, 0.2])
    result = helpers._vector_search(
        SimpleNamespace(search=lambda **kwargs: []),
        None,
        "PRIVATE_VECTOR_QUERY",
        None,
        5,
        set(),
    )
    assert result == []
    assert "Generating embedding" in stream.getvalue()
    assert "PRIVATE_VECTOR_QUERY" not in stream.getvalue()


def test_access_log_redaction_does_not_mutate_the_original_record(log_output):
    args = ("GET /recall?query=PRIVATE_RECORD_QUERY HTTP/1.1", "200", "123")
    record = logging.LogRecord("werkzeug", logging.INFO, __file__, 0, '"%s" %s %s', args, None)
    logging.getLogger("werkzeug").handle(record)
    assert "PRIVATE_RECORD_QUERY" not in log_output[1].getvalue()
    assert record.args is args
    assert "PRIVATE_RECORD_QUERY" in record.getMessage()
