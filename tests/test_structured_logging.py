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
