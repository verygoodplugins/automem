from __future__ import annotations

import logging
from types import SimpleNamespace

from automem.classification.memory_classifier import MemoryClassifier
from automem.service_state import EnrichmentCircuit
from automem.utils.text import summarize_content


def test_quota_failure_opens_circuit_and_skips_requests():
    circuit = EnrichmentCircuit(cooldown_seconds=60, clock=lambda: 100.0)

    allowed, _ = circuit.begin_request()
    assert allowed is True
    assert circuit.record_failure("429 insufficient_quota") is True
    allowed, _ = circuit.begin_request()
    assert allowed is False
    assert circuit.to_dict()["circuit_open_skips"] == 1


def test_non_quota_failure_does_not_open_circuit():
    circuit = EnrichmentCircuit(cooldown_seconds=60, clock=lambda: 100.0)

    assert circuit.record_failure("connection reset") is False
    allowed, _ = circuit.begin_request()
    assert allowed is True


def test_successful_probe_closes_circuit_and_records_recovery():
    now = [100.0]
    circuit = EnrichmentCircuit(cooldown_seconds=60, clock=lambda: now[0])

    circuit.record_failure("insufficient_quota")
    now[0] += 60
    allowed, was_probe = circuit.begin_request()
    assert allowed is True
    assert was_probe is True
    circuit.record_success(was_probe)

    allowed, _ = circuit.begin_request()
    assert allowed is True
    assert circuit.to_dict()["recoveries"] == 1


def test_failed_probe_does_not_permanently_block_future_requests():
    now = [100.0]
    circuit = EnrichmentCircuit(cooldown_seconds=60, clock=lambda: now[0])

    circuit.record_failure("insufficient_quota")
    now[0] += 60
    allowed, was_probe = circuit.begin_request()
    assert allowed is True
    assert was_probe is True
    assert circuit.record_failure("connection reset", was_probe) is False
    allowed, _ = circuit.begin_request()
    assert allowed is True


def test_inflight_success_does_not_close_quota_circuit():
    now = [100.0]
    circuit = EnrichmentCircuit(cooldown_seconds=60, clock=lambda: now[0])

    first_allowed, _ = circuit.begin_request()
    second_allowed, second_was_probe = circuit.begin_request()
    assert first_allowed is True
    assert second_allowed is True
    assert circuit.record_failure("insufficient_quota") is True

    circuit.record_success(second_was_probe)

    allowed, _ = circuit.begin_request()
    assert allowed is False


def test_classifier_makes_no_second_request_while_circuit_is_open():
    calls = []

    def create(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("insufficient_quota")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    circuit = EnrichmentCircuit(cooldown_seconds=60, clock=lambda: 100.0)
    classifier = MemoryClassifier(
        normalize_memory_type=lambda raw: (raw, False),
        ensure_openai_client=lambda: None,
        get_openai_client=lambda: client,
        classification_model="gpt-4o-mini",
        logger=logging.getLogger(__name__),
        circuit=circuit,
    )

    classifier.classify("qwxz flibber jabberwock snorkelblatt")
    classifier.classify("qwxz flibber jabberwock snorkelblatt")

    assert len(calls) == 1


def test_summarizer_makes_no_second_request_while_circuit_is_open():
    calls = []

    def create(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("insufficient_quota")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    circuit = EnrichmentCircuit(cooldown_seconds=60, clock=lambda: 100.0)
    content = "x" * 600

    summarize_content(content, client, "gpt-4o-mini", 300, circuit)
    summarize_content(content, client, "gpt-4o-mini", 300, circuit)

    assert len(calls) == 1
