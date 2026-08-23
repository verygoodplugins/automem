from __future__ import annotations

from datetime import timedelta

from automem.service_state import EnrichmentCircuit


def test_quota_failure_opens_circuit_and_skips_requests():
    circuit = EnrichmentCircuit(cooldown_seconds=60, clock=lambda: 100.0)

    assert circuit.allow_request() is True
    assert circuit.record_failure("429 insufficient_quota") is True
    assert circuit.allow_request() is False
    assert circuit.to_dict()["circuit_open_skips"] == 1


def test_non_quota_failure_does_not_open_circuit():
    circuit = EnrichmentCircuit(cooldown_seconds=60, clock=lambda: 100.0)

    assert circuit.record_failure("connection reset") is False
    assert circuit.allow_request() is True


def test_successful_probe_closes_circuit_and_records_recovery():
    now = [100.0]
    circuit = EnrichmentCircuit(cooldown_seconds=60, clock=lambda: now[0])

    circuit.record_failure("insufficient_quota")
    now[0] += 60
    assert circuit.allow_request() is True
    circuit.record_success()

    assert circuit.allow_request() is True
    assert circuit.to_dict()["recoveries"] == 1
