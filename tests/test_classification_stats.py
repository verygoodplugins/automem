"""Tests for ClassificationStats and MemoryClassifier stats instrumentation."""

from __future__ import annotations

import logging
from threading import Thread
from types import SimpleNamespace

from automem.classification.memory_classifier import MemoryClassifier
from automem.service_state import ClassificationStats, ServiceState

LOGGER = logging.getLogger("test_classification_stats")

# Content that matches none of the MemoryClassifier.PATTERNS regexes.
NON_PATTERN_CONTENT = "qwxz flibber jabberwock snorkelblatt"


def _stub_client(create_fn):
    completions = SimpleNamespace(create=create_fn)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def _failing_client(message: str = "429 insufficient_quota"):
    def _raise(*args, **kwargs):
        raise RuntimeError(message)

    return _stub_client(_raise)


def _succeeding_client(memory_type: str = "Insight", confidence: float = 0.9):
    def _create(*args, **kwargs):
        content = '{"type": "%s", "confidence": %s}' % (memory_type, confidence)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    return _stub_client(_create)


def _make_classifier(client, stats):
    return MemoryClassifier(
        normalize_memory_type=lambda raw: (raw, False),
        ensure_openai_client=lambda: None,
        get_openai_client=lambda: client,
        classification_model="gpt-4o-mini",
        logger=LOGGER,
        stats=stats,
    )


def test_record_methods_increment_counters():
    stats = ClassificationStats()
    stats.record_pattern()
    stats.record_llm_attempt()
    stats.record_llm_success()
    stats.record_fallback()

    assert stats.pattern_classifications == 1
    assert stats.llm_attempts == 1
    assert stats.llm_successes == 1
    assert stats.fallbacks == 1


def test_record_fallback_with_error_sets_error_fields():
    stats = ClassificationStats()
    stats.record_fallback("boom")

    assert stats.fallbacks == 1
    assert stats.last_error == "boom"
    assert stats.last_error_at is not None


def test_record_fallback_without_error_keeps_error_fields():
    stats = ClassificationStats()
    stats.record_fallback()

    assert stats.fallbacks == 1
    assert stats.last_error is None
    assert stats.last_error_at is None


def test_to_dict_shape():
    stats = ClassificationStats()
    assert stats.to_dict() == {
        "llm_attempts": 0,
        "llm_successes": 0,
        "fallbacks": 0,
        "pattern_classifications": 0,
        "last_error": None,
        "last_error_at": None,
    }


def test_service_state_has_classification_stats():
    state = ServiceState()
    assert isinstance(state.classification_stats, ClassificationStats)


def test_pattern_hit_counts_pattern_classification():
    stats = ClassificationStats()
    classifier = _make_classifier(_failing_client(), stats)

    memory_type, _ = classifier.classify("decided to use FalkorDB for the graph")

    assert memory_type == "Decision"
    assert stats.pattern_classifications == 1
    assert stats.llm_attempts == 0
    assert stats.fallbacks == 0


def test_llm_failure_counts_fallback_and_records_error():
    stats = ClassificationStats()
    classifier = _make_classifier(_failing_client(), stats)

    result = classifier.classify(NON_PATTERN_CONTENT)

    assert result == ("Memory", 0.3)
    assert stats.llm_attempts == 1
    assert stats.llm_successes == 0
    assert stats.fallbacks == 1
    assert "429" in stats.last_error
    assert stats.last_error_at is not None
    assert not hasattr(classifier, "_last_llm_error")


def test_classification_stats_records_thread_safely():
    stats = ClassificationStats()

    def worker():
        for _ in range(200):
            stats.record_pattern()
            stats.record_llm_attempt()
            stats.record_llm_success()
            stats.record_fallback("boom")

    threads = [Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    snapshot = stats.to_dict()
    assert snapshot["pattern_classifications"] == 1600
    assert snapshot["llm_attempts"] == 1600
    assert snapshot["llm_successes"] == 1600
    assert snapshot["fallbacks"] == 1600
    assert snapshot["last_error"] == "boom"
    assert snapshot["last_error_at"] is not None


def test_llm_success_counts_success():
    stats = ClassificationStats()
    classifier = _make_classifier(_succeeding_client(), stats)

    result = classifier.classify(NON_PATTERN_CONTENT)

    assert result == ("Insight", 0.9)
    assert stats.llm_attempts == 1
    assert stats.llm_successes == 1
    assert stats.fallbacks == 0
    assert stats.last_error is None


def test_classifier_without_stats_still_works():
    classifier = _make_classifier(_failing_client(), None)

    assert classifier.classify("decided to use FalkorDB")[0] == "Decision"
    assert classifier.classify(NON_PATTERN_CONTENT) == ("Memory", 0.3)
