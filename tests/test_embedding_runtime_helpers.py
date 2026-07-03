"""Tests for automem.embedding.runtime_helpers batch embedding fallback logic.

Regression coverage for the silent-placeholder bug: when the provider's
batch endpoint fails (e.g. Voyage hanging on multi-input requests), the
helper must fall back to per-item real embedding calls before resorting
to placeholder (hash-based) embeddings, which are invisible to semantic
search.
"""

import logging

import pytest

from automem.embedding.runtime_helpers import generate_real_embeddings_batch

DIM = 4

logger = logging.getLogger("test_embedding_runtime_helpers")


def _placeholder(content: str) -> list:
    """Sentinel placeholder vector, distinguishable from real embeddings."""
    return [-1.0] * DIM


def _real_vector(content: str) -> list:
    """Deterministic per-content 'real' embedding."""
    return [float(len(content))] * DIM


class _State:
    def __init__(self, provider):
        self.embedding_provider = provider
        self.effective_vector_size = DIM


def _run(contents, provider, caplog, *, allow_placeholder_fallback=True):
    with caplog.at_level(logging.DEBUG):
        return generate_real_embeddings_batch(
            contents,
            init_embedding_provider=lambda: None,
            state=_State(provider),
            logger=logger,
            placeholder_embedding=_placeholder,
            allow_placeholder_fallback=allow_placeholder_fallback,
        )


def _warning_messages(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


# ==================== Happy path ====================


def test_batch_success_returns_batch_result(caplog):
    """When the provider batch call succeeds, its result is returned as-is."""

    class Provider:
        single_calls = 0

        def provider_name(self):
            return "voyage:voyage-4"

        def generate_embeddings_batch(self, contents):
            return [_real_vector(c) for c in contents]

        def generate_embedding(self, content):
            Provider.single_calls += 1
            return _real_vector(content)

    contents = ["alpha", "bravo!", "charlie7"]
    result = _run(contents, Provider(), caplog)

    assert result == [_real_vector(c) for c in contents]
    assert Provider.single_calls == 0, "per-item fallback should not run on success"
    assert _warning_messages(caplog) == []


# ==================== Batch failure, singles work ====================


def test_batch_raises_single_works_uses_real_embeddings(caplog):
    """Batch call raises (Voyage multi-input hang): fall back to per-item
    real embeddings — never placeholders."""

    class Provider:
        def provider_name(self):
            return "voyage:voyage-4"

        def generate_embeddings_batch(self, contents):
            raise RuntimeError("read timeout")

        def generate_embedding(self, content):
            return _real_vector(content)

    contents = ["alpha", "bravo!", "charlie7"]
    result = _run(contents, Provider(), caplog)

    assert len(result) == len(contents)
    assert result == [_real_vector(c) for c in contents]
    assert _placeholder("x") not in result, "no item should fall back to placeholder"

    warnings = _warning_messages(caplog)
    assert len(warnings) == 1, f"expected exactly one warning, got: {warnings}"
    assert "voyage:voyage-4" in warnings[0]
    assert "read timeout" in warnings[0]


def test_batch_wrong_count_single_works_uses_real_embeddings(caplog):
    """Batch returns the wrong number of vectors: fall back to per-item
    real embeddings."""

    class Provider:
        def provider_name(self):
            return "voyage:voyage-4"

        def generate_embeddings_batch(self, contents):
            return [_real_vector(contents[0])]  # 1 vector for 3 contents

        def generate_embedding(self, content):
            return _real_vector(content)

    contents = ["alpha", "bravo!", "charlie7"]
    result = _run(contents, Provider(), caplog)

    assert len(result) == len(contents)
    assert result == [_real_vector(c) for c in contents]
    assert _placeholder("x") not in result

    warnings = _warning_messages(caplog)
    assert len(warnings) == 1, f"expected exactly one warning, got: {warnings}"
    assert "voyage:voyage-4" in warnings[0]


def test_batch_wrong_dims_single_works_uses_real_embeddings(caplog):
    """Batch returns vectors with wrong dimensions: fall back to per-item
    real embeddings."""

    class Provider:
        def provider_name(self):
            return "voyage:voyage-4"

        def generate_embeddings_batch(self, contents):
            return [[0.5] * (DIM + 1) for _ in contents]

        def generate_embedding(self, content):
            return _real_vector(content)

    contents = ["alpha", "bravo!"]
    result = _run(contents, Provider(), caplog)

    assert result == [_real_vector(c) for c in contents]
    assert _placeholder("x") not in result


def test_missing_provider_name_still_falls_back_to_single_embeddings(caplog):
    """A custom provider without provider_name() should not bypass fallback."""

    class Provider:
        def generate_embeddings_batch(self, contents):
            raise RuntimeError("read timeout")

        def generate_embedding(self, content):
            return _real_vector(content)

    contents = ["alpha", "bravo!"]
    result = _run(contents, Provider(), caplog)

    assert result == [_real_vector(c) for c in contents]
    warnings = _warning_messages(caplog)
    assert len(warnings) == 1, f"expected exactly one warning, got: {warnings}"
    assert "Provider" in warnings[0]
    assert "read timeout" in warnings[0]


# ==================== Batch and singles both fail ====================


def test_batch_and_single_fail_uses_placeholders_with_summary_warning(caplog):
    """Only when both batch AND per-item calls fail do we use placeholders,
    and we must log a loud summary warning about semantic-search invisibility."""

    class Provider:
        def provider_name(self):
            return "voyage:voyage-4"

        def generate_embeddings_batch(self, contents):
            raise RuntimeError("read timeout")

        def generate_embedding(self, content):
            raise RuntimeError("connection refused")

    contents = ["alpha", "bravo!", "charlie7"]
    result = _run(contents, Provider(), caplog)

    assert result == [_placeholder(c) for c in contents]

    warnings = _warning_messages(caplog)
    summary = [w for w in warnings if "placeholder" in w]
    assert summary, f"expected a placeholder summary warning, got: {warnings}"
    assert "3/3" in summary[0]
    assert "invisible to semantic search" in summary[0]


def test_partial_single_failure_only_failed_items_get_placeholders(caplog):
    """Items whose individual call fails get placeholders; the rest keep
    real embeddings. Summary warning counts only the fallbacks."""

    class Provider:
        def provider_name(self):
            return "voyage:voyage-4"

        def generate_embeddings_batch(self, contents):
            raise RuntimeError("read timeout")

        def generate_embedding(self, content):
            if content == "bravo!":
                raise RuntimeError("transient failure")
            return _real_vector(content)

    contents = ["alpha", "bravo!", "charlie7"]
    result = _run(contents, Provider(), caplog)

    assert result == [
        _real_vector("alpha"),
        _placeholder("bravo!"),
        _real_vector("charlie7"),
    ]

    summary = [w for w in _warning_messages(caplog) if "placeholder" in w]
    assert summary, "expected a placeholder summary warning"
    assert "1/3" in summary[0]
    assert "invisible to semantic search" in summary[0]


def test_strict_mode_raises_instead_of_using_placeholder_fallback(caplog):
    """Repair callers can reject placeholder fallback when real embeddings fail."""

    class Provider:
        def provider_name(self):
            return "voyage:voyage-4"

        def generate_embeddings_batch(self, contents):
            raise RuntimeError("read timeout")

        def generate_embedding(self, content):
            raise RuntimeError("connection refused")

    with pytest.raises(RuntimeError, match="Failed to generate provider embedding"):
        _run(["alpha"], Provider(), caplog, allow_placeholder_fallback=False)


# ==================== Contract ====================


def test_empty_contents_returns_empty_list(caplog):
    class Provider:
        def provider_name(self):
            return "voyage:voyage-4"

        def generate_embeddings_batch(self, contents):
            raise AssertionError("should not be called")

        def generate_embedding(self, content):
            raise AssertionError("should not be called")

    assert _run([], Provider(), caplog) == []


def test_no_provider_uses_placeholders(caplog):
    """No provider configured at all: placeholders (pre-existing behavior)."""
    with caplog.at_level(logging.DEBUG):
        result = generate_real_embeddings_batch(
            ["alpha", "bravo!"],
            init_embedding_provider=lambda: None,
            state=_State(None),
            logger=logger,
            placeholder_embedding=_placeholder,
        )
    assert result == [_placeholder("alpha"), _placeholder("bravo!")]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
