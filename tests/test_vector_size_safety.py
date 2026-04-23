"""Tests for vector dimension mismatch detection and safety net (issue #107).

Validates that:
- Autodetect (default) adopts existing collection dimension on mismatch
- Strict mode raises VectorDimensionMismatchError (fatal, not swallowed)
- init_qdrant propagates VectorDimensionMismatchError instead of catching it
- Matching dimensions work normally
- Missing collections fall back to config default
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from automem.utils.validation import VectorDimensionMismatchError, get_effective_vector_size


def _make_qdrant_client(collection_dim):
    """Build a mock QdrantClient whose collection reports *collection_dim*."""
    client = MagicMock()
    info = MagicMock()
    info.config.params.vectors.size = collection_dim
    client.get_collection.return_value = info
    return client


def _make_qdrant_client_no_collection():
    """Build a mock QdrantClient where the collection doesn't exist yet."""
    client = MagicMock()
    client.get_collection.side_effect = Exception("Collection 'memories' doesn't exist")
    return client


# ---------------------------------------------------------------------------
# get_effective_vector_size
# ---------------------------------------------------------------------------


class TestAutodetectDefault:
    """VECTOR_SIZE_AUTODETECT defaults to true — mismatch adopts collection dim."""

    @patch.dict("os.environ", {}, clear=False)
    def test_mismatch_adopts_collection_dim(self):
        """When autodetect=true (default) and collection is 3072d, adopt 3072."""
        import os

        os.environ.pop("VECTOR_SIZE_AUTODETECT", None)

        client = _make_qdrant_client(3072)
        with patch("automem.config.VECTOR_SIZE", 1024), patch(
            "automem.config.COLLECTION_NAME", "memories"
        ):
            dim, source = get_effective_vector_size(client)

        assert dim == 3072
        assert source == "collection"

    @patch.dict("os.environ", {"VECTOR_SIZE_AUTODETECT": "true"}, clear=False)
    def test_explicit_autodetect_true(self):
        client = _make_qdrant_client(3072)
        with patch("automem.config.VECTOR_SIZE", 1024), patch(
            "automem.config.COLLECTION_NAME", "memories"
        ):
            dim, source = get_effective_vector_size(client)

        assert dim == 3072
        assert source == "collection"

    @patch.dict("os.environ", {"VECTOR_SIZE_AUTODETECT": "1"}, clear=False)
    def test_autodetect_truthy_variants(self):
        client = _make_qdrant_client(768)
        with patch("automem.config.VECTOR_SIZE", 1024), patch(
            "automem.config.COLLECTION_NAME", "memories"
        ):
            dim, source = get_effective_vector_size(client)

        assert dim == 768
        assert source == "collection"


class TestStrictMode:
    """VECTOR_SIZE_AUTODETECT=false raises VectorDimensionMismatchError on mismatch."""

    @patch.dict("os.environ", {"VECTOR_SIZE_AUTODETECT": "false"}, clear=False)
    def test_mismatch_raises_fatal_error(self):
        client = _make_qdrant_client(3072)
        with patch("automem.config.VECTOR_SIZE", 1024), patch(
            "automem.config.COLLECTION_NAME", "memories"
        ):
            with pytest.raises(VectorDimensionMismatchError) as exc_info:
                get_effective_vector_size(client)

        assert exc_info.value.collection_dim == 3072
        assert exc_info.value.config_dim == 1024
        assert "3072" in str(exc_info.value)
        assert "1024" in str(exc_info.value)

    @patch.dict("os.environ", {"VECTOR_SIZE_AUTODETECT": "0"}, clear=False)
    def test_strict_with_zero(self):
        client = _make_qdrant_client(3072)
        with patch("automem.config.VECTOR_SIZE", 1024), patch(
            "automem.config.COLLECTION_NAME", "memories"
        ):
            with pytest.raises(VectorDimensionMismatchError):
                get_effective_vector_size(client)

    @patch.dict("os.environ", {"VECTOR_SIZE_AUTODETECT": "no"}, clear=False)
    def test_strict_with_no(self):
        client = _make_qdrant_client(768)
        with patch("automem.config.VECTOR_SIZE", 1024), patch(
            "automem.config.COLLECTION_NAME", "memories"
        ):
            with pytest.raises(VectorDimensionMismatchError):
                get_effective_vector_size(client)


class TestMatchingDimensions:
    """No error when collection and config agree."""

    def test_matching_dimensions_returns_collection_source(self):
        client = _make_qdrant_client(1024)
        with patch("automem.config.VECTOR_SIZE", 1024), patch(
            "automem.config.COLLECTION_NAME", "memories"
        ):
            dim, source = get_effective_vector_size(client)

        assert dim == 1024
        assert source == "collection"


class TestNewInstall:
    """No existing collection → uses config default."""

    def test_no_collection_uses_config(self):
        client = _make_qdrant_client_no_collection()
        with patch("automem.config.VECTOR_SIZE", 1024), patch(
            "automem.config.COLLECTION_NAME", "memories"
        ):
            dim, source = get_effective_vector_size(client)

        assert dim == 1024
        assert source == "config"

    def test_no_client_uses_config(self):
        with patch("automem.config.VECTOR_SIZE", 1024):
            dim, source = get_effective_vector_size(None)

        assert dim == 1024
        assert source == "config"


# ---------------------------------------------------------------------------
# VectorDimensionMismatchError is NOT a ValueError
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    """VectorDimensionMismatchError must NOT be caught by 'except ValueError'."""

    def test_not_a_value_error(self):
        err = VectorDimensionMismatchError(3072, 1024)
        assert not isinstance(err, ValueError)
        assert isinstance(err, RuntimeError)

    def test_error_message_is_actionable(self):
        err = VectorDimensionMismatchError(3072, 1024)
        msg = str(err)
        assert "FATAL" in msg
        assert "VECTOR_SIZE=3072" in msg
        assert "VECTOR_SIZE_AUTODETECT=true" in msg
        assert "reembed" in msg


# ---------------------------------------------------------------------------
# init_qdrant propagation
# ---------------------------------------------------------------------------


class TestInitQdrantPropagation:
    """init_qdrant must let VectorDimensionMismatchError propagate (fatal)."""

    def test_dimension_mismatch_propagates(self):
        from automem.stores.runtime_clients import init_qdrant

        state = SimpleNamespace(qdrant=None)
        logger = MagicMock()

        def boom():
            raise VectorDimensionMismatchError(3072, 1024)

        with patch("automem.config.QDRANT_URL", "http://localhost:6333"), patch(
            "automem.config.QDRANT_API_KEY", None
        ):
            with pytest.raises(VectorDimensionMismatchError):
                init_qdrant(
                    state=state,
                    logger=logger,
                    qdrant_client_cls=MagicMock(),
                    ensure_collection_fn=boom,
                )

        assert state.qdrant is None

    def test_value_error_still_caught_gracefully(self):
        """Regular ValueError (e.g. bad URL) should still be caught, not fatal."""
        from automem.stores.runtime_clients import init_qdrant

        state = SimpleNamespace(qdrant=None)
        logger = MagicMock()

        def bad_config():
            raise ValueError("invalid URL format")

        with patch("automem.config.QDRANT_URL", "http://localhost:6333"), patch(
            "automem.config.QDRANT_API_KEY", None
        ):
            init_qdrant(
                state=state,
                logger=logger,
                qdrant_client_cls=MagicMock(),
                ensure_collection_fn=bad_config,
            )

        assert state.qdrant is None
        logger.exception.assert_called_once()


# ---------------------------------------------------------------------------
# OpenAI model auto-upgrade for large dimensions
# ---------------------------------------------------------------------------


class TestOpenAIModelAutoUpgrade:
    """text-embedding-3-small can't produce >1536d; auto-upgrade to large."""

    def test_upgrade_for_3072d(self):
        from automem.embedding.provider_init import _resolve_openai_model

        logger = MagicMock()
        result = _resolve_openai_model("text-embedding-3-small", 3072, logger)
        assert result == "text-embedding-3-large"
        logger.warning.assert_called_once()

    def test_no_upgrade_for_1024d(self):
        from automem.embedding.provider_init import _resolve_openai_model

        logger = MagicMock()
        result = _resolve_openai_model("text-embedding-3-small", 1024, logger)
        assert result == "text-embedding-3-small"
        logger.warning.assert_not_called()

    def test_no_upgrade_for_768d(self):
        from automem.embedding.provider_init import _resolve_openai_model

        logger = MagicMock()
        result = _resolve_openai_model("text-embedding-3-small", 768, logger)
        assert result == "text-embedding-3-small"
        logger.warning.assert_not_called()

    def test_large_model_unchanged(self):
        """text-embedding-3-large should never be downgraded."""
        from automem.embedding.provider_init import _resolve_openai_model

        logger = MagicMock()
        result = _resolve_openai_model("text-embedding-3-large", 3072, logger)
        assert result == "text-embedding-3-large"
        logger.warning.assert_not_called()

    def test_at_boundary_1536(self):
        """At exactly 1536d, small model should still work (no upgrade)."""
        from automem.embedding.provider_init import _resolve_openai_model

        logger = MagicMock()
        result = _resolve_openai_model("text-embedding-3-small", 1536, logger)
        assert result == "text-embedding-3-small"
        logger.warning.assert_not_called()

    def test_above_boundary_1537(self):
        """At 1537d, small model should be upgraded."""
        from automem.embedding.provider_init import _resolve_openai_model

        logger = MagicMock()
        result = _resolve_openai_model("text-embedding-3-small", 1537, logger)
        assert result == "text-embedding-3-large"
        logger.warning.assert_called_once()
