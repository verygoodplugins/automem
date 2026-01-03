"""Tests for per-request isolation headers (X-Graph-Name, X-Collection-Name).

This module tests the implementation of per-request isolation as specified in:
services/automem-federation/docs/automem-isolation-headers-spec.md
"""

import json
import os
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import app


# Test fixtures and setup
@pytest.fixture
def mock_graph():
    """Mock FalkorDB graph instance."""
    graph = MagicMock()
    graph.query = MagicMock(return_value=SimpleNamespace(result_set=[]))
    return graph


@pytest.fixture
def mock_qdrant():
    """Mock Qdrant client instance."""
    qdrant = MagicMock()
    qdrant.upsert = MagicMock()
    qdrant.retrieve = MagicMock(return_value=[])
    qdrant.delete = MagicMock()
    return qdrant


class TestHeaderExtraction:
    """Test header extraction and validation logic."""

    def test_get_graph_name_with_valid_header(self):
        """Test that valid X-Graph-Name header is accepted."""
        with app.app.test_request_context(
            headers={"X-Graph-Name": "test_graph"}
        ):
            assert app._get_graph_name() == "test_graph"

    def test_get_graph_name_with_underscore(self):
        """Test that graph names with underscores are valid."""
        with app.app.test_request_context(
            headers={"X-Graph-Name": "test_graph_123"}
        ):
            assert app._get_graph_name() == "test_graph_123"

    def test_get_graph_name_with_hyphen(self):
        """Test that graph names with hyphens are valid."""
        with app.app.test_request_context(
            headers={"X-Graph-Name": "test-graph-123"}
        ):
            assert app._get_graph_name() == "test-graph-123"

    def test_get_graph_name_defaults_to_env(self):
        """Test that missing header falls back to environment default."""
        with app.app.test_request_context():
            assert app._get_graph_name() == app.GRAPH_NAME

    def test_get_graph_name_empty_header_uses_default(self):
        """Test that empty header value uses default."""
        with app.app.test_request_context(
            headers={"X-Graph-Name": ""}
        ):
            assert app._get_graph_name() == app.GRAPH_NAME

    def test_get_graph_name_whitespace_header_uses_default(self):
        """Test that whitespace-only header uses default."""
        with app.app.test_request_context(
            headers={"X-Graph-Name": "   "}
        ):
            assert app._get_graph_name() == app.GRAPH_NAME

    def test_get_collection_name_with_valid_header(self):
        """Test that valid X-Collection-Name header is accepted."""
        with app.app.test_request_context(
            headers={"X-Collection-Name": "test_collection"}
        ):
            assert app._get_collection_name() == "test_collection"

    def test_get_collection_name_defaults_to_env(self):
        """Test that missing header falls back to environment default."""
        with app.app.test_request_context():
            assert app._get_collection_name() == app.COLLECTION_NAME


class TestHeaderValidation:
    """Test header validation rules (regex, length, format)."""

    def test_invalid_graph_name_special_chars(self):
        """Test that special characters in graph name are rejected (400)."""
        with app.app.test_request_context(
            headers={"X-Graph-Name": "invalid@name"}
        ):
            from werkzeug.exceptions import BadRequest
            with pytest.raises(BadRequest) as exc_info:
                app._get_graph_name()
            assert exc_info.value.code == 400
            assert "Invalid X-Graph-Name" in str(exc_info.value.description)

    def test_invalid_graph_name_spaces(self):
        """Test that spaces in graph name are rejected (400)."""
        with app.app.test_request_context(
            headers={"X-Graph-Name": "invalid name"}
        ):
            from werkzeug.exceptions import BadRequest
            with pytest.raises(BadRequest) as exc_info:
                app._get_graph_name()
            assert exc_info.value.code == 400

    def test_invalid_graph_name_too_long(self):
        """Test that graph names over 64 chars are rejected (400)."""
        with app.app.test_request_context(
            headers={"X-Graph-Name": "a" * 65}
        ):
            from werkzeug.exceptions import BadRequest
            with pytest.raises(BadRequest) as exc_info:
                app._get_graph_name()
            assert exc_info.value.code == 400

    def test_valid_graph_name_max_length(self):
        """Test that graph names at exactly 64 chars are accepted."""
        with app.app.test_request_context(
            headers={"X-Graph-Name": "a" * 64}
        ):
            assert app._get_graph_name() == "a" * 64

    def test_invalid_collection_name_special_chars(self):
        """Test that special characters in collection name are rejected (400)."""
        with app.app.test_request_context(
            headers={"X-Collection-Name": "invalid$collection"}
        ):
            from werkzeug.exceptions import BadRequest
            with pytest.raises(BadRequest) as exc_info:
                app._get_collection_name()
            assert exc_info.value.code == 400
            assert "Invalid X-Collection-Name" in str(exc_info.value.description)


class TestWhitelistEnforcement:
    """Test whitelist enforcement via ALLOWED_GRAPHS and ALLOWED_COLLECTIONS."""

    def test_graph_whitelist_allows_listed_graph(self, monkeypatch):
        """Test that whitelisted graph names are accepted."""
        monkeypatch.setenv("ALLOWED_GRAPHS", "graph1,graph2,graph3")
        with app.app.test_request_context(
            headers={"X-Graph-Name": "graph2"}
        ):
            assert app._get_graph_name() == "graph2"

    def test_graph_whitelist_rejects_unlisted_graph(self, monkeypatch):
        """Test that non-whitelisted graph names are rejected (403)."""
        monkeypatch.setenv("ALLOWED_GRAPHS", "graph1,graph2")
        with app.app.test_request_context(
            headers={"X-Graph-Name": "graph3"}
        ):
            from werkzeug.exceptions import Forbidden
            with pytest.raises(Forbidden) as exc_info:
                app._get_graph_name()
            assert exc_info.value.code == 403
            assert "not allowed" in str(exc_info.value.description)

    def test_graph_whitelist_with_spaces(self, monkeypatch):
        """Test that whitelist parsing handles spaces correctly."""
        monkeypatch.setenv("ALLOWED_GRAPHS", "graph1, graph2, graph3")
        with app.app.test_request_context(
            headers={"X-Graph-Name": "graph2"}
        ):
            assert app._get_graph_name() == "graph2"

    def test_no_whitelist_allows_any_valid_name(self, monkeypatch):
        """Test that without whitelist, any valid name is accepted."""
        monkeypatch.delenv("ALLOWED_GRAPHS", raising=False)
        with app.app.test_request_context(
            headers={"X-Graph-Name": "any_valid_graph"}
        ):
            assert app._get_graph_name() == "any_valid_graph"

    def test_collection_whitelist_allows_listed_collection(self, monkeypatch):
        """Test that whitelisted collection names are accepted."""
        monkeypatch.setenv("ALLOWED_COLLECTIONS", "coll1,coll2,coll3")
        with app.app.test_request_context(
            headers={"X-Collection-Name": "coll2"}
        ):
            assert app._get_collection_name() == "coll2"

    def test_collection_whitelist_rejects_unlisted_collection(self, monkeypatch):
        """Test that non-whitelisted collection names are rejected (403)."""
        monkeypatch.setenv("ALLOWED_COLLECTIONS", "coll1,coll2")
        with app.app.test_request_context(
            headers={"X-Collection-Name": "coll3"}
        ):
            from werkzeug.exceptions import Forbidden
            with pytest.raises(Forbidden) as exc_info:
                app._get_collection_name()
            assert exc_info.value.code == 403


class TestBackwardsCompatibility:
    """Test that existing behavior is preserved when headers are not provided."""

    def test_get_memory_graph_without_request_context_uses_default(self):
        """Test that get_memory_graph() works outside request context."""
        # This should not raise an error and should use the default
        # Note: This will fail if FalkorDB is not available, but that's expected
        # in a unit test environment. The key is that it doesn't raise a RuntimeError
        # about missing request context.
        try:
            graph = app.get_memory_graph()
            # If it returns None, that's fine - FalkorDB might not be running
            # We're just testing it doesn't crash
        except RuntimeError as e:
            if "request context" in str(e).lower():
                pytest.fail("get_memory_graph() should not require request context")
            # Other RuntimeErrors are acceptable (e.g., connection failures)


class TestConnectionCaching:
    """Test that graph connections are cached per graph name."""

    @patch('app.state')
    def test_graph_caching_same_name(self, mock_state):
        """Test that requesting same graph name returns cached instance."""
        mock_falkordb = MagicMock()
        mock_state.falkordb = mock_falkordb
        mock_state.memory_graph = MagicMock()

        # Create a mock graph instance
        mock_graph_instance = MagicMock()
        mock_falkordb.select_graph.return_value = mock_graph_instance

        # Clear cache for clean test
        app._graph_cache.clear()

        # First call should create new instance
        graph1 = app.get_memory_graph("test_graph")
        assert mock_falkordb.select_graph.call_count == 1
        assert mock_falkordb.select_graph.call_args[0][0] == "test_graph"

        # Second call with same name should use cache
        graph2 = app.get_memory_graph("test_graph")
        assert mock_falkordb.select_graph.call_count == 1  # Still 1, not 2
        assert graph1 is graph2  # Same instance

    @patch('app.state')
    def test_graph_caching_different_names(self, mock_state):
        """Test that different graph names create separate instances."""
        mock_falkordb = MagicMock()
        mock_state.falkordb = mock_falkordb
        mock_state.memory_graph = MagicMock()

        # Return different instances for different graph names
        mock_graph1 = MagicMock()
        mock_graph2 = MagicMock()
        mock_falkordb.select_graph.side_effect = [mock_graph1, mock_graph2]

        # Clear cache for clean test
        app._graph_cache.clear()

        # Two different graph names should create two instances
        graph1 = app.get_memory_graph("graph1")
        graph2 = app.get_memory_graph("graph2")

        assert mock_falkordb.select_graph.call_count == 2
        assert graph1 is mock_graph1
        assert graph2 is mock_graph2
        assert graph1 is not graph2
