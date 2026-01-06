"""Tests for memory content size governance (auto-summarization and limits)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from automem.utils.text import should_summarize_content, summarize_content


class TestShouldSummarizeContent:
    """Tests for the should_summarize_content helper."""

    def test_empty_content_returns_ok(self):
        assert should_summarize_content("", 500, 2000) == "ok"
        assert should_summarize_content(None, 500, 2000) == "ok"  # type: ignore

    def test_short_content_returns_ok(self):
        content = "Short memory content."
        assert should_summarize_content(content, 500, 2000) == "ok"

    def test_content_at_soft_limit_returns_ok(self):
        content = "x" * 500
        assert should_summarize_content(content, 500, 2000) == "ok"

    def test_content_above_soft_limit_returns_summarize(self):
        content = "x" * 501
        assert should_summarize_content(content, 500, 2000) == "summarize"

    def test_content_at_hard_limit_returns_summarize(self):
        content = "x" * 2000
        assert should_summarize_content(content, 500, 2000) == "summarize"

    def test_content_above_hard_limit_returns_reject(self):
        content = "x" * 2001
        assert should_summarize_content(content, 500, 2000) == "reject"


class TestSummarizeContent:
    """Tests for the summarize_content function."""

    def test_returns_none_when_client_is_none(self):
        result = summarize_content("Some long content", None, "gpt-4o-mini", 300)
        assert result is None

    def test_returns_content_when_already_short(self):
        content = "Short content"
        mock_client = MagicMock()
        result = summarize_content(content, mock_client, "gpt-4o-mini", 300)
        assert result == content
        mock_client.chat.completions.create.assert_not_called()

    def test_calls_openai_for_long_content(self):
        long_content = "x" * 600
        expected_summary = "Brief summary of the content."

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=expected_summary))]
        )

        result = summarize_content(long_content, mock_client, "gpt-4o-mini", 300)

        assert result == expected_summary
        mock_client.chat.completions.create.assert_called_once()

    def test_returns_none_when_summary_not_shorter(self):
        long_content = "x" * 600
        # Summary is same length as original
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=long_content))]
        )

        result = summarize_content(long_content, mock_client, "gpt-4o-mini", 300)
        assert result is None

    def test_returns_none_on_api_error(self):
        long_content = "x" * 600
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        result = summarize_content(long_content, mock_client, "gpt-4o-mini", 300)
        assert result is None


class TestMemoryStoreContentSize:
    """Integration tests for content size handling in store endpoint."""

    @pytest.fixture
    def app_client(self):
        """Create a test client with mocked dependencies."""
        # Import here to ensure stubs are installed
        from app import create_app

        app = create_app()
        app.config["TESTING"] = True
        return app.test_client()

    @patch("automem.api.memory.MEMORY_CONTENT_HARD_LIMIT", 100)
    def test_rejects_content_exceeding_hard_limit(self, app_client):
        """Content exceeding hard limit should be rejected with 400."""
        response = app_client.post(
            "/memory",
            json={
                "content": "x" * 150,
                "tags": ["test"],
            },
        )
        assert response.status_code == 400
        assert b"exceeds maximum length" in response.data

    @patch("automem.api.memory.MEMORY_CONTENT_SOFT_LIMIT", 50)
    @patch("automem.api.memory.MEMORY_CONTENT_HARD_LIMIT", 200)
    @patch("automem.api.memory.MEMORY_AUTO_SUMMARIZE", True)
    def test_auto_summarizes_content_above_soft_limit(self, app_client):
        """Content above soft limit should be auto-summarized when enabled."""
        with patch("automem.api.memory.summarize_content") as mock_summarize:
            mock_summarize.return_value = "Brief summary."

            response = app_client.post(
                "/memory",
                json={
                    "content": "x" * 100,  # Above soft limit
                    "tags": ["test"],
                },
            )

            # Should attempt summarization
            mock_summarize.assert_called_once()

    @patch("automem.api.memory.MEMORY_CONTENT_SOFT_LIMIT", 50)
    @patch("automem.api.memory.MEMORY_CONTENT_HARD_LIMIT", 200)
    @patch("automem.api.memory.MEMORY_AUTO_SUMMARIZE", False)
    def test_skips_summarization_when_disabled(self, app_client):
        """Content above soft limit should not be summarized when disabled."""
        with patch("automem.api.memory.summarize_content") as mock_summarize:
            response = app_client.post(
                "/memory",
                json={
                    "content": "x" * 100,
                    "tags": ["test"],
                },
            )

            mock_summarize.assert_not_called()
