"""Integration tests for MiniMax LLM provider.

These tests require a valid MINIMAX_API_KEY environment variable.
They are skipped when the key is not available.

Run with:
    MINIMAX_API_KEY=your-key pytest tests/test_minimax_integration.py -v
"""

import json
import os

import pytest

pytestmark = pytest.mark.integration

MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
skip_no_key = pytest.mark.skipif(
    not MINIMAX_API_KEY,
    reason="MINIMAX_API_KEY not set — skipping MiniMax integration tests",
)


@skip_no_key
class TestMiniMaxLiveClassification:
    """Integration tests for MiniMax-powered memory type classification."""

    def _make_classifier(self):
        """Create a MemoryClassifier wired to the real MiniMax API."""
        import openai

        from automem.classification.memory_classifier import MemoryClassifier
        from automem.config import MINIMAX_BASE_URL, MINIMAX_DEFAULT_MODEL, normalize_memory_type

        client = openai.OpenAI(
            api_key=MINIMAX_API_KEY,
            base_url=MINIMAX_BASE_URL,
        )

        import logging

        logger = logging.getLogger("test_minimax_integration")

        classifier = MemoryClassifier(
            normalize_memory_type=normalize_memory_type,
            ensure_openai_client=lambda: None,
            get_openai_client=lambda: client,
            classification_model=MINIMAX_DEFAULT_MODEL,
            logger=logger,
        )
        return classifier

    def test_classify_decision(self):
        """MiniMax should classify a decision-type memory."""
        classifier = self._make_classifier()
        # Content that won't match regex patterns
        content = (
            "After evaluating the options, the team committed to "
            "adopting PostgreSQL as the primary database for the new service."
        )
        memory_type, confidence = classifier.classify(content, use_llm=True)
        assert memory_type in {"Decision", "Context", "Insight"}, (
            f"Expected Decision/Context/Insight, got {memory_type}"
        )
        assert 0.0 < confidence <= 1.0

    def test_classify_returns_valid_type(self):
        """MiniMax should return a valid canonical memory type."""
        from automem.config import MEMORY_TYPES

        classifier = self._make_classifier()
        content = "The CI pipeline keeps failing because of flaky network tests."
        memory_type, confidence = classifier.classify(content, use_llm=True)
        # Should be a canonical type or "Memory" (fallback)
        assert memory_type in MEMORY_TYPES or memory_type == "Memory", (
            f"Got non-canonical type: {memory_type}"
        )
        assert 0.0 <= confidence <= 1.0

    def test_classify_returns_json_with_confidence(self):
        """MiniMax should return a valid confidence score."""
        classifier = self._make_classifier()
        content = "I notice I always check email first thing in the morning before coding."
        memory_type, confidence = classifier.classify(content, use_llm=True)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0


@skip_no_key
class TestMiniMaxLiveSummarization:
    """Integration tests for MiniMax-powered auto-summarization."""

    def test_summarize_long_content(self):
        """MiniMax should summarize long content to a shorter string."""
        import openai

        from automem.config import MINIMAX_BASE_URL, MINIMAX_DEFAULT_MODEL
        from automem.utils.text import summarize_content

        client = openai.OpenAI(
            api_key=MINIMAX_API_KEY,
            base_url=MINIMAX_BASE_URL,
        )

        long_content = (
            "During the architecture review meeting on January 15th, the team discussed "
            "the pros and cons of migrating from a monolithic architecture to microservices. "
            "The main arguments in favor included independent deployment, technology diversity, "
            "and team autonomy. The concerns raised were about increased operational complexity, "
            "data consistency challenges, and the need for better monitoring infrastructure. "
            "After a two-hour discussion, the team decided to start with a pilot project "
            "by extracting the notification service as the first microservice, while keeping "
            "the rest of the system as a monolith for now. The target date for the pilot "
            "was set for March 1st, with a go/no-go decision after a month of production use."
        )

        result = summarize_content(
            long_content,
            client,
            MINIMAX_DEFAULT_MODEL,
            target_length=200,
        )

        assert result is not None, "Summarization returned None"
        assert len(result) < len(long_content), (
            f"Summary ({len(result)} chars) not shorter than original ({len(long_content)} chars)"
        )
        assert len(result) > 0, "Summary is empty"
