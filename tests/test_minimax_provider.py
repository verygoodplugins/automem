"""Tests for MiniMax LLM provider integration.

Covers:
- init_openai() with MINIMAX_API_KEY auto-detection and explicit LLM_PROVIDER
- Temperature clamping for MiniMax models in classification and summarization
- <think>…</think> tag stripping from MiniMax responses
- Default model selection when using MiniMax
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from automem.classification.memory_classifier import (
    MemoryClassifier,
    _MINIMAX_PREFIXES,
    _THINK_TAG_RE,
)
from automem.config import MINIMAX_BASE_URL, MINIMAX_DEFAULT_MODEL
from automem.service_runtime import init_openai
from automem.utils.text import summarize_content


# ---------------------------------------------------------------------------
# init_openai() — MiniMax provider selection
# ---------------------------------------------------------------------------


class TestInitOpenAIMiniMax:
    """Test MiniMax auto-detection and explicit provider selection in init_openai()."""

    def test_minimax_explicit_provider(self):
        """LLM_PROVIDER=minimax should use MINIMAX_API_KEY and MiniMax base URL."""
        state = SimpleNamespace(openai_client=None)
        logger = Mock()
        openai_cls = Mock()
        openai_client = Mock()
        openai_cls.return_value = openai_client

        env = {
            "LLM_PROVIDER": "minimax",
            "MINIMAX_API_KEY": "minimax-test-key",
        }

        init_openai(
            state=state,
            logger=logger,
            openai_cls=openai_cls,
            get_env_fn=env.get,
        )

        openai_cls.assert_called_once_with(
            api_key="minimax-test-key",
            base_url=MINIMAX_BASE_URL,
        )
        assert state.openai_client is openai_client
        assert state.llm_provider == "minimax"

    def test_minimax_explicit_provider_no_key(self):
        """LLM_PROVIDER=minimax without MINIMAX_API_KEY should not initialize."""
        state = SimpleNamespace(openai_client=None)
        logger = Mock()
        openai_cls = Mock()

        env = {"LLM_PROVIDER": "minimax"}

        init_openai(
            state=state,
            logger=logger,
            openai_cls=openai_cls,
            get_env_fn=env.get,
        )

        openai_cls.assert_not_called()
        assert state.openai_client is None

    def test_minimax_auto_detection(self):
        """Auto mode with only MINIMAX_API_KEY should use MiniMax."""
        state = SimpleNamespace(openai_client=None)
        logger = Mock()
        openai_cls = Mock()
        openai_client = Mock()
        openai_cls.return_value = openai_client

        env = {
            "LLM_PROVIDER": "auto",
            "MINIMAX_API_KEY": "minimax-test-key",
        }

        init_openai(
            state=state,
            logger=logger,
            openai_cls=openai_cls,
            get_env_fn=env.get,
        )

        openai_cls.assert_called_once_with(
            api_key="minimax-test-key",
            base_url=MINIMAX_BASE_URL,
        )
        assert state.openai_client is openai_client
        assert state.llm_provider == "minimax"

    def test_auto_prefers_openai_over_minimax(self):
        """Auto mode with both keys should prefer OpenAI."""
        state = SimpleNamespace(openai_client=None)
        logger = Mock()
        openai_cls = Mock()
        openai_client = Mock()
        openai_cls.return_value = openai_client

        env = {
            "LLM_PROVIDER": "auto",
            "OPENAI_API_KEY": "openai-test-key",
            "MINIMAX_API_KEY": "minimax-test-key",
        }

        init_openai(
            state=state,
            logger=logger,
            openai_cls=openai_cls,
            get_env_fn=env.get,
        )

        openai_cls.assert_called_once_with(api_key="openai-test-key")
        assert state.openai_client is openai_client
        assert state.llm_provider == "openai"

    def test_openai_explicit_provider(self):
        """LLM_PROVIDER=openai should use OPENAI_API_KEY."""
        state = SimpleNamespace(openai_client=None)
        logger = Mock()
        openai_cls = Mock()
        openai_client = Mock()
        openai_cls.return_value = openai_client

        env = {
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "openai-test-key",
            "OPENAI_BASE_URL": "",
        }

        init_openai(
            state=state,
            logger=logger,
            openai_cls=openai_cls,
            get_env_fn=env.get,
        )

        openai_cls.assert_called_once_with(api_key="openai-test-key")
        assert state.openai_client is openai_client
        assert state.llm_provider == "openai"

    def test_openai_explicit_with_base_url(self):
        """LLM_PROVIDER=openai with OPENAI_BASE_URL should pass base_url."""
        state = SimpleNamespace(openai_client=None)
        logger = Mock()
        openai_cls = Mock()
        openai_client = Mock()
        openai_cls.return_value = openai_client

        env = {
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "openai-test-key",
            "OPENAI_BASE_URL": "https://custom.endpoint/v1",
        }

        init_openai(
            state=state,
            logger=logger,
            openai_cls=openai_cls,
            get_env_fn=env.get,
        )

        openai_cls.assert_called_once_with(
            api_key="openai-test-key",
            base_url="https://custom.endpoint/v1",
        )
        assert state.openai_client is openai_client

    def test_no_keys_logs_message(self):
        """Auto mode with no API keys should log a helpful message."""
        state = SimpleNamespace(openai_client=None)
        logger = Mock()
        openai_cls = Mock()

        env = {"LLM_PROVIDER": "auto"}

        init_openai(
            state=state,
            logger=logger,
            openai_cls=openai_cls,
            get_env_fn=env.get,
        )

        openai_cls.assert_not_called()
        assert state.openai_client is None
        # Should log about missing keys
        logger.info.assert_called()
        log_msg = logger.info.call_args[0][0]
        assert "MINIMAX_API_KEY" in log_msg or "OPENAI_API_KEY" in log_msg

    def test_minimax_init_failure_handled(self):
        """If MiniMax client initialization fails, state.openai_client stays None."""
        state = SimpleNamespace(openai_client=None)
        logger = Mock()
        openai_cls = Mock(side_effect=RuntimeError("connection failed"))

        env = {
            "LLM_PROVIDER": "minimax",
            "MINIMAX_API_KEY": "minimax-test-key",
        }

        init_openai(
            state=state,
            logger=logger,
            openai_cls=openai_cls,
            get_env_fn=env.get,
        )

        assert state.openai_client is None

    def test_already_initialized_skips(self):
        """If client is already initialized, skip re-initialization."""
        existing_client = Mock()
        state = SimpleNamespace(openai_client=existing_client)
        logger = Mock()
        openai_cls = Mock()

        env = {"LLM_PROVIDER": "minimax", "MINIMAX_API_KEY": "key"}

        init_openai(
            state=state,
            logger=logger,
            openai_cls=openai_cls,
            get_env_fn=env.get,
        )

        openai_cls.assert_not_called()
        assert state.openai_client is existing_client

    def test_no_openai_package(self):
        """If openai package is not installed, log and skip."""
        state = SimpleNamespace(openai_client=None)
        logger = Mock()

        env = {"LLM_PROVIDER": "minimax", "MINIMAX_API_KEY": "key"}

        init_openai(
            state=state,
            logger=logger,
            openai_cls=None,
            get_env_fn=env.get,
        )

        assert state.openai_client is None


# ---------------------------------------------------------------------------
# Temperature clamping for MiniMax models
# ---------------------------------------------------------------------------


class TestMiniMaxTemperatureClamping:
    """Test that MiniMax models get temperature clamped to (0, 1]."""

    def _make_classifier(self, model: str) -> tuple:
        """Create a MemoryClassifier and mock client for testing."""
        mock_client = Mock()
        logger = Mock()

        classifier = MemoryClassifier(
            normalize_memory_type=lambda t: (t, False) if t else ("", True),
            ensure_openai_client=lambda: None,
            get_openai_client=lambda: mock_client,
            classification_model=model,
            logger=logger,
        )
        return classifier, mock_client

    def test_minimax_model_temperature_clamped(self):
        """MiniMax-M2.7 model should have temperature clamped to (0, 1]."""
        classifier, mock_client = self._make_classifier("MiniMax-M2.7")

        # Set up mock response
        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"type": "Decision", "confidence": 0.9}'
                    )
                )
            ]
        )
        mock_client.chat.completions.create.return_value = mock_response

        result = classifier.classify("This is a test that won't match patterns", use_llm=True)

        # Verify the call was made with clamped temperature
        call_kwargs = mock_client.chat.completions.create.call_args
        assert "temperature" in call_kwargs.kwargs
        temp = call_kwargs.kwargs["temperature"]
        assert 0 < temp <= 1.0, f"Temperature {temp} not in (0, 1]"
        assert temp == 0.3  # 0.3 is already in range, so stays 0.3

    def test_openai_model_no_clamping(self):
        """OpenAI models should not have special temperature clamping."""
        classifier, mock_client = self._make_classifier("gpt-4o-mini")

        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"type": "Context", "confidence": 0.7}'
                    )
                )
            ]
        )
        mock_client.chat.completions.create.return_value = mock_response

        # gpt-4o-mini starts with "gpt-4o" which is in MAX_COMPLETION_PREFIXES
        # so it uses max_completion_tokens, not temperature
        result = classifier.classify("Test content for classification", use_llm=True)

        call_kwargs = mock_client.chat.completions.create.call_args
        assert "max_completion_tokens" in call_kwargs.kwargs

    def test_minimax_highspeed_model_clamped(self):
        """MiniMax-M2.7-highspeed model should also be temperature clamped."""
        classifier, mock_client = self._make_classifier("MiniMax-M2.7-highspeed")

        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"type": "Insight", "confidence": 0.85}'
                    )
                )
            ]
        )
        mock_client.chat.completions.create.return_value = mock_response

        result = classifier.classify("Unique test content 12345", use_llm=True)

        call_kwargs = mock_client.chat.completions.create.call_args
        assert "temperature" in call_kwargs.kwargs
        temp = call_kwargs.kwargs["temperature"]
        assert 0 < temp <= 1.0


# ---------------------------------------------------------------------------
# Think-tag stripping
# ---------------------------------------------------------------------------


class TestThinkTagStripping:
    """Test <think>…</think> tag stripping from MiniMax responses."""

    def test_think_tag_regex(self):
        """The regex should strip <think> blocks."""
        text = '<think>I need to classify this...</think>{"type": "Decision", "confidence": 0.9}'
        cleaned = _THINK_TAG_RE.sub("", text).strip()
        assert cleaned == '{"type": "Decision", "confidence": 0.9}'

    def test_multiline_think_tag(self):
        """Multi-line think tags should be stripped."""
        text = (
            "<think>\nLet me think about this.\n"
            "The content mentions a decision.\n</think>\n"
            '{"type": "Decision", "confidence": 0.85}'
        )
        cleaned = _THINK_TAG_RE.sub("", text).strip()
        assert cleaned == '{"type": "Decision", "confidence": 0.85}'

    def test_no_think_tag(self):
        """Content without think tags should be unchanged."""
        text = '{"type": "Context", "confidence": 0.7}'
        cleaned = _THINK_TAG_RE.sub("", text).strip()
        assert cleaned == text

    def test_classifier_strips_think_tags(self):
        """MemoryClassifier should strip think tags from MiniMax responses."""
        mock_client = Mock()
        logger = Mock()

        classifier = MemoryClassifier(
            normalize_memory_type=lambda t: (t, False) if t else ("", True),
            ensure_openai_client=lambda: None,
            get_openai_client=lambda: mock_client,
            classification_model="MiniMax-M2.7",
            logger=logger,
        )

        # Response with think tags
        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='<think>Analyzing...</think>{"type": "Insight", "confidence": 0.88}'
                    )
                )
            ]
        )
        mock_client.chat.completions.create.return_value = mock_response

        result = classifier.classify("Very unique test xyzzy 99999", use_llm=True)
        assert result == ("Insight", 0.88)

    def test_classifier_only_think_tags_returns_none(self):
        """If response is only think tags, classifier should return None (fallback)."""
        mock_client = Mock()
        logger = Mock()

        classifier = MemoryClassifier(
            normalize_memory_type=lambda t: (t, False) if t else ("", True),
            ensure_openai_client=lambda: None,
            get_openai_client=lambda: mock_client,
            classification_model="MiniMax-M2.7",
            logger=logger,
        )

        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="<think>Just thinking, no actual response.</think>"
                    )
                )
            ]
        )
        mock_client.chat.completions.create.return_value = mock_response

        # Should fall back to default since LLM returns None
        result = classifier.classify("Unique content abcdef 11111", use_llm=True)
        assert result == ("Memory", 0.3)  # fallback

    def test_openai_model_no_think_tag_strip(self):
        """Non-MiniMax models should not attempt think-tag stripping."""
        mock_client = Mock()
        logger = Mock()

        classifier = MemoryClassifier(
            normalize_memory_type=lambda t: (t, False) if t else ("", True),
            ensure_openai_client=lambda: None,
            get_openai_client=lambda: mock_client,
            classification_model="gpt-3.5-turbo",
            logger=logger,
        )

        # Response without think tags, normal model
        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"type": "Pattern", "confidence": 0.75}'
                    )
                )
            ]
        )
        mock_client.chat.completions.create.return_value = mock_response

        result = classifier.classify("Unique test qwerty 22222", use_llm=True)
        assert result == ("Pattern", 0.75)


# ---------------------------------------------------------------------------
# Summarization with MiniMax
# ---------------------------------------------------------------------------


class TestMiniMaxSummarization:
    """Test summarize_content() with MiniMax models."""

    def test_summarize_minimax_model_temperature_clamped(self):
        """summarize_content() should clamp temperature for MiniMax models."""
        mock_client = Mock()
        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Short summary of content.")
                )
            ]
        )
        mock_client.chat.completions.create.return_value = mock_response

        long_content = "A" * 600  # exceeds default target_length

        result = summarize_content(long_content, mock_client, "MiniMax-M2.7")

        call_kwargs = mock_client.chat.completions.create.call_args
        assert "temperature" in call_kwargs.kwargs
        temp = call_kwargs.kwargs["temperature"]
        assert 0 < temp <= 1.0

    def test_summarize_minimax_strips_think_tags(self):
        """summarize_content() should strip think tags from MiniMax responses."""
        mock_client = Mock()
        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="<think>Let me summarize...</think>Short summary here."
                    )
                )
            ]
        )
        mock_client.chat.completions.create.return_value = mock_response

        long_content = "B" * 600

        result = summarize_content(long_content, mock_client, "MiniMax-M2.7")
        assert result == "Short summary here."

    def test_summarize_non_minimax_no_strip(self):
        """Non-MiniMax models should not strip think tags in summarization."""
        mock_client = Mock()
        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="A good summary of the content.")
                )
            ]
        )
        mock_client.chat.completions.create.return_value = mock_response

        long_content = "C" * 600

        result = summarize_content(long_content, mock_client, "gpt-3.5-turbo")
        assert result == "A good summary of the content."


# ---------------------------------------------------------------------------
# Config constants
# ---------------------------------------------------------------------------


class TestConfigConstants:
    """Test that MiniMax config constants are correct."""

    def test_minimax_base_url(self):
        assert MINIMAX_BASE_URL == "https://api.minimax.io/v1"

    def test_minimax_default_model(self):
        assert MINIMAX_DEFAULT_MODEL == "MiniMax-M2.7"

    def test_minimax_prefixes(self):
        assert "MiniMax-" in _MINIMAX_PREFIXES
        assert "minimax-" in _MINIMAX_PREFIXES

    def test_minimax_model_detected(self):
        """MiniMax model names should be detected by the prefix check."""
        for model in ("MiniMax-M2.7", "MiniMax-M2.7-highspeed", "MiniMax-M2.5"):
            assert model.startswith(_MINIMAX_PREFIXES), f"{model} not detected"
