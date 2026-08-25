from __future__ import annotations

import importlib


def test_identity_synthesis_defaults_disabled(monkeypatch) -> None:
    import automem.config as config

    with monkeypatch.context() as mp:
        mp.delenv("IDENTITY_SYNTHESIS_ENABLED", raising=False)
        mp.delenv("CONSOLIDATION_IDENTITY_INTERVAL_SECONDS", raising=False)
        mp.delenv("IDENTITY_SYNTHESIS_MODEL", raising=False)
        mp.setenv("CLASSIFICATION_MODEL", "test-classifier")

        config = importlib.reload(config)

        assert config.IDENTITY_SYNTHESIS_ENABLED is False
        assert config.CONSOLIDATION_IDENTITY_INTERVAL_SECONDS == 0
        assert config.IDENTITY_SYNTHESIS_MODEL == "test-classifier"

    importlib.reload(config)


def test_identity_synthesis_enabled_restores_weekly_default(monkeypatch) -> None:
    import automem.config as config

    with monkeypatch.context() as mp:
        mp.setenv("IDENTITY_SYNTHESIS_ENABLED", "true")
        mp.delenv("CONSOLIDATION_IDENTITY_INTERVAL_SECONDS", raising=False)

        config = importlib.reload(config)

        assert config.IDENTITY_SYNTHESIS_ENABLED is True
        assert config.CONSOLIDATION_IDENTITY_INTERVAL_SECONDS == 604800

    importlib.reload(config)


def test_llm_output_budgets_are_independently_configurable(monkeypatch) -> None:
    import automem.config as config

    with monkeypatch.context() as mp:
        mp.setenv("CLASSIFICATION_MAX_TOKENS", "320")
        mp.setenv("ENRICHMENT_MAX_TOKENS", "640")

        config = importlib.reload(config)

        assert config.CLASSIFICATION_MAX_TOKENS == 320
        assert config.ENRICHMENT_MAX_TOKENS == 640

    importlib.reload(config)
