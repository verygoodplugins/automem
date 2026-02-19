from __future__ import annotations

import os
from typing import Any


def init_embedding_provider(
    *,
    state: Any,
    logger: Any,
    vector_size_config: int,
    embedding_model: str,
) -> None:
    """Initialize embedding provider with auto-selection fallback.

    Priority order:
    1. Voyage API (if VOYAGE_API_KEY is set)
    2. OpenAI API (if OPENAI_API_KEY is set)
    3. Ollama local server (if configured)
    4. Local fastembed model (no API key needed)
    5. Placeholder hash-based embeddings (fallback)

    Can be controlled via EMBEDDING_PROVIDER env var:
    - "auto" (default): Try Voyage, then OpenAI, then Ollama, then fastembed, then placeholder
    - "voyage": Use Voyage only, fail if unavailable
    - "openai": Use OpenAI only, fail if unavailable
    - "local": Use fastembed only, fail if unavailable
    - "ollama": Use Ollama only, fail if unavailable
    - "placeholder": Use placeholder embeddings
    """
    if state.embedding_provider is not None:
        return

    provider_config = (os.getenv("EMBEDDING_PROVIDER", "auto") or "auto").strip().lower()
    # Use effective dimension (auto-detected from existing collection or config default).
    # If Qdrant hasn't set it (or config was changed in-process), align to vector_size_config.
    if state.qdrant is None and state.effective_vector_size != vector_size_config:
        state.effective_vector_size = vector_size_config
    vector_size = state.effective_vector_size

    # Explicit provider selection
    if provider_config == "voyage":
        api_key = os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError("EMBEDDING_PROVIDER=voyage but VOYAGE_API_KEY not set")
        try:
            from automem.embedding.voyage import VoyageEmbeddingProvider

            voyage_model = os.getenv("VOYAGE_MODEL", "voyage-4")
            state.embedding_provider = VoyageEmbeddingProvider(
                api_key=api_key, model=voyage_model, dimension=vector_size
            )
            logger.info("Embedding provider: %s", state.embedding_provider.provider_name())
            return
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Voyage provider: {e}") from e

    elif provider_config == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("EMBEDDING_PROVIDER=openai but OPENAI_API_KEY not set")
        openai_base_url = (os.getenv("OPENAI_BASE_URL") or "").strip() or None
        try:
            from automem.embedding.openai import OpenAIEmbeddingProvider

            state.embedding_provider = OpenAIEmbeddingProvider(
                api_key=api_key,
                model=embedding_model,
                dimension=vector_size,
                base_url=openai_base_url,
            )
            logger.info("Embedding provider: %s", state.embedding_provider.provider_name())
            return
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI provider: {e}") from e

    elif provider_config == "local":
        try:
            from automem.embedding.fastembed import FastEmbedProvider

            state.embedding_provider = FastEmbedProvider(dimension=vector_size)
            logger.info("Embedding provider: %s", state.embedding_provider.provider_name())
            return
        except Exception as e:
            raise RuntimeError(f"Failed to initialize local fastembed provider: {e}") from e

    elif provider_config == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "nomic-embed-text")
        try:
            timeout = float(os.getenv("OLLAMA_TIMEOUT", "30"))
            max_retries = int(os.getenv("OLLAMA_MAX_RETRIES", "2"))
        except ValueError as ve:
            raise RuntimeError(f"Invalid OLLAMA_TIMEOUT or OLLAMA_MAX_RETRIES value: {ve}") from ve
        try:
            from automem.embedding.ollama import OllamaEmbeddingProvider

            state.embedding_provider = OllamaEmbeddingProvider(
                base_url=base_url,
                model=model,
                dimension=vector_size,
                timeout=timeout,
                max_retries=max_retries,
            )
            logger.info("Embedding provider: %s", state.embedding_provider.provider_name())
            return
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Ollama provider: {e}") from e

    elif provider_config == "placeholder":
        from automem.embedding.placeholder import PlaceholderEmbeddingProvider

        state.embedding_provider = PlaceholderEmbeddingProvider(dimension=vector_size)
        logger.info("Embedding provider: %s", state.embedding_provider.provider_name())
        return

    # Auto-selection: Try Voyage → OpenAI → Ollama → fastembed → placeholder
    if provider_config == "auto":
        # Try Voyage first (preferred)
        voyage_key = os.getenv("VOYAGE_API_KEY")
        if voyage_key:
            try:
                from automem.embedding.voyage import VoyageEmbeddingProvider

                voyage_model = os.getenv("VOYAGE_MODEL", "voyage-4")
                state.embedding_provider = VoyageEmbeddingProvider(
                    api_key=voyage_key, model=voyage_model, dimension=vector_size
                )
                logger.info(
                    "Embedding provider (auto-selected): %s",
                    state.embedding_provider.provider_name(),
                )
                return
            except Exception as e:
                logger.warning("Failed to initialize Voyage provider, trying OpenAI: %s", str(e))

        # Try OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                from automem.embedding.openai import OpenAIEmbeddingProvider

                openai_base_url = (os.getenv("OPENAI_BASE_URL") or "").strip() or None
                state.embedding_provider = OpenAIEmbeddingProvider(
                    api_key=api_key,
                    model=embedding_model,
                    dimension=vector_size,
                    base_url=openai_base_url,
                )
                logger.info(
                    "Embedding provider (auto-selected): %s",
                    state.embedding_provider.provider_name(),
                )
                return
            except Exception as e:
                logger.warning(
                    "Failed to initialize OpenAI provider, trying local model: %s", str(e)
                )

        ollama_base_url = os.getenv("OLLAMA_BASE_URL")
        ollama_model = os.getenv("OLLAMA_MODEL")
        if ollama_base_url or ollama_model:
            try:
                from automem.embedding.ollama import OllamaEmbeddingProvider

                base_url = ollama_base_url or "http://localhost:11434"
                model = ollama_model or "nomic-embed-text"
                try:
                    timeout = float(os.getenv("OLLAMA_TIMEOUT", "30"))
                    max_retries = int(os.getenv("OLLAMA_MAX_RETRIES", "2"))
                except ValueError:
                    logger.warning("Invalid OLLAMA_TIMEOUT or OLLAMA_MAX_RETRIES, using defaults")
                    timeout = 30.0
                    max_retries = 2
                state.embedding_provider = OllamaEmbeddingProvider(
                    base_url=base_url,
                    model=model,
                    dimension=vector_size,
                    timeout=timeout,
                    max_retries=max_retries,
                )
                logger.info(
                    "Embedding provider (auto-selected): %s",
                    state.embedding_provider.provider_name(),
                )
                return
            except Exception as e:
                logger.warning(
                    "Failed to initialize Ollama provider, trying local model: %s", str(e)
                )

        # Try local fastembed
        try:
            from automem.embedding.fastembed import FastEmbedProvider

            state.embedding_provider = FastEmbedProvider(dimension=vector_size)
            logger.info(
                "Embedding provider (auto-selected): %s", state.embedding_provider.provider_name()
            )
            return
        except Exception as e:
            logger.warning("Failed to initialize fastembed provider, using placeholder: %s", str(e))

        # Fallback to placeholder
        from automem.embedding.placeholder import PlaceholderEmbeddingProvider

        state.embedding_provider = PlaceholderEmbeddingProvider(dimension=vector_size)
        logger.warning(
            "Using placeholder embeddings (no semantic search). "
            "Install fastembed or set VOYAGE_API_KEY/OPENAI_API_KEY for semantic embeddings."
        )
        logger.info(
            "Embedding provider (auto-selected): %s", state.embedding_provider.provider_name()
        )
        return

    # Invalid config
    raise ValueError(
        f"Invalid EMBEDDING_PROVIDER={provider_config}. "
        f"Valid options: auto, voyage, openai, local, ollama, placeholder"
    )
