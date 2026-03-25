from __future__ import annotations

import os
from typing import Any

# text-embedding-3-small native output is 1536d; it can be truncated to
# any dimension <= 1536 via the Matryoshka ``dimensions`` parameter but
# CANNOT produce vectors larger than 1536d.
_SMALL_MODEL_MAX_DIM = 1536


_PROVIDER_DIMENSION_CONSTRAINTS: dict[str, set[int]] = {
    "voyage": {256, 512, 1024, 2048},
}


def _validate_provider_dimension(provider_name: str, vector_size: int, logger: Any) -> None:
    """Raise early if autodetected dimension is incompatible with the provider."""
    constraints = _PROVIDER_DIMENSION_CONSTRAINTS.get(provider_name)
    if constraints is None or vector_size in constraints:
        return
    sorted_dims = sorted(constraints)
    raise RuntimeError(
        f"\n{'=' * 72}\n"
        f"FATAL: Dimension mismatch between existing Qdrant collection and "
        f"{provider_name} provider!\n"
        f"  Autodetected collection dimension: {vector_size}d\n"
        f"  {provider_name} supported dimensions: {sorted_dims}\n"
        f"\n"
        f"This happens when VECTOR_SIZE_AUTODETECT adopts a dimension from a\n"
        f"previous provider that the new provider cannot produce.\n"
        f"\n"
        f"Fix options:\n"
        f"  1. Keep your current provider (remove EMBEDDING_PROVIDER={provider_name})\n"
        f"  2. Set VECTOR_SIZE={sorted_dims[-1]} and re-embed:\n"
        f"     python scripts/reembed_embeddings.py\n"
        f"  3. Delete the Qdrant collection and let AutoMem recreate it at the\n"
        f"     correct dimension (loses existing embeddings, triggers re-embed)\n"
        f"{'=' * 72}"
    )


def _resolve_openai_model(embedding_model: str, vector_size: int, logger: Any) -> str:
    """Auto-upgrade to text-embedding-3-large when the dimension exceeds small model capacity."""
    small_name = "text-embedding-3-small"
    large_name = "text-embedding-3-large"
    if vector_size > _SMALL_MODEL_MAX_DIM and embedding_model.endswith(small_name):
        logger.warning(
            "VECTOR_SIZE=%d exceeds text-embedding-3-small capacity (%dd). "
            "Auto-upgrading to text-embedding-3-large. "
            "Set EMBEDDING_MODEL=text-embedding-3-large explicitly to silence this warning.",
            vector_size,
            _SMALL_MODEL_MAX_DIM,
        )
        return embedding_model[: -len(small_name)] + large_name
    return embedding_model


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
        _validate_provider_dimension("voyage", vector_size, logger)
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
        effective_model = _resolve_openai_model(embedding_model, vector_size, logger)
        try:
            from automem.embedding.openai import OpenAIEmbeddingProvider

            state.embedding_provider = OpenAIEmbeddingProvider(
                api_key=api_key,
                model=effective_model,
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
                _validate_provider_dimension("voyage", vector_size, logger)
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
                effective_model = _resolve_openai_model(embedding_model, vector_size, logger)
                state.embedding_provider = OpenAIEmbeddingProvider(
                    api_key=api_key,
                    model=effective_model,
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
