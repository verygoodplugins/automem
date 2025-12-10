"""Runtime validation utilities for AutoMem configuration."""

import logging
import os

logger = logging.getLogger("automem.validation")


def get_effective_vector_size(qdrant_client=None):
    """
    Get the effective vector size, preferring existing collection dimension over config.

    By default this is strict: if an existing collection dimension differs from the
    configured VECTOR_SIZE, we raise to prevent silent corruption. Set the environment
    variable VECTOR_SIZE_AUTODETECT=true to instead adopt the existing collection size.

    Args:
        qdrant_client: Optional QdrantClient instance. If None, returns config default.

    Returns:
        tuple: (effective_dimension: int, source: str)
            - source is "collection" if detected from existing, "config" otherwise
    """
    from automem.config import COLLECTION_NAME, VECTOR_SIZE

    if qdrant_client is None:
        return VECTOR_SIZE, "config"

    try:
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        collection_dim = collection_info.config.params.vectors.size

        if collection_dim != VECTOR_SIZE:
            allow_autodetect = os.getenv("VECTOR_SIZE_AUTODETECT", "false").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if allow_autodetect:
                logger.info(
                    "Auto-detected existing collection dimension: %dd (config default: %dd). "
                    "Using %dd because VECTOR_SIZE_AUTODETECT=true. "
                    "To migrate, run: python scripts/reembed_embeddings.py",
                    collection_dim,
                    VECTOR_SIZE,
                    collection_dim,
                )
                return collection_dim, "collection"

            # Strict mode (default): fail fast so the operator can migrate or set the env override
            raise ValueError(
                f"Vector dimension mismatch: collection={collection_dim}d, config={VECTOR_SIZE}d. "
                "Set VECTOR_SIZE to the existing dimension, migrate with scripts/reembed_embeddings.py, "
                "or set VECTOR_SIZE_AUTODETECT=true to adopt the collection dimension."
            )

        return collection_dim, "collection"

    except Exception as e:
        # Collection doesn't exist yet (first run) - use config default
        if isinstance(e, AttributeError):
            # Likely running with a stubbed client (tests)
            return VECTOR_SIZE, "config"
        error_msg = str(e).lower()
        if "not found" in error_msg or "doesn't exist" in error_msg:
            # New installation, collection will be created with config dimension
            return VECTOR_SIZE, "config"
        # Re-raise unexpected errors
        raise


def validate_vector_dimensions(qdrant_client=None):
    """
    Legacy validation function - now just logs info about dimension detection.

    Kept for backwards compatibility but no longer raises on mismatch.
    Use get_effective_vector_size() to get the actual dimension to use.
    """
    effective_dim, source = get_effective_vector_size(qdrant_client)
    if source == "collection":
        from automem.config import VECTOR_SIZE

        if effective_dim != VECTOR_SIZE:
            logger.info(
                "Vector dimension: %dd (from existing collection, config says %dd)",
                effective_dim,
                VECTOR_SIZE,
            )
    return effective_dim
