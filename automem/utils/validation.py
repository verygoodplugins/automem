"""Runtime validation utilities for AutoMem configuration."""

import logging
import os

logger = logging.getLogger("automem.validation")


class VectorDimensionMismatchError(RuntimeError):
    """Raised when the Qdrant collection dimension doesn't match the configured VECTOR_SIZE.

    This is intentionally NOT a ValueError so it propagates through init_qdrant's
    generic ValueError handler and causes a fatal startup failure with actionable
    instructions instead of silently disabling vector search.
    """

    def __init__(self, collection_dim: int, config_dim: int):
        self.collection_dim = collection_dim
        self.config_dim = config_dim
        super().__init__(
            f"\n{'=' * 72}\n"
            f"FATAL: Vector dimension mismatch detected!\n"
            f"  Existing Qdrant collection: {collection_dim}d\n"
            f"  Configured VECTOR_SIZE:     {config_dim}d\n"
            f"\n"
            f"This usually happens when docker-compose defaults change between updates.\n"
            f"\n"
            f"Fix options (pick one):\n"
            f"  1. Set VECTOR_SIZE={collection_dim} in your .env to match your existing data\n"
            f"  2. Set VECTOR_SIZE_AUTODETECT=true to always adopt the existing collection size\n"
            f"  3. Delete the collection and re-embed: python scripts/reembed_embeddings.py\n"
            f"{'=' * 72}"
        )


def get_effective_vector_size(qdrant_client=None):
    """
    Get the effective vector size, preferring existing collection dimension over config.

    By default, autodetect is enabled: if an existing collection has a different
    dimension than VECTOR_SIZE, the existing dimension is adopted and a warning is
    logged. Set VECTOR_SIZE_AUTODETECT=false to enforce strict matching (raises
    VectorDimensionMismatchError on mismatch).

    Args:
        qdrant_client: Optional QdrantClient instance. If None, returns config default.

    Returns:
        tuple: (effective_dimension: int, source: str)
            - source is "collection" if detected from existing, "config" otherwise

    Raises:
        VectorDimensionMismatchError: When strict mode is enabled and dimensions don't match.
    """
    from automem.config import COLLECTION_NAME, VECTOR_SIZE

    if qdrant_client is None:
        return VECTOR_SIZE, "config"

    try:
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        collection_dim = collection_info.config.params.vectors.size

        if collection_dim != VECTOR_SIZE:
            deny_autodetect = os.getenv("VECTOR_SIZE_AUTODETECT", "true").lower() in {
                "0",
                "false",
                "no",
                "off",
            }
            if deny_autodetect:
                raise VectorDimensionMismatchError(collection_dim, VECTOR_SIZE)

            logger.warning(
                "Vector dimension mismatch: collection=%dd, config=%dd. "
                "Auto-adopting existing collection dimension (%dd) because "
                "VECTOR_SIZE_AUTODETECT is enabled. To silence this warning, "
                "set VECTOR_SIZE=%d in your .env file. "
                "To migrate to a new dimension, run: python scripts/reembed_embeddings.py",
                collection_dim,
                VECTOR_SIZE,
                collection_dim,
                collection_dim,
            )
            return collection_dim, "collection"

        return collection_dim, "collection"

    except VectorDimensionMismatchError:
        raise
    except Exception as e:
        if isinstance(e, AttributeError):
            return VECTOR_SIZE, "config"
        error_msg = str(e).lower()
        if "not found" in error_msg or "doesn't exist" in error_msg:
            return VECTOR_SIZE, "config"
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
