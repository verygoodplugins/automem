from __future__ import annotations

import hashlib
import random
from typing import Any, Callable, List, Optional


def normalize_tags(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(tag, str) for tag in value):
        return value
    raise ValueError("'tags' must be a list of strings or a single string")


def coerce_importance(value: Any) -> float:
    if value is None:
        return 0.5
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("'importance' must be a number") from exc
    if score < 0 or score > 1:
        raise ValueError("'importance' must be between 0 and 1")
    return score


def coerce_embedding(value: Any, expected_dim: int) -> Optional[List[float]]:
    if value is None or value == "":
        return None
    vector: List[Any]
    if isinstance(value, list):
        vector = value
    elif isinstance(value, str):
        vector = [part.strip() for part in value.split(",") if part.strip()]
    else:
        raise ValueError("Embedding must be a list of floats or a comma-separated string")

    if len(vector) != expected_dim:
        raise ValueError(f"Embedding must contain exactly {expected_dim} values")

    try:
        return [float(component) for component in vector]
    except ValueError as exc:
        raise ValueError("Embedding must contain numeric values") from exc


def generate_placeholder_embedding(content: str, expected_dim: int) -> List[float]:
    """Generate a deterministic embedding vector from the content."""
    digest = hashlib.sha256(content.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = random.Random(seed)
    return [rng.random() for _ in range(expected_dim)]


def generate_real_embedding(
    content: str,
    *,
    init_embedding_provider: Callable[[], None],
    state: Any,
    logger: Any,
    placeholder_embedding: Callable[[str], List[float]],
) -> List[float]:
    """Generate an embedding using the configured provider."""
    init_embedding_provider()

    if state.embedding_provider is None:
        logger.warning("No embedding provider available, using placeholder")
        return placeholder_embedding(content)

    expected_dim = state.effective_vector_size
    try:
        embedding = state.embedding_provider.generate_embedding(content)
        if not isinstance(embedding, list) or len(embedding) != expected_dim:
            logger.warning(
                "Provider %s returned %s dims (expected %d); falling back to placeholder",
                state.embedding_provider.provider_name(),
                len(embedding) if isinstance(embedding, list) else "invalid",
                expected_dim,
            )
            return placeholder_embedding(content)
        return embedding
    except Exception as exc:
        logger.warning("Failed to generate embedding: %s", str(exc))
        return placeholder_embedding(content)


def generate_real_embeddings_batch(
    contents: List[str],
    *,
    init_embedding_provider: Callable[[], None],
    state: Any,
    logger: Any,
    placeholder_embedding: Callable[[str], List[float]],
) -> List[List[float]]:
    """Generate multiple embeddings in a single batch for efficiency."""
    init_embedding_provider()

    if not contents:
        return []

    if state.embedding_provider is None:
        logger.debug("No embedding provider available, falling back to placeholder embeddings")
        return [placeholder_embedding(c) for c in contents]

    expected_dim = state.effective_vector_size
    try:
        embeddings = state.embedding_provider.generate_embeddings_batch(contents)
        if not embeddings or any(len(e) != expected_dim for e in embeddings):
            logger.warning(
                "Provider %s returned invalid dims in batch; using placeholders",
                state.embedding_provider.provider_name() if state.embedding_provider else "unknown",
            )
            return [placeholder_embedding(c) for c in contents]
        return embeddings
    except Exception as exc:
        logger.warning("Failed to generate batch embeddings: %s", str(exc))
        return [placeholder_embedding(c) for c in contents]
