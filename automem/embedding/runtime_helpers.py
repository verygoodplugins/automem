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
    allow_placeholder_fallback: bool = True,
) -> List[List[float]]:
    """Generate multiple embeddings in a single batch for efficiency."""
    init_embedding_provider()

    if not contents:
        return []

    if state.embedding_provider is None:
        if not allow_placeholder_fallback:
            raise RuntimeError("No embedding provider available")
        logger.debug("No embedding provider available, falling back to placeholder embeddings")
        return [placeholder_embedding(c) for c in contents]

    provider = state.embedding_provider
    try:
        provider_name = provider.provider_name()
    except Exception:
        provider_name = provider.__class__.__name__ or "unknown"
    if not allow_placeholder_fallback and str(provider_name).strip().lower() == "placeholder":
        raise RuntimeError("Placeholder embedding provider is not allowed in strict mode")
    expected_dim = state.effective_vector_size
    try:
        embeddings = provider.generate_embeddings_batch(contents)
        if (
            not isinstance(embeddings, list)
            or len(embeddings) != len(contents)
            or any(not isinstance(e, list) or len(e) != expected_dim for e in embeddings)
        ):
            raise ValueError(
                f"invalid batch result: expected {len(contents)} embeddings "
                f"of {expected_dim} dims"
            )
        return embeddings
    except Exception as exc:
        logger.warning(
            "Batch embedding via provider %s failed (%s); "
            "falling back to per-item embedding calls",
            provider_name,
            exc,
        )

    # Per-item fallback: single-input calls may succeed even when the batch
    # endpoint fails (e.g. Voyage hanging on multi-input requests).
    results: List[List[float]] = []
    placeholder_count = 0
    for content in contents:
        try:
            embedding = provider.generate_embedding(content)
            if not isinstance(embedding, list) or len(embedding) != expected_dim:
                raise ValueError(
                    f"expected {expected_dim} dims, got "
                    f"{len(embedding) if isinstance(embedding, list) else 'invalid'}"
                )
            results.append(embedding)
        except Exception as exc:
            if not allow_placeholder_fallback:
                raise RuntimeError(
                    f"Failed to generate provider embedding for item "
                    f"{len(results) + 1}/{len(contents)}"
                ) from exc
            logger.debug("Per-item embedding failed (%s); using placeholder", exc)
            results.append(placeholder_embedding(content))
            placeholder_count += 1

    if placeholder_count:
        logger.warning(
            "%d/%d items fell back to placeholder embeddings — these will be "
            "invisible to semantic search until re-embedded",
            placeholder_count,
            len(contents),
        )
    return results
