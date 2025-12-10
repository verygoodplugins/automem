"""Placeholder embedding provider using deterministic hash-based embeddings."""

import hashlib
import random
from typing import List

from automem.embedding.provider import EmbeddingProvider


class PlaceholderEmbeddingProvider(EmbeddingProvider):
    """Generates deterministic embeddings from content hash.

    This provider creates embeddings without semantic meaning, useful as a
    fallback when no real embedding model is available. Embeddings are
    deterministic (same content always produces same embedding) but have no
    semantic similarity properties.
    """

    def __init__(self, dimension: int = 768):
        """Initialize placeholder provider.

        Args:
            dimension: Number of dimensions for embedding vectors
        """
        self._dimension = dimension

    def generate_embedding(self, text: str) -> List[float]:
        """Generate a deterministic embedding from text hash.

        Args:
            text: The text to embed

        Returns:
            A deterministic vector based on content hash
        """
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little", signed=False)
        rng = random.Random(
            seed
        )  # nosec: B311 - Deterministic RNG is intentional for placeholder embeddings
        return [rng.random() for _ in range(self._dimension)]

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of deterministic vectors
        """
        return [self.generate_embedding(text) for text in texts]

    def dimension(self) -> int:
        """Return embedding dimensionality.

        Returns:
            The number of dimensions in the embedding vectors
        """
        return self._dimension

    def provider_name(self) -> str:
        """Return provider name.

        Returns:
            Provider identifier
        """
        return "placeholder"
