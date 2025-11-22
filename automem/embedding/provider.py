"""Base embedding provider interface."""

from abc import ABC, abstractmethod
from typing import List


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    Provides a common interface for generating embeddings from text,
    allowing AutoMem to support multiple embedding backends.
    """

    @abstractmethod
    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for a single text.

        Args:
            text: The text to embed

        Returns:
            A list of floats representing the embedding vector

        Raises:
            Exception: If embedding generation fails
        """
        pass

    @abstractmethod
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in a single batch.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors, one per input text

        Raises:
            Exception: If batch embedding generation fails
        """
        pass

    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of embeddings produced by this provider.

        Returns:
            The number of dimensions in the embedding vectors
        """
        pass

    @abstractmethod
    def provider_name(self) -> str:
        """Return a human-readable name for this provider.

        Returns:
            Provider name (e.g., "openai", "fastembed:bge-base-en-v1.5")
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dimension={self.dimension()}, provider={self.provider_name()})"
