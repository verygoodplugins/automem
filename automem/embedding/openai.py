"""OpenAI embedding provider using OpenAI API."""

import logging
from typing import List

from openai import OpenAI

from automem.embedding.provider import EmbeddingProvider

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Generates embeddings using OpenAI's embedding API.

    Requires an OpenAI API key and makes network requests to OpenAI.
    Provides high-quality semantic embeddings but requires API costs.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimension: int = 768,
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        """Initialize OpenAI embedding provider.

        Args:
            api_key: OpenAI API key
            model: OpenAI embedding model to use
            dimension: Number of dimensions for embeddings
            timeout: Request timeout in seconds (default 30)
            max_retries: Maximum number of retries (default 2)

        Raises:
            Exception: If OpenAI client initialization fails
        """
        self.client = OpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)
        self.model = model
        self._dimension = dimension
        logger.info(
            "OpenAI embedding provider initialized (model=%s, dimensions=%d, timeout=%.1fs, retries=%d)",
            model,
            dimension,
            timeout,
            max_retries,
        )

    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding using OpenAI API.

        Args:
            text: The text to embed

        Returns:
            Embedding vector from OpenAI

        Raises:
            Exception: If API call fails
        """
        response = self.client.embeddings.create(
            input=text, model=self.model, dimensions=self._dimension
        )
        embedding = response.data[0].embedding
        if len(embedding) != self._dimension:
            raise ValueError(
                f"OpenAI embedding length {len(embedding)} != configured dimension {self._dimension} "
                f"(model={self.model}). Ensure 'dimensions' is supported and vector store size matches."
            )
        logger.debug("Generated OpenAI embedding for text (length: %d)", len(text))
        return embedding

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in one API call.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors from OpenAI

        Raises:
            Exception: If API call fails
        """
        if not texts:
            return []

        response = self.client.embeddings.create(
            input=texts, model=self.model, dimensions=self._dimension
        )
        embeddings = [item.embedding for item in response.data]
        bad = next((i for i, e in enumerate(embeddings) if len(e) != self._dimension), None)
        if bad is not None:
            raise ValueError(
                f"OpenAI batch embedding length {len(embeddings[bad])} != configured dimension {self._dimension} "
                f"at index {bad} (model={self.model})."
            )
        logger.info(
            "Generated %d OpenAI embeddings in batch (avg length: %d)",
            len(embeddings),
            sum(len(t) for t in texts) // len(texts) if texts else 0,
        )
        return embeddings

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
        return f"openai:{self.model}"
