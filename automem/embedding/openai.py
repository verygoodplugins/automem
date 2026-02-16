"""OpenAI-compatible embedding provider.

Works with OpenAI and any provider exposing an OpenAI-compatible
``/v1/embeddings`` endpoint (e.g. OpenRouter, LiteLLM, Azure, vLLM).
Set ``base_url`` (or the ``OPENAI_BASE_URL`` env var) to point at a
custom gateway.
"""

import logging
from typing import List, Optional

from openai import OpenAI

from automem.embedding.provider import EmbeddingProvider

logger = logging.getLogger(__name__)

# The ``dimensions`` request parameter is an OpenAI extension.  Not every
# compatible provider supports it, so we only send it when talking to
# OpenAI's own API.
_OPENAI_API_HOST = "api.openai.com"


def _is_openai_native(base_url: Optional[str]) -> bool:
    """Return True when *base_url* points at OpenAI's first-party API."""
    if base_url is None:
        return True  # SDK default → api.openai.com
    return _OPENAI_API_HOST in base_url


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Generates embeddings using an OpenAI-compatible embedding API.

    Works out-of-the-box with OpenAI. For third-party providers (OpenRouter,
    LiteLLM, vLLM, Azure, etc.) pass *base_url* or set the ``OPENAI_BASE_URL``
    environment variable.

    The ``dimensions`` parameter is only sent when talking to OpenAI's own API
    because many compatible providers do not support it.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimension: int = 768,
        timeout: float = 30.0,
        max_retries: int = 2,
        base_url: Optional[str] = None,
    ):
        """Initialize OpenAI-compatible embedding provider.

        Args:
            api_key: API key for the embedding service.
            model: Embedding model name (provider-specific).
            dimension: Expected number of dimensions for embeddings.
            timeout: Request timeout in seconds (default 30).
            max_retries: Maximum number of retries (default 2).
            base_url: Optional base URL for an OpenAI-compatible API.
                      When *None* the OpenAI SDK falls back to the
                      ``OPENAI_BASE_URL`` env var, then ``https://api.openai.com/v1``.

        Raises:
            Exception: If client initialization fails.
        """
        client_kwargs = dict(api_key=api_key, timeout=timeout, max_retries=max_retries)
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        self.model = model
        self._dimension = dimension
        self._base_url = base_url
        self._send_dimensions = _is_openai_native(base_url)
        logger.info(
            "OpenAI-compatible embedding provider initialized "
            "(model=%s, dimensions=%d, base_url=%s, send_dimensions=%s, timeout=%.1fs, retries=%d)",
            model,
            dimension,
            base_url or "(default: api.openai.com)",
            self._send_dimensions,
            timeout,
            max_retries,
        )

    def _create_kwargs(self, input_val, model: str) -> dict:
        """Build kwargs for ``client.embeddings.create``."""
        kwargs = dict(input=input_val, model=model)
        if self._send_dimensions:
            kwargs["dimensions"] = self._dimension
        return kwargs

    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding using the configured API.

        Args:
            text: The text to embed

        Returns:
            Embedding vector

        Raises:
            Exception: If API call fails
        """
        response = self.client.embeddings.create(**self._create_kwargs(text, self.model))
        embedding = response.data[0].embedding
        if len(embedding) != self._dimension:
            raise ValueError(
                f"Embedding length {len(embedding)} != configured dimension {self._dimension} "
                f"(model={self.model}). "
                + (
                    "Ensure 'dimensions' is supported and vector store size matches."
                    if self._send_dimensions
                    else "The model's native dimension may differ from VECTOR_SIZE. "
                    "Adjust VECTOR_SIZE to match this model's output or use a model "
                    "that produces the expected number of dimensions."
                )
            )
        logger.debug("Generated embedding for text (length: %d)", len(text))
        return embedding

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in one API call.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            Exception: If API call fails
        """
        if not texts:
            return []

        response = self.client.embeddings.create(**self._create_kwargs(texts, self.model))
        embeddings = [item.embedding for item in response.data]
        bad = next((i for i, e in enumerate(embeddings) if len(e) != self._dimension), None)
        if bad is not None:
            raise ValueError(
                f"Batch embedding length {len(embeddings[bad])} != configured dimension "
                f"{self._dimension} at index {bad} (model={self.model})."
            )
        logger.info(
            "Generated %d embeddings in batch (avg length: %d)",
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
            Provider identifier including base URL hint when non-default.
        """
        if self._base_url:
            return f"openai-compatible:{self.model}"
        return f"openai:{self.model}"
