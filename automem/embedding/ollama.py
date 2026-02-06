"""Ollama embedding provider using local Ollama HTTP API."""

import logging
from typing import List, Optional

import requests

from automem.embedding.provider import EmbeddingProvider

logger = logging.getLogger(__name__)


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Generates embeddings using an Ollama server.

    Requires a running Ollama instance (default: http://localhost:11434) with
    the requested embedding model pulled locally (e.g., nomic-embed-text).
    """

    def __init__(
        self,
        base_url: str,
        model: str = "nomic-embed-text",
        dimension: int = 768,
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        """Initialize Ollama embedding provider.

        Args:
            base_url: Base URL for the Ollama API (e.g., http://localhost:11434)
            model: Ollama embedding model name
            dimension: Expected embedding dimension
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for transient errors
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._dimension = dimension
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        logger.info(
            "Ollama embedding provider initialized (model=%s, dimensions=%d, base_url=%s)",
            model,
            dimension,
            self.base_url,
        )

    def _request_embedding(self, text: str) -> List[float]:
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.model, "prompt": text}
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                if "embedding" in data:
                    return data["embedding"]
                if "data" in data and data["data"]:
                    candidate = data["data"][0]
                    if isinstance(candidate, dict) and "embedding" in candidate:
                        return candidate["embedding"]
                raise ValueError(f"Unexpected Ollama embedding response format: {data}")
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
        raise RuntimeError(f"Ollama embedding request failed: {last_error}") from last_error

    def generate_embedding(self, text: str) -> List[float]:
        embedding = self._request_embedding(text)
        if len(embedding) != self._dimension:
            raise ValueError(
                f"Ollama embedding length {len(embedding)} != configured dimension {self._dimension} "
                f"(model={self.model})."
            )
        logger.debug("Generated Ollama embedding for text (length: %d)", len(text))
        return embedding

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = [self._request_embedding(text) for text in texts]
        bad = next((i for i, e in enumerate(embeddings) if len(e) != self._dimension), None)
        if bad is not None:
            raise ValueError(
                f"Ollama batch embedding length {len(embeddings[bad])} != configured dimension "
                f"{self._dimension} at index {bad} (model={self.model})."
            )
        logger.info(
            "Generated %d Ollama embeddings in batch (avg length: %d)",
            len(embeddings),
            sum(len(t) for t in texts) // len(texts) if texts else 0,
        )
        return embeddings

    def dimension(self) -> int:
        return self._dimension

    def provider_name(self) -> str:
        return f"ollama:{self.model}"
