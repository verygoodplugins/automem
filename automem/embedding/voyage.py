"""Voyage AI embedding provider using Voyage API."""

import logging
import os
import time
from typing import List, Optional

import httpx

from automem.embedding.provider import EmbeddingProvider

logger = logging.getLogger(__name__)

VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"


class VoyageEmbeddingProvider(EmbeddingProvider):
    """Generates embeddings using Voyage AI's embedding API.

    Requires a Voyage API key and makes network requests to Voyage AI.
    Provides high-quality semantic embeddings with generous free tier.
    
    Supports the Voyage 4 family with shared embedding spaces:
    - voyage-4-large: Best quality, MoE architecture
    - voyage-4: Approaches voyage-3-large quality, mid-sized
    - voyage-4-lite: Optimized for latency/cost
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "voyage-4",
        dimension: int = 1024,
        timeout: float = 30.0,
        max_retries: int = 2,
        input_type: Optional[str] = None,
    ):
        """Initialize Voyage embedding provider.

        Args:
            api_key: Voyage API key (falls back to VOYAGE_API_KEY env var)
            model: Voyage embedding model to use (voyage-4, voyage-4-large, voyage-4-lite)
            dimension: Number of dimensions for embeddings (256, 512, 1024, 2048)
            timeout: Request timeout in seconds (default 30)
            max_retries: Maximum number of retries (default 2)
            input_type: Optional input type hint ("query" or "document")

        Raises:
            ValueError: If API key is not provided or dimension is invalid
        """
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        if not self.api_key:
            raise ValueError("Voyage API key required (pass api_key or set VOYAGE_API_KEY)")
        
        valid_dimensions = {256, 512, 1024, 2048}
        if dimension not in valid_dimensions:
            raise ValueError(f"Invalid dimension {dimension}. Must be one of: {valid_dimensions}")
        
        self.model = model
        self._dimension = dimension
        self.timeout = timeout
        self.max_retries = max_retries
        self.input_type = input_type
        
        self.client = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        
        logger.info(
            "Voyage embedding provider initialized (model=%s, dimensions=%d, timeout=%.1fs, retries=%d)",
            model,
            dimension,
            timeout,
            max_retries,
        )

    def _make_request(self, texts: List[str], input_type: Optional[str] = None) -> List[List[float]]:
        """Make embedding request to Voyage API.
        
        Args:
            texts: List of texts to embed
            input_type: Optional input type ("query" or "document")
            
        Returns:
            List of embedding vectors
            
        Raises:
            Exception: If API call fails after retries
        """
        payload = {
            "input": texts,
            "model": self.model,
            "output_dimension": self._dimension,
        }
        
        # Use instance input_type if not overridden
        effective_input_type = input_type or self.input_type
        if effective_input_type:
            payload["input_type"] = effective_input_type
        
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.post(VOYAGE_API_URL, json=payload)
                response.raise_for_status()
                data = response.json()
                
                # Defensive validation of API response structure
                if "data" not in data:
                    raise ValueError("Voyage API response missing 'data' field")
                if not isinstance(data["data"], list):
                    raise ValueError("Voyage API response 'data' field is not a list")
                
                embeddings = []
                for i, item in enumerate(data["data"]):
                    if "embedding" not in item:
                        raise ValueError(
                            f"Voyage API response item {i} "
                            f"missing 'embedding' field"
                        )
                    embeddings.append(item["embedding"])
                
                # Validate dimensions
                for i, emb in enumerate(embeddings):
                    if len(emb) != self._dimension:
                        raise ValueError(
                            f"Voyage embedding length {len(emb)} != configured dimension {self._dimension} "
                            f"at index {i} (model={self.model})"
                        )
                
                return embeddings
                
            except httpx.HTTPStatusError as e:
                last_error = e
                status = e.response.status_code
                if status == 429 or 500 <= status < 600:
                    error_type = (
                        "rate limited" if status == 429
                        else "server error"
                    )
                    logger.warning(
                        "Voyage %s (status %d), attempt %d/%d",
                        error_type, status,
                        attempt + 1, self.max_retries + 1,
                    )
                    if attempt < self.max_retries:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                raise
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    logger.warning(
                        "Voyage request failed, attempt %d/%d: %s",
                        attempt + 1, self.max_retries + 1, e,
                    )
                    continue
                raise
        
        raise last_error or RuntimeError("Voyage request failed after retries")

    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding using Voyage API.

        Args:
            text: The text to embed

        Returns:
            Embedding vector from Voyage

        Raises:
            Exception: If API call fails
        """
        embeddings = self._make_request([text])
        logger.debug("Generated Voyage embedding for text (length: %d)", len(text))
        return embeddings[0]

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in one API call.

        Args:
            texts: List of texts to embed (max 128 per batch for Voyage)

        Returns:
            List of embedding vectors from Voyage

        Raises:
            Exception: If API call fails
        """
        if not texts:
            return []
        
        # Voyage supports up to 128 texts per batch
        batch_size = 128
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self._make_request(batch)
            all_embeddings.extend(embeddings)
        
        logger.info(
            "Generated %d Voyage embeddings in batch (avg length: %d)",
            len(all_embeddings),
            sum(len(t) for t in texts) // len(texts) if texts else 0,
        )
        return all_embeddings

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
        return f"voyage:{self.model}"
    
    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, 'client'):
            self.client.close()
