"""FastEmbed local embedding provider using ONNX models."""

import logging
import os
from pathlib import Path
from typing import List, Optional

from fastembed import TextEmbedding

from automem.embedding.provider import EmbeddingProvider

logger = logging.getLogger(__name__)


# Model dimension mapping
FASTEMBED_MODELS = {
    384: "BAAI/bge-small-en-v1.5",  # 67MB, fast
    768: "BAAI/bge-base-en-v1.5",  # 210MB, matches OpenAI dimension
    1024: "BAAI/bge-large-en-v1.5",  # 1.2GB, high quality
}


class FastEmbedProvider(EmbeddingProvider):
    """Generates embeddings using local ONNX models via fastembed.

    Downloads model on first use and caches locally. Requires no API key
    and works offline after initial download. Provides good quality semantic
    embeddings without API costs.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        dimension: int = 768,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize FastEmbed embedding provider.

        Args:
            model_name: Specific model to use, or None to auto-select by dimension
            dimension: Number of dimensions (used for auto-selection if model_name is None)
            cache_dir: Directory for model cache (defaults to ~/.config/automem/models/)

        Raises:
            Exception: If model initialization fails
        """
        # Auto-select model by dimension if not specified
        if model_name is None:
            model_name = FASTEMBED_MODELS.get(dimension, "BAAI/bge-base-en-v1.5")
            logger.info(
                "Auto-selected fastembed model for %d dimensions: %s",
                dimension,
                model_name,
            )

        # Set cache directory; allow env override for portability across users/containers
        if cache_dir is None:
            env_dir = os.getenv("AUTOMEM_MODELS_DIR")
            cache_dir = (
                Path(env_dir) if env_dir else (Path.home() / ".config" / "automem" / "models")
            )
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Check if model is cached
        model_dir_name = model_name.replace("/", "--").replace(":", "--")
        model_cached = any(
            d.name.startswith(model_dir_name) for d in cache_dir.iterdir() if d.is_dir()
        )

        if model_cached:
            logger.info("Loading %s from cache...", model_name)
        else:
            logger.info(
                "Downloading %s embedding model (~%s, first time only)\n"
                "Model will be cached to: %s\n"
                "This may take 1-2 minutes...",
                model_name,
                self._get_model_size_description(dimension),
                cache_dir,
            )

        # Initialize model (fastembed handles download automatically)
        # Try with progress_bar parameter, fall back if not supported
        try:
            self.model = TextEmbedding(
                model_name=model_name, cache_dir=str(cache_dir), progress_bar=True
            )
        except TypeError:
            # Fallback if progress_bar parameter doesn't exist
            self.model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))

        self.model_name = model_name
        # Derive actual embedding dimension to avoid mismatches
        try:
            _probe = next(self.model.embed([" "]))
            actual_dim = len(_probe)
        except Exception:
            # Fallback: assume requested dimension if probe fails (keeps behavior unchanged)
            actual_dim = dimension

        if actual_dim != dimension:
            logger.warning(
                "fastembed actual dimension %d != configured %d for model %s. "
                "Using actual dimension; ensure your VECTOR_SIZE/Qdrant collection matches.",
                actual_dim,
                dimension,
                model_name,
            )
        self._dimension = actual_dim

        logger.info("✓ %s ready (dimension=%d)", model_name, self._dimension)

    @staticmethod
    def _get_model_size_description(dimension: int) -> str:
        """Get human-readable model size description."""
        size_map = {384: "67MB", 768: "210MB", 1024: "1.2GB"}
        return size_map.get(dimension, "~200MB")

    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding using local ONNX model.

        Args:
            text: The text to embed

        Returns:
            Embedding vector from fastembed

        Raises:
            Exception: If embedding generation fails
        """
        embeddings = list(self.model.embed([text]))
        embedding = embeddings[0].tolist()
        if len(embedding) != self._dimension:
            raise ValueError(
                f"fastembed embedding length {len(embedding)} != configured dimension {self._dimension} "
                f"(model={self.model_name})"
            )
        logger.debug("Generated fastembed embedding for text (length: %d)", len(text))
        return embedding

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors from fastembed

        Raises:
            Exception: If batch embedding generation fails
        """
        if not texts:
            return []

        embeddings = [emb.tolist() for emb in self.model.embed(texts)]
        bad = next((i for i, e in enumerate(embeddings) if len(e) != self._dimension), None)
        if bad is not None:
            raise ValueError(
                f"fastembed batch embedding length {len(embeddings[bad])} != configured dimension {self._dimension} "
                f"at index {bad} (model={self.model_name})"
            )
        logger.info(
            "Generated %d fastembed embeddings in batch (avg length: %d)",
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
        return f"fastembed:{self.model_name}"
