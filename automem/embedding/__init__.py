"""Embedding provider module for AutoMem.

Provides abstraction over different embedding backends:
- OpenAI (API-based, requires key)
- FastEmbed (local model, no API key needed)
- Placeholder (hash-based fallback)
"""

from .provider import EmbeddingProvider
# Optional backends: guard imports to avoid hard dependencies at import time
try:
    from .openai import OpenAIEmbeddingProvider  # type: ignore
except ImportError:
    OpenAIEmbeddingProvider = None  # type: ignore[assignment]
try:
    from .fastembed import FastEmbedProvider  # type: ignore
except ImportError:
    FastEmbedProvider = None  # type: ignore[assignment]
from .placeholder import PlaceholderEmbeddingProvider

__all__ = [
    "EmbeddingProvider",
    "FastEmbedProvider",
    "OpenAIEmbeddingProvider",
    "PlaceholderEmbeddingProvider",
]
