"""Embedding provider module for AutoMem.

Provides abstraction over different embedding backends:
- OpenAI (API-based, requires key)
- FastEmbed (local model, no API key needed)
- Placeholder (hash-based fallback)
"""

from automem.embedding.provider import EmbeddingProvider
from automem.embedding.openai import OpenAIEmbeddingProvider
from automem.embedding.fastembed import FastEmbedProvider
from automem.embedding.placeholder import PlaceholderEmbeddingProvider

__all__ = [
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "FastEmbedProvider",
    "PlaceholderEmbeddingProvider",
]
