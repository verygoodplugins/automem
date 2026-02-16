"""LLM-based query expansion for recall.

Before searching, generates alternative phrasings of the query so we catch
memories stored with different wording. For example:
  "Jeff's workout routine" → ["exercise program", "fitness schedule", "gym sessions"]

Uses a cheap, fast model to keep latency low. Results are merged with the
original query via the existing multi-query/RRF pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EXPAND_ENABLED = os.environ.get("QUERY_EXPAND_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)
EXPAND_MODEL = os.environ.get("QUERY_EXPAND_MODEL", "gpt-4.1-nano")
EXPAND_TIMEOUT = float(os.environ.get("QUERY_EXPAND_TIMEOUT", "10.0"))
EXPAND_MAX_ALTERNATIVES = int(os.environ.get("QUERY_EXPAND_MAX", "3"))
# Reuse the rerank API key/base since it's the same cheap-model endpoint
EXPAND_API_KEY = os.environ.get("RERANK_API_KEY", "")
EXPAND_BASE_URL = os.environ.get("RERANK_BASE_URL", "")

# Minimum query length to bother expanding (very short queries are already broad)
MIN_QUERY_LENGTH = 8

# Cache recent expansions to avoid repeated LLM calls for the same query
_expansion_cache: Dict[str, List[str]] = {}
_CACHE_MAX_SIZE = 100

_expand_client: Any = None
_expand_client_type: str = ""


def _get_expand_client() -> Any:
    """Get or create a dedicated client for query expansion."""
    global _expand_client, _expand_client_type
    if _expand_client is not None:
        return _expand_client
    if not EXPAND_API_KEY:
        return None

    if EXPAND_API_KEY.startswith("sk-ant-"):
        try:
            import anthropic

            _expand_client = anthropic.Anthropic(api_key=EXPAND_API_KEY)
            _expand_client_type = "anthropic"
            return _expand_client
        except ImportError:
            logger.warning("anthropic package not installed for query expansion")
            return None
    else:
        try:
            from openai import OpenAI

            kwargs: Dict[str, Any] = {"api_key": EXPAND_API_KEY}
            if EXPAND_BASE_URL:
                kwargs["base_url"] = EXPAND_BASE_URL
            _expand_client = OpenAI(**kwargs)
            _expand_client_type = "openai"
            return _expand_client
        except Exception:
            logger.warning("Failed to create query expansion client", exc_info=True)
            return None


SYSTEM_PROMPT = """You generate alternative search queries to find relevant memories in a personal knowledge base.

Given a search query, produce alternative phrasings that would match memories stored with different wording.

Rules:
- Generate 2-3 short alternative queries (2-5 words each)
- Focus on synonyms, related terms, and different framings
- Include both broader and more specific alternatives
- Don't repeat the original query
- Think about how a memory might have been STORED vs how it's being SEARCHED

Return a JSON array of strings. Example: ["alternative 1", "alternative 2", "alternative 3"]"""


def expand_query(
    query: str,
    openai_client: Any = None,
    max_alternatives: int = EXPAND_MAX_ALTERNATIVES,
    model: Optional[str] = None,
) -> List[str]:
    """Generate alternative phrasings for a search query.

    Args:
        query: Original search query
        openai_client: Fallback OpenAI client
        max_alternatives: Max number of alternatives to generate
        model: Override model name

    Returns:
        List of alternative query strings (may be empty on failure)
    """
    if not EXPAND_ENABLED or not query or len(query.strip()) < MIN_QUERY_LENGTH:
        return []

    query_key = query.strip().lower()

    # Check cache
    if query_key in _expansion_cache:
        logger.debug("Query expansion cache hit: %s", query_key[:50])
        return _expansion_cache[query_key]

    client = _get_expand_client() or openai_client
    if client is None:
        return []

    model = model or EXPAND_MODEL

    try:
        t0 = time.monotonic()

        if (
            _expand_client_type == "anthropic"
            and client is not None
            and hasattr(client, "messages")
        ):
            response = client.messages.create(
                model=model,
                max_tokens=200,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": f"Query: {query}\n\nRespond with ONLY a JSON array of alternative queries.",
                    },
                ],
                timeout=EXPAND_TIMEOUT,
            )
            raw = response.content[0].text.strip()
        else:
            extra_params: Dict[str, Any] = {"max_tokens": 200}
            if not model.startswith(("o", "gpt-5")):
                extra_params["temperature"] = 0.7  # Some creativity for diverse alternatives

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Query: {query}\n\nRespond with ONLY a JSON array of alternative queries.",
                    },
                ],
                timeout=EXPAND_TIMEOUT,
                **extra_params,
            )
            raw = response.choices[0].message.content.strip()

        elapsed_ms = (time.monotonic() - t0) * 1000

        # Extract JSON array
        import re as _re

        json_match = _re.search(r"\[.*\]", raw, _re.DOTALL)
        if json_match:
            raw = json_match.group(0)

        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            return []

        alternatives = [
            str(item).strip() for item in parsed if isinstance(item, str) and len(item.strip()) >= 3
        ][:max_alternatives]

        # Cache the result
        if len(_expansion_cache) >= _CACHE_MAX_SIZE:
            # Evict oldest entries (simple FIFO via dict ordering)
            keys_to_remove = list(_expansion_cache.keys())[: _CACHE_MAX_SIZE // 2]
            for k in keys_to_remove:
                del _expansion_cache[k]
        _expansion_cache[query_key] = alternatives

        logger.info(
            "Query expansion: query=%r model=%s alternatives=%d time=%.0fms",
            query[:50],
            model,
            len(alternatives),
            elapsed_ms,
        )

        return alternatives

    except Exception:
        logger.warning("Query expansion failed, using original query only", exc_info=True)
        return []
