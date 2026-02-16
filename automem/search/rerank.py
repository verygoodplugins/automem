"""LLM-based reranking for recall results.

After initial retrieval (vector + graph + BM25), an LLM scores each
result's actual relevance to the query. This catches false positives
from vector search and promotes results that are genuinely relevant.

Uses a cheap, fast model (gpt-4.1-nano by default) to keep latency low.
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

RERANK_ENABLED = os.environ.get("RERANK_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)
RERANK_MODEL = os.environ.get("RERANK_MODEL", "gpt-4.1-nano")
RERANK_TOP_N = int(os.environ.get("RERANK_TOP_N", "20"))  # How many candidates to rerank
RERANK_TIMEOUT = float(os.environ.get("RERANK_TIMEOUT", "15.0"))  # Seconds
# Separate API key/base for reranking (falls back to main OpenAI client if not set)
RERANK_API_KEY = os.environ.get("RERANK_API_KEY", "")
RERANK_BASE_URL = os.environ.get("RERANK_BASE_URL", "")

_rerank_client: Any = None
_rerank_client_type: str = ""  # "anthropic" or "openai"


def _get_rerank_client() -> Any:
    """Get or create a dedicated client for reranking."""
    global _rerank_client, _rerank_client_type
    if _rerank_client is not None:
        return _rerank_client
    if not RERANK_API_KEY:
        return None

    # Detect Anthropic key
    if RERANK_API_KEY.startswith("sk-ant-"):
        try:
            import anthropic

            _rerank_client = anthropic.Anthropic(api_key=RERANK_API_KEY)
            _rerank_client_type = "anthropic"
            return _rerank_client
        except ImportError:
            logger.warning("anthropic package not installed for reranking")
            return None
    else:
        try:
            from openai import OpenAI

            kwargs: Dict[str, Any] = {"api_key": RERANK_API_KEY}
            if RERANK_BASE_URL:
                kwargs["base_url"] = RERANK_BASE_URL
            _rerank_client = OpenAI(**kwargs)
            _rerank_client_type = "openai"
            return _rerank_client
        except Exception:
            logger.warning("Failed to create rerank OpenAI client", exc_info=True)
            return None


SYSTEM_PROMPT = """You are a memory relevance scorer. Given a search query and a list of memory snippets, score each snippet's relevance to the query on a scale of 0-10.

Scoring guide:
- 10: Directly answers the query or is exactly what was asked for
- 7-9: Highly relevant, contains key information related to the query
- 4-6: Somewhat relevant, tangentially related
- 1-3: Barely relevant, only loosely connected
- 0: Completely irrelevant

Return a JSON array of objects with "index" (0-based) and "score" (0-10) for each snippet.
Example: [{"index": 0, "score": 8}, {"index": 1, "score": 3}]

Be strict — only high scores for genuinely relevant results."""


def rerank(
    query: str,
    results: List[Dict[str, Any]],
    openai_client: Any,
    top_n: int = RERANK_TOP_N,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Rerank results using an LLM. Returns reordered list with rerank_score added.

    Args:
        query: The original search query
        results: List of recall results (must have 'memory' dict with 'content')
        openai_client: OpenAI client instance
        top_n: Max number of candidates to send to the LLM
        model: Override model name

    Returns:
        Reordered results list with 'rerank_score' added to each result
    """
    if not RERANK_ENABLED or not results or not query:
        return results

    # Try dedicated rerank client first, then fall back to passed-in client
    client = _get_rerank_client() or openai_client
    if client is None:
        return results

    model = model or RERANK_MODEL
    candidates = results[:top_n]
    remainder = results[top_n:]

    # Build the prompt with numbered snippets
    snippets = []
    for i, r in enumerate(candidates):
        mem = r.get("memory") or r
        content = mem.get("content", "")
        # Truncate long content to keep prompt manageable
        if len(content) > 300:
            content = content[:300] + "..."
        snippets.append(f"[{i}] {content}")

    user_prompt = f"Query: {query}\n\nSnippets:\n" + "\n".join(snippets)

    try:
        t0 = time.monotonic()

        # Route to appropriate API
        if (
            _rerank_client_type == "anthropic"
            and client is not None
            and hasattr(client, "messages")
        ):
            response = client.messages.create(
                model=model,
                max_tokens=500,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt
                        + "\n\nRespond with ONLY a JSON array, no other text.",
                    },
                ],
                timeout=RERANK_TIMEOUT,
            )
            elapsed_ms = (time.monotonic() - t0) * 1000
            raw = response.content[0].text.strip()
        else:
            # OpenAI-compatible path
            extra_params: Dict[str, Any] = {"max_tokens": 500}
            if not model.startswith(("o", "gpt-5")):
                extra_params["temperature"] = 0.0

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": user_prompt
                        + "\n\nRespond with ONLY a JSON array, no other text.",
                    },
                ],
                timeout=RERANK_TIMEOUT,
                **extra_params,
            )
            elapsed_ms = (time.monotonic() - t0) * 1000
            raw = response.choices[0].message.content.strip()

        # Extract JSON from response (may have markdown fences or preamble)
        import re as _re

        json_match = _re.search(r"[\[{].*[\]}]", raw, _re.DOTALL)
        if json_match:
            raw = json_match.group(0)

        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            scores_list = (
                parsed.get("results") or parsed.get("scores") or parsed.get("rankings") or []
            )
            if not scores_list:
                # Try to find any list value
                for v in parsed.values():
                    if isinstance(v, list):
                        scores_list = v
                        break
        elif isinstance(parsed, list):
            scores_list = parsed
        else:
            scores_list = []

        # Build index -> score map
        score_map: Dict[int, float] = {}
        for item in scores_list:
            if isinstance(item, dict) and "index" in item and "score" in item:
                idx = int(item["index"])
                score = float(item["score"])
                if 0 <= idx < len(candidates):
                    score_map[idx] = score

        # Apply scores and sort
        for i, r in enumerate(candidates):
            rerank_score = score_map.get(i, 5.0)  # Default to middle if not scored
            r["rerank_score"] = rerank_score
            r.setdefault("score_components", {})
            r["score_components"]["rerank"] = rerank_score

        # Sort by rerank score descending
        candidates.sort(key=lambda r: -r.get("rerank_score", 0))

        # Remainder gets a default score
        for r in remainder:
            r["rerank_score"] = 0.0
            r.setdefault("score_components", {})
            r["score_components"]["rerank"] = 0.0

        logger.info(
            "LLM rerank: model=%s candidates=%d scored=%d time=%.0fms",
            model,
            len(candidates),
            len(score_map),
            elapsed_ms,
        )

        return candidates + remainder

    except Exception:
        logger.warning("LLM rerank failed, returning original order", exc_info=True)
        return results
