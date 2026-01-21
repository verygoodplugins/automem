from __future__ import annotations

import logging
import re
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# Common stopwords to exclude from search tokens
SEARCH_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "using",
    "have",
    "will",
    "your",
    "about",
    "after",
    "before",
    "when",
    "then",
    "than",
    "also",
    "just",
    "very",
    "more",
    "less",
    "over",
    "under",
}

# Entity-level stopwords and blocklist for extraction filtering
ENTITY_STOPWORDS = {
    "you",
    "your",
    "yours",
    "whatever",
    "today",
    "tomorrow",
    "project",
    "projects",
    "office",
    "session",
    "meeting",
}

# Common error codes and technical strings to exclude from entity extraction
ENTITY_BLOCKLIST = {
    # HTTP errors
    "bad request",
    "not found",
    "unauthorized",
    "forbidden",
    "internal server error",
    "service unavailable",
    "gateway timeout",
    # Network errors
    "econnreset",
    "econnrefused",
    "etimedout",
    "enotfound",
    "enetunreach",
    "ehostunreach",
    "epipe",
    "eaddrinuse",
    # Common error patterns
    "error",
    "warning",
    "exception",
    "failed",
    "failure",
}


def _extract_keywords(text: str) -> List[str]:
    """Convert a raw query string into normalized keyword tokens."""
    if not text:
        return []

    words = re.findall(r"[A-Za-z0-9_\-]+", text.lower())
    keywords: List[str] = []
    seen: set[str] = set()

    for word in words:
        cleaned = word.strip("-_")
        if len(cleaned) < 3:
            continue
        if cleaned in SEARCH_STOPWORDS:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        keywords.append(cleaned)

    return keywords


# Summarization system prompt for auto-condensing oversized memories
SUMMARIZE_SYSTEM_PROMPT = """You are a memory summarization assistant. Your task is to condense a memory into a brief, information-dense summary.

Rules:
1. Preserve the key decision, insight, pattern, or context
2. Keep critical details: what happened, why it matters, the outcome
3. Remove filler words, redundant phrasing, and unnecessary detail
4. Maintain any specific names, files, or technical terms mentioned
5. Output ONLY the summary text, no JSON or formatting

Target length: Under {target_length} characters.

Format: "Brief title. Key context. Impact/outcome."
"""


def summarize_content(
    content: str,
    openai_client: Any,
    model: str,
    target_length: int = 300,
) -> Optional[str]:
    """Summarize content using an LLM to fit within target length.

    Args:
        content: The original content to summarize
        openai_client: An initialized OpenAI client instance
        model: The model to use for summarization (e.g., gpt-4o-mini)
        target_length: Target character length for the summary (default 300)

    Returns:
        Summarized content string, or None if summarization fails
    """
    if openai_client is None:
        logger.warning("Cannot summarize: OpenAI client not available")
        return None

    if not content or len(content) <= target_length:
        return content

    try:
        system_prompt = SUMMARIZE_SYSTEM_PROMPT.format(target_length=target_length)

        # Estimate tokens from target character length (~4 chars/token), cap at 150
        token_limit = min(150, max(1, int(target_length / 4)))

        # Use appropriate token parameter based on model family
        if model.startswith("o"):  # o-series (o1, o3, etc.)
            token_param = {"max_completion_tokens": token_limit}
        elif model.startswith("gpt-5"):  # gpt-5 family
            token_param = {"max_output_tokens": token_limit}
        else:  # gpt-4o-mini, gpt-4, etc.
            token_param = {"max_tokens": token_limit}

        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            temperature=0.3,
            **token_param,
        )

        summary = response.choices[0].message.content.strip()

        # Validate we actually got a shorter result
        if summary and len(summary) < len(content):
            logger.info(
                "Summarized memory content: %d -> %d chars (%.0f%% reduction)",
                len(content),
                len(summary),
                (1 - len(summary) / len(content)) * 100,
            )
            return summary

        logger.warning(
            "Summarization did not reduce content length (%d -> %d), returning original",
            len(content),
            len(summary) if summary else 0,
        )
        return None

    except Exception:
        logger.exception("Memory summarization failed")
        return None


def should_summarize_content(content: str, soft_limit: int, hard_limit: int) -> str:
    """Check if content should be summarized or rejected.

    Args:
        content: The content to check
        soft_limit: Character count above which summarization is triggered
        hard_limit: Character count above which content is rejected

    Returns:
        "ok" if content is fine as-is
        "summarize" if content should be summarized
        "reject" if content exceeds hard limit
    """
    if not content:
        return "ok"

    length = len(content)

    if length > hard_limit:
        return "reject"
    if length > soft_limit:
        return "summarize"
    return "ok"
