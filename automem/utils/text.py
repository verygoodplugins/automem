from __future__ import annotations

import re
from typing import List

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
