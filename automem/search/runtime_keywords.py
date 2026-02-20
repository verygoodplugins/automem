from __future__ import annotations

import logging
import re
from typing import Callable, List, Set, Tuple


def load_keyword_runtime() -> Tuple[Set[str], Set[str], Set[str], Callable[[str], List[str]]]:
    search_stopwords: Set[str] = set()
    entity_stopwords: Set[str] = set()
    entity_blocklist: Set[str] = set()
    logger = logging.getLogger(__name__)

    try:
        from automem.utils.text import ENTITY_BLOCKLIST as _am_entity_blocklist
        from automem.utils.text import ENTITY_STOPWORDS as _am_entity_stopwords
        from automem.utils.text import SEARCH_STOPWORDS as _am_search_stopwords
        from automem.utils.text import _extract_keywords as _am_extract_keywords

        search_stopwords = _am_search_stopwords
        entity_stopwords = _am_entity_stopwords
        entity_blocklist = _am_entity_blocklist
        return (
            search_stopwords,
            entity_stopwords,
            entity_blocklist,
            _am_extract_keywords,
        )
    except ImportError:
        pass
    except Exception:
        logger.exception(
            "Unexpected error importing keyword helpers; falling back to local extractor"
        )

    def _extract_keywords(text: str) -> List[str]:
        if not text:
            return []
        words = re.findall(r"[A-Za-z0-9_\-]+", text.lower())
        keywords: List[str] = []
        seen: set[str] = set()
        for word in words:
            cleaned = word.strip("-_")
            if len(cleaned) < 3:
                continue
            if cleaned in search_stopwords:
                continue
            if cleaned in seen:
                continue
            seen.add(cleaned)
            keywords.append(cleaned)
        return keywords

    return search_stopwords, entity_stopwords, entity_blocklist, _extract_keywords
