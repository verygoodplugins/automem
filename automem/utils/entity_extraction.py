from __future__ import annotations

import logging
import re
from threading import Lock
from typing import Any, Dict, List, Optional, Set

try:
    import spacy  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    spacy = None

logger = logging.getLogger(__name__)

_SEARCH_STOPWORDS: Set[str] = set()
_ENTITY_STOPWORDS: Set[str] = set()
_ENTITY_BLOCKLIST: Set[str] = set()
_SPACY_MODEL = "en_core_web_sm"
_SPACY_NLP: Any = None
_SPACY_INIT_LOCK = Lock()


def configure_entity_extraction(
    *,
    search_stopwords: Set[str],
    entity_stopwords: Set[str],
    entity_blocklist: Set[str],
    spacy_model: str,
) -> None:
    """Configure shared extraction state from app-level config."""
    global _SEARCH_STOPWORDS, _ENTITY_STOPWORDS, _ENTITY_BLOCKLIST, _SPACY_MODEL, _SPACY_NLP

    _SEARCH_STOPWORDS = set(search_stopwords)
    _ENTITY_STOPWORDS = set(entity_stopwords)
    _ENTITY_BLOCKLIST = set(entity_blocklist)

    if spacy_model and spacy_model != _SPACY_MODEL:
        _SPACY_MODEL = spacy_model
        _SPACY_NLP = None


def _get_spacy_nlp() -> Any:
    global _SPACY_NLP
    if spacy is None:
        return None

    with _SPACY_INIT_LOCK:
        if _SPACY_NLP is not None:
            return _SPACY_NLP

        try:
            _SPACY_NLP = spacy.load(_SPACY_MODEL)
            logger.info("Loaded spaCy model '%s' for enrichment", _SPACY_MODEL)
        except Exception:  # pragma: no cover - optional dependency
            logger.warning("Failed to load spaCy model '%s'", _SPACY_MODEL)
            _SPACY_NLP = None

        return _SPACY_NLP


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower())
    return cleaned.strip("-")


def _is_valid_entity(
    value: str, *, allow_lower: bool = False, max_words: Optional[int] = None
) -> bool:
    if not value:
        return False

    cleaned = value.strip()
    if len(cleaned) < 3:
        return False

    words = cleaned.split()
    if max_words is not None and len(words) > max_words:
        return False

    lowered = cleaned.lower()
    if lowered in _SEARCH_STOPWORDS or lowered in _ENTITY_STOPWORDS:
        return False

    # Reject error codes and technical noise
    if lowered in _ENTITY_BLOCKLIST:
        return False

    if not any(ch.isalpha() for ch in cleaned):
        return False

    if not allow_lower and cleaned[0].islower() and not cleaned.isupper():
        return False

    # Reject strings starting with markdown/formatting or code characters
    if cleaned[0] in {"-", "*", "#", ">", "|", "[", "]", "{", "}", "(", ")", "_", "'", '"'}:
        return False

    # Reject common code artifacts (suffixes that indicate class names)
    code_suffixes = (
        "Adapter",
        "Handler",
        "Manager",
        "Service",
        "Controller",
        "Provider",
        "Factory",
        "Builder",
        "Helper",
        "Util",
    )
    if any(cleaned.endswith(suffix) for suffix in code_suffixes):
        return False

    # Reject boolean/null literals and common JSON noise
    if lowered in {"true", "false", "null", "none", "undefined"}:
        return False

    # Reject environment vars (ALL_CAPS_WITH_UNDERSCORES) and fragments ending with ':'
    if ("_" in cleaned and cleaned.isupper()) or cleaned.endswith(":"):
        return False

    return True


def generate_summary(
    content: str, fallback: Optional[str] = None, *, max_length: int = 240
) -> Optional[str]:
    text = (content or "").strip()
    if not text:
        return fallback

    sentences = re.split(r"(?<=[.!?])\s+", text)
    summary = sentences[0] if sentences else text
    summary = summary.strip()

    if not summary:
        return fallback

    if len(summary) > max_length:
        truncated = summary[:max_length].rsplit(" ", 1)[0]
        summary = truncated.strip() if truncated else summary[:max_length].strip()

    if fallback and fallback.strip() == summary:
        return fallback

    return summary


def extract_entities(content: str) -> Dict[str, List[str]]:
    """Extract entities from memory content using spaCy when available."""
    result: Dict[str, Set[str]] = {
        "tools": set(),
        "projects": set(),
        "people": set(),
        "concepts": set(),
        "organizations": set(),
    }

    text = (content or "").strip()
    if not text:
        return {key: [] for key in result}

    nlp = _get_spacy_nlp()
    if nlp is not None:
        try:
            doc = nlp(text)
            for ent in doc.ents:
                value = ent.text.strip()
                if not _is_valid_entity(value, allow_lower=False, max_words=6):
                    continue
                if ent.label_ in {"PERSON"}:
                    result["people"].add(value)
                elif ent.label_ in {"ORG"}:
                    result["organizations"].add(value)
                elif ent.label_ in {"PRODUCT", "WORK_OF_ART", "LAW"}:
                    result["tools"].add(value)
                elif ent.label_ in {"EVENT", "GPE", "LOC", "NORP"}:
                    result["concepts"].add(value)
        except Exception:  # pragma: no cover - defensive
            logger.exception("spaCy entity extraction failed")

    # Regex-based fallbacks to capture simple patterns
    for match in re.findall(
        r"(?:with|met with|meeting with|talked to|spoke with)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        text,
    ):
        result["people"].add(match.strip())

    tool_patterns = [
        r"(?:use|using|deploy|deployed|with|via)\s+([A-Z][\w\-]+)",
        r"([A-Z][\w\-]+)\s+(?:vs|versus|over|instead of)",
    ]
    for pattern in tool_patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            cleaned = match.strip()
            if _is_valid_entity(cleaned):
                result["tools"].add(cleaned)

    for match in re.findall(r"`([^`]+)`", text):
        cleaned = match.strip()
        if _is_valid_entity(cleaned, allow_lower=False, max_words=4):
            result["projects"].add(cleaned)

    # Extract project names from "project called/named 'X'" pattern
    for match in re.findall(
        r'(?:project|repo|repository)\s+(?:called|named)\s+"([^"]+)"', text, re.IGNORECASE
    ):
        cleaned = match.strip()
        if _is_valid_entity(cleaned, allow_lower=False, max_words=4):
            result["projects"].add(cleaned)

    # Extract project names from 'project "X"' pattern
    for match in re.findall(r'(?:project|repo|repository)\s+"([^"]+)"', text, re.IGNORECASE):
        cleaned = match.strip()
        if _is_valid_entity(cleaned, allow_lower=False, max_words=4):
            result["projects"].add(cleaned)

    for match in re.findall(r"Project\s+([A-Z][\w\-]+)", text):
        cleaned = match.strip()
        if _is_valid_entity(cleaned):
            result["projects"].add(cleaned)

    # Extract project names from "project: project-name" pattern (common in session starts)
    for match in re.findall(r"(?:in |on )?project:\s+([a-z][a-z0-9\-]+)", text, re.IGNORECASE):
        cleaned = match.strip()
        if _is_valid_entity(cleaned, allow_lower=True):
            result["projects"].add(cleaned)

    result["tools"].difference_update(result["people"])

    cleaned = {key: sorted({value for value in values if value}) for key, values in result.items()}
    return cleaned
