"""AutoMem Memory Service API.

Provides a small Flask API that stores memories in FalkorDB and Qdrant.
This module focuses on being resilient: it validates requests, handles
transient outages, and degrades gracefully when one of the backing services
is unavailable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from threading import Thread, Event, Lock
from queue import Empty, Queue
import time

from dotenv import load_dotenv
from flask import Flask, abort, jsonify, request
from falkordb import FalkorDB
from qdrant_client import QdrantClient, models as qdrant_models
from qdrant_client.models import Distance, PointStruct, VectorParams

try:
    from qdrant_client.models import PayloadSchemaType
except ImportError:
    # Fallback for test environments where PayloadSchemaType might not be available
    PayloadSchemaType = None
from werkzeug.exceptions import HTTPException
from consolidation import MemoryConsolidator, ConsolidationScheduler

# Make OpenAI import optional to allow running without it
try:
    from openai import OpenAI  # type: ignore
except ImportError:
    OpenAI = None  # type: ignore

# Import only the interface; import backends lazily in init_embedding_provider()
from automem.embedding.provider import EmbeddingProvider

try:
    import spacy  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    spacy = None

# Load environment variables before configuring the application.
load_dotenv()
load_dotenv(Path.home() / ".config" / "automem" / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,  # Write to stdout so Railway correctly parses log levels
)
logger = logging.getLogger("automem.api")

# Configure Flask and Werkzeug loggers to use stdout instead of stderr
# This ensures Railway correctly parses log levels instead of treating everything as "error"
for logger_name in ["werkzeug", "flask.app"]:
    framework_logger = logging.getLogger(logger_name)
    framework_logger.handlers.clear()
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    framework_logger.addHandler(stdout_handler)
    framework_logger.setLevel(logging.INFO)

# Ensure local package imports work when only app.py is copied
try:
    import automem  # type: ignore
except Exception:
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

app = Flask(__name__)

# Configuration constants
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "memories")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE") or os.getenv("QDRANT_VECTOR_SIZE", "768"))
GRAPH_NAME = os.getenv("FALKORDB_GRAPH", "memories")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))

# Consolidation scheduling defaults (seconds unless noted)
CONSOLIDATION_TICK_SECONDS = int(os.getenv("CONSOLIDATION_TICK_SECONDS", "60"))
CONSOLIDATION_DECAY_INTERVAL_SECONDS = int(
    os.getenv("CONSOLIDATION_DECAY_INTERVAL_SECONDS", str(3600))
)
CONSOLIDATION_CREATIVE_INTERVAL_SECONDS = int(os.getenv("CONSOLIDATION_CREATIVE_INTERVAL_SECONDS", str(3600)))
CONSOLIDATION_CLUSTER_INTERVAL_SECONDS = int(os.getenv("CONSOLIDATION_CLUSTER_INTERVAL_SECONDS", str(21600)))
CONSOLIDATION_FORGET_INTERVAL_SECONDS = int(os.getenv("CONSOLIDATION_FORGET_INTERVAL_SECONDS", str(86400)))
_DECAY_THRESHOLD_RAW = os.getenv("CONSOLIDATION_DECAY_IMPORTANCE_THRESHOLD", "0.3").strip()
CONSOLIDATION_DECAY_IMPORTANCE_THRESHOLD = (
    float(_DECAY_THRESHOLD_RAW) if _DECAY_THRESHOLD_RAW else None
)
CONSOLIDATION_HISTORY_LIMIT = int(os.getenv("CONSOLIDATION_HISTORY_LIMIT", "20"))
CONSOLIDATION_CONTROL_LABEL = "ConsolidationControl"
CONSOLIDATION_RUN_LABEL = "ConsolidationRun"
CONSOLIDATION_CONTROL_NODE_ID = os.getenv("CONSOLIDATION_CONTROL_NODE_ID", "global")
CONSOLIDATION_TASK_FIELDS = {
    "decay": "decay_last_run",
    "creative": "creative_last_run",
    "cluster": "cluster_last_run",
    "forget": "forget_last_run",
    "full": "full_last_run",
}

# Enrichment configuration
ENRICHMENT_MAX_ATTEMPTS = int(os.getenv("ENRICHMENT_MAX_ATTEMPTS", "3"))
ENRICHMENT_SIMILARITY_LIMIT = int(os.getenv("ENRICHMENT_SIMILARITY_LIMIT", "5"))
ENRICHMENT_SIMILARITY_THRESHOLD = float(os.getenv("ENRICHMENT_SIMILARITY_THRESHOLD", "0.8"))
ENRICHMENT_IDLE_SLEEP_SECONDS = float(os.getenv("ENRICHMENT_IDLE_SLEEP_SECONDS", "2"))
ENRICHMENT_FAILURE_BACKOFF_SECONDS = float(os.getenv("ENRICHMENT_FAILURE_BACKOFF_SECONDS", "5"))
ENRICHMENT_ENABLE_SUMMARIES = os.getenv("ENRICHMENT_ENABLE_SUMMARIES", "true").lower() not in {"0", "false", "no"}
ENRICHMENT_SPACY_MODEL = os.getenv("ENRICHMENT_SPACY_MODEL", "en_core_web_sm")
RECALL_RELATION_LIMIT = int(os.getenv("RECALL_RELATION_LIMIT", "5"))

# Embedding batching configuration
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "20"))
EMBEDDING_BATCH_TIMEOUT_SECONDS = float(os.getenv("EMBEDDING_BATCH_TIMEOUT_SECONDS", "2.0"))

# Memory types for classification
MEMORY_TYPES = {
    "Decision", "Pattern", "Preference", "Style",
    "Habit", "Insight", "Context"
}

# Note: "Memory" used as internal fallback only, not a valid classification

# Enhanced relationship types with their properties
RELATIONSHIP_TYPES = {
    # Original relationships
    "RELATES_TO": {"description": "General relationship"},
    "LEADS_TO": {"description": "Causal relationship"},
    "OCCURRED_BEFORE": {"description": "Temporal relationship"},

    # New PKG relationships
    "PREFERS_OVER": {"description": "Preference relationship", "properties": ["context", "strength", "reason"]},
    "EXEMPLIFIES": {"description": "Pattern example", "properties": ["pattern_type", "confidence"]},
    "CONTRADICTS": {"description": "Conflicting information", "properties": ["resolution", "reason"]},
    "REINFORCES": {"description": "Strengthens pattern", "properties": ["strength", "observations"]},
    "INVALIDATED_BY": {"description": "Superseded information", "properties": ["reason", "timestamp"]},
    "EVOLVED_INTO": {"description": "Evolution of knowledge", "properties": ["confidence", "reason"]},
    "DERIVED_FROM": {"description": "Derived knowledge", "properties": ["transformation", "confidence"]},
    "PART_OF": {"description": "Hierarchical relationship", "properties": ["role", "context"]},
}

ALLOWED_RELATIONS = set(RELATIONSHIP_TYPES.keys())

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
    "bad request", "not found", "unauthorized", "forbidden", "internal server error",
    "service unavailable", "gateway timeout",
    # Network errors
    "econnreset", "econnrefused", "etimedout", "enotfound", "enetunreach",
    "ehostunreach", "epipe", "eaddrinuse",
    # Common error patterns
    "error", "warning", "exception", "failed", "failure",
}

# Search weighting parameters (can be overridden via environment variables)
SEARCH_WEIGHT_VECTOR = float(os.getenv("SEARCH_WEIGHT_VECTOR", "0.35"))
SEARCH_WEIGHT_KEYWORD = float(os.getenv("SEARCH_WEIGHT_KEYWORD", "0.35"))
SEARCH_WEIGHT_TAG = float(os.getenv("SEARCH_WEIGHT_TAG", "0.15"))
SEARCH_WEIGHT_IMPORTANCE = float(os.getenv("SEARCH_WEIGHT_IMPORTANCE", "0.1"))
SEARCH_WEIGHT_CONFIDENCE = float(os.getenv("SEARCH_WEIGHT_CONFIDENCE", "0.05"))
SEARCH_WEIGHT_RECENCY = float(os.getenv("SEARCH_WEIGHT_RECENCY", "0.1"))
SEARCH_WEIGHT_EXACT = float(os.getenv("SEARCH_WEIGHT_EXACT", "0.15"))

API_TOKEN = os.getenv("AUTOMEM_API_TOKEN")
ADMIN_TOKEN = os.getenv("ADMIN_API_TOKEN")


def _normalize_tag_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        if not raw.strip():
            return []
        return [part.strip() for part in raw.split(",") if part.strip()]
    if isinstance(raw, (list, tuple, set)):
        tags: List[str] = []
        for item in raw:
            if isinstance(item, str) and item.strip():
                tags.append(item.strip())
        return tags
    return []


def _expand_tag_prefixes(tag: str) -> List[str]:
    """Expand a tag into all prefixes using ':' as the canonical delimiter."""
    parts = re.split(r"[:/]", tag)
    prefixes: List[str] = []
    accumulator: List[str] = []
    for part in parts:
        if not part:
            continue
        accumulator.append(part)
        prefixes.append(":".join(accumulator))
    return prefixes


def _compute_tag_prefixes(tags: List[str]) -> List[str]:
    """Compute unique, lowercased tag prefixes for fast prefix filtering."""
    seen: Set[str] = set()

try:
    from automem.utils.text import (
        SEARCH_STOPWORDS as _AM_SEARCH_STOPWORDS,
        ENTITY_STOPWORDS as _AM_ENTITY_STOPWORDS,
        ENTITY_BLOCKLIST as _AM_ENTITY_BLOCKLIST,
        _extract_keywords as _AM_extract_keywords,
    )
    # Override local constants if package is available
    SEARCH_STOPWORDS = _AM_SEARCH_STOPWORDS
    ENTITY_STOPWORDS = _AM_ENTITY_STOPWORDS
    ENTITY_BLOCKLIST = _AM_ENTITY_BLOCKLIST
    _extract_keywords = _AM_extract_keywords
except Exception:
    # Define local fallback for keyword extraction
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
            if cleaned in SEARCH_STOPWORDS:
                continue
            if cleaned in seen:
                continue
            seen.add(cleaned)
            keywords.append(cleaned)
        return keywords


# Local tag helpers (keep in-app for compatibility)
def _normalize_tag_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        if not raw.strip():
            return []
        return [part.strip() for part in raw.split(",") if part.strip()]
    if isinstance(raw, (list, tuple, set)):
        tags: List[str] = []
        for item in raw:
            if isinstance(item, str) and item.strip():
                tags.append(item.strip())
        return tags
    return []


def _expand_tag_prefixes(tag: str) -> List[str]:
    parts = re.split(r"[:/]", tag)
    prefixes: List[str] = []
    accumulator: List[str] = []
    for part in parts:
        if not part:
            continue
        accumulator.append(part)
        prefixes.append(":".join(accumulator))
    return prefixes


def _compute_tag_prefixes(tags: List[str]) -> List[str]:
    """Compute unique, lowercased tag prefixes for fast prefix filtering."""
    seen: Set[str] = set()
    prefixes: List[str] = []
    for tag in tags or []:
        normalized = (tag or "").strip().lower()
        if not normalized:
            continue
        for prefix in _expand_tag_prefixes(normalized):
            if prefix not in seen:
                seen.add(prefix)
                prefixes.append(prefix)
    return prefixes


def _prepare_tag_filters(tag_filters: Optional[List[str]]) -> List[str]:
    return [
        tag.strip().lower()
        for tag in (tag_filters or [])
        if isinstance(tag, str) and tag.strip()
    ]


# Local time helpers (fallback if package not available)
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    candidate = value.strip()
    if not candidate:
        return None
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(candidate)
    except ValueError:
        return None


def _normalize_timestamp(raw: Any) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Timestamp must be a non-empty ISO formatted string")
    candidate = raw.strip()
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError("Invalid ISO timestamp") from exc
    return parsed.astimezone(timezone.utc).isoformat()


def _parse_time_expression(expression: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not expression:
        return None, None
    expr = expression.strip().lower()
    if not expr:
        return None, None
    now = datetime.now(timezone.utc)
    def start_of_day(dt: datetime) -> datetime:
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    def end_of_day(dt: datetime) -> datetime:
        return start_of_day(dt) + timedelta(days=1)
    if expr in {"today", "this day"}:
        start = start_of_day(now)
        end = end_of_day(now)
    elif expr in {"yesterday"}:
        start = start_of_day(now - timedelta(days=1))
        end = start + timedelta(days=1)
    elif expr in {"last 24 hours", "past 24 hours"}:
        end = now
        start = now - timedelta(hours=24)
    elif expr in {"last 48 hours", "past 48 hours"}:
        end = now
        start = now - timedelta(hours=48)
    elif expr in {"this week"}:
        start = start_of_day(now - timedelta(days=now.weekday()))
        end = start + timedelta(days=7)
    elif expr in {"last week", "past week"}:
        end = start_of_day(now - timedelta(days=now.weekday()))
        start = end - timedelta(days=7)
    elif expr in {"this month"}:
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if start.month == 12:
            end = start.replace(year=start.year + 1, month=1)
        else:
            end = start.replace(month=start.month + 1)
    elif expr in {"last month", "past month"}:
        current_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if current_month_start.month == 1:
            previous_month_start = current_month_start.replace(year=current_month_start.year - 1, month=12)
        else:
            previous_month_start = current_month_start.replace(month=current_month_start.month - 1)
        start = previous_month_start
        end = current_month_start
    elif expr.startswith("last ") and expr.endswith(" days"):
        try:
            days = int(expr.split()[1])
            end = now
            start = now - timedelta(days=days)
        except ValueError:
            return None, None
    elif expr in {"last year", "past year", "this year"}:
        start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        if expr.startswith("last") or expr.startswith("past"):
            end = start
            start = start.replace(year=start.year - 1)
        else:
            if start.year == 9999:
                end = now
            else:
                end = start.replace(year=start.year + 1)
    else:
        return None, None
    return start.isoformat(), end.isoformat()


# Local scoring/metadata helpers (fallback if package not available)
def _parse_metadata_field(value: Any) -> Any:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            decoded = json.loads(value)
            if isinstance(decoded, dict):
                return decoded
        except json.JSONDecodeError:
            return value
    return value


def _collect_metadata_terms(metadata: Dict[str, Any]) -> Set[str]:
    terms: Set[str] = set()
    def visit(item: Any) -> None:
        if isinstance(item, str):
            trimmed = item.strip()
            if not trimmed:
                return
            if len(trimmed) <= 256:
                lower = trimmed.lower()
                terms.add(lower)
                for token in re.findall(r"[a-z0-9_\-]+", lower):
                    terms.add(token)
        elif isinstance(item, (list, tuple, set)):
            for sub in item:
                visit(sub)
        elif isinstance(item, dict):
            for sub in item.values():
                visit(sub)
    visit(metadata)
    return terms


def _compute_recency_score(timestamp: Optional[str]) -> float:
    if not timestamp:
        return 0.0
    parsed = _parse_iso_datetime(timestamp)
    if not parsed:
        return 0.0
    age_days = max((datetime.now(timezone.utc) - parsed).total_seconds() / 86400.0, 0.0)
    if age_days <= 0:
        return 1.0
    return max(0.0, 1.0 - (age_days / 180.0))


def _compute_metadata_score(
    result: Dict[str, Any],
    query: str,
    tokens: List[str],
) -> Tuple[float, Dict[str, float]]:
    memory = result.get("memory", {})
    metadata = _parse_metadata_field(memory.get("metadata")) if memory else {}
    metadata_terms = _collect_metadata_terms(metadata) if isinstance(metadata, dict) else set()
    tags = memory.get("tags") or []
    tag_terms = {str(tag).lower() for tag in tags if isinstance(tag, str)}
    token_hits = 0
    for token in tokens:
        if token in tag_terms or token in metadata_terms:
            token_hits += 1
    exact_match = 0.0
    normalized_query = query.lower().strip()
    if normalized_query and normalized_query in metadata_terms:
        exact_match = 1.0
    importance = memory.get("importance")
    importance_score = float(importance) if isinstance(importance, (int, float)) else 0.0
    confidence = memory.get("confidence")
    confidence_score = float(confidence) if isinstance(confidence, (int, float)) else 0.0
    recency_score = _compute_recency_score(memory.get("timestamp"))
    tag_score = token_hits / max(len(tokens), 1) if tokens else 0.0
    vector_component = result.get("match_score", 0.0) if result.get("match_type") == "vector" else 0.0
    keyword_component = result.get("match_score", 0.0) if result.get("match_type") in {"keyword", "trending"} else 0.0
    final = (
        SEARCH_WEIGHT_VECTOR * vector_component
        + SEARCH_WEIGHT_KEYWORD * keyword_component
        + SEARCH_WEIGHT_TAG * tag_score
        + SEARCH_WEIGHT_IMPORTANCE * importance_score
        + SEARCH_WEIGHT_CONFIDENCE * confidence_score
        + SEARCH_WEIGHT_RECENCY * recency_score
        + SEARCH_WEIGHT_EXACT * exact_match
    )
    components = {
        "vector": vector_component,
        "keyword": keyword_component,
        "tag": tag_score,
        "importance": importance_score,
        "confidence": confidence_score,
        "recency": recency_score,
        "exact": exact_match,
    }
    return final, components


def _build_graph_tag_predicate(tag_mode: str, tag_match: str) -> str:
    """Construct a Cypher predicate for tag filtering with mode/match semantics."""
    normalized_mode = "all" if tag_mode == "all" else "any"
    normalized_match = "prefix" if tag_match == "prefix" else "exact"
    tags_expr = "[tag IN coalesce(m.tags, []) | toLower(tag)]"

    if normalized_match == "exact":
        if normalized_mode == "all":
            return f"ALL(req IN $tag_filters WHERE req IN {tags_expr})"
        return f"ANY(tag IN {tags_expr} WHERE tag IN $tag_filters)"

    prefixes_expr = "coalesce(m.tag_prefixes, [])"
    prefix_any = f"ANY(req IN $tag_filters WHERE req IN {prefixes_expr})"
    prefix_all = f"ALL(req IN $tag_filters WHERE req IN {prefixes_expr})"
    fallback_any = (
        f"ANY(req IN $tag_filters WHERE ANY(tag IN {tags_expr} WHERE tag STARTS WITH req))"
    )
    fallback_all = (
        f"ALL(req IN $tag_filters WHERE ANY(tag IN {tags_expr} WHERE tag STARTS WITH req))"
    )

    if normalized_mode == "all":
        return (
            f"((size({prefixes_expr}) > 0 AND {prefix_all}) "
            f"OR (size({prefixes_expr}) = 0 AND {fallback_all}))"
        )

    return (
        f"((size({prefixes_expr}) > 0 AND {prefix_any}) "
        f"OR (size({prefixes_expr}) = 0 AND {fallback_any}))"
    )








def _result_passes_filters(
    result: Dict[str, Any],
    start_time: Optional[str],
    end_time: Optional[str],
    tag_filters: Optional[List[str]] = None,
    tag_mode: str = "any",
    tag_match: str = "prefix",
) -> bool:
    memory = result.get("memory", {}) or {}
    timestamp = memory.get("timestamp")
    if start_time or end_time:
        parsed = _parse_iso_datetime(timestamp) if timestamp else None
        parsed_start = _parse_iso_datetime(start_time) if start_time else None
        parsed_end = _parse_iso_datetime(end_time) if end_time else None
        if parsed is None:
            return False
        if parsed_start and parsed < parsed_start:
            return False
        if parsed_end and parsed > parsed_end:
            return False

    if tag_filters:
        normalized_filters = _prepare_tag_filters(tag_filters)
        if normalized_filters:
            normalized_mode = "all" if tag_mode == "all" else "any"
            normalized_match = "prefix" if tag_match == "prefix" else "exact"

            tags = memory.get("tags") or []
            lowered_tags = [
                str(tag).strip().lower()
                for tag in tags
                if isinstance(tag, str) and str(tag).strip()
            ]

            if normalized_match == "exact":
                tag_set = set(lowered_tags)
                if not tag_set:
                    return False
                if normalized_mode == "all":
                    if not all(filter_tag in tag_set for filter_tag in normalized_filters):
                        return False
                else:
                    if not any(filter_tag in tag_set for filter_tag in normalized_filters):
                        return False
            else:
                prefixes = memory.get("tag_prefixes") or []
                prefix_set = {
                    str(prefix).strip().lower()
                    for prefix in prefixes
                    if isinstance(prefix, str) and str(prefix).strip()
                }

                def _tags_start_with() -> bool:
                    if not lowered_tags:
                        return False
                    if normalized_mode == "all":
                        return all(
                            any(tag.startswith(filter_tag) for tag in lowered_tags)
                            for filter_tag in normalized_filters
                        )
                    return any(
                        tag.startswith(filter_tag)
                        for filter_tag in normalized_filters
                        for tag in lowered_tags
                    )

                if prefix_set:
                    if normalized_mode == "all":
                        if not all(filter_tag in prefix_set for filter_tag in normalized_filters):
                            return False
                    else:
                        if not any(filter_tag in prefix_set for filter_tag in normalized_filters):
                            return False
                else:
                    if not _tags_start_with():
                        return False

    return True


def _format_graph_result(
    graph: Any,
    node: Any,
    score: Optional[float],
    match_type: str,
    seen_ids: set[str],
) -> Optional[Dict[str, Any]]:
    data = _serialize_node(node)
    memory_id = str(data.get("id")) if data.get("id") is not None else None
    if not memory_id or memory_id in seen_ids:
        return None

    seen_ids.add(memory_id)
    relations: List[Dict[str, Any]] = _fetch_relations(graph, memory_id)

    numeric_score = float(score) if score is not None else 0.0
    return {
        "id": memory_id,
        "score": numeric_score,
        "match_score": numeric_score,
        "match_type": match_type,
        "source": "graph",
        "memory": data,
        "relations": relations,
    }


def _graph_trending_results(
    graph: Any,
    limit: int,
    seen_ids: set[str],
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    tag_filters: Optional[List[str]] = None,
    tag_mode: str = "any",
    tag_match: str = "prefix",
) -> List[Dict[str, Any]]:
    """Return high-importance memories when no specific query is supplied."""
    try:
        where_clauses = ["coalesce(m.archived, false) = false"]
        params: Dict[str, Any] = {"limit": limit}
        if start_time:
            where_clauses.append("m.timestamp >= $start_time")
            params["start_time"] = start_time
        if end_time:
            where_clauses.append("m.timestamp <= $end_time")
            params["end_time"] = end_time
        if tag_filters:
            normalized_filters = _prepare_tag_filters(tag_filters)
            if normalized_filters:
                where_clauses.append(_build_graph_tag_predicate(tag_mode, tag_match))
                params["tag_filters"] = normalized_filters

        query = f"""
            MATCH (m:Memory)
            WHERE {' AND '.join(where_clauses)}
            RETURN m
            ORDER BY m.importance DESC, m.timestamp DESC
            LIMIT $limit
        """
        result = graph.query(query, params)
    except Exception:
        logger.exception("Failed to load trending memories")
        return []

    trending: List[Dict[str, Any]] = []
    for row in getattr(result, "result_set", []) or []:
        record = _format_graph_result(graph, row[0], None, "trending", seen_ids)
        if record is None:
            continue
        # Use importance as a pseudo-score for ordering consistency
        importance = record["memory"].get("importance")
        record["score"] = float(importance) if isinstance(importance, (int, float)) else 0.0
        record["match_score"] = record["score"]
        trending.append(record)

    return trending


def _graph_keyword_search(
    graph: Any,
    query_text: str,
    limit: int,
    seen_ids: set[str],
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    tag_filters: Optional[List[str]] = None,
    tag_mode: str = "any",
    tag_match: str = "prefix",
) -> List[Dict[str, Any]]:
    """Perform a keyword-oriented search in the graph store."""
    normalized = query_text.strip().lower()
    if not normalized or normalized == "*":
        return _graph_trending_results(
            graph,
            limit,
            seen_ids,
            start_time,
            end_time,
            tag_filters,
            tag_mode,
            tag_match,
        )

    keywords = _extract_keywords(normalized)
    phrase = normalized if len(normalized) >= 3 else ""

    try:
        base_where = ["m.content IS NOT NULL"]
        params: Dict[str, Any] = {"limit": limit}
        if start_time:
            base_where.append("m.timestamp >= $start_time")
            params["start_time"] = start_time
        if end_time:
            base_where.append("m.timestamp <= $end_time")
            params["end_time"] = end_time
        if tag_filters:
            normalized_filters = _prepare_tag_filters(tag_filters)
            if normalized_filters:
                base_where.append(_build_graph_tag_predicate(tag_mode, tag_match))
                params["tag_filters"] = normalized_filters

        where_clause = " AND ".join(base_where)

        if keywords:
            params.update({"keywords": keywords, "phrase": phrase})
            query = f"""
                MATCH (m:Memory)
                WHERE {where_clause}
                WITH m,
                     toLower(m.content) AS content,
                     [tag IN coalesce(m.tags, []) | toLower(tag)] AS tags
                UNWIND $keywords AS kw
                WITH m, content, tags, kw,
                     CASE WHEN content CONTAINS kw THEN 2 ELSE 0 END +
                     CASE WHEN any(tag IN tags WHERE tag CONTAINS kw) THEN 1 ELSE 0 END AS kw_score
                WITH m, content, tags, SUM(kw_score) AS keyword_score
                WITH m, keyword_score +
                     CASE WHEN $phrase <> '' AND content CONTAINS $phrase THEN 2 ELSE 0 END +
                     CASE WHEN $phrase <> '' AND any(tag IN tags WHERE tag CONTAINS $phrase) THEN 1 ELSE 0 END AS score
                WHERE score > 0
                RETURN m, score
                ORDER BY score DESC, m.importance DESC, m.timestamp DESC
                LIMIT $limit
            """
            result = graph.query(query, params)
        elif phrase:
            params.update({"phrase": phrase})
            query = f"""
                MATCH (m:Memory)
                WHERE {where_clause}
                WITH m,
                     toLower(m.content) AS content,
                     [tag IN coalesce(m.tags, []) | toLower(tag)] AS tags
                WITH m,
                     CASE WHEN content CONTAINS $phrase THEN 2 ELSE 0 END +
                     CASE WHEN any(tag IN tags WHERE tag CONTAINS $phrase) THEN 1 ELSE 0 END AS score
                WHERE score > 0
                RETURN m, score
                ORDER BY score DESC, m.importance DESC, m.timestamp DESC
                LIMIT $limit
            """
            result = graph.query(query, params)
        else:
            return _graph_trending_results(
                graph,
                limit,
                seen_ids,
                start_time,
                end_time,
                tag_filters,
                tag_mode,
                tag_match,
            )
    except Exception:
        logger.exception("Graph keyword search failed")
        return []

    matches: List[Dict[str, Any]] = []
    for row in getattr(result, "result_set", []) or []:
        node = row[0]
        score = row[1] if len(row) > 1 else None
        record = _format_graph_result(graph, node, score, "keyword", seen_ids)
        if record is None:
            continue
        matches.append(record)

    return matches

def _build_qdrant_tag_filter(
    tags: Optional[List[str]],
    mode: str = "any",
    match: str = "exact",
):
    """Build a Qdrant filter for tag constraints, supporting mode/match semantics."""
    normalized_tags = _prepare_tag_filters(tags)
    if not normalized_tags:
        return None

    target_key = "tag_prefixes" if match == "prefix" else "tags"
    normalized_mode = "all" if mode == "all" else "any"

    if normalized_mode == "any":
        return qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key=target_key,
                    match=qdrant_models.MatchAny(any=normalized_tags),
                )
            ]
        )

    must_conditions = [
        qdrant_models.FieldCondition(
            key=target_key,
            match=qdrant_models.MatchValue(value=tag),
        )
        for tag in normalized_tags
    ]

    return qdrant_models.Filter(must=must_conditions)


def _vector_filter_only_tag_search(
    qdrant_client: Optional[QdrantClient],
    tag_filters: Optional[List[str]],
    tag_mode: str,
    tag_match: str,
    limit: int,
    seen_ids: set[str],
) -> List[Dict[str, Any]]:
    """Fallback scroll search when only tags are provided."""
    if qdrant_client is None or not tag_filters or limit <= 0:
        return []

    query_filter = _build_qdrant_tag_filter(tag_filters, tag_mode, tag_match)
    if query_filter is None:
        return []

    try:
        points, _ = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
    except Exception:
        logger.exception("Qdrant tag-only scroll failed")
        return []

    results: List[Dict[str, Any]] = []
    for point in points or []:
        memory_id = str(point.id)
        if memory_id in seen_ids:
            continue
        seen_ids.add(memory_id)

        payload = point.payload or {}
        importance = payload.get("importance")
        if isinstance(importance, (int, float)):
            score = float(importance)
        else:
            try:
                score = float(importance) if importance is not None else 0.0
            except (TypeError, ValueError):
                score = 0.0

        results.append(
            {
                "id": memory_id,
                "score": score,
                "match_score": score,
                "match_type": "tag",
                "source": "qdrant",
                "memory": payload,
                "relations": [],
            }
        )

    return results


def _vector_search(
    qdrant_client: Optional[QdrantClient],
    graph: Any,
    query_text: str,
    embedding_param: Optional[str],
    limit: int,
    seen_ids: set[str],
    tag_filters: Optional[List[str]] = None,
    tag_mode: str = "any",
    tag_match: str = "prefix",
) -> List[Dict[str, Any]]:
    """Perform vector search against Qdrant when configured."""
    if qdrant_client is None:
        return []

    normalized = (query_text or "").strip()
    if not embedding_param and normalized in {"", "*"}:
        return []

    embedding: Optional[List[float]] = None

    if embedding_param:
        try:
            embedding = _coerce_embedding(embedding_param)
        except ValueError as exc:
            abort(400, description=str(exc))
    elif normalized:
        logger.debug("Generating embedding for query: %s", normalized)
        embedding = _generate_real_embedding(normalized)

    if not embedding:
        return []

    query_filter = _build_qdrant_tag_filter(tag_filters, tag_mode, tag_match)

    try:
        vector_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=embedding,
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
        )
    except Exception:
        logger.exception("Qdrant search failed")
        return []

    matches: List[Dict[str, Any]] = []
    for hit in vector_results:
        memory_id = str(hit.id)
        if memory_id in seen_ids:
            continue

        seen_ids.add(memory_id)
        payload = hit.payload or {}
        relations = _fetch_relations(graph, memory_id) if graph is not None else []
        score = float(hit.score) if hit.score is not None else 0.0

        matches.append(
            {
                "id": memory_id,
                "score": score,
                "match_score": score,
                "match_type": "vector",
                "source": "qdrant",
                "memory": payload,
                "relations": relations,
            }
        )

    return matches

class MemoryClassifier:
    """Classifies memories into specific types based on content patterns."""

    PATTERNS = {
        "Decision": [
            r"decided to", r"chose (\w+) over", r"going with", r"picked",
            r"selected", r"will use", r"choosing", r"opted for"
        ],
        "Pattern": [
            r"usually", r"typically", r"tend to", r"pattern i noticed",
            r"often", r"frequently", r"regularly", r"consistently"
        ],
        "Preference": [
            r"prefer", r"like.*better", r"favorite", r"always use",
            r"rather than", r"instead of", r"favor"
        ],
        "Style": [
            r"wrote.*in.*style", r"communicated", r"responded to",
            r"formatted as", r"using.*tone", r"expressed as"
        ],
        "Habit": [
            r"always", r"every time", r"habitually", r"routine",
            r"daily", r"weekly", r"monthly"
        ],
        "Insight": [
            r"realized", r"discovered", r"learned that", r"understood",
            r"figured out", r"insight", r"revelation"
        ],
        "Context": [
            r"during", r"while working on", r"in the context of",
            r"when", r"at the time", r"situation was"
        ],
    }

    SYSTEM_PROMPT = """You are a memory classification system. Classify each memory into exactly ONE of these types:

- **Decision**: Choices made, selected options, what was decided
- **Pattern**: Recurring behaviors, typical approaches, consistent tendencies  
- **Preference**: Likes/dislikes, favorites, personal tastes
- **Style**: Communication approach, formatting, tone used
- **Habit**: Regular routines, repeated actions, schedules
- **Insight**: Discoveries, learnings, realizations, key findings
- **Context**: Situational background, what was happening, circumstances

Return JSON with: {"type": "<type>", "confidence": <0.0-1.0>}"""

    def classify(self, content: str, *, use_llm: bool = True) -> tuple[str, float]:
        """
        Classify memory type and return confidence score.
        Returns: (type, confidence)
        
        Args:
            content: Memory content to classify
            use_llm: If True, falls back to LLM when regex patterns don't match
        """
        content_lower = content.lower()

        # Try regex patterns first (fast, free)
        for memory_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    # Start with base confidence based on pattern match
                    confidence = 0.6

                    # Boost confidence for multiple pattern matches
                    matches = sum(1 for p in patterns if re.search(p, content_lower))
                    if matches > 1:
                        confidence = min(0.95, confidence + (matches * 0.1))

                    return memory_type, confidence

        # If no regex match and LLM enabled, use LLM classification
        if use_llm:
            try:
                result = self._classify_with_llm(content)
                if result:
                    return result
            except Exception:
                logger.exception("LLM classification failed, using fallback")

        # Default to base Memory type with lower confidence
        return "Memory", 0.3

    def _classify_with_llm(self, content: str) -> Optional[tuple[str, float]]:
        """Use OpenAI to classify memory type (fallback for complex content)."""
        # Reuse existing client if available
        if state.openai_client is None:
            init_openai()
        
        if state.openai_client is None:
            return None
        
        try:
            response = state.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": content[:1000]}  # Limit to 1000 chars
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=50
            )
            
            result = json.loads(response.choices[0].message.content)
            memory_type = result.get("type", "Memory")
            confidence = float(result.get("confidence", 0.7))
            
            # Validate type
            if memory_type not in MEMORY_TYPES:
                logger.warning("LLM returned invalid type '%s', using Context", memory_type)
                return "Context", 0.5
            
            logger.info("LLM classified as %s (confidence: %.2f)", memory_type, confidence)
            return memory_type, confidence
            
        except Exception as exc:
            logger.warning("LLM classification failed: %s", exc)
            return None


memory_classifier = MemoryClassifier()


_SPACY_NLP = None
_SPACY_INIT_LOCK = Lock()


def _get_spacy_nlp():  # type: ignore[return-type]
    global _SPACY_NLP
    if spacy is None:
        return None

    with _SPACY_INIT_LOCK:
        if _SPACY_NLP is not None:
            return _SPACY_NLP

        try:
            _SPACY_NLP = spacy.load(ENRICHMENT_SPACY_MODEL)
            logger.info("Loaded spaCy model '%s' for enrichment", ENRICHMENT_SPACY_MODEL)
        except Exception:  # pragma: no cover - optional dependency
            logger.warning("Failed to load spaCy model '%s'", ENRICHMENT_SPACY_MODEL)
            _SPACY_NLP = None

        return _SPACY_NLP


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower())
    return cleaned.strip("-")


def _is_valid_entity(value: str, *, allow_lower: bool = False, max_words: Optional[int] = None) -> bool:
    if not value:
        return False

    cleaned = value.strip()
    if len(cleaned) < 3:
        return False

    words = cleaned.split()
    if max_words is not None and len(words) > max_words:
        return False

    lowered = cleaned.lower()
    if lowered in SEARCH_STOPWORDS or lowered in ENTITY_STOPWORDS:
        return False
    
    # Reject error codes and technical noise
    if lowered in ENTITY_BLOCKLIST:
        return False

    if not any(ch.isalpha() for ch in cleaned):
        return False

    if not allow_lower and cleaned[0].islower() and not cleaned.isupper():
        return False
    
    # Reject strings starting with markdown/formatting or code characters
    if cleaned[0] in {'-', '*', '#', '>', '|', '[', ']', '{', '}', '(', ')', '_', "'", '"'}:
        return False
    
    # Reject common code artifacts (suffixes that indicate class names)
    code_suffixes = ('Adapter', 'Handler', 'Manager', 'Service', 'Controller', 
                     'Provider', 'Factory', 'Builder', 'Helper', 'Util')
    if any(cleaned.endswith(suffix) for suffix in code_suffixes):
        return False
    
    # Reject boolean/null literals and common JSON noise
    if lowered in {'true', 'false', 'null', 'none', 'undefined'}:
        return False
    
    # Reject environment variables (all caps with underscores) and text fragments ending with colons
    if ('_' in cleaned and cleaned.isupper()) or cleaned.endswith(':'):
        return False

    return True


def generate_summary(content: str, fallback: Optional[str] = None, *, max_length: int = 240) -> Optional[str]:
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
    for match in re.findall(r"(?:with|met with|meeting with|talked to|spoke with)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", text):
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

    for match in re.findall(r'`([^`]+)`', text):
        cleaned = match.strip()
        if _is_valid_entity(cleaned, allow_lower=False, max_words=4):
            result["projects"].add(cleaned)

    # Extract project names from "project called/named 'X'" pattern
    for match in re.findall(r'(?:project|repo|repository)\s+(?:called|named)\s+"([^"]+)"', text, re.IGNORECASE):
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


@dataclass
class EnrichmentStats:
    processed_total: int = 0
    successes: int = 0
    failures: int = 0
    last_success_id: Optional[str] = None
    last_success_at: Optional[str] = None
    last_error: Optional[str] = None
    last_error_at: Optional[str] = None

    def record_success(self, memory_id: str) -> None:
        self.processed_total += 1
        self.successes += 1
        self.last_success_id = memory_id
        self.last_success_at = utc_now()

    def record_failure(self, error: str) -> None:
        self.processed_total += 1
        self.failures += 1
        self.last_error = error
        self.last_error_at = utc_now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "processed_total": self.processed_total,
            "successes": self.successes,
            "failures": self.failures,
            "last_success_id": self.last_success_id,
            "last_success_at": self.last_success_at,
            "last_error": self.last_error,
            "last_error_at": self.last_error_at,
        }


@dataclass
class EnrichmentJob:
    memory_id: str
    attempt: int = 0
    forced: bool = False


@dataclass
class ServiceState:
    falkordb: Optional[FalkorDB] = None
    memory_graph: Any = None
    qdrant: Optional[QdrantClient] = None
    openai_client: Optional[OpenAI] = None  # Keep for backward compatibility (e.g., memory type classification)
    embedding_provider: Optional[EmbeddingProvider] = None  # New provider pattern for embeddings
    enrichment_queue: Optional[Queue] = None
    enrichment_thread: Optional[Thread] = None
    enrichment_stats: EnrichmentStats = field(default_factory=EnrichmentStats)
    enrichment_inflight: Set[str] = field(default_factory=set)
    enrichment_pending: Set[str] = field(default_factory=set)
    enrichment_lock: Lock = field(default_factory=Lock)
    consolidation_thread: Optional[Thread] = None
    consolidation_stop_event: Optional[Event] = None
    # Async embedding generation
    embedding_queue: Optional[Queue] = None
    embedding_thread: Optional[Thread] = None
    embedding_inflight: Set[str] = field(default_factory=set)
    embedding_pending: Set[str] = field(default_factory=set)
    embedding_lock: Lock = field(default_factory=Lock)


state = ServiceState()


def _extract_api_token() -> Optional[str]:
    if not API_TOKEN:
        return None

    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()

    api_key_header = request.headers.get("X-API-Key")
    if api_key_header:
        return api_key_header.strip()

    api_key_param = request.args.get("api_key")
    if api_key_param:
        return api_key_param.strip()

    return None


def _require_admin_token() -> None:
    if not ADMIN_TOKEN:
        abort(403, description="Admin token not configured")

    provided = (
        request.headers.get("X-Admin-Token")
        or request.headers.get("X-Admin-Api-Key")
        or request.args.get("admin_token")
    )

    if provided != ADMIN_TOKEN:
        abort(401, description="Admin authorization required")


@app.before_request
def require_api_token() -> None:
    if not API_TOKEN:
        return

    if request.endpoint in {None, 'health'}:
        return

    token = _extract_api_token()
    if token != API_TOKEN:
        abort(401, description="Unauthorized")


def init_openai() -> None:
    """Initialize OpenAI client for memory type classification (not embeddings)."""
    if state.openai_client is not None:
        return

    # Check if OpenAI is available at all
    if OpenAI is None:
        logger.info("OpenAI package not installed (used for memory type classification)")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.info("OpenAI API key not provided (used for memory type classification)")
        return

    try:
        state.openai_client = OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized for memory type classification")
    except Exception:
        logger.exception("Failed to initialize OpenAI client")
        state.openai_client = None


def init_embedding_provider() -> None:
    """Initialize embedding provider with auto-selection fallback.

    Priority order:
    1. OpenAI API (if OPENAI_API_KEY is set)
    2. Local fastembed model (no API key needed)
    3. Placeholder hash-based embeddings (fallback)

    Can be controlled via EMBEDDING_PROVIDER env var:
    - "auto" (default): Try OpenAI, then fastembed, then placeholder
    - "openai": Use OpenAI only, fail if unavailable
    - "local": Use fastembed only, fail if unavailable
    - "placeholder": Use placeholder embeddings
    """
    if state.embedding_provider is not None:
        return

    provider_config = (os.getenv("EMBEDDING_PROVIDER", "auto") or "auto").strip().lower()
    vector_size = VECTOR_SIZE

    # Explicit provider selection
    if provider_config == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("EMBEDDING_PROVIDER=openai but OPENAI_API_KEY not set")
        try:
            from automem.embedding.openai import OpenAIEmbeddingProvider
            state.embedding_provider = OpenAIEmbeddingProvider(
                api_key=api_key,
                model="text-embedding-3-small",
                dimension=vector_size
            )
            logger.info("Embedding provider: %s", state.embedding_provider.provider_name())
            return
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI provider: {e}") from e

    elif provider_config == "local":
        try:
            from automem.embedding.fastembed import FastEmbedProvider
            state.embedding_provider = FastEmbedProvider(dimension=vector_size)
            logger.info("Embedding provider: %s", state.embedding_provider.provider_name())
            return
        except Exception as e:
            raise RuntimeError(f"Failed to initialize local fastembed provider: {e}") from e

    elif provider_config == "placeholder":
        from automem.embedding.placeholder import PlaceholderEmbeddingProvider
        state.embedding_provider = PlaceholderEmbeddingProvider(dimension=vector_size)
        logger.info("Embedding provider: %s", state.embedding_provider.provider_name())
        return

    # Auto-selection: Try OpenAI → fastembed → placeholder
    if provider_config == "auto":
        # Try OpenAI first
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                from automem.embedding.openai import OpenAIEmbeddingProvider
                state.embedding_provider = OpenAIEmbeddingProvider(
                    api_key=api_key,
                    model="text-embedding-3-small",
                    dimension=vector_size
                )
                logger.info("Embedding provider (auto-selected): %s", state.embedding_provider.provider_name())
                return
            except Exception as e:
                logger.warning("Failed to initialize OpenAI provider, trying local model: %s", str(e))

        # Try local fastembed
        try:
            from automem.embedding.fastembed import FastEmbedProvider
            state.embedding_provider = FastEmbedProvider(dimension=vector_size)
            logger.info("Embedding provider (auto-selected): %s", state.embedding_provider.provider_name())
            return
        except Exception as e:
            logger.warning("Failed to initialize fastembed provider, using placeholder: %s", str(e))

        # Fallback to placeholder
        from automem.embedding.placeholder import PlaceholderEmbeddingProvider
        state.embedding_provider = PlaceholderEmbeddingProvider(dimension=vector_size)
        logger.warning(
            "Using placeholder embeddings (no semantic search). "
            "Install fastembed or set OPENAI_API_KEY for semantic embeddings."
        )
        logger.info("Embedding provider (auto-selected): %s", state.embedding_provider.provider_name())
        return

    # Invalid config
    raise ValueError(
        f"Invalid EMBEDDING_PROVIDER={provider_config}. "
        f"Valid options: auto, openai, local, placeholder"
    )


def init_falkordb() -> None:
    """Initialize FalkorDB connection if not already connected."""
    if state.memory_graph is not None:
        return

    host = (
        os.getenv("FALKORDB_HOST")
        or os.getenv("RAILWAY_PRIVATE_DOMAIN")
        or os.getenv("RAILWAY_PUBLIC_DOMAIN")
        or "localhost"
    )
    password = os.getenv("FALKORDB_PASSWORD")

    try:
        logger.info("Connecting to FalkorDB at %s:%s", host, FALKORDB_PORT)
        state.falkordb = FalkorDB(
            host=host,
            port=FALKORDB_PORT,
            password=password,
            username="default" if password else None
        )
        state.memory_graph = state.falkordb.select_graph(GRAPH_NAME)
        logger.info("FalkorDB connection established")
    except Exception:  # pragma: no cover - log full stack trace in production
        logger.exception("Failed to initialize FalkorDB connection")
        state.falkordb = None
        state.memory_graph = None


def init_qdrant() -> None:
    """Initialize Qdrant connection and ensure the collection exists."""
    if state.qdrant is not None:
        return

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url:
        logger.info("Qdrant URL not provided; skipping client initialization")
        return

    try:
        logger.info("Connecting to Qdrant at %s", url)
        state.qdrant = QdrantClient(url=url, api_key=api_key)
        _ensure_qdrant_collection()
        logger.info("Qdrant connection established")
    except Exception:  # pragma: no cover - log full stack trace in production
        logger.exception("Failed to initialize Qdrant client")
        state.qdrant = None


def _ensure_qdrant_collection() -> None:
    """Create the Qdrant collection if it does not already exist."""
    if state.qdrant is None:
        return

    try:
        collections = state.qdrant.get_collections()
        existing = {collection.name for collection in collections.collections}
        if COLLECTION_NAME not in existing:
            logger.info("Creating Qdrant collection '%s'", COLLECTION_NAME)
            state.qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
        
        # Ensure payload indexes exist for tag filtering
        logger.info("Ensuring Qdrant payload indexes for collection '%s'", COLLECTION_NAME)
        if PayloadSchemaType:
            # Use enum if available
            state.qdrant.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="tags",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            state.qdrant.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="tag_prefixes",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        else:
            # Fallback to string values when enum not available
            state.qdrant.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="tags",
                field_schema="keyword",
            )
            state.qdrant.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="tag_prefixes",
                field_schema="keyword",
            )
    except Exception:  # pragma: no cover - log full stack trace in production
        logger.exception("Failed to ensure Qdrant collection; disabling client")
        state.qdrant = None


def get_memory_graph() -> Any:
    init_falkordb()
    return state.memory_graph


def get_qdrant_client() -> Optional[QdrantClient]:
    init_qdrant()
    return state.qdrant


def init_enrichment_pipeline() -> None:
    """Initialize the background enrichment pipeline."""
    if state.enrichment_queue is not None:
        return

    state.enrichment_queue = Queue()
    state.enrichment_thread = Thread(target=enrichment_worker, daemon=True)
    state.enrichment_thread.start()
    logger.info("Enrichment pipeline initialized")


def enqueue_enrichment(memory_id: str, *, forced: bool = False, attempt: int = 0) -> None:
    if not memory_id or state.enrichment_queue is None:
        return

    job = EnrichmentJob(memory_id=memory_id, attempt=attempt, forced=forced)

    with state.enrichment_lock:
        if not forced and (
            memory_id in state.enrichment_pending or memory_id in state.enrichment_inflight
        ):
            return

        state.enrichment_pending.add(memory_id)
        state.enrichment_queue.put(job)


def _load_control_record(graph: Any) -> Dict[str, Any]:
    """Fetch or create the consolidation control node."""
    try:
        result = graph.query(
            f"""
            MERGE (c:{CONSOLIDATION_CONTROL_LABEL} {{id: $id}})
            RETURN c
            """,
            {"id": CONSOLIDATION_CONTROL_NODE_ID},
        )
    except Exception:
        logger.exception("Failed to load consolidation control record")
        return {}

    if not getattr(result, "result_set", None):
        return {}

    node = result.result_set[0][0]
    properties = getattr(node, "properties", None)
    if isinstance(properties, dict):
        return dict(properties)
    if isinstance(node, dict):
        return dict(node)
    return {}


def _load_recent_runs(graph: Any, limit: int) -> List[Dict[str, Any]]:
    """Return recent consolidation run records."""
    try:
        result = graph.query(
            f"""
            MATCH (r:{CONSOLIDATION_RUN_LABEL})
            RETURN r
            ORDER BY r.started_at DESC
            LIMIT $limit
            """,
            {"limit": limit},
        )
    except Exception:
        logger.exception("Failed to load consolidation history")
        return []

    runs: List[Dict[str, Any]] = []
    for row in getattr(result, "result_set", []) or []:
        node = row[0]
        properties = getattr(node, "properties", None)
        if isinstance(properties, dict):
            runs.append(dict(properties))
        elif isinstance(node, dict):
            runs.append(dict(node))
    return runs


def _apply_scheduler_overrides(scheduler: ConsolidationScheduler) -> None:
    """Override default scheduler intervals using configuration."""
    overrides = {
        'decay': timedelta(seconds=CONSOLIDATION_DECAY_INTERVAL_SECONDS),
        'creative': timedelta(seconds=CONSOLIDATION_CREATIVE_INTERVAL_SECONDS),
        'cluster': timedelta(seconds=CONSOLIDATION_CLUSTER_INTERVAL_SECONDS),
        'forget': timedelta(seconds=CONSOLIDATION_FORGET_INTERVAL_SECONDS),
    }

    for task, interval in overrides.items():
        if task in scheduler.schedules:
            scheduler.schedules[task]['interval'] = interval


def _tasks_for_mode(mode: str) -> List[str]:
    """Map a consolidation mode to its task identifiers."""
    if mode == 'full':
        return ['decay', 'creative', 'cluster', 'forget', 'full']
    if mode in CONSOLIDATION_TASK_FIELDS:
        return [mode]
    return [mode]


def _persist_consolidation_run(graph: Any, result: Dict[str, Any]) -> None:
    """Record consolidation outcomes and update scheduler metadata."""
    mode = result.get('mode', 'unknown')
    completed_at = result.get('completed_at') or utc_now()
    started_at = result.get('started_at') or completed_at
    success = bool(result.get('success'))
    dry_run = bool(result.get('dry_run'))

    try:
        graph.query(
            f"""
            CREATE (r:{CONSOLIDATION_RUN_LABEL} {{
                id: $id,
                mode: $mode,
                task: $task,
                success: $success,
                dry_run: $dry_run,
                started_at: $started_at,
                completed_at: $completed_at,
                result: $result
            }})
            """,
            {
                "id": uuid.uuid4().hex,
                "mode": mode,
                "task": mode,
                "success": success,
                "dry_run": dry_run,
                "started_at": started_at,
                "completed_at": completed_at,
                "result": json.dumps(result, default=str),
            },
        )
    except Exception:
        logger.exception("Failed to record consolidation run history")

    for task in _tasks_for_mode(mode):
        field = CONSOLIDATION_TASK_FIELDS.get(task)
        if not field:
            continue
        try:
            graph.query(
                f"""
                MERGE (c:{CONSOLIDATION_CONTROL_LABEL} {{id: $id}})
                SET c.{field} = $timestamp
                """,
                {
                    "id": CONSOLIDATION_CONTROL_NODE_ID,
                    "timestamp": completed_at,
                },
            )
        except Exception:
            logger.exception("Failed to update consolidation control for task %s", task)

    try:
        graph.query(
            f"""
            MATCH (r:{CONSOLIDATION_RUN_LABEL})
            WITH r ORDER BY r.started_at DESC
            SKIP $keep
            DELETE r
            """,
            {"keep": CONSOLIDATION_HISTORY_LIMIT},
        )
    except Exception:
        logger.exception("Failed to prune consolidation history")


def _build_scheduler_from_graph(graph: Any) -> Optional[ConsolidationScheduler]:
    vector_store = get_qdrant_client()
    consolidator = MemoryConsolidator(graph, vector_store)
    scheduler = ConsolidationScheduler(consolidator)
    _apply_scheduler_overrides(scheduler)

    control = _load_control_record(graph)
    for task, field in CONSOLIDATION_TASK_FIELDS.items():
        iso_value = control.get(field)
        last_run = _parse_iso_datetime(iso_value)
        if last_run and task in scheduler.schedules:
            scheduler.schedules[task]['last_run'] = last_run

    return scheduler


def _run_consolidation_tick() -> None:
    graph = get_memory_graph()
    if graph is None:
        return

    scheduler = _build_scheduler_from_graph(graph)
    if scheduler is None:
        return

    try:
        results = scheduler.run_scheduled_tasks(
            decay_threshold=CONSOLIDATION_DECAY_IMPORTANCE_THRESHOLD
        )
        for result in results:
            _persist_consolidation_run(graph, result)
    except Exception:
        logger.exception("Consolidation scheduler tick failed")


def consolidation_worker() -> None:
    """Background loop that triggers consolidation tasks."""
    logger.info("Consolidation scheduler thread started")
    while state.consolidation_stop_event and not state.consolidation_stop_event.wait(CONSOLIDATION_TICK_SECONDS):
        _run_consolidation_tick()


def init_consolidation_scheduler() -> None:
    """Ensure the background consolidation scheduler is running."""
    if state.consolidation_thread and state.consolidation_thread.is_alive():
        return

    stop_event = Event()
    state.consolidation_stop_event = stop_event
    state.consolidation_thread = Thread(
        target=consolidation_worker,
        daemon=True,
        name="consolidation-scheduler",
    )
    state.consolidation_thread.start()
    # Kick off an initial tick so schedules are populated quickly.
    _run_consolidation_tick()
    logger.info("Consolidation scheduler initialized")

def enrichment_worker() -> None:
    """Background worker that processes memories for enrichment."""
    while True:
        try:
            if state.enrichment_queue is None:
                time.sleep(ENRICHMENT_IDLE_SLEEP_SECONDS)
                continue

            try:
                job: EnrichmentJob = state.enrichment_queue.get(timeout=ENRICHMENT_IDLE_SLEEP_SECONDS)
            except Empty:
                continue

            with state.enrichment_lock:
                state.enrichment_pending.discard(job.memory_id)
                state.enrichment_inflight.add(job.memory_id)

            try:
                processed = enrich_memory(job.memory_id, forced=job.forced)
                state.enrichment_stats.record_success(job.memory_id)
                if not processed:
                    logger.debug("Enrichment skipped for %s (already processed)", job.memory_id)
            except Exception as exc:  # pragma: no cover - background thread
                state.enrichment_stats.record_failure(str(exc))
                logger.exception("Failed to enrich memory %s", job.memory_id)
                if job.attempt + 1 < ENRICHMENT_MAX_ATTEMPTS:
                    time.sleep(ENRICHMENT_FAILURE_BACKOFF_SECONDS)
                    enqueue_enrichment(job.memory_id, forced=job.forced, attempt=job.attempt + 1)
                else:
                    logger.error(
                        "Giving up on enrichment for %s after %s attempts",
                        job.memory_id,
                        job.attempt + 1,
                    )
            finally:
                with state.enrichment_lock:
                    state.enrichment_inflight.discard(job.memory_id)
                state.enrichment_queue.task_done()
        except Exception:  # pragma: no cover - defensive catch-all
            logger.exception("Error in enrichment worker loop")
            time.sleep(ENRICHMENT_FAILURE_BACKOFF_SECONDS)


def init_embedding_pipeline() -> None:
    """Initialize the background embedding generation pipeline."""
    if state.embedding_queue is not None:
        return

    state.embedding_queue = Queue()
    state.embedding_thread = Thread(target=embedding_worker, daemon=True)
    state.embedding_thread.start()
    logger.info("Embedding pipeline initialized")


def enqueue_embedding(memory_id: str, content: str) -> None:
    """Queue a memory for async embedding generation."""
    if not memory_id or not content or state.embedding_queue is None:
        return

    with state.embedding_lock:
        if memory_id in state.embedding_pending or memory_id in state.embedding_inflight:
            return
        
        state.embedding_pending.add(memory_id)
        state.embedding_queue.put((memory_id, content))


def embedding_worker() -> None:
    """Background worker that generates embeddings and stores them in Qdrant with batching."""
    batch: List[Tuple[str, str]] = []  # List of (memory_id, content) tuples
    batch_deadline = time.time() + EMBEDDING_BATCH_TIMEOUT_SECONDS
    
    while True:
        try:
            if state.embedding_queue is None:
                time.sleep(1)
                continue

            # Calculate remaining time until batch deadline
            timeout = max(0.1, batch_deadline - time.time())
            
            try:
                memory_id, content = state.embedding_queue.get(timeout=timeout)
                batch.append((memory_id, content))
                
                # Process batch if full
                if len(batch) >= EMBEDDING_BATCH_SIZE:
                    _process_embedding_batch(batch)
                    batch = []
                    batch_deadline = time.time() + EMBEDDING_BATCH_TIMEOUT_SECONDS
                    
            except Empty:
                # Timeout reached - process whatever we have
                if batch:
                    _process_embedding_batch(batch)
                    batch = []
                batch_deadline = time.time() + EMBEDDING_BATCH_TIMEOUT_SECONDS
                continue
                
        except Exception:  # pragma: no cover - defensive catch-all
            logger.exception("Error in embedding worker loop")
            # Process any pending batch before sleeping
            if batch:
                try:
                    _process_embedding_batch(batch)
                except Exception:
                    logger.exception("Failed to process batch during error recovery")
                batch = []
            time.sleep(1)
            batch_deadline = time.time() + EMBEDDING_BATCH_TIMEOUT_SECONDS


def _process_embedding_batch(batch: List[Tuple[str, str]]) -> None:
    """Process a batch of embeddings efficiently."""
    if not batch:
        return
    
    memory_ids = [item[0] for item in batch]
    contents = [item[1] for item in batch]
    
    # Mark all as inflight
    with state.embedding_lock:
        for memory_id in memory_ids:
            state.embedding_pending.discard(memory_id)
            state.embedding_inflight.add(memory_id)
    
    try:
        # Generate embeddings in batch
        embeddings = _generate_real_embeddings_batch(contents)
        
        # Store each embedding individually (Qdrant operations are fast)
        for memory_id, content, embedding in zip(memory_ids, contents, embeddings):
            try:
                _store_embedding_in_qdrant(memory_id, content, embedding)
                logger.debug("Generated and stored embedding for %s", memory_id)
            except Exception:  # pragma: no cover
                logger.exception("Failed to store embedding for %s", memory_id)
    except Exception:  # pragma: no cover
        logger.exception("Failed to generate batch embeddings")
    finally:
        # Mark all as complete
        with state.embedding_lock:
            for memory_id in memory_ids:
                state.embedding_inflight.discard(memory_id)
        
        # Mark all queue items as done
        for _ in batch:
            state.embedding_queue.task_done()


def _store_embedding_in_qdrant(memory_id: str, content: str, embedding: List[float]) -> None:
    """Store a pre-generated embedding in Qdrant with memory metadata."""
    qdrant_client = get_qdrant_client()
    if qdrant_client is None:
        return
    
    graph = get_memory_graph()
    if graph is None:
        return
    
    # Fetch latest memory data from FalkorDB for payload
    result = graph.query("MATCH (m:Memory {id: $id}) RETURN m", {"id": memory_id})
    if not getattr(result, "result_set", None):
        logger.warning("Memory %s not found in FalkorDB, skipping Qdrant update", memory_id)
        return
    
    node = result.result_set[0][0]
    properties = getattr(node, "properties", {})
    
    # Store in Qdrant
    try:
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=memory_id,
                    vector=embedding,
                    payload={
                        "content": properties.get("content", content),
                        "tags": properties.get("tags", []),
                        "tag_prefixes": properties.get("tag_prefixes", []),
                        "importance": properties.get("importance", 0.5),
                        "timestamp": properties.get("timestamp", utc_now()),
                        "type": properties.get("type", "Context"),
                        "confidence": properties.get("confidence", 0.5),
                        "updated_at": properties.get("updated_at", utc_now()),
                        "last_accessed": properties.get("last_accessed", utc_now()),
                        "metadata": json.loads(properties.get("metadata", "{}")),
                    },
                )
            ],
        )
        logger.info("Stored embedding for %s in Qdrant", memory_id)
    except Exception:  # pragma: no cover - log full stack trace
        logger.exception("Qdrant upsert failed for %s", memory_id)


def generate_and_store_embedding(memory_id: str, content: str) -> None:
    """Generate embedding for content and store in Qdrant (legacy single-item API)."""
    embedding = _generate_real_embedding(content)
    _store_embedding_in_qdrant(memory_id, content, embedding)


def enrich_memory(memory_id: str, *, forced: bool = False) -> bool:
    """Enrich a memory with relationships, patterns, and entity extraction."""
    graph = get_memory_graph()
    if graph is None:
        raise RuntimeError("FalkorDB unavailable for enrichment")

    result = graph.query(
        "MATCH (m:Memory {id: $id}) RETURN m",
        {"id": memory_id}
    )

    if not result.result_set:
        logger.debug("Skipping enrichment for %s; memory not found", memory_id)
        return False

    node = result.result_set[0][0]
    properties = getattr(node, "properties", None)
    if not isinstance(properties, dict):
        properties = dict(getattr(node, "__dict__", {}))

    metadata_raw = properties.get("metadata")
    metadata = _parse_metadata_field(metadata_raw) or {}
    if not isinstance(metadata, dict):
        metadata = {"_raw_metadata": metadata}

    already_processed = bool(properties.get("processed"))
    if already_processed and not forced:
        return False

    content = properties.get("content", "") or ""
    entities = extract_entities(content)

    tags = list(dict.fromkeys(_normalize_tag_list(properties.get("tags"))))
    entity_tags: Set[str] = set()

    if entities:
        entities_section = metadata.setdefault("entities", {})
        if not isinstance(entities_section, dict):
            entities_section = {}
        for category, values in entities.items():
            if not values:
                continue
            entities_section[category] = sorted(values)
            for value in values:
                slug = _slugify(value)
                if slug:
                    entity_tags.add(f"entity:{category}:{slug}")
        metadata["entities"] = entities_section

    if entity_tags:
        tags = list(dict.fromkeys(tags + sorted(entity_tags)))

    temporal_links = find_temporal_relationships(graph, memory_id)
    pattern_info = detect_patterns(graph, memory_id, content)
    semantic_neighbors = link_semantic_neighbors(graph, memory_id)

    if ENRICHMENT_ENABLE_SUMMARIES:
        existing_summary = properties.get("summary")
        summary = generate_summary(content, existing_summary if forced else None)
    else:
        summary = properties.get("summary")

    enrichment_meta = metadata.setdefault("enrichment", {})
    if not isinstance(enrichment_meta, dict):
        enrichment_meta = {}
    enrichment_meta.update(
        {
            "last_run": utc_now(),
            "forced": forced,
            "temporal_links": temporal_links,
            "patterns_detected": pattern_info,
            "semantic_neighbors": [
                {"id": neighbour_id, "score": score}
                for neighbour_id, score in semantic_neighbors
            ],
        }
    )
    metadata["enrichment"] = enrichment_meta

    update_payload = {
        "id": memory_id,
        "metadata": json.dumps(metadata, default=str),
        "tags": tags,
        "summary": summary,
        "enriched_at": utc_now(),
    }

    graph.query(
        """
        MATCH (m:Memory {id: $id})
        SET m.metadata = $metadata,
            m.tags = $tags,
            m.summary = $summary,
            m.enriched = true,
            m.enriched_at = $enriched_at,
            m.processed = true
        """,
        update_payload,
    )

    logger.debug(
        "Enriched memory %s (temporal=%s, patterns=%s, semantic=%s)",
        memory_id,
        temporal_links,
        pattern_info,
        len(semantic_neighbors),
    )

    return True


def find_temporal_relationships(graph: Any, memory_id: str, limit: int = 5) -> int:
    """Find and create temporal relationships with recent memories."""
    created = 0
    try:
        result = graph.query(
            """
            MATCH (m1:Memory {id: $id})
            MATCH (m2:Memory)
            WHERE m2.id <> $id
                AND m2.timestamp IS NOT NULL
                AND m1.timestamp IS NOT NULL
                AND m2.timestamp < m1.timestamp
            RETURN m2.id
            ORDER BY m2.timestamp DESC
            LIMIT $limit
            """,
            {"id": memory_id, "limit": limit}
        )

        timestamp = utc_now()
        for related_id, in result.result_set:
            if not related_id:
                continue
            graph.query(
                """
                MATCH (m1:Memory {id: $id1})
                MATCH (m2:Memory {id: $id2})
                MERGE (m1)-[r:PRECEDED_BY]->(m2)
                SET r.updated_at = $timestamp,
                    r.count = COALESCE(r.count, 0) + 1
                """,
                {"id1": memory_id, "id2": related_id, "timestamp": timestamp},
            )
            created += 1
    except Exception:
        logger.exception("Failed to find temporal relationships")

    return created


def detect_patterns(graph: Any, memory_id: str, content: str) -> List[Dict[str, Any]]:
    """Detect if this memory exemplifies or creates patterns."""
    detected: List[Dict[str, Any]] = []

    try:
        memory_type, confidence = memory_classifier.classify(content)
        result = graph.query(
            """
            MATCH (m:Memory)
            WHERE m.type = $type
                AND m.id <> $id
                AND m.confidence > 0.5
            RETURN m.id, m.content
            LIMIT 10
            """,
            {"type": memory_type, "id": memory_id}
        )

        similar_texts = [content]
        similar_texts.extend(row[1] for row in result.result_set if len(row) > 1)
        similar_count = len(result.result_set)

        if similar_count >= 3:
            tokens = Counter()
            for text in similar_texts:
                for token in re.findall(r"[a-zA-Z]{4,}", (text or "").lower()):
                    if token in SEARCH_STOPWORDS:
                        continue
                    tokens[token] += 1

            top_terms = [term for term, _ in tokens.most_common(5)]
            pattern_id = f"pattern-{memory_type}-{uuid.uuid4().hex[:8]}"
            description = (
                f"Pattern across {similar_count + 1} {memory_type} memories"
                + (f" highlighting {', '.join(top_terms)}" if top_terms else "")
            )

            graph.query(
                """
                MERGE (p:Pattern {type: $type})
                ON CREATE SET
                    p.id = $pattern_id,
                    p.content = $description,
                    p.confidence = $initial_confidence,
                    p.observations = 1,
                    p.key_terms = $key_terms,
                    p.created_at = $timestamp
                ON MATCH SET
                    p.confidence = CASE
                        WHEN p.confidence < 0.95 THEN p.confidence + 0.05
                        ELSE 0.95
                    END,
                    p.observations = p.observations + 1,
                    p.key_terms = $key_terms,
                    p.updated_at = $timestamp
                """,
                {
                    "type": memory_type,
                    "pattern_id": pattern_id,
                    "description": description,
                    "initial_confidence": 0.35,
                    "key_terms": top_terms,
                    "timestamp": utc_now(),
                },
            )

            graph.query(
                """
                MATCH (m:Memory {id: $memory_id})
                MATCH (p:Pattern {type: $type})
                MERGE (m)-[r:EXEMPLIFIES]->(p)
                SET r.confidence = $confidence,
                    r.updated_at = $timestamp
                """,
                {
                    "type": memory_type,
                    "memory_id": memory_id,
                    "confidence": confidence,
                    "timestamp": utc_now(),
                },
            )

            detected.append(
                {
                    "type": memory_type,
                    "similar_memories": similar_count,
                    "key_terms": top_terms,
                }
            )
    except Exception:
        logger.exception("Failed to detect patterns")

    return detected


def link_semantic_neighbors(graph: Any, memory_id: str) -> List[Tuple[str, float]]:
    client = get_qdrant_client()
    if client is None:
        return []

    try:
        points = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[memory_id],
            with_vectors=True,
            with_payload=False,
        )
    except Exception:
        logger.exception("Failed to fetch vector for memory %s", memory_id)
        return []

    if not points or getattr(points[0], "vector", None) is None:
        return []

    query_vector = points[0].vector

    try:
        neighbors = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=ENRICHMENT_SIMILARITY_LIMIT + 1,
            with_payload=False,
        )
    except Exception:
        logger.exception("Semantic neighbor search failed for %s", memory_id)
        return []

    created: List[Tuple[str, float]] = []
    timestamp = utc_now()

    for neighbour in neighbors:
        neighbour_id = str(neighbour.id)
        if neighbour_id == memory_id:
            continue

        score = float(neighbour.score or 0.0)
        if score < ENRICHMENT_SIMILARITY_THRESHOLD:
            continue

        params = {
            "id1": memory_id,
            "id2": neighbour_id,
            "score": score,
            "timestamp": timestamp,
        }

        graph.query(
            """
            MATCH (a:Memory {id: $id1})
            MATCH (b:Memory {id: $id2})
            MERGE (a)-[r:SIMILAR_TO]->(b)
            SET r.score = $score,
                r.updated_at = $timestamp
            """,
            params,
        )

        graph.query(
            """
            MATCH (a:Memory {id: $id1})
            MATCH (b:Memory {id: $id2})
            MERGE (b)-[r:SIMILAR_TO]->(a)
            SET r.score = $score,
                r.updated_at = $timestamp
            """,
            params,
        )

        created.append((neighbour_id, score))

    return created


@app.route("/enrichment/status", methods=["GET"])
def enrichment_status() -> Any:
    queue_size = state.enrichment_queue.qsize() if state.enrichment_queue else 0
    thread_alive = bool(state.enrichment_thread and state.enrichment_thread.is_alive())

    with state.enrichment_lock:
        pending = len(state.enrichment_pending)
        inflight = len(state.enrichment_inflight)

    response = {
        "status": "running" if thread_alive else "stopped",
        "queue_size": queue_size,
        "pending": pending,
        "inflight": inflight,
        "max_attempts": ENRICHMENT_MAX_ATTEMPTS,
        "stats": state.enrichment_stats.to_dict(),
    }

    return jsonify(response)


@app.route("/enrichment/reprocess", methods=["POST"])
def enrichment_reprocess() -> Any:
    _require_admin_token()

    payload = request.get_json(silent=True) or {}
    ids: Set[str] = set()

    raw_ids = payload.get("ids") or request.args.get("ids")
    if isinstance(raw_ids, str):
        ids.update(part.strip() for part in raw_ids.split(",") if part.strip())
    elif isinstance(raw_ids, list):
        for item in raw_ids:
            if isinstance(item, str) and item.strip():
                ids.add(item.strip())

    if not ids:
        abort(400, description="No memory ids provided for reprocessing")

    for memory_id in ids:
        enqueue_enrichment(memory_id, forced=True)

    return jsonify({
        "status": "queued",
        "count": len(ids),
        "ids": sorted(ids),
    }), 202


@app.route("/admin/reembed", methods=["POST"])
def admin_reembed() -> Any:
    """Regenerate embeddings for existing memories using OpenAI API.

    Requires admin token and OpenAI API key configured.

    Parameters:
    - batch_size: Number of memories to process at once (default 32, max 100)
    - limit: Total number of memories to reembed (default all)
    - force: Regenerate even if embedding exists (default false)
    """
    _require_admin_token()

    # Check if embedding provider is available
    init_embedding_provider()
    if state.embedding_provider is None:
        abort(503, description="Embedding provider not available")

    # Check Qdrant is available
    qdrant_client = get_qdrant_client()
    if qdrant_client is None:
        abort(503, description="Qdrant is not available - cannot store embeddings")

    # Parse parameters
    payload = request.get_json(silent=True) or {}
    batch_size = min(int(payload.get("batch_size", 32)), 100)
    limit = payload.get("limit")
    force_reembed = payload.get("force", False)

    # Get graph connection
    graph = get_memory_graph()
    if graph is None:
        abort(503, description="FalkorDB is unavailable")

    # Query memories to reembed
    if force_reembed:
        query = "MATCH (m:Memory) RETURN m.id, m.content ORDER BY m.timestamp DESC"
    else:
        # Only reembed memories that don't have real embeddings yet
        # We'll check by seeing if they have the default placeholder pattern
        query = """
            MATCH (m:Memory)
            WHERE m.content IS NOT NULL
            RETURN m.id, m.content
            ORDER BY m.timestamp DESC
        """

    if limit:
        query += f" LIMIT {int(limit)}"

    result = graph.query(query)
    memories_to_process = []

    for row in result.result_set:
        memory_id = row[0]
        content = row[1]
        if content:
            memories_to_process.append((memory_id, content))

    if not memories_to_process:
        return jsonify({
            "status": "complete",
            "message": "No memories found to reembed",
            "processed": 0,
            "total": 0
        })

    # Process in batches
    processed = 0
    failed = 0
    failed_ids = []

    for i in range(0, len(memories_to_process), batch_size):
        batch = memories_to_process[i:i + batch_size]
        points = []

        for memory_id, content in batch:
            try:
                # Generate real embedding using OpenAI
                embedding = _generate_real_embedding(content)

                # Retrieve existing metadata from Qdrant if available
                try:
                    existing = qdrant_client.retrieve(
                        collection_name=COLLECTION_NAME,
                        ids=[memory_id],
                        with_payload=True
                    )
                    if existing:
                        payload_data = existing[0].payload
                    else:
                        # Fallback: query from graph for metadata
                        meta_result = graph.query(
                            "MATCH (m:Memory {id: $id}) RETURN m",
                            {"id": memory_id}
                        )
                        if meta_result.result_set:
                            node = meta_result.result_set[0][0]
                            props = _serialize_node(node)
                            payload_data = {
                                "content": content,
                                "tags": props.get("tags", []),
                                "importance": props.get("importance", 0.5),
                                "timestamp": props.get("timestamp"),
                            "type": props.get("type", "Context"),  # Default to Context instead of Memory
                            "confidence": props.get("confidence", 0.6),
                                "updated_at": props.get("updated_at"),
                                "last_accessed": props.get("last_accessed"),
                                "metadata": props.get("metadata", {}),
                            }
                        else:
                            payload_data = {
                                "content": content,
                                "tags": [],
                                "importance": 0.5,
                                "timestamp": utc_now(),
                                "type": "Context",
                                "confidence": 0.6,
                                "metadata": {},
                            }
                except Exception as e:
                    logger.warning(f"Failed to retrieve metadata for {memory_id}: {e}")
                    # Use minimal payload
                    payload_data = {
                        "content": content,
                        "tags": [],
                        "importance": 0.5,
                        "timestamp": utc_now(),
                        "type": "Context",
                        "confidence": 0.6,
                        "metadata": {},
                    }

                points.append(
                    PointStruct(
                        id=memory_id,
                        vector=embedding,
                        payload=payload_data
                    )
                )
                processed += 1

            except Exception as e:
                logger.error(f"Failed to generate embedding for memory {memory_id}: {e}")
                failed += 1
                failed_ids.append(memory_id)

        # Batch upsert to Qdrant
        if points:
            try:
                qdrant_client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points
                )
                logger.info(f"Successfully reembedded batch of {len(points)} memories")
            except Exception as e:
                logger.error(f"Failed to upsert batch to Qdrant: {e}")
                failed += len(points)
                failed_ids.extend([p.id for p in points])
                processed -= len(points)

    response = {
        "status": "complete",
        "processed": processed,
        "failed": failed,
        "total": len(memories_to_process),
        "batch_size": batch_size,
    }

    if failed_ids:
        response["failed_ids"] = failed_ids[:10]  # Limit to first 10 for response size
        if len(failed_ids) > 10:
            response["failed_ids_truncated"] = True

    return jsonify(response)


@app.errorhandler(Exception)
def handle_exceptions(exc: Exception):
    """Return JSON responses for both HTTP and unexpected errors."""
    if isinstance(exc, HTTPException):
        response = {
            "status": "error",
            "code": exc.code,
            "message": exc.description or exc.name,
        }
        return jsonify(response), exc.code

    logger.exception("Unhandled error")
    response = {
        "status": "error",
        "code": 500,
        "message": "Internal server error",
    }
    return jsonify(response), 500


@app.route("/health", methods=["GET"])
def health() -> Any:
    graph_available = get_memory_graph() is not None
    qdrant_available = get_qdrant_client() is not None

    status = "healthy" if graph_available and qdrant_available else "degraded"
    
    # Get enrichment queue stats (non-authenticated for monitoring)
    enrichment_thread_alive = bool(state.enrichment_thread and state.enrichment_thread.is_alive())
    with state.enrichment_lock:
        enrichment_pending = len(state.enrichment_pending)
        enrichment_inflight = len(state.enrichment_inflight)
    
    health_data = {
        "status": status,
        "falkordb": "connected" if graph_available else "disconnected",
        "qdrant": "connected" if qdrant_available else "disconnected",
        "enrichment": {
            "status": "running" if enrichment_thread_alive else "stopped",
            "queue_depth": state.enrichment_queue.qsize() if state.enrichment_queue else 0,
            "pending": enrichment_pending,
            "inflight": enrichment_inflight,
            "processed": state.enrichment_stats.successes,
            "failed": state.enrichment_stats.failures,
        },
        "timestamp": utc_now(),
        "graph": GRAPH_NAME,
    }
    return jsonify(health_data)


@app.route("/memory", methods=["POST"])
def store_memory() -> Any:
    query_start = time.perf_counter()
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        abort(400, description="JSON body is required")

    content = (payload.get("content") or "").strip()
    if not content:
        abort(400, description="'content' is required")

    tags = _normalize_tags(payload.get("tags"))
    tags_lower = [t.strip().lower() for t in tags if isinstance(t, str) and t.strip()]
    tag_prefixes = _compute_tag_prefixes(tags_lower)
    importance = _coerce_importance(payload.get("importance"))
    memory_id = payload.get("id") or str(uuid.uuid4())

    metadata_raw = payload.get("metadata")
    if metadata_raw is None:
        metadata: Dict[str, Any] = {}
    elif isinstance(metadata_raw, dict):
        metadata = metadata_raw
    else:
        abort(400, description="'metadata' must be an object")
    metadata_json = json.dumps(metadata, default=str)

    # Accept explicit type/confidence or classify automatically
    memory_type = payload.get("type")
    type_confidence = payload.get("confidence")
    
    if memory_type:
        # Validate explicit type
        if memory_type not in MEMORY_TYPES:
            abort(400, description=f"Invalid memory type '{memory_type}'. Must be one of: {', '.join(sorted(MEMORY_TYPES))}")
        # Use provided confidence or default
        if type_confidence is None:
            type_confidence = 0.9  # High confidence for explicit types
        else:
            type_confidence = _coerce_importance(type_confidence)
    else:
        # Auto-classify if no type provided
        memory_type, type_confidence = memory_classifier.classify(content)

    # Handle temporal validity fields
    t_valid = payload.get("t_valid")
    t_invalid = payload.get("t_invalid")
    if t_valid:
        try:
            t_valid = _normalize_timestamp(t_valid)
        except ValueError as exc:
            abort(400, description=f"Invalid t_valid: {exc}")
    if t_invalid:
        try:
            t_invalid = _normalize_timestamp(t_invalid)
        except ValueError as exc:
            abort(400, description=f"Invalid t_invalid: {exc}")

    try:
        embedding = _coerce_embedding(payload.get("embedding"))
    except ValueError as exc:
        abort(400, description=str(exc))

    graph = get_memory_graph()
    if graph is None:
        abort(503, description="FalkorDB is unavailable")

    created_at = payload.get("timestamp")
    if created_at:
        try:
            created_at = _normalize_timestamp(created_at)
        except ValueError as exc:
            abort(400, description=str(exc))
    else:
        created_at = utc_now()

    updated_at = payload.get("updated_at")
    if updated_at:
        try:
            updated_at = _normalize_timestamp(updated_at)
        except ValueError as exc:
            abort(400, description=f"Invalid updated_at: {exc}")
    else:
        updated_at = created_at

    last_accessed = payload.get("last_accessed")
    if last_accessed:
        try:
            last_accessed = _normalize_timestamp(last_accessed)
        except ValueError as exc:
            abort(400, description=f"Invalid last_accessed: {exc}")
    else:
        last_accessed = updated_at

    try:
        graph.query(
            """
            MERGE (m:Memory {id: $id})
            SET m.content = $content,
                m.timestamp = $timestamp,
                m.importance = $importance,
                m.tags = $tags,
                m.tag_prefixes = $tag_prefixes,
                m.type = $type,
                m.confidence = $confidence,
                m.t_valid = $t_valid,
                m.t_invalid = $t_invalid,
                m.updated_at = $updated_at,
                m.last_accessed = $last_accessed,
                m.metadata = $metadata,
                m.processed = false
            RETURN m
            """,
            {
                "id": memory_id,
                "content": content,
                "timestamp": created_at,
                "importance": importance,
                "tags": tags,
                "tag_prefixes": tag_prefixes,
                "type": memory_type,
                "confidence": type_confidence,
                "t_valid": t_valid or created_at,
                "t_invalid": t_invalid,
                "updated_at": updated_at,
                "last_accessed": last_accessed,
                "metadata": metadata_json,
            },
        )
    except Exception:  # pragma: no cover - log full stack trace in production
        logger.exception("Failed to persist memory in FalkorDB")
        abort(500, description="Failed to store memory in FalkorDB")

    # Queue for enrichment
    enqueue_enrichment(memory_id)

    # Queue for async embedding generation (if no embedding provided)
    embedding_status = "skipped"
    qdrant_client = get_qdrant_client()
    
    if embedding is not None:
        # Sync path: User provided embedding, store immediately
        embedding_status = "provided"
        qdrant_result = None
        if qdrant_client is not None:
            try:
                qdrant_client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[
                        PointStruct(
                            id=memory_id,
                            vector=embedding,
                            payload={
                                "content": content,
                                "tags": tags,
                                "tag_prefixes": tag_prefixes,
                                "importance": importance,
                                "timestamp": created_at,
                                "type": memory_type,
                                "confidence": type_confidence,
                                "updated_at": updated_at,
                                "last_accessed": last_accessed,
                                "metadata": metadata,
                            },
                        )
                    ],
                )
                qdrant_result = "stored"
            except Exception:  # pragma: no cover - log full stack trace in production
                logger.exception("Qdrant upsert failed")
                qdrant_result = "failed"
    elif qdrant_client is not None:
        # Async path: Queue embedding generation
        enqueue_embedding(memory_id, content)
        embedding_status = "queued"
        qdrant_result = "queued"
    else:
        qdrant_result = "unconfigured"

    response = {
        "status": "success",
        "memory_id": memory_id,
        "stored_at": created_at,
        "type": memory_type,
        "confidence": type_confidence,
        "qdrant": qdrant_result,
        "embedding_status": embedding_status,
        "enrichment": "queued" if state.enrichment_queue else "disabled",
        "metadata": metadata,
        "timestamp": created_at,
        "updated_at": updated_at,
        "last_accessed": last_accessed,
        "query_time_ms": round((time.perf_counter() - query_start) * 1000, 2),
    }
    
    # Structured logging for performance analysis
    logger.info(
        "memory_stored",
        extra={
            "memory_id": memory_id,
            "type": memory_type,
            "importance": importance,
            "tags_count": len(tags),
            "content_length": len(content),
            "latency_ms": response["query_time_ms"],
            "embedding_status": embedding_status,
            "qdrant_status": qdrant_result,
            "enrichment_queued": bool(state.enrichment_queue),
        }
    )
    
    return jsonify(response), 201


@app.route("/memory/<memory_id>", methods=["PATCH"])
def update_memory(memory_id: str) -> Any:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        abort(400, description="JSON body is required")

    graph = get_memory_graph()
    if graph is None:
        abort(503, description="FalkorDB is unavailable")

    result = graph.query("MATCH (m:Memory {id: $id}) RETURN m", {"id": memory_id})
    if not getattr(result, "result_set", None):
        abort(404, description="Memory not found")

    current_node = result.result_set[0][0]
    current = _serialize_node(current_node)

    new_content = payload.get("content", current.get("content"))
    tags = _normalize_tag_list(payload.get("tags", current.get("tags")))
    tags_lower = [t.strip().lower() for t in tags if isinstance(t, str) and t.strip()]
    tag_prefixes = _compute_tag_prefixes(tags_lower)
    importance = payload.get("importance", current.get("importance"))
    memory_type = payload.get("type", current.get("type"))
    confidence = payload.get("confidence", current.get("confidence"))
    timestamp = payload.get("timestamp", current.get("timestamp"))
    metadata_raw = payload.get("metadata", _parse_metadata_field(current.get("metadata")))
    updated_at = payload.get("updated_at", current.get("updated_at", utc_now()))
    last_accessed = payload.get("last_accessed", current.get("last_accessed"))

    if metadata_raw is None:
        metadata: Dict[str, Any] = {}
    elif isinstance(metadata_raw, dict):
        metadata = metadata_raw
    else:
        abort(400, description="'metadata' must be an object")
    metadata_json = json.dumps(metadata, default=str)

    if timestamp:
        try:
            timestamp = _normalize_timestamp(timestamp)
        except ValueError as exc:
            abort(400, description=f"Invalid timestamp: {exc}")

    if updated_at:
        try:
            updated_at = _normalize_timestamp(updated_at)
        except ValueError as exc:
            abort(400, description=f"Invalid updated_at: {exc}")

    if last_accessed:
        try:
            last_accessed = _normalize_timestamp(last_accessed)
        except ValueError as exc:
            abort(400, description=f"Invalid last_accessed: {exc}")

    update_query = """
        MATCH (m:Memory {id: $id})
        SET m.content = $content,
            m.tags = $tags,
            m.tag_prefixes = $tag_prefixes,
            m.importance = $importance,
            m.type = $type,
            m.confidence = $confidence,
            m.timestamp = $timestamp,
            m.metadata = $metadata,
            m.updated_at = $updated_at,
            m.last_accessed = $last_accessed
        RETURN m
    """

    graph.query(
        update_query,
        {
            "id": memory_id,
            "content": new_content,
            "tags": tags,
            "tag_prefixes": tag_prefixes,
            "importance": importance,
            "type": memory_type,
            "confidence": confidence,
            "timestamp": timestamp,
            "metadata": metadata_json,
            "updated_at": updated_at,
            "last_accessed": last_accessed,
        },
    )

    qdrant_client = get_qdrant_client()
    vector = None
    if qdrant_client is not None:
        if new_content != current.get("content"):
            vector = _generate_real_embedding(new_content)
        else:
            try:
                existing = qdrant_client.retrieve(
                    collection_name=COLLECTION_NAME,
                    ids=[memory_id],
                    with_vectors=True,
                )
                if existing:
                    vector = existing[0].vector
            except Exception:
                logger.exception("Failed to retrieve existing vector; regenerating")
                vector = _generate_real_embedding(new_content)

        if vector is not None:
            payload = {
                "content": new_content,
                "tags": tags,
                "tag_prefixes": tag_prefixes,
                "importance": importance,
                "timestamp": timestamp,
                "type": memory_type,
                "confidence": confidence,
                "updated_at": updated_at,
                "last_accessed": last_accessed,
                "metadata": metadata,
            }
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=[PointStruct(id=memory_id, vector=vector, payload=payload)],
            )

    return jsonify({"status": "success", "memory_id": memory_id})


@app.route("/memory/<memory_id>", methods=["DELETE"])
def delete_memory(memory_id: str) -> Any:
    graph = get_memory_graph()
    if graph is None:
        abort(503, description="FalkorDB is unavailable")

    result = graph.query("MATCH (m:Memory {id: $id}) RETURN m", {"id": memory_id})
    if not getattr(result, "result_set", None):
        abort(404, description="Memory not found")

    graph.query("MATCH (m:Memory {id: $id}) DETACH DELETE m", {"id": memory_id})

    qdrant_client = get_qdrant_client()
    if qdrant_client is not None:
        try:
            if 'qdrant_models' in globals() and qdrant_models is not None:
                selector = qdrant_models.PointIdsList(points=[memory_id])
            else:
                selector = {"points": [memory_id]}
            qdrant_client.delete(collection_name=COLLECTION_NAME, points_selector=selector)
        except Exception:
            logger.exception("Failed to delete vector for memory %s", memory_id)

    return jsonify({"status": "success", "memory_id": memory_id})


@app.route("/memory/by-tag", methods=["GET"])
def memories_by_tag() -> Any:
    raw_tags = request.args.getlist("tags") or request.args.get("tags")
    tags = _normalize_tag_list(raw_tags)
    if not tags:
        abort(400, description="'tags' query parameter is required")

    limit = max(1, min(int(request.args.get("limit", 20)), 200))

    graph = get_memory_graph()
    if graph is None:
        abort(503, description="FalkorDB is unavailable")

    params = {
        "tags": [tag.lower() for tag in tags],
        "limit": limit,
    }

    query = """
        MATCH (m:Memory)
        WHERE ANY(tag IN coalesce(m.tags, []) WHERE toLower(tag) IN $tags)
        RETURN m
        ORDER BY m.importance DESC, m.timestamp DESC
        LIMIT $limit
    """

    try:
        result = graph.query(query, params)
    except Exception:
        logger.exception("Tag search failed")
        abort(500, description="Failed to search by tag")

    memories: List[Dict[str, Any]] = []
    for row in getattr(result, "result_set", []) or []:
        data = _serialize_node(row[0])
        data["metadata"] = _parse_metadata_field(data.get("metadata"))
        memories.append(data)

    return jsonify({"status": "success", "tags": tags, "count": len(memories), "memories": memories})


@app.route("/recall", methods=["GET"])
def recall_memories() -> Any:
    query_start = time.perf_counter()
    query_text = (request.args.get("query") or "").strip()
    limit = max(1, min(int(request.args.get("limit", 5)), 50))
    embedding_param = request.args.get("embedding")
    time_query = request.args.get("time_query") or request.args.get("time")
    start_param = request.args.get("start")
    end_param = request.args.get("end")
    tags_param = request.args.getlist("tags") or request.args.get("tags")

    tag_mode = (request.args.get("tag_mode") or "any").strip().lower()
    if tag_mode not in {"any", "all"}:
        tag_mode = "any"

    tag_match = (request.args.get("tag_match") or "prefix").strip().lower()
    if tag_match not in {"exact", "prefix"}:
        tag_match = "prefix"

    time_start, time_end = _parse_time_expression(time_query)

    start_time = time_start
    end_time = time_end

    if start_param:
        try:
            start_time = _normalize_timestamp(start_param)
        except ValueError as exc:
            abort(400, description=f"Invalid start time: {exc}")

    if end_param:
        try:
            end_time = _normalize_timestamp(end_param)
        except ValueError as exc:
            abort(400, description=f"Invalid end time: {exc}")

    tag_filters = _normalize_tag_list(tags_param)

    seen_ids: set[str] = set()
    graph = get_memory_graph()
    qdrant_client = get_qdrant_client()

    results: List[Dict[str, Any]] = []
    vector_matches: List[Dict[str, Any]] = []
    if qdrant_client is not None:
        vector_matches = _vector_search(
            qdrant_client,
            graph,
            query_text,
            embedding_param,
            limit,
            seen_ids,
            tag_filters,
            tag_mode,
            tag_match,
        )
        if start_time or end_time or tag_filters:
            vector_matches = [
                res
                for res in vector_matches
                if _result_passes_filters(res, start_time, end_time, tag_filters, tag_mode, tag_match)
            ]
    results.extend(vector_matches[:limit])

    remaining_slots = max(0, limit - len(results))

    if remaining_slots and graph is not None:
        graph_matches = _graph_keyword_search(
            graph,
            query_text,
            remaining_slots,
            seen_ids,
            start_time=start_time,
            end_time=end_time,
            tag_filters=tag_filters,
            tag_mode=tag_mode,
            tag_match=tag_match,
        )
        results.extend(graph_matches[:remaining_slots])

    tags_only_request = (
        not query_text
        and not (embedding_param and embedding_param.strip())
        and bool(tag_filters)
    )

    if (
        tags_only_request
        and qdrant_client is not None
        and len(results) < limit
    ):
        tag_only_results = _vector_filter_only_tag_search(
            qdrant_client,
            tag_filters,
            tag_mode,
            tag_match,
            limit - len(results),
            seen_ids,
        )
        results.extend(tag_only_results)

    query_tokens = _extract_keywords(query_text.lower()) if query_text else []
    for result in results:
        final_score, components = _compute_metadata_score(result, query_text or "", query_tokens)
        result.setdefault("score_components", components)
        result["score_components"].update(components)
        result["final_score"] = final_score
        result["original_score"] = result.get("score", 0.0)
        result["score"] = final_score

    results = [
        res
        for res in results
        if _result_passes_filters(res, start_time, end_time, tag_filters, tag_mode, tag_match)
    ]

    results.sort(
        key=lambda r: (
            -float(r.get("final_score", 0.0)),
            r.get("source") != "qdrant",
            -float(r.get("original_score", 0.0)),
            -float((r.get("memory") or {}).get("importance", 0.0) or 0.0),
        )
    )

    response = {
        "status": "success",
        "query": query_text,
        "results": results,
        "count": len(results),
        "vector_search": {
            "enabled": qdrant_client is not None,
            "matched": bool(vector_matches),
        },
    }

    if query_text:
        response["keywords"] = query_tokens
    if start_time or end_time:
        response["time_window"] = {"start": start_time, "end": end_time}
    if tag_filters:
        response["tags"] = tag_filters
    response["tag_mode"] = tag_mode
    response["tag_match"] = tag_match
    response["query_time_ms"] = round((time.perf_counter() - query_start) * 1000, 2)

    # Structured logging for performance analysis
    logger.info(
        "recall_complete",
        extra={
            "query": query_text[:100] if query_text else "",  # Truncate for logs
            "results": len(results),
            "latency_ms": response["query_time_ms"],
            "vector_enabled": qdrant_client is not None,
            "vector_matches": len(vector_matches),
            "has_time_filter": bool(start_time or end_time),
            "has_tag_filter": bool(tag_filters),
            "limit": limit,
        }
    )

    return jsonify(response)


@app.route("/associate", methods=["POST"])
def create_association() -> Any:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        abort(400, description="JSON body is required")

    memory1_id = (payload.get("memory1_id") or "").strip()
    memory2_id = (payload.get("memory2_id") or "").strip()
    relation_type = (payload.get("type") or "RELATES_TO").upper()
    strength = _coerce_importance(payload.get("strength", 0.5))

    if not memory1_id or not memory2_id:
        abort(400, description="'memory1_id' and 'memory2_id' are required")
    if memory1_id == memory2_id:
        abort(400, description="Cannot associate a memory with itself")
    if relation_type not in ALLOWED_RELATIONS:
        abort(
            400, description=f"Relation type must be one of {sorted(ALLOWED_RELATIONS)}"
        )

    graph = get_memory_graph()
    if graph is None:
        abort(503, description="FalkorDB is unavailable")

    timestamp = utc_now()

    # Build relationship properties based on type
    relationship_props = {
        "strength": strength,
        "updated_at": timestamp,
    }

    # Add type-specific properties if provided
    relation_config = RELATIONSHIP_TYPES.get(relation_type, {})
    if "properties" in relation_config:
        for prop in relation_config["properties"]:
            if prop in payload:
                relationship_props[prop] = payload[prop]

    # Build the SET clause dynamically
    set_clauses = [f"r.{key} = ${key}" for key in relationship_props]
    set_clause = ", ".join(set_clauses)

    try:
        result = graph.query(
            f"""
            MATCH (m1:Memory {{id: $id1}})
            MATCH (m2:Memory {{id: $id2}})
            MERGE (m1)-[r:{relation_type}]->(m2)
            SET {set_clause}
            RETURN r
            """,
            {
                "id1": memory1_id,
                "id2": memory2_id,
                **relationship_props,
            },
        )
    except Exception:  # pragma: no cover - log full stack trace in production
        logger.exception("Failed to create association")
        abort(500, description="Failed to create association")

    if not result.result_set:
        abort(404, description="One or both memories do not exist")

    response = {
        "status": "success",
        "message": f"Association created between {memory1_id} and {memory2_id}",
        "relation_type": relation_type,
        "strength": strength,
    }

    # Add additional properties to response
    for prop in relation_config.get("properties", []):
        if prop in relationship_props:
            response[prop] = relationship_props[prop]

    return jsonify(response), 201


@app.route("/consolidate", methods=["POST"])
def consolidate_memories() -> Any:
    """Run memory consolidation."""
    data = request.get_json() or {}
    mode = data.get('mode', 'full')
    dry_run = data.get('dry_run', True)

    graph = get_memory_graph()
    if graph is None:
        abort(503, description="FalkorDB is unavailable")

    init_consolidation_scheduler()

    try:
        vector_store = get_qdrant_client()
        consolidator = MemoryConsolidator(graph, vector_store)
        results = consolidator.consolidate(mode=mode, dry_run=dry_run)

        if not dry_run:
            _persist_consolidation_run(graph, results)

        return jsonify({
            "status": "success",
            "consolidation": results
        }), 200
    except Exception as e:
        logger.error(f"Consolidation failed: {e}")
        return jsonify({
            "error": "Consolidation failed",
            "details": str(e)
        }), 500


@app.route("/consolidate/status", methods=["GET"])
def consolidation_status() -> Any:
    """Get consolidation scheduler status."""
    graph = get_memory_graph()
    if graph is None:
        abort(503, description="FalkorDB is unavailable")

    try:
        init_consolidation_scheduler()
        scheduler = _build_scheduler_from_graph(graph)
        history = _load_recent_runs(graph, CONSOLIDATION_HISTORY_LIMIT)
        next_runs = scheduler.get_next_runs() if scheduler else {}

        return jsonify({
            "status": "success",
            "next_runs": next_runs,
            "history": history,
            "thread_alive": bool(state.consolidation_thread and state.consolidation_thread.is_alive()),
            "tick_seconds": CONSOLIDATION_TICK_SECONDS,
        }), 200
    except Exception as e:
        logger.error(f"Failed to get consolidation status: {e}")
        return jsonify({
            "error": "Failed to get status",
            "details": str(e)
        }), 500


@app.route("/startup-recall", methods=["GET"])
def startup_recall() -> Any:
    """Recall critical lessons at session startup."""
    graph = get_memory_graph()
    if graph is None:
        abort(503, description="FalkorDB is unavailable")

    try:
        # Search for critical lessons and system rules
        lesson_query = """
            MATCH (m:Memory)
            WHERE 'critical' IN m.tags OR 'lesson' IN m.tags OR 'ai-assistant' IN m.tags
            RETURN m.id as id, m.content as content, m.tags as tags,
                   m.importance as importance, m.type as type, m.metadata as metadata
            ORDER BY m.importance DESC
            LIMIT 10
        """

        lesson_results = graph.query(lesson_query)
        lessons = []

        if lesson_results.result_set:
            for row in lesson_results.result_set:
                lessons.append({
                    'id': row[0],
                    'content': row[1],
                    'tags': row[2] if row[2] else [],
                    'importance': row[3] if row[3] else 0.5,
                    'type': row[4] if row[4] else 'Context',
                    'metadata': json.loads(row[5]) if row[5] else {}
                })

        # Get system rules
        system_query = """
            MATCH (m:Memory)
            WHERE 'system' IN m.tags OR 'memory-recall' IN m.tags
            RETURN m.id as id, m.content as content, m.tags as tags
            LIMIT 5
        """

        system_results = graph.query(system_query)
        system_rules = []

        if system_results.result_set:
            for row in system_results.result_set:
                system_rules.append({
                    'id': row[0],
                    'content': row[1],
                    'tags': row[2] if row[2] else []
                })

        response = {
            'status': 'success',
            'critical_lessons': lessons,
            'system_rules': system_rules,
            'lesson_count': len(lessons),
            'has_critical': any(l.get('importance', 0) >= 0.9 for l in lessons),
            'summary': f"Recalled {len(lessons)} lesson(s) and {len(system_rules)} system rule(s)"
        }

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Startup recall failed: {e}")
        return jsonify({
            "error": "Startup recall failed",
            "details": str(e)
        }), 500


@app.route("/analyze", methods=["GET"])
def analyze_memories() -> Any:
    """Analyze memory patterns, preferences, and insights."""
    query_start = time.perf_counter()
    graph = get_memory_graph()
    if graph is None:
        abort(503, description="FalkorDB is unavailable")

    analytics = {
        "memory_types": {},
        "patterns": [],
        "preferences": [],
        "temporal_insights": {},
        "entity_frequency": {},
        "confidence_distribution": {},
    }

    try:
        # Analyze memory type distribution
        type_result = graph.query(
            """
            MATCH (m:Memory)
            WHERE m.type IS NOT NULL
            RETURN m.type, COUNT(m) as count, AVG(m.confidence) as avg_confidence
            ORDER BY count DESC
            """
        )

        for mem_type, count, avg_conf in type_result.result_set:
            analytics["memory_types"][mem_type] = {
                "count": count,
                "average_confidence": round(avg_conf, 3) if avg_conf else 0,
            }

        # Find patterns with high confidence
        pattern_result = graph.query(
            """
            MATCH (p:Pattern)
            WHERE p.confidence > 0.6
            RETURN p.type, p.content, p.confidence, p.observations
            ORDER BY p.confidence DESC
            LIMIT 10
            """
        )

        for p_type, content, confidence, observations in pattern_result.result_set:
            analytics["patterns"].append({
                "type": p_type,
                "description": content,
                "confidence": round(confidence, 3) if confidence else 0,
                "observations": observations or 0,
            })

        # Find preferences (PREFERS_OVER relationships)
        pref_result = graph.query(
            """
            MATCH (m1:Memory)-[r:PREFERS_OVER]->(m2:Memory)
            RETURN m1.content, m2.content, r.context, r.strength
            ORDER BY r.strength DESC
            LIMIT 10
            """
        )

        for preferred, over, context, strength in pref_result.result_set:
            analytics["preferences"].append({
                "prefers": preferred,
                "over": over,
                "context": context,
                "strength": round(strength, 3) if strength else 0,
            })

        # Temporal insights - simplified for FalkorDB compatibility
        try:
            temporal_result = graph.query(
                """
                MATCH (m:Memory)
                WHERE m.timestamp IS NOT NULL
                RETURN m.timestamp, m.importance
                LIMIT 100
                """
            )

            # Process temporal data in Python
            from collections import defaultdict
            hour_data = defaultdict(lambda: {"count": 0, "total_importance": 0})

            for timestamp, importance in temporal_result.result_set:
                if timestamp and len(timestamp) > 13:
                    # Extract hour from timestamp string
                    hour_str = timestamp[11:13]
                    if hour_str.isdigit():
                        hour = int(hour_str)
                        hour_data[hour]["count"] += 1
                        hour_data[hour]["total_importance"] += importance or 0.5

            # Calculate averages
            for hour, data in hour_data.items():
                if data["count"] > 0:
                    analytics["temporal_insights"][f"hour_{hour:02d}"] = {
                        "count": data["count"],
                        "avg_importance": round(data["total_importance"] / data["count"], 3)
                    }
        except Exception:
            # Skip temporal insights if query fails
            pass

        # Confidence distribution
        conf_result = graph.query(
            """
            MATCH (m:Memory)
            WHERE m.confidence IS NOT NULL
            RETURN
                CASE
                    WHEN m.confidence < 0.3 THEN 'low'
                    WHEN m.confidence < 0.7 THEN 'medium'
                    ELSE 'high'
                END as level,
                COUNT(m) as count
            """
        )

        for level, count in conf_result.result_set:
            analytics["confidence_distribution"][level] = count

        # Entity extraction insights (top mentioned tools/projects)
        entity_result = graph.query(
            """
            MATCH (m:Memory)
            WHERE m.content IS NOT NULL
            RETURN m.content
            LIMIT 100
            """
        )

        entity_counts: Dict[str, Dict[str, int]] = {
            "tools": {},
            "projects": {},
        }

        for (content,) in entity_result.result_set:
            entities = extract_entities(content)
            for tool in entities.get("tools", []):
                entity_counts["tools"][tool] = entity_counts["tools"].get(tool, 0) + 1
            for project in entities.get("projects", []):
                entity_counts["projects"][project] = entity_counts["projects"].get(project, 0) + 1

        # Top 5 most mentioned
        analytics["entity_frequency"]["tools"] = sorted(
            entity_counts["tools"].items(), key=lambda x: x[1], reverse=True
        )[:5]
        analytics["entity_frequency"]["projects"] = sorted(
            entity_counts["projects"].items(), key=lambda x: x[1], reverse=True
        )[:5]

    except Exception:
        logger.exception("Failed to generate analytics")
        abort(500, description="Failed to generate analytics")

    return jsonify({
        "status": "success",
        "analytics": analytics,
        "generated_at": utc_now(),
        "query_time_ms": round((time.perf_counter() - query_start) * 1000, 2),
    })


def _normalize_tags(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(tag, str) for tag in value):
        return value
    abort(400, description="'tags' must be a list of strings or a single string")


def _coerce_importance(value: Any) -> float:
    if value is None:
        return 0.5
    try:
        score = float(value)
    except (TypeError, ValueError):
        abort(400, description="'importance' must be a number")
    if score < 0 or score > 1:
        abort(400, description="'importance' must be between 0 and 1")
    return score


def _coerce_embedding(value: Any) -> Optional[List[float]]:
    if value is None or value == "":
        return None
    vector: List[Any]
    if isinstance(value, list):
        vector = value
    elif isinstance(value, str):
        vector = [part.strip() for part in value.split(",") if part.strip()]
    else:
        raise ValueError(
            "Embedding must be a list of floats or a comma-separated string"
        )

    if len(vector) != VECTOR_SIZE:
        raise ValueError(f"Embedding must contain exactly {VECTOR_SIZE} values")

    try:
        return [float(component) for component in vector]
    except ValueError as exc:
        raise ValueError("Embedding must contain numeric values") from exc


def _generate_placeholder_embedding(content: str) -> List[float]:
    """Generate a deterministic embedding vector from the content."""
    digest = hashlib.sha256(content.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = random.Random(seed)
    return [rng.random() for _ in range(VECTOR_SIZE)]


def _generate_real_embedding(content: str) -> List[float]:
    """Generate an embedding using the configured provider."""
    init_embedding_provider()

    if state.embedding_provider is None:
        logger.warning("No embedding provider available, using placeholder")
        return _generate_placeholder_embedding(content)

    try:
        embedding = state.embedding_provider.generate_embedding(content)
        if not isinstance(embedding, list) or len(embedding) != VECTOR_SIZE:
            logger.warning(
                "Provider %s returned %s dims (expected %d); falling back to placeholder",
                state.embedding_provider.provider_name(),
                len(embedding) if isinstance(embedding, list) else "invalid",
                VECTOR_SIZE,
            )
            return _generate_placeholder_embedding(content)
        return embedding
    except Exception as e:
        logger.warning("Failed to generate embedding: %s", str(e))
        return _generate_placeholder_embedding(content)


def _generate_real_embeddings_batch(contents: List[str]) -> List[List[float]]:
    """Generate multiple embeddings in a single batch for efficiency."""
    init_embedding_provider()

    if not contents:
        return []

    if state.embedding_provider is None:
        logger.debug("No embedding provider available, falling back to placeholder embeddings")
        return [_generate_placeholder_embedding(c) for c in contents]

    try:
        embeddings = state.embedding_provider.generate_embeddings_batch(contents)
        if not embeddings or any(len(e) != VECTOR_SIZE for e in embeddings):
            logger.warning(
                "Provider %s returned invalid dims in batch; using placeholders",
                state.embedding_provider.provider_name() if state.embedding_provider else "unknown",
            )
            return [_generate_placeholder_embedding(c) for c in contents]
        return embeddings
    except Exception as e:
        logger.warning("Failed to generate batch embeddings: %s", str(e))
        return [_generate_placeholder_embedding(c) for c in contents]


def _serialize_node(node: Any) -> Dict[str, Any]:
    properties = getattr(node, "properties", None)
    if isinstance(properties, dict):
        data = dict(properties)
    elif isinstance(node, dict):
        data = dict(node)
    else:
        return {"value": node}

    if "metadata" in data:
        data["metadata"] = _parse_metadata_field(data["metadata"])

    return data


def _summarize_relation_node(data: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}

    for key in ("id", "type", "timestamp", "summary", "importance", "confidence"):
        if key in data:
            summary[key] = data[key]

    content = data.get("content")
    if "summary" not in summary and isinstance(content, str):
        snippet = content.strip()
        if len(snippet) > 160:
            snippet = snippet[:157].rsplit(" ", 1)[0] + "…"
        summary["content"] = snippet

    tags = data.get("tags")
    if isinstance(tags, list) and tags:
        summary["tags"] = tags[:5]

    return summary


def _fetch_relations(graph: Any, memory_id: str) -> List[Dict[str, Any]]:
    try:
        records = graph.query(
            """
            MATCH (m:Memory {id: $id})-[r]->(related:Memory)
            RETURN type(r) as relation_type, r.strength as strength, related
            ORDER BY coalesce(r.updated_at, related.timestamp) DESC
            LIMIT $limit
            """,
            {"id": memory_id, "limit": RECALL_RELATION_LIMIT},
        )
    except Exception:  # pragma: no cover - log full stack trace in production
        logger.exception("Failed to fetch relations for memory %s", memory_id)
        return []

    connections: List[Dict[str, Any]] = []
    for relation_type, strength, related in records.result_set:
        connections.append(
            {
                "type": relation_type,
                "strength": strength,
                "memory": _summarize_relation_node(_serialize_node(related)),
            }
        )
    return connections


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8001"))
    logger.info("Starting Flask API on port %s", port)
    init_falkordb()
    init_qdrant()
    init_openai()  # Still needed for memory type classification
    init_embedding_provider()  # New provider pattern for embeddings
    init_enrichment_pipeline()
    init_embedding_pipeline()
    init_consolidation_scheduler()
    app.run(host="0.0.0.0", port=port, debug=False)
