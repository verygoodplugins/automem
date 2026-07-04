from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable

from dotenv import load_dotenv

# Load environment variables before configuring the application.
load_dotenv()
load_dotenv(Path.home() / ".config" / "automem" / ".env")

# Qdrant / FalkorDB configuration
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "memories")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE") or os.getenv("QDRANT_VECTOR_SIZE", "1024"))
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY")

# QDRANT_URL takes precedence; otherwise construct from QDRANT_HOST + QDRANT_PORT.
# This keeps Railway templates simple (just set QDRANT_HOST=qdrant.railway.internal).
_qdrant_host = os.getenv("QDRANT_HOST")
QDRANT_URL: str | None = os.getenv("QDRANT_URL") or (
    f"http://{_qdrant_host}:{QDRANT_PORT}" if _qdrant_host else None
)

GRAPH_NAME = os.getenv("FALKORDB_GRAPH", "memories")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))

# Consolidation scheduling defaults (seconds unless noted)
CONSOLIDATION_TICK_SECONDS = int(os.getenv("CONSOLIDATION_TICK_SECONDS", "60"))
CONSOLIDATION_DECAY_INTERVAL_SECONDS = int(
    os.getenv("CONSOLIDATION_DECAY_INTERVAL_SECONDS", str(86400))
)
CONSOLIDATION_CREATIVE_INTERVAL_SECONDS = int(
    os.getenv("CONSOLIDATION_CREATIVE_INTERVAL_SECONDS", str(604800))
)
CONSOLIDATION_CLUSTER_INTERVAL_SECONDS = int(
    os.getenv("CONSOLIDATION_CLUSTER_INTERVAL_SECONDS", str(2592000))
)
CONSOLIDATION_FORGET_INTERVAL_SECONDS = int(
    os.getenv("CONSOLIDATION_FORGET_INTERVAL_SECONDS", str(0))
)

# Clustering tuning is deployment-specific and depends on embedding geometry,
# corpus shape, and acceptable merge risk. Defaults preserve existing behavior;
# operators can lower thresholds or min cluster size after measuring local data.
CONSOLIDATION_CLUSTER_SIMILARITY_THRESHOLD = float(
    os.getenv("CONSOLIDATION_CLUSTER_SIMILARITY_THRESHOLD", "0.75")
)
CONSOLIDATION_MIN_CLUSTER_SIZE = int(os.getenv("CONSOLIDATION_MIN_CLUSTER_SIZE", "3"))
_DECAY_THRESHOLD_RAW = os.getenv("CONSOLIDATION_DECAY_IMPORTANCE_THRESHOLD", "0.3").strip()
CONSOLIDATION_DECAY_IMPORTANCE_THRESHOLD = (
    float(_DECAY_THRESHOLD_RAW) if _DECAY_THRESHOLD_RAW else None
)
CONSOLIDATION_HISTORY_LIMIT = int(os.getenv("CONSOLIDATION_HISTORY_LIMIT", "20"))

# Decay formula tuning
CONSOLIDATION_BASE_DECAY_RATE = float(os.getenv("CONSOLIDATION_BASE_DECAY_RATE", "0.01"))
CONSOLIDATION_IMPORTANCE_FLOOR_FACTOR = float(
    os.getenv("CONSOLIDATION_IMPORTANCE_FLOOR_FACTOR", "0.3")
)

# Memory protection configuration (prevents accidental data loss)
CONSOLIDATION_DELETE_THRESHOLD = float(os.getenv("CONSOLIDATION_DELETE_THRESHOLD", "0.0"))
CONSOLIDATION_ARCHIVE_THRESHOLD = float(os.getenv("CONSOLIDATION_ARCHIVE_THRESHOLD", "0.0"))
CONSOLIDATION_GRACE_PERIOD_DAYS = int(os.getenv("CONSOLIDATION_GRACE_PERIOD_DAYS", "90"))
CONSOLIDATION_IMPORTANCE_PROTECTION_THRESHOLD = float(
    os.getenv("CONSOLIDATION_IMPORTANCE_PROTECTION_THRESHOLD", "0.7")
)
_PROTECTED_TYPES_RAW = os.getenv("CONSOLIDATION_PROTECTED_TYPES", "Decision,Insight").strip()
CONSOLIDATION_PROTECTED_TYPES = (
    frozenset(t.strip() for t in _PROTECTED_TYPES_RAW.split(",") if t.strip())
    if _PROTECTED_TYPES_RAW
    else frozenset()
)
CONSOLIDATION_CONTROL_LABEL = "ConsolidationControl"
CONSOLIDATION_RUN_LABEL = "ConsolidationRun"
CONSOLIDATION_CONTROL_NODE_ID = os.getenv("CONSOLIDATION_CONTROL_NODE_ID", "global")
IDENTITY_SYNTHESIS_ENABLED = os.getenv("IDENTITY_SYNTHESIS_ENABLED", "false").lower() in {
    "1",
    "true",
    "yes",
}
_IDENTITY_INTERVAL_DEFAULT = "604800" if IDENTITY_SYNTHESIS_ENABLED else "0"
CONSOLIDATION_IDENTITY_INTERVAL_SECONDS = int(
    os.getenv("CONSOLIDATION_IDENTITY_INTERVAL_SECONDS", _IDENTITY_INTERVAL_DEFAULT)
)
IDENTITY_SYNTHESIS_MODEL = os.getenv("IDENTITY_SYNTHESIS_MODEL") or os.getenv(
    "CLASSIFICATION_MODEL",
    "gpt-4o-mini",
)

CONSOLIDATION_TASK_FIELDS = {
    "decay": "decay_last_run",
    "creative": "creative_last_run",
    "cluster": "cluster_last_run",
    "forget": "forget_last_run",
    "identity": "identity_last_run",
    "full": "full_last_run",
}

# Sync configuration (background sync worker)
SYNC_CHECK_INTERVAL_SECONDS = int(os.getenv("SYNC_CHECK_INTERVAL_SECONDS", "3600"))  # 1 hour
SYNC_AUTO_REPAIR = os.getenv("SYNC_AUTO_REPAIR", "true").lower() not in {"0", "false", "no"}

# Enrichment configuration
ENRICHMENT_MAX_ATTEMPTS = int(os.getenv("ENRICHMENT_MAX_ATTEMPTS", "3"))
ENRICHMENT_SIMILARITY_LIMIT = int(os.getenv("ENRICHMENT_SIMILARITY_LIMIT", "5"))
ENRICHMENT_SIMILARITY_THRESHOLD = float(os.getenv("ENRICHMENT_SIMILARITY_THRESHOLD", "0.8"))
ENRICHMENT_IDLE_SLEEP_SECONDS = float(os.getenv("ENRICHMENT_IDLE_SLEEP_SECONDS", "2"))
ENRICHMENT_FAILURE_BACKOFF_SECONDS = float(os.getenv("ENRICHMENT_FAILURE_BACKOFF_SECONDS", "5"))
ENRICHMENT_ENABLE_SUMMARIES = os.getenv("ENRICHMENT_ENABLE_SUMMARIES", "true").lower() not in {
    "0",
    "false",
    "no",
}
ENRICHMENT_SPACY_MODEL = os.getenv("ENRICHMENT_SPACY_MODEL", "en_core_web_sm")

# JIT (just-in-time) enrichment: run lightweight enrichment inline during recall
# for memories that haven't been processed by the async worker yet.
JIT_ENRICHMENT_ENABLED = os.getenv("JIT_ENRICHMENT_ENABLED", "true").lower() not in {
    "0",
    "false",
    "no",
}

# Model configuration
# voyage-4 (1024d): Recommended default via EMBEDDING_PROVIDER=auto
# text-embedding-3-small (1536d native): OpenAI fallback; truncated to VECTOR_SIZE via
#   Matryoshka when the upstream API supports the ``dimensions`` parameter.
#   For OpenAI-compatible endpoints that don't support ``dimensions``, the model
#   returns its native 1536-d output and VECTOR_SIZE is ignored.
#   If VECTOR_SIZE > 1536, auto-upgrades to text-embedding-3-large.
# text-embedding-3-large: OpenAI high-precision, use VECTOR_SIZE=3072
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CLASSIFICATION_MODEL = os.getenv("CLASSIFICATION_MODEL", "gpt-4o-mini")

RECALL_RELATION_LIMIT = int(os.getenv("RECALL_RELATION_LIMIT", "5"))
RECALL_EXPANSION_LIMIT = int(os.getenv("RECALL_EXPANSION_LIMIT", "25"))
RECALL_MIN_SCORE = float(os.getenv("RECALL_MIN_SCORE", "0.0"))
RECALL_ADAPTIVE_FLOOR = os.getenv("RECALL_ADAPTIVE_FLOOR", "true").lower() in ("true", "1", "yes")
RECALL_METADATA_SEARCH_ENABLED = os.getenv(
    "RECALL_METADATA_SEARCH_ENABLED", "true"
).lower() not in {
    "0",
    "false",
    "no",
}

# Recall candidate-pool over-fetch. The vector store returns its top-K by raw
# similarity, but the richer final score (importance, exact-match, tags) runs
# only on those survivors — so a high-importance, exact-topic memory that is a
# slightly weaker pure-vector match gets cut before its signal can lift it.
# Over-fetch widens the pool the re-rank sees; the response is still trimmed to
# the requested limit. Set OVERFETCH=1 to restore the legacy 1x behavior.
RECALL_VECTOR_OVERFETCH = max(1, int(os.getenv("RECALL_VECTOR_OVERFETCH", "4")))
# Absolute ceiling on the over-fetched candidate count, kept separate from the
# user-facing RECALL_MAX_LIMIT so over-fetch isn't strangled by the response cap.
RECALL_VECTOR_FETCH_CAP = max(1, int(os.getenv("RECALL_VECTOR_FETCH_CAP", "200")))

# Internal artifact memory types that must never surface in user-facing /recall
# results or vector sync accounting (e.g. consolidation cluster summaries).
# Comma-separated env override.
RECALL_EXCLUDED_TYPES = frozenset(
    t.strip() for t in os.getenv("RECALL_EXCLUDED_TYPES", "MetaPattern").split(",") if t.strip()
)

# Memory content size limits (governs auto-summarization on store)
# Soft limit: Content above this triggers auto-summarization
MEMORY_CONTENT_SOFT_LIMIT = int(os.getenv("MEMORY_CONTENT_SOFT_LIMIT", "500"))
# Hard limit: Content above this is rejected outright
MEMORY_CONTENT_HARD_LIMIT = int(os.getenv("MEMORY_CONTENT_HARD_LIMIT", "2000"))
# Enable/disable auto-summarization (if disabled, content above soft limit is stored as-is)
MEMORY_AUTO_SUMMARIZE = os.getenv("MEMORY_AUTO_SUMMARIZE", "true").lower() not in {
    "0",
    "false",
    "no",
}
# Target length for summarized content
MEMORY_SUMMARY_TARGET_LENGTH = int(os.getenv("MEMORY_SUMMARY_TARGET_LENGTH", "300"))

# Memory types for classification
MEMORY_TYPES = {"Decision", "Pattern", "Preference", "Style", "Habit", "Insight", "Context"}

# Type aliases for normalization (lowercase and legacy types → canonical)
# Non-canonical types are auto-mapped to canonical types on store
TYPE_ALIASES: dict[str, str] = {
    # Lowercase versions of canonical types
    "decision": "Decision",
    "pattern": "Pattern",
    "preference": "Preference",
    "style": "Style",
    "habit": "Habit",
    "insight": "Insight",
    "context": "Context",
    # Legacy/alternative types
    "memory": "Context",
    "milestone": "Context",
    "analysis": "Insight",
    "observation": "Insight",
    "document": "Context",
    "meeting_notes": "Context",
    "template": "Pattern",
    "project": "Context",
    "issue": "Insight",
    "timeline": "Context",
    "organization": "Context",
    "person": "Context",
    "interests": "Preference",
    "personality": "Preference",
    "emotional_patterns": "Preference",
    "relationship_dynamics": "Preference",
    "personal_situation": "Context",
    "health_habits": "Habit",
    "practical_info": "Context",
    "communication": "Preference",
    "legal_analysis": "Insight",
}


def normalize_memory_type(raw_type: str | None) -> tuple[str, bool]:
    """Normalize a memory type to a canonical type.

    Returns:
        tuple of (normalized_type, was_modified)
        - normalized_type: The canonical type (e.g., "Decision", "Context")
        - was_modified: True if the type was changed, False if already canonical
    """
    if not raw_type:
        return "Context", True

    # Already canonical
    if raw_type in MEMORY_TYPES:
        return raw_type, False

    # Check aliases
    if raw_type in TYPE_ALIASES:
        return TYPE_ALIASES[raw_type], True

    # Unknown type - reject by returning None marker
    return "", True  # Empty string signals rejection


LEGACY_DISCOVERED_RELATIONS: Dict[str, str] = {
    "EXPLAINS": "explains",
    "SHARES_THEME": "shares_theme",
    "PARALLEL_CONTEXT": "parallel_context",
}


def _relation_config(
    description: str,
    *,
    properties: Iterable[str] = (),
    authorable: bool,
    system_generated: bool,
    default_expand: bool,
    public_visible: bool,
    color: str,
) -> Dict[str, Any]:
    return {
        "description": description,
        "properties": list(properties),
        "authorable": authorable,
        "system_generated": system_generated,
        "default_expand": default_expand,
        "public_visible": public_visible,
        "color": color,
    }


# Relationship registry. This is the source of truth for API/MCP/docs/viewer behavior.
RELATIONSHIP_TYPES: Dict[str, Dict[str, Any]] = {
    "RELATES_TO": _relation_config(
        "General relationship",
        authorable=True,
        system_generated=False,
        default_expand=True,
        public_visible=True,
        color="#94A3B8",
    ),
    "LEADS_TO": _relation_config(
        "Causal relationship",
        authorable=True,
        system_generated=False,
        default_expand=True,
        public_visible=True,
        color="#3B82F6",
    ),
    "OCCURRED_BEFORE": _relation_config(
        "Temporal relationship",
        authorable=True,
        system_generated=False,
        default_expand=True,
        public_visible=True,
        color="#6B7280",
    ),
    "SIMILAR_TO": _relation_config(
        "Semantic similarity",
        properties=["score", "updated_at"],
        authorable=False,
        system_generated=True,
        default_expand=False,
        public_visible=True,
        color="#0EA5E9",
    ),
    "PRECEDED_BY": _relation_config(
        "Prior in time",
        properties=["count", "updated_at"],
        authorable=False,
        system_generated=True,
        default_expand=False,
        public_visible=True,
        color="#475569",
    ),
    "PREFERS_OVER": _relation_config(
        "Preference relationship",
        properties=["context", "strength", "reason"],
        authorable=True,
        system_generated=False,
        default_expand=True,
        public_visible=True,
        color="#8B5CF6",
    ),
    "EXEMPLIFIES": _relation_config(
        "Pattern example",
        properties=["pattern_type", "confidence"],
        authorable=True,
        system_generated=False,
        default_expand=True,
        public_visible=True,
        color="#10B981",
    ),
    "CONTRADICTS": _relation_config(
        "Conflicting information",
        properties=["resolution", "reason"],
        authorable=True,
        system_generated=False,
        default_expand=True,
        public_visible=True,
        color="#EF4444",
    ),
    "REINFORCES": _relation_config(
        "Strengthens pattern",
        properties=["strength", "observations"],
        authorable=True,
        system_generated=False,
        default_expand=True,
        public_visible=True,
        color="#22C55E",
    ),
    "INVALIDATED_BY": _relation_config(
        "Superseded information",
        properties=["reason", "timestamp"],
        authorable=True,
        system_generated=False,
        default_expand=True,
        public_visible=True,
        color="#F97316",
    ),
    "EVOLVED_INTO": _relation_config(
        "Evolution of knowledge",
        properties=["confidence", "reason"],
        authorable=True,
        system_generated=False,
        default_expand=True,
        public_visible=True,
        color="#06B6D4",
    ),
    "DERIVED_FROM": _relation_config(
        "Derived knowledge",
        properties=["transformation", "confidence"],
        authorable=True,
        system_generated=False,
        default_expand=True,
        public_visible=True,
        color="#A855F7",
    ),
    "PART_OF": _relation_config(
        "Hierarchical relationship",
        properties=["role", "context"],
        authorable=True,
        system_generated=False,
        default_expand=True,
        public_visible=True,
        color="#64748B",
    ),
    "DISCOVERED": _relation_config(
        "Heuristic relationship inferred by consolidation",
        properties=["kind", "confidence", "similarity", "updated_at", "origin"],
        authorable=False,
        system_generated=True,
        default_expand=False,
        public_visible=False,
        color="#A1A1AA",
    ),
}

AUTHORABLE_RELATIONS = frozenset(
    name for name, meta in RELATIONSHIP_TYPES.items() if meta.get("authorable")
)
PUBLIC_RELATIONS = frozenset(
    name for name, meta in RELATIONSHIP_TYPES.items() if meta.get("public_visible")
)
SYSTEM_RELATIONS = frozenset(
    name for name, meta in RELATIONSHIP_TYPES.items() if meta.get("system_generated")
)
DEFAULT_EXPAND_RELATIONS = frozenset(
    name for name, meta in RELATIONSHIP_TYPES.items() if meta.get("default_expand")
)
FILTERABLE_RELATIONS = frozenset(RELATIONSHIP_TYPES.keys())
ALLOWED_RELATIONS = frozenset(FILTERABLE_RELATIONS)
RELATION_COLORS = {
    name: str(meta["color"])
    for name, meta in RELATIONSHIP_TYPES.items()
    if meta.get("public_visible")
}


def canonicalize_relation_type(raw_type: str | None) -> str:
    relation_type = (raw_type or "").strip().upper()
    if relation_type in LEGACY_DISCOVERED_RELATIONS:
        return "DISCOVERED"
    return relation_type


def relation_kind_for_storage(
    relation_type: str,
    properties: Dict[str, Any] | None = None,
) -> str | None:
    normalized_type = canonicalize_relation_type(relation_type)
    relation_props = dict(properties or {})
    if relation_type in LEGACY_DISCOVERED_RELATIONS:
        return LEGACY_DISCOVERED_RELATIONS[relation_type]
    if normalized_type == "DISCOVERED":
        kind = relation_props.get("kind")
        if kind:
            return str(kind).strip().lower()
    return None


def normalize_relation_type(
    relation_type: str | None,
    properties: Dict[str, Any] | None = None,
) -> tuple[str, Dict[str, Any]]:
    raw_type = (relation_type or "").strip().upper()
    normalized_type = canonicalize_relation_type(raw_type)
    normalized_props = dict(properties or {})
    kind = relation_kind_for_storage(raw_type, normalized_props)
    if kind and normalized_type == "DISCOVERED":
        normalized_props["kind"] = kind
    return normalized_type, normalized_props


def expand_relation_query_types(relation_types: Iterable[str]) -> list[str]:
    expanded: list[str] = []
    seen: set[str] = set()

    for relation_type in relation_types:
        normalized_type = canonicalize_relation_type(relation_type)
        db_types = [normalized_type]
        if normalized_type == "DISCOVERED":
            db_types = ["DISCOVERED", *LEGACY_DISCOVERED_RELATIONS.keys()]
        for db_type in db_types:
            if db_type not in seen:
                expanded.append(db_type)
                seen.add(db_type)

    return expanded


# Search weighting parameters (can be overridden via environment variables)
SEARCH_WEIGHT_VECTOR = float(os.getenv("SEARCH_WEIGHT_VECTOR", "0.35"))
SEARCH_WEIGHT_KEYWORD = float(os.getenv("SEARCH_WEIGHT_KEYWORD", "0.35"))
SEARCH_WEIGHT_METADATA = float(os.getenv("SEARCH_WEIGHT_METADATA", "0.35"))
SEARCH_WEIGHT_TAG = float(os.getenv("SEARCH_WEIGHT_TAG", "0.2"))
SEARCH_WEIGHT_IMPORTANCE = float(os.getenv("SEARCH_WEIGHT_IMPORTANCE", "0.1"))
SEARCH_WEIGHT_CONFIDENCE = float(os.getenv("SEARCH_WEIGHT_CONFIDENCE", "0.05"))
SEARCH_WEIGHT_RECENCY = float(os.getenv("SEARCH_WEIGHT_RECENCY", "0.1"))
SEARCH_WEIGHT_EXACT = float(os.getenv("SEARCH_WEIGHT_EXACT", "0.2"))
SEARCH_WEIGHT_RELATION = float(os.getenv("SEARCH_WEIGHT_RELATION", "0.25"))
SEARCH_WEIGHT_RELEVANCE = float(os.getenv("SEARCH_WEIGHT_RELEVANCE", "0.0"))


def _positive_or_default(raw: str, default: float) -> float:
    """Parse a float env value, falling back to ``default`` when not strictly positive.

    Unparseable values raise ValueError, matching the neighboring float() parses;
    only the domain (value > 0) is guarded here.
    """
    value = float(raw)
    return value if value > 0 else default


# Recency decay tuning: window in days, curve "linear" (score hits 0 at window)
# or "exp" (window acts as half-life). Invalid curve values fall back to linear;
# non-positive window values fall back to 180 (a window <= 0 would divide by
# zero or produce unbounded scores at recall time).
SEARCH_RECENCY_WINDOW_DAYS = _positive_or_default(
    os.getenv("SEARCH_RECENCY_WINDOW_DAYS", "180"), 180.0
)
_RECENCY_CURVE_RAW = os.getenv("SEARCH_RECENCY_CURVE", "linear").strip().lower()
SEARCH_RECENCY_CURVE = _RECENCY_CURVE_RAW if _RECENCY_CURVE_RAW in {"linear", "exp"} else "linear"


def _non_negative_int_or_default(raw: str, default: int) -> int:
    """Parse an int env value, falling back to ``default`` when negative.

    Unparseable values raise ValueError, matching the neighboring int()/float()
    parses. 0 is a valid sentinel here (it selects legacy behavior), so only
    negative values fall back — mirroring ``_positive_or_default``'s
    fail-safe-to-default spirit rather than silently meaning "legacy".
    """
    value = int(raw)
    return value if value >= 0 else default


# Tag-score query-length normalization: the tag-overlap score divides token
# hits by min(len(query_tokens), cap) so long queries aren't penalized
# relative to short ones. 0 disables the cap (legacy: denominator = full
# query length). Default is 0 (opt-in): a production-corpus A/B (2026-06-11,
# 200 queries, 10k-memory clone) showed cap values 2/3/4 regress Recall@5 by
# 14/7/4pp on ungated queries — the capped denominator inflates tag scores
# and amplifies tag noise over vector/keyword evidence. Negative values fall
# back to the default — falling back is safer than treating a typo'd
# negative as intentional.
SEARCH_TAG_SCORE_TOKEN_CAP = _non_negative_int_or_default(
    os.getenv("SEARCH_TAG_SCORE_TOKEN_CAP", "0"), 0
)


def _clamped_unit_interval(raw: str) -> float:
    """Parse a float env value and clamp it into [0.0, 1.0].

    Unparseable values raise ValueError, matching the neighboring float()
    parses. Negative values clamp to 0.0 (the gate-disabled sentinel).
    Values above 1.0 clamp to 1.0 rather than falling back: the evidence
    components the gate compares against are themselves bounded at ~1.0, so
    a gate above 1.0 can never be exceeded and would only act as a uniform
    score dampener — clamping preserves the strongest gate the caller could
    have meant.
    """
    value = float(raw)
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


# Within-pool relevance gate (issue #130). When a query has tokens and a
# result's best query-topical evidence (max of the vector, keyword, metadata,
# and exact-match components) falls below this threshold, the
# query-independent components (importance, confidence, recency, tag overlap)
# are scaled by evidence / gate — a linear ramp, not a cliff — so
# high-importance but off-topic memories cannot ride query-independent score
# to the top of a tag-scoped pool. 0.0 (default) disables the gate and
# preserves legacy scoring exactly. The context bonus is never gated:
# `context_tags` remains the explicit soft-boost channel.
RECALL_RELEVANCE_GATE = _clamped_unit_interval(os.getenv("RECALL_RELEVANCE_GATE", "0.0"))


def _non_negative_or_zero(raw: str) -> float:
    """Parse a float env value, clamping negatives to 0.0 (the no-op value).

    Unparseable values raise ValueError, matching the neighboring float()
    parses. Unlike ``_positive_or_default``, 0.0 is a meaningful caller
    choice here (weight disabled), so negatives clamp instead of falling
    back to the default.
    """
    value = float(raw)
    return value if value > 0.0 else 0.0


# Relative-recency re-rank weight (issues #158/#159). When a recall request
# activates recency_bias, candidate timestamps are min-max normalized across
# the current candidate set and this weight times that relative recency is
# added to each final score. Inert unless the re-rank runs (RECALL_RECENCY_BIAS
# env or the per-request recency_bias param), so the default changes nothing.
SEARCH_WEIGHT_TEMPORAL = _non_negative_or_zero(os.getenv("SEARCH_WEIGHT_TEMPORAL", "0.1"))

# Default recency-bias mode for /recall: "off" (never re-rank), "on" (always),
# "auto" (only when the query expresses temporal intent — "latest", "current",
# ...). Per-request override via the recency_bias query param. Invalid values
# fall back to "off" so a typo cannot silently enable re-ranking.
_RECALL_RECENCY_BIAS_RAW = os.getenv("RECALL_RECENCY_BIAS", "off").strip().lower()
RECALL_RECENCY_BIAS = (
    _RECALL_RECENCY_BIAS_RAW if _RECALL_RECENCY_BIAS_RAW in {"auto", "on", "off"} else "off"
)

# API tokens
API_TOKEN = os.getenv("AUTOMEM_API_TOKEN")
ADMIN_TOKEN = os.getenv("ADMIN_API_TOKEN")
