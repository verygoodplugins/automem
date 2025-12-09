from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables before configuring the application.
load_dotenv()
load_dotenv(Path.home() / ".config" / "automem" / ".env")

# Qdrant / FalkorDB configuration
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "memories")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE") or os.getenv("QDRANT_VECTOR_SIZE", "3072"))
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

# Sync configuration (background sync worker)
SYNC_CHECK_INTERVAL_SECONDS = int(os.getenv("SYNC_CHECK_INTERVAL_SECONDS", "3600"))  # 1 hour
SYNC_AUTO_REPAIR = os.getenv("SYNC_AUTO_REPAIR", "true").lower() not in {"0", "false", "no"}

# Enrichment configuration
ENRICHMENT_MAX_ATTEMPTS = int(os.getenv("ENRICHMENT_MAX_ATTEMPTS", "3"))
ENRICHMENT_SIMILARITY_LIMIT = int(os.getenv("ENRICHMENT_SIMILARITY_LIMIT", "5"))
ENRICHMENT_SIMILARITY_THRESHOLD = float(os.getenv("ENRICHMENT_SIMILARITY_THRESHOLD", "0.8"))
ENRICHMENT_IDLE_SLEEP_SECONDS = float(os.getenv("ENRICHMENT_IDLE_SLEEP_SECONDS", "2"))
ENRICHMENT_FAILURE_BACKOFF_SECONDS = float(os.getenv("ENRICHMENT_FAILURE_BACKOFF_SECONDS", "5"))
ENRICHMENT_ENABLE_SUMMARIES = os.getenv("ENRICHMENT_ENABLE_SUMMARIES", "true").lower() not in {"0", "false", "no"}
ENRICHMENT_SPACY_MODEL = os.getenv("ENRICHMENT_SPACY_MODEL", "en_core_web_sm")

# Model configuration
# text-embedding-3-large (3072d): Better semantic precision, recommended for production
# text-embedding-3-small (768d): Cheaper, use VECTOR_SIZE=768 if switching
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
CLASSIFICATION_MODEL = os.getenv("CLASSIFICATION_MODEL", "gpt-4o-mini")

RECALL_RELATION_LIMIT = int(os.getenv("RECALL_RELATION_LIMIT", "5"))
RECALL_EXPANSION_LIMIT = int(os.getenv("RECALL_EXPANSION_LIMIT", "25"))

# Memory types for classification
MEMORY_TYPES = {
    "Decision", "Pattern", "Preference", "Style",
    "Habit", "Insight", "Context"
}

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

# Enhanced relationship types with their properties
RELATIONSHIP_TYPES = {
    # Original relationships
    "RELATES_TO": {"description": "General relationship"},
    "LEADS_TO": {"description": "Causal relationship"},
    "OCCURRED_BEFORE": {"description": "Temporal relationship"},
    # Frequently created by enrichment/temporal linking
    "SIMILAR_TO": {"description": "Semantic similarity", "properties": ["score", "updated_at"]},
    "PRECEDED_BY": {"description": "Prior in time", "properties": ["count", "updated_at"]},

    # New PKG relationships
    "PREFERS_OVER": {"description": "Preference relationship", "properties": ["context", "strength", "reason"]},
    "EXEMPLIFIES": {"description": "Pattern example", "properties": ["pattern_type", "confidence"]},
    "CONTRADICTS": {"description": "Conflicting information", "properties": ["resolution", "reason"]},
    "REINFORCES": {"description": "Strengthens pattern", "properties": ["strength", "observations"]},
    "INVALIDATED_BY": {"description": "Superseded information", "properties": ["reason", "timestamp"]},
    "EVOLVED_INTO": {"description": "Evolution of knowledge", "properties": ["confidence", "reason"]},
    "DERIVED_FROM": {"description": "Derived knowledge", "properties": ["transformation", "confidence"]},
    "PART_OF": {"description": "Hierarchical relationship", "properties": ["role", "context"]},

    # Discovered/creative connections used by consolidator/analysis
    "EXPLAINS": {"description": "Provides explanation for another memory", "properties": ["confidence", "updated_at"]},
    "SHARES_THEME": {"description": "Shares a common theme", "properties": ["similarity", "updated_at"]},
    "PARALLEL_CONTEXT": {"description": "Parallel events or contexts", "properties": ["confidence", "updated_at"]},
}

ALLOWED_RELATIONS = set(RELATIONSHIP_TYPES.keys())

# Search weighting parameters (can be overridden via environment variables)
SEARCH_WEIGHT_VECTOR = float(os.getenv("SEARCH_WEIGHT_VECTOR", "0.35"))
SEARCH_WEIGHT_KEYWORD = float(os.getenv("SEARCH_WEIGHT_KEYWORD", "0.35"))
SEARCH_WEIGHT_TAG = float(os.getenv("SEARCH_WEIGHT_TAG", "0.2"))
SEARCH_WEIGHT_IMPORTANCE = float(os.getenv("SEARCH_WEIGHT_IMPORTANCE", "0.1"))
SEARCH_WEIGHT_CONFIDENCE = float(os.getenv("SEARCH_WEIGHT_CONFIDENCE", "0.05"))
SEARCH_WEIGHT_RECENCY = float(os.getenv("SEARCH_WEIGHT_RECENCY", "0.1"))
SEARCH_WEIGHT_EXACT = float(os.getenv("SEARCH_WEIGHT_EXACT", "0.2"))
SEARCH_WEIGHT_RELATION = float(os.getenv("SEARCH_WEIGHT_RELATION", "0.25"))

# API tokens
API_TOKEN = os.getenv("AUTOMEM_API_TOKEN")
ADMIN_TOKEN = os.getenv("ADMIN_API_TOKEN")
