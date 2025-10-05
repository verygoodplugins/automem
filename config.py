"""Configuration constants for AutoMem service."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
load_dotenv(Path.home() / ".config" / "automem" / ".env")

# Database configuration
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

# Memory types for classification
MEMORY_TYPES = {
    "Decision", "Pattern", "Preference", "Style",
    "Habit", "Insight", "Context", "Memory"  # Memory is the default base type
}

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

# Search stopwords
SEARCH_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "using",
    "have", "will", "your", "about", "after", "before", "when", "then",
    "than", "also", "just", "very", "more", "less", "over", "under",
}

# Entity extraction stopwords
ENTITY_STOPWORDS = {
    "you", "your", "yours", "whatever", "today", "tomorrow",
    "project", "projects", "office", "session", "meeting",
}

# Search weighting parameters (can be overridden via environment variables)
SEARCH_WEIGHT_VECTOR = float(os.getenv("SEARCH_WEIGHT_VECTOR", "0.35"))
SEARCH_WEIGHT_KEYWORD = float(os.getenv("SEARCH_WEIGHT_KEYWORD", "0.35"))
SEARCH_WEIGHT_TAG = float(os.getenv("SEARCH_WEIGHT_TAG", "0.15"))
SEARCH_WEIGHT_IMPORTANCE = float(os.getenv("SEARCH_WEIGHT_IMPORTANCE", "0.1"))
SEARCH_WEIGHT_CONFIDENCE = float(os.getenv("SEARCH_WEIGHT_CONFIDENCE", "0.05"))
SEARCH_WEIGHT_RECENCY = float(os.getenv("SEARCH_WEIGHT_RECENCY", "0.1"))
SEARCH_WEIGHT_EXACT = float(os.getenv("SEARCH_WEIGHT_EXACT", "0.15"))

# API authentication tokens
API_TOKEN = os.getenv("AUTOMEM_API_TOKEN")
ADMIN_TOKEN = os.getenv("ADMIN_API_TOKEN")
