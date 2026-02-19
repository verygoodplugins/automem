"""AutoMem Memory Service API.

Provides a small Flask API that stores memories in FalkorDB and Qdrant.
This module focuses on being resilient: it validates requests, handles
transient outages, and degrades gracefully when one of the backing services
is unavailable.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from falkordb import FalkorDB
from flask import Blueprint, Flask, abort, jsonify, request
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models

try:
    from qdrant_client.http.exceptions import UnexpectedResponse
except ImportError:  # Allow tests to import without full qdrant client installed
    UnexpectedResponse = Exception  # type: ignore[misc,assignment]

try:  # Allow tests to import without full qdrant client installed
    from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams
except Exception:  # pragma: no cover - degraded import path
    try:
        from qdrant_client.http import models as _qmodels

        Distance = getattr(_qmodels, "Distance", None)
        PointStruct = getattr(_qmodels, "PointStruct", None)
        VectorParams = getattr(_qmodels, "VectorParams", None)
        PayloadSchemaType = getattr(_qmodels, "PayloadSchemaType", None)
    except Exception:
        Distance = PointStruct = VectorParams = None
        PayloadSchemaType = None

# Provide a simple PointStruct shim for tests/environments lacking qdrant models
if PointStruct is None:  # pragma: no cover - test shim

    class PointStruct:  # type: ignore[no-redef]
        def __init__(self, id: str, vector: List[float], payload: Dict[str, Any]):
            self.id = id
            self.vector = vector
            self.payload = payload


from werkzeug.exceptions import HTTPException

from consolidation import ConsolidationScheduler, MemoryConsolidator

# Make OpenAI import optional to allow running without it
try:
    from openai import OpenAI  # type: ignore
except ImportError:
    OpenAI = None  # type: ignore

# SSE streaming for real-time observability
from automem.api.auth_helpers import extract_api_token as _extract_api_token_helper
from automem.api.auth_helpers import require_admin_token as _require_admin_token_helper
from automem.api.auth_helpers import require_api_token as _require_api_token_helper
from automem.api.stream import create_stream_blueprint, emit_event
from automem.embedding.runtime_helpers import coerce_embedding as _coerce_embedding_value
from automem.embedding.runtime_helpers import coerce_importance as _coerce_importance_value
from automem.embedding.runtime_helpers import (
    generate_placeholder_embedding as _generate_placeholder_embedding_value,
)
from automem.embedding.runtime_helpers import (
    generate_real_embedding as _generate_real_embedding_value,
)
from automem.embedding.runtime_helpers import (
    generate_real_embeddings_batch as _generate_real_embeddings_batch_value,
)
from automem.embedding.runtime_helpers import normalize_tags as _normalize_tags_value
from automem.service_state import EnrichmentJob, EnrichmentStats, ServiceState

# Environment is loaded by automem.config

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
    stdout_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
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

# Legacy blueprint placeholders for deprecated route definitions below.
# These are not registered with the app and are safe to keep until full removal.
admin_bp = Blueprint("admin_legacy", __name__)
memory_bp = Blueprint("memory_legacy", __name__)
recall_bp = Blueprint("recall_legacy", __name__)
consolidation_bp = Blueprint("consolidation_legacy", __name__)

# Import canonical configuration constants
from automem.config import (
    ADMIN_TOKEN,
    ALLOWED_RELATIONS,
    API_TOKEN,
    CLASSIFICATION_MODEL,
    COLLECTION_NAME,
    CONSOLIDATION_ARCHIVE_THRESHOLD,
    CONSOLIDATION_CLUSTER_INTERVAL_SECONDS,
    CONSOLIDATION_CONTROL_LABEL,
    CONSOLIDATION_CONTROL_NODE_ID,
    CONSOLIDATION_CREATIVE_INTERVAL_SECONDS,
    CONSOLIDATION_DECAY_IMPORTANCE_THRESHOLD,
    CONSOLIDATION_DECAY_INTERVAL_SECONDS,
    CONSOLIDATION_DELETE_THRESHOLD,
    CONSOLIDATION_FORGET_INTERVAL_SECONDS,
    CONSOLIDATION_GRACE_PERIOD_DAYS,
    CONSOLIDATION_HISTORY_LIMIT,
    CONSOLIDATION_IMPORTANCE_PROTECTION_THRESHOLD,
    CONSOLIDATION_PROTECTED_TYPES,
    CONSOLIDATION_RUN_LABEL,
    CONSOLIDATION_TASK_FIELDS,
    CONSOLIDATION_TICK_SECONDS,
    EMBEDDING_MODEL,
    ENRICHMENT_ENABLE_SUMMARIES,
    ENRICHMENT_FAILURE_BACKOFF_SECONDS,
    ENRICHMENT_IDLE_SLEEP_SECONDS,
    ENRICHMENT_MAX_ATTEMPTS,
    ENRICHMENT_SIMILARITY_LIMIT,
    ENRICHMENT_SIMILARITY_THRESHOLD,
    ENRICHMENT_SPACY_MODEL,
    FALKORDB_PORT,
    GRAPH_NAME,
    JIT_ENRICHMENT_ENABLED,
    MEMORY_TYPES,
    RECALL_EXPANSION_LIMIT,
    RECALL_RELATION_LIMIT,
    RELATIONSHIP_TYPES,
    SEARCH_WEIGHT_CONFIDENCE,
    SEARCH_WEIGHT_EXACT,
    SEARCH_WEIGHT_IMPORTANCE,
    SEARCH_WEIGHT_KEYWORD,
    SEARCH_WEIGHT_RECENCY,
    SEARCH_WEIGHT_TAG,
    SEARCH_WEIGHT_VECTOR,
    SYNC_AUTO_REPAIR,
    SYNC_CHECK_INTERVAL_SECONDS,
    TYPE_ALIASES,
    VECTOR_SIZE,
    normalize_memory_type,
)
from automem.search.runtime_recall_helpers import (
    _graph_keyword_search,
    _result_passes_filters,
    _vector_filter_only_tag_search,
    _vector_search,
    configure_recall_helpers,
)
from automem.stores.graph_store import _build_graph_tag_predicate
from automem.stores.vector_store import _build_qdrant_tag_filter
from automem.utils.entity_extraction import (
    _slugify,
    configure_entity_extraction,
    extract_entities,
    generate_summary,
)
from automem.utils.graph import _serialize_node, _summarize_relation_node
from automem.utils.scoring import _compute_metadata_score, _parse_metadata_field
from automem.utils.tags import (
    _compute_tag_prefixes,
    _expand_tag_prefixes,
    _normalize_tag_list,
    _prepare_tag_filters,
)

# Shared utils and helpers
from automem.utils.time import (
    _normalize_timestamp,
    _parse_iso_datetime,
    _parse_time_expression,
    utc_now,
)
from automem.utils.validation import get_effective_vector_size, validate_vector_dimensions

# Embedding batching configuration
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "20"))
EMBEDDING_BATCH_TIMEOUT_SECONDS = float(os.getenv("EMBEDDING_BATCH_TIMEOUT_SECONDS", "2.0"))

"""Note: default types/relations/weights are imported from automem.config"""

# Keyword/NER constants come from automem.utils.text if available
SEARCH_STOPWORDS: Set[str] = set()
ENTITY_STOPWORDS: Set[str] = set()
ENTITY_BLOCKLIST: Set[str] = set()

# Search weights are imported from automem.config

# Maximum number of results returned by /recall
RECALL_MAX_LIMIT = int(os.getenv("RECALL_MAX_LIMIT", "100"))

# API tokens are imported from automem.config


try:
    from automem.utils.text import ENTITY_BLOCKLIST as _AM_ENTITY_BLOCKLIST
    from automem.utils.text import ENTITY_STOPWORDS as _AM_ENTITY_STOPWORDS
    from automem.utils.text import SEARCH_STOPWORDS as _AM_SEARCH_STOPWORDS
    from automem.utils.text import _extract_keywords as _AM_extract_keywords

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


configure_entity_extraction(
    search_stopwords=SEARCH_STOPWORDS,
    entity_stopwords=ENTITY_STOPWORDS,
    entity_blocklist=ENTITY_BLOCKLIST,
    spacy_model=ENRICHMENT_SPACY_MODEL,
)


class MemoryClassifier:
    """Classifies memories into specific types based on content patterns."""

    PATTERNS = {
        "Decision": [
            r"decided to",
            r"chose (\w+) over",
            r"going with",
            r"picked",
            r"selected",
            r"will use",
            r"choosing",
            r"opted for",
        ],
        "Pattern": [
            r"usually",
            r"typically",
            r"tend to",
            r"pattern i noticed",
            r"often",
            r"frequently",
            r"regularly",
            r"consistently",
        ],
        "Preference": [
            r"prefer",
            r"like.*better",
            r"favorite",
            r"always use",
            r"rather than",
            r"instead of",
            r"favor",
        ],
        "Style": [
            r"wrote.*in.*style",
            r"communicated",
            r"responded to",
            r"formatted as",
            r"using.*tone",
            r"expressed as",
        ],
        "Habit": [
            r"always",
            r"every time",
            r"habitually",
            r"routine",
            r"daily",
            r"weekly",
            r"monthly",
        ],
        "Insight": [
            r"realized",
            r"discovered",
            r"learned that",
            r"understood",
            r"figured out",
            r"insight",
            r"revelation",
        ],
        "Context": [
            r"during",
            r"while working on",
            r"in the context of",
            r"when",
            r"at the time",
            r"situation was",
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
            # Build model-specific params (o-series and gpt-5 don't support temperature)
            extra_params: dict = {}
            if CLASSIFICATION_MODEL.startswith(("o", "gpt-5")):
                extra_params["max_completion_tokens"] = 50
            else:
                extra_params["max_tokens"] = 50
                extra_params["temperature"] = 0.3
            response = state.openai_client.chat.completions.create(
                model=CLASSIFICATION_MODEL,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": content[:1000]},  # Limit to 1000 chars
                ],
                response_format={"type": "json_object"},
                **extra_params,
            )

            result = json.loads(response.choices[0].message.content)
            raw_type = result.get("type", "Memory")
            confidence = float(result.get("confidence", 0.7))

            # Normalize type (handles aliases and case variations)
            memory_type, was_normalized = normalize_memory_type(raw_type)
            if not memory_type:
                logger.warning("LLM returned unmappable type '%s', using Context", raw_type)
                return "Context", 0.5

            if was_normalized and memory_type != raw_type:
                logger.debug("LLM type normalized '%s' -> '%s'", raw_type, memory_type)

            logger.info("LLM classified as %s (confidence: %.2f)", memory_type, confidence)
            return memory_type, confidence

        except Exception as exc:
            logger.warning("LLM classification failed: %s", exc)
            return None


memory_classifier = MemoryClassifier()


state = ServiceState()


def _extract_api_token() -> Optional[str]:
    return _extract_api_token_helper(request, API_TOKEN)


def get_openai_client() -> Optional[OpenAI]:
    return state.openai_client


def _require_admin_token() -> None:
    _require_admin_token_helper(
        request_obj=request,
        admin_token=ADMIN_TOKEN,
        abort_fn=abort,
    )


@app.before_request
def require_api_token() -> None:
    _require_api_token_helper(
        request_obj=request,
        api_token=API_TOKEN,
        extract_api_token_fn=_extract_api_token,
        abort_fn=abort,
    )


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
    1. Voyage API (if VOYAGE_API_KEY is set)
    2. OpenAI API (if OPENAI_API_KEY is set)
    3. Ollama local server (if configured)
    4. Local fastembed model (no API key needed)
    5. Placeholder hash-based embeddings (fallback)

    Can be controlled via EMBEDDING_PROVIDER env var:
    - "auto" (default): Try Voyage, then OpenAI, then Ollama, then fastembed, then placeholder
    - "voyage": Use Voyage only, fail if unavailable
    - "openai": Use OpenAI only, fail if unavailable
    - "local": Use fastembed only, fail if unavailable
    - "ollama": Use Ollama only, fail if unavailable
    - "placeholder": Use placeholder embeddings
    """
    if state.embedding_provider is not None:
        return

    provider_config = (os.getenv("EMBEDDING_PROVIDER", "auto") or "auto").strip().lower()
    # Use effective dimension (auto-detected from existing collection or config default).
    # If Qdrant hasn't set it (or config was changed in-process), align to VECTOR_SIZE.
    if state.qdrant is None and state.effective_vector_size != VECTOR_SIZE:
        state.effective_vector_size = VECTOR_SIZE
    vector_size = state.effective_vector_size

    # Explicit provider selection
    if provider_config == "voyage":
        api_key = os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError("EMBEDDING_PROVIDER=voyage but VOYAGE_API_KEY not set")
        try:
            from automem.embedding.voyage import VoyageEmbeddingProvider

            voyage_model = os.getenv("VOYAGE_MODEL", "voyage-4")
            state.embedding_provider = VoyageEmbeddingProvider(
                api_key=api_key, model=voyage_model, dimension=vector_size
            )
            logger.info("Embedding provider: %s", state.embedding_provider.provider_name())
            return
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Voyage provider: {e}") from e

    elif provider_config == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("EMBEDDING_PROVIDER=openai but OPENAI_API_KEY not set")
        openai_base_url = (os.getenv("OPENAI_BASE_URL") or "").strip() or None
        try:
            from automem.embedding.openai import OpenAIEmbeddingProvider

            state.embedding_provider = OpenAIEmbeddingProvider(
                api_key=api_key,
                model=EMBEDDING_MODEL,
                dimension=vector_size,
                base_url=openai_base_url,
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

    elif provider_config == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "nomic-embed-text")
        try:
            timeout = float(os.getenv("OLLAMA_TIMEOUT", "30"))
            max_retries = int(os.getenv("OLLAMA_MAX_RETRIES", "2"))
        except ValueError as ve:
            raise RuntimeError(f"Invalid OLLAMA_TIMEOUT or OLLAMA_MAX_RETRIES value: {ve}") from ve
        try:
            from automem.embedding.ollama import OllamaEmbeddingProvider

            state.embedding_provider = OllamaEmbeddingProvider(
                base_url=base_url,
                model=model,
                dimension=vector_size,
                timeout=timeout,
                max_retries=max_retries,
            )
            logger.info("Embedding provider: %s", state.embedding_provider.provider_name())
            return
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Ollama provider: {e}") from e

    elif provider_config == "placeholder":
        from automem.embedding.placeholder import PlaceholderEmbeddingProvider

        state.embedding_provider = PlaceholderEmbeddingProvider(dimension=vector_size)
        logger.info("Embedding provider: %s", state.embedding_provider.provider_name())
        return

    # Auto-selection: Try Voyage → OpenAI → Ollama → fastembed → placeholder
    if provider_config == "auto":
        # Try Voyage first (preferred)
        voyage_key = os.getenv("VOYAGE_API_KEY")
        if voyage_key:
            try:
                from automem.embedding.voyage import VoyageEmbeddingProvider

                voyage_model = os.getenv("VOYAGE_MODEL", "voyage-4")
                state.embedding_provider = VoyageEmbeddingProvider(
                    api_key=voyage_key, model=voyage_model, dimension=vector_size
                )
                logger.info(
                    "Embedding provider (auto-selected): %s",
                    state.embedding_provider.provider_name(),
                )
                return
            except Exception as e:
                logger.warning("Failed to initialize Voyage provider, trying OpenAI: %s", str(e))

        # Try OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                from automem.embedding.openai import OpenAIEmbeddingProvider

                openai_base_url = (os.getenv("OPENAI_BASE_URL") or "").strip() or None
                state.embedding_provider = OpenAIEmbeddingProvider(
                    api_key=api_key,
                    model=EMBEDDING_MODEL,
                    dimension=vector_size,
                    base_url=openai_base_url,
                )
                logger.info(
                    "Embedding provider (auto-selected): %s",
                    state.embedding_provider.provider_name(),
                )
                return
            except Exception as e:
                logger.warning(
                    "Failed to initialize OpenAI provider, trying local model: %s", str(e)
                )

        ollama_base_url = os.getenv("OLLAMA_BASE_URL")
        ollama_model = os.getenv("OLLAMA_MODEL")
        if ollama_base_url or ollama_model:
            try:
                from automem.embedding.ollama import OllamaEmbeddingProvider

                base_url = ollama_base_url or "http://localhost:11434"
                model = ollama_model or "nomic-embed-text"
                try:
                    timeout = float(os.getenv("OLLAMA_TIMEOUT", "30"))
                    max_retries = int(os.getenv("OLLAMA_MAX_RETRIES", "2"))
                except ValueError:
                    logger.warning("Invalid OLLAMA_TIMEOUT or OLLAMA_MAX_RETRIES, using defaults")
                    timeout = 30.0
                    max_retries = 2
                state.embedding_provider = OllamaEmbeddingProvider(
                    base_url=base_url,
                    model=model,
                    dimension=vector_size,
                    timeout=timeout,
                    max_retries=max_retries,
                )
                logger.info(
                    "Embedding provider (auto-selected): %s",
                    state.embedding_provider.provider_name(),
                )
                return
            except Exception as e:
                logger.warning(
                    "Failed to initialize Ollama provider, trying local model: %s", str(e)
                )

        # Try local fastembed
        try:
            from automem.embedding.fastembed import FastEmbedProvider

            state.embedding_provider = FastEmbedProvider(dimension=vector_size)
            logger.info(
                "Embedding provider (auto-selected): %s", state.embedding_provider.provider_name()
            )
            return
        except Exception as e:
            logger.warning("Failed to initialize fastembed provider, using placeholder: %s", str(e))

        # Fallback to placeholder
        from automem.embedding.placeholder import PlaceholderEmbeddingProvider

        state.embedding_provider = PlaceholderEmbeddingProvider(dimension=vector_size)
        logger.warning(
            "Using placeholder embeddings (no semantic search). "
            "Install fastembed or set VOYAGE_API_KEY/OPENAI_API_KEY for semantic embeddings."
        )
        logger.info(
            "Embedding provider (auto-selected): %s", state.embedding_provider.provider_name()
        )
        return

    # Invalid config
    raise ValueError(
        f"Invalid EMBEDDING_PROVIDER={provider_config}. "
        f"Valid options: auto, voyage, openai, local, ollama, placeholder"
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

        # Only pass authentication if password is actually configured
        connection_params = {
            "host": host,
            "port": FALKORDB_PORT,
        }
        if password:
            connection_params["password"] = password
            connection_params["username"] = "default"

        state.falkordb = FalkorDB(**connection_params)
        state.memory_graph = state.falkordb.select_graph(GRAPH_NAME)
        logger.info(
            "FalkorDB connection established (auth: %s)", "enabled" if password else "disabled"
        )
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
    except ValueError:
        # Surface migration guidance (e.g., vector dimension mismatch) and halt startup
        state.qdrant = None
        raise
    except Exception:  # pragma: no cover - log full stack trace in production
        logger.exception("Failed to initialize Qdrant client")
        state.qdrant = None


def _ensure_qdrant_collection() -> None:
    """Create the Qdrant collection if it does not already exist."""
    if state.qdrant is None:
        return

    try:
        # Auto-detect vector dimension from existing collection (backwards compatibility)
        # This ensures users with 768d embeddings aren't broken by default change to 3072d
        effective_dim, source = get_effective_vector_size(state.qdrant)
        state.effective_vector_size = effective_dim

        if source == "collection":
            logger.info(
                "Using existing collection dimension: %dd (config default: %dd)",
                effective_dim,
                VECTOR_SIZE,
            )
        else:
            logger.info("Using configured vector dimension: %dd", effective_dim)

        collections = state.qdrant.get_collections()
        existing = {collection.name for collection in collections.collections}
        if COLLECTION_NAME not in existing:
            logger.info(
                "Creating Qdrant collection '%s' with %dd vectors", COLLECTION_NAME, effective_dim
            )
            state.qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=effective_dim, distance=Distance.COSINE),
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
    except ValueError:
        # Bubble up migration guidance so the service fails fast instead of silently degrading
        raise
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


# ---------------------------------------------------------------------------
# Access Tracking (updates last_accessed on recall)
# ---------------------------------------------------------------------------


def update_last_accessed(memory_ids: List[str]) -> None:
    """Update last_accessed timestamp for retrieved memories (direct, synchronous)."""
    if not memory_ids:
        return

    graph = get_memory_graph()
    if graph is None:
        return

    timestamp = utc_now()
    try:
        graph.query(
            """
            UNWIND $ids AS mid
            MATCH (m:Memory {id: mid})
            SET m.last_accessed = $ts
            """,
            {"ids": memory_ids, "ts": timestamp},
        )
        logger.debug("Updated last_accessed for %d memories", len(memory_ids))
    except Exception:
        logger.exception("Failed to update last_accessed for memories")


def _load_control_record(graph: Any) -> Dict[str, Any]:
    """Fetch or create the consolidation control node."""
    bootstrap_timestamp = utc_now()
    bootstrap_fields = sorted(set(CONSOLIDATION_TASK_FIELDS.values()))
    bootstrap_set_clause = ",\n                ".join(
        f"c.{field} = coalesce(c.{field}, $now)" for field in bootstrap_fields
    )
    try:
        result = graph.query(
            f"""
            MERGE (c:{CONSOLIDATION_CONTROL_LABEL} {{id: $id}})
            SET {bootstrap_set_clause}
            RETURN c
            """,
            {"id": CONSOLIDATION_CONTROL_NODE_ID, "now": bootstrap_timestamp},
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
        "decay": timedelta(seconds=CONSOLIDATION_DECAY_INTERVAL_SECONDS),
        "creative": timedelta(seconds=CONSOLIDATION_CREATIVE_INTERVAL_SECONDS),
        "cluster": timedelta(seconds=CONSOLIDATION_CLUSTER_INTERVAL_SECONDS),
        "forget": timedelta(seconds=CONSOLIDATION_FORGET_INTERVAL_SECONDS),
    }

    for task, interval in overrides.items():
        if task in scheduler.schedules:
            scheduler.schedules[task]["interval"] = interval


def _tasks_for_mode(mode: str) -> List[str]:
    """Map a consolidation mode to its task identifiers."""
    if mode == "full":
        return ["decay", "creative", "cluster", "forget", "full"]
    if mode in CONSOLIDATION_TASK_FIELDS:
        return [mode]
    return [mode]


def _persist_consolidation_run(graph: Any, result: Dict[str, Any]) -> None:
    """Record consolidation outcomes and update scheduler metadata."""
    mode = result.get("mode", "unknown")
    completed_at = result.get("completed_at") or utc_now()
    started_at = result.get("started_at") or completed_at
    success = bool(result.get("success"))
    dry_run = bool(result.get("dry_run"))

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
    consolidator = _build_consolidator_from_config(graph, vector_store)
    scheduler = ConsolidationScheduler(consolidator)
    _apply_scheduler_overrides(scheduler)

    control = _load_control_record(graph)
    for task, field in CONSOLIDATION_TASK_FIELDS.items():
        iso_value = control.get(field)
        last_run = _parse_iso_datetime(iso_value)
        if last_run and task in scheduler.schedules:
            scheduler.schedules[task]["last_run"] = last_run

    return scheduler


def _build_consolidator_from_config(graph: Any, vector_store: Any) -> MemoryConsolidator:
    return MemoryConsolidator(
        graph,
        vector_store,
        delete_threshold=CONSOLIDATION_DELETE_THRESHOLD,
        archive_threshold=CONSOLIDATION_ARCHIVE_THRESHOLD,
        grace_period_days=CONSOLIDATION_GRACE_PERIOD_DAYS,
        importance_protection_threshold=CONSOLIDATION_IMPORTANCE_PROTECTION_THRESHOLD,
        protected_types=set(CONSOLIDATION_PROTECTED_TYPES),
    )


def _run_consolidation_tick() -> None:
    graph = get_memory_graph()
    if graph is None:
        return

    scheduler = _build_scheduler_from_graph(graph)
    if scheduler is None:
        return

    try:
        tick_start = time.perf_counter()
        results = scheduler.run_scheduled_tasks(
            decay_threshold=CONSOLIDATION_DECAY_IMPORTANCE_THRESHOLD
        )
        for result in results:
            _persist_consolidation_run(graph, result)

            # Emit SSE event for real-time monitoring
            task_type = result.get("mode", "unknown")
            steps = result.get("steps", {})
            affected_count = 0

            # Count affected memories from each step
            if "decay" in steps:
                affected_count += steps["decay"].get("updated", 0)
            if "creative" in steps:
                affected_count += steps["creative"].get("created", 0)
            if "cluster" in steps:
                affected_count += steps["cluster"].get("meta_memories_created", 0)
            if "forget" in steps:
                affected_count += steps["forget"].get("archived", 0)
                affected_count += steps["forget"].get("deleted", 0)

            elapsed_ms = int((time.perf_counter() - tick_start) * 1000)
            next_runs = scheduler.get_next_runs()

            emit_event(
                "consolidation.run",
                {
                    "task_type": task_type,
                    "affected_count": affected_count,
                    "elapsed_ms": elapsed_ms,
                    "success": result.get("success", False),
                    "next_scheduled": next_runs.get(task_type, "unknown"),
                    "steps": list(steps.keys()),
                },
                utc_now,
            )
    except Exception:
        logger.exception("Consolidation scheduler tick failed")


def consolidation_worker() -> None:
    """Background loop that triggers consolidation tasks."""
    logger.info("Consolidation scheduler thread started")
    while state.consolidation_stop_event and not state.consolidation_stop_event.wait(
        CONSOLIDATION_TICK_SECONDS
    ):
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
                job: EnrichmentJob = state.enrichment_queue.get(
                    timeout=ENRICHMENT_IDLE_SLEEP_SECONDS
                )
            except Empty:
                continue

            with state.enrichment_lock:
                state.enrichment_pending.discard(job.memory_id)
                state.enrichment_inflight.add(job.memory_id)

            enrich_start = time.perf_counter()
            emit_event(
                "enrichment.start",
                {
                    "memory_id": job.memory_id,
                    "attempt": job.attempt + 1,
                },
                utc_now,
            )

            try:
                processed = enrich_memory(job.memory_id, forced=job.forced)
                state.enrichment_stats.record_success(job.memory_id)
                elapsed_ms = int((time.perf_counter() - enrich_start) * 1000)
                emit_event(
                    "enrichment.complete",
                    {
                        "memory_id": job.memory_id,
                        "success": True,
                        "elapsed_ms": elapsed_ms,
                        "skipped": not processed,
                    },
                    utc_now,
                )
                if not processed:
                    logger.debug("Enrichment skipped for %s (already processed)", job.memory_id)
            except Exception as exc:  # pragma: no cover - background thread
                state.enrichment_stats.record_failure(str(exc))
                elapsed_ms = int((time.perf_counter() - enrich_start) * 1000)
                emit_event(
                    "enrichment.failed",
                    {
                        "memory_id": job.memory_id,
                        "error": str(exc)[:100],
                        "attempt": job.attempt + 1,
                        "elapsed_ms": elapsed_ms,
                        "will_retry": job.attempt + 1 < ENRICHMENT_MAX_ATTEMPTS,
                    },
                    utc_now,
                )
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
                        "relevance_score": properties.get("relevance_score"),
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


# ---------------------------------------------------------------------------
# Background Sync Worker
# ---------------------------------------------------------------------------


def init_sync_worker() -> None:
    """Initialize the background sync worker if auto-repair is enabled."""
    if not SYNC_AUTO_REPAIR:
        logger.info("Sync auto-repair disabled (SYNC_AUTO_REPAIR=false)")
        return

    if state.sync_thread is not None:
        return

    state.sync_stop_event = Event()
    state.sync_thread = Thread(target=sync_worker, daemon=True)
    state.sync_thread.start()
    logger.info("Sync worker initialized (interval: %ds)", SYNC_CHECK_INTERVAL_SECONDS)


def sync_worker() -> None:
    """Background worker that detects and repairs FalkorDB/Qdrant sync drift.

    This is non-destructive: only adds missing embeddings, never removes existing ones.
    """
    while not state.sync_stop_event.is_set():
        try:
            # Wait for the interval (or until stop event)
            if state.sync_stop_event.wait(timeout=SYNC_CHECK_INTERVAL_SECONDS):
                break  # Stop event set

            _run_sync_check()

        except Exception:
            logger.exception("Error in sync worker")
            # Sleep briefly on error before retrying
            time.sleep(60)


def _run_sync_check() -> None:
    """Check for sync drift and repair if needed."""
    graph = get_memory_graph()
    qdrant = get_qdrant_client()

    if graph is None or qdrant is None:
        logger.debug("Sync check skipped: services unavailable")
        return

    try:
        # Get memory IDs from FalkorDB
        falkor_result = graph.query("MATCH (m:Memory) RETURN m.id AS id")
        falkor_ids: Set[str] = set()
        for row in getattr(falkor_result, "result_set", []) or []:
            if row[0]:
                falkor_ids.add(str(row[0]))

        # Get point IDs from Qdrant
        qdrant_ids: Set[str] = set()
        offset = None
        while True:
            result = qdrant.scroll(
                collection_name=COLLECTION_NAME,
                limit=1000,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            points, next_offset = result
            for point in points:
                qdrant_ids.add(str(point.id))
            if next_offset is None:
                break
            offset = next_offset

        # Check for drift
        missing_ids = falkor_ids - qdrant_ids

        state.sync_last_run = utc_now()
        state.sync_last_result = {
            "falkordb_count": len(falkor_ids),
            "qdrant_count": len(qdrant_ids),
            "missing_count": len(missing_ids),
        }

        if not missing_ids:
            logger.debug("Sync check: no drift detected (%d memories)", len(falkor_ids))
            return

        logger.warning(
            "Sync drift detected: %d memories missing from Qdrant (will auto-repair)",
            len(missing_ids),
        )

        # Queue missing memories for embedding
        for memory_id in missing_ids:
            # Fetch content to queue
            mem_result = graph.query(
                "MATCH (m:Memory {id: $id}) RETURN m.content", {"id": memory_id}
            )
            if getattr(mem_result, "result_set", None):
                content = mem_result.result_set[0][0]
                if content:
                    enqueue_embedding(memory_id, content)

        logger.info("Queued %d memories for sync repair", len(missing_ids))

    except Exception:
        logger.exception("Sync check failed")


def jit_enrich_lightweight(memory_id: str, properties: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Run lightweight JIT enrichment inline during recall.

    Extracts entities, generates summary, and updates tags — the cheap parts
    of enrichment (~50ms).  Does NOT set ``processed=true`` so the async worker
    still runs the expensive operations (temporal links, patterns, neighbors).

    Returns the updated *properties* dict on success, or ``None`` on failure.
    """
    graph = get_memory_graph()
    if graph is None:
        return None

    # Check canonical enrichment state to avoid re-enriching Qdrant-sourced memories
    try:
        check = graph.query(
            "MATCH (m:Memory {id: $id}) RETURN m.enriched, m.processed",
            {"id": memory_id},
        )
        if getattr(check, "result_set", None):
            row = check.result_set[0]
            if row[0] or row[1]:  # already enriched or processed — skip
                logger.debug("JIT skipped for %s (already enriched/processed)", memory_id)
                return None
    except Exception as exc:  # noqa: BLE001 - best-effort guard, proceed if check fails
        logger.debug("JIT state-check failed for %s: %s", memory_id, exc)

    content = properties.get("content", "") or ""
    if not content:
        return None

    # --- cheap enrichment steps -------------------------------------------
    entities = extract_entities(content)

    tags = list(dict.fromkeys(_normalize_tag_list(properties.get("tags"))))
    entity_tags: Set[str] = set()

    metadata_raw = properties.get("metadata")
    metadata = _parse_metadata_field(metadata_raw) or {}
    if not isinstance(metadata, dict):
        metadata = {"_raw_metadata": metadata}

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

    tag_prefixes = _compute_tag_prefixes(tags)

    summary = None
    if ENRICHMENT_ENABLE_SUMMARIES:
        summary = generate_summary(content, properties.get("summary"))
    else:
        summary = properties.get("summary")

    enriched_at = utc_now()

    # Record that JIT ran (the full worker will overwrite this later)
    enrichment_meta = metadata.setdefault("enrichment", {})
    if not isinstance(enrichment_meta, dict):
        enrichment_meta = {}
    enrichment_meta["jit"] = True
    enrichment_meta["jit_at"] = enriched_at
    metadata["enrichment"] = enrichment_meta

    # --- persist to graph (do NOT set processed=true) ---------------------
    update_payload = {
        "id": memory_id,
        "metadata": json.dumps(metadata, default=str),
        "tags": tags,
        "tag_prefixes": tag_prefixes,
        "summary": summary,
        "enriched_at": enriched_at,
    }
    try:
        graph.query(
            """
            MATCH (m:Memory {id: $id})
            SET m.metadata = $metadata,
                m.tags = $tags,
                m.tag_prefixes = $tag_prefixes,
                m.summary = $summary,
                m.enriched = true,
                m.enriched_at = $enriched_at
            """,
            update_payload,
        )
    except Exception:
        logger.exception("JIT enrichment graph update failed for %s", memory_id)
        return None

    # --- sync to Qdrant payload -------------------------------------------
    qdrant_client = get_qdrant_client()
    if qdrant_client is not None:
        try:
            qdrant_client.set_payload(
                collection_name=COLLECTION_NAME,
                points=[memory_id],
                payload={
                    "tags": tags,
                    "tag_prefixes": tag_prefixes,
                    "metadata": metadata,
                },
            )
        except Exception as exc:  # noqa: BLE001 - Qdrant client raises multiple exception types
            logger.debug("JIT Qdrant payload sync skipped for %s: %s", memory_id, exc)

    # --- return updated properties for the current recall response --------
    updated = dict(properties)
    updated["tags"] = tags
    updated["tag_prefixes"] = tag_prefixes
    updated["metadata"] = metadata
    updated["summary"] = summary
    updated["enriched"] = True
    updated["enriched_at"] = enriched_at

    logger.debug("JIT-enriched memory %s (entities=%s)", memory_id, bool(entities))
    return updated


def enrich_memory(memory_id: str, *, forced: bool = False) -> bool:
    """Enrich a memory with relationships, patterns, and entity extraction."""
    graph = get_memory_graph()
    if graph is None:
        raise RuntimeError("FalkorDB unavailable for enrichment")

    result = graph.query("MATCH (m:Memory {id: $id}) RETURN m", {"id": memory_id})

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

    tag_prefixes = _compute_tag_prefixes(tags)

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
                {"id": neighbour_id, "score": score} for neighbour_id, score in semantic_neighbors
            ],
        }
    )
    metadata["enrichment"] = enrichment_meta

    update_payload = {
        "id": memory_id,
        "metadata": json.dumps(metadata, default=str),
        "tags": tags,
        "tag_prefixes": tag_prefixes,
        "summary": summary,
        "enriched_at": utc_now(),
    }

    graph.query(
        """
        MATCH (m:Memory {id: $id})
        SET m.metadata = $metadata,
            m.tags = $tags,
            m.tag_prefixes = $tag_prefixes,
            m.summary = $summary,
            m.enriched = true,
            m.enriched_at = $enriched_at,
            m.processed = true
        """,
        update_payload,
    )

    qdrant_client = get_qdrant_client()
    if qdrant_client is not None:
        try:
            qdrant_client.set_payload(
                collection_name=COLLECTION_NAME,
                points=[memory_id],
                payload={
                    "tags": tags,
                    "tag_prefixes": tag_prefixes,
                    "metadata": metadata,
                },
            )
        except UnexpectedResponse as exc:
            # 404 means embedding upload hasn't completed yet (race condition)
            if exc.status_code == 404:
                logger.debug(
                    "Qdrant payload sync skipped - point not yet uploaded: %s", memory_id[:8]
                )
            else:
                logger.warning("Qdrant payload sync failed (%d): %s", exc.status_code, memory_id)
        except Exception:
            logger.exception("Failed to sync Qdrant payload for enriched memory %s", memory_id)

    logger.debug(
        "Enriched memory %s (temporal=%s, patterns=%s, semantic=%s)",
        memory_id,
        temporal_links,
        pattern_info,
        len(semantic_neighbors),
    )

    return True


def _temporal_cutoff() -> str:
    """Return an ISO timestamp 7 days ago to bound temporal queries."""
    return (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()


def find_temporal_relationships(graph: Any, memory_id: str, limit: int = 5) -> int:
    """Find and create temporal relationships with recent memories."""
    created = 0
    try:
        result = graph.query(
            """
            MATCH (m1:Memory {id: $id})
            WITH m1, m1.timestamp AS ts
            WHERE ts IS NOT NULL
            MATCH (m2:Memory)
            WHERE m2.id <> $id
                AND m2.timestamp IS NOT NULL
                AND m2.timestamp < ts
                AND m2.timestamp > $cutoff
            RETURN m2.id
            ORDER BY m2.timestamp DESC
            LIMIT $limit
            """,
            {"id": memory_id, "limit": limit, "cutoff": _temporal_cutoff()},
            timeout=5000,
        )

        timestamp = utc_now()
        for (related_id,) in result.result_set:
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
            {"type": memory_type, "id": memory_id},
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
            description = f"Pattern across {similar_count + 1} {memory_type} memories" + (
                f" highlighting {', '.join(top_terms)}" if top_terms else ""
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


# Legacy route implementations retained for reference only.
# NOTE: These are bound to unregistered "*_legacy" blueprints and are not active.
# Active endpoints live in automem/api/* blueprints registered above.


@admin_bp.route("/admin/reembed", methods=["POST"])
def admin_reembed() -> Any:
    """Legacy admin handler; route now provided by automem.api.admin blueprint."""
    abort(410, description="/admin/reembed moved to blueprint")


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


@memory_bp.route("/memory", methods=["POST"])
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
    raw_type = payload.get("type")
    type_confidence = payload.get("confidence")

    if raw_type:
        # Normalize type (handles aliases and case variations)
        memory_type, was_normalized = normalize_memory_type(raw_type)

        # Empty string means unknown type that couldn't be mapped
        if not memory_type:
            valid_types = sorted(MEMORY_TYPES)
            alias_examples = ", ".join(f"'{k}'" for k in list(TYPE_ALIASES.keys())[:5])
            abort(
                400,
                description=(
                    f"Invalid memory type '{raw_type}'. "
                    f"Must be one of: {', '.join(valid_types)}, "
                    f"or aliases like {alias_examples}..."
                ),
            )

        if was_normalized and memory_type != raw_type:
            logger.debug("Normalized type '%s' -> '%s'", raw_type, memory_type)

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
        },
    )

    # Emit SSE event for real-time monitoring
    emit_event(
        "memory.store",
        {
            "id": memory_id,
            "content_preview": content[:100] + "..." if len(content) > 100 else content,
            "type": memory_type,
            "importance": importance,
            "tags": tags[:5],
            "size_bytes": len(content),
            "elapsed_ms": int(response["query_time_ms"]),
        },
        utc_now,
    )

    return jsonify(response), 201


@memory_bp.route("/memory/<memory_id>", methods=["PATCH"])
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


@memory_bp.route("/memory/<memory_id>", methods=["DELETE"])
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
            if "qdrant_models" in globals() and qdrant_models is not None:
                selector = qdrant_models.PointIdsList(points=[memory_id])
            else:
                selector = {"points": [memory_id]}
            qdrant_client.delete(collection_name=COLLECTION_NAME, points_selector=selector)
        except Exception:
            logger.exception("Failed to delete vector for memory %s", memory_id)

    return jsonify({"status": "success", "memory_id": memory_id})


@memory_bp.route("/memory/by-tag", methods=["GET"])
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

    return jsonify(
        {"status": "success", "tags": tags, "count": len(memories), "memories": memories}
    )


@recall_bp.route("/recall", methods=["GET"])
def recall_memories() -> Any:
    query_start = time.perf_counter()
    query_text = (request.args.get("query") or "").strip()
    try:
        requested_limit = int(request.args.get("limit", 5))
    except (TypeError, ValueError):
        requested_limit = 5
    limit = max(1, min(requested_limit, RECALL_MAX_LIMIT))
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
    # Delegate implementation to recall blueprint module (kept for backward-compatibility)
    from automem.api.recall import handle_recall  # local import to avoid cycles

    response = handle_recall(
        get_memory_graph,
        get_qdrant_client,
        _normalize_tag_list,
        _normalize_timestamp,
        _parse_time_expression,
        _extract_keywords,
        _compute_metadata_score,
        _result_passes_filters,
        _graph_keyword_search,
        _vector_search,
        _vector_filter_only_tag_search,
        RECALL_MAX_LIMIT,
        logger,
        allowed_relations=ALLOWED_RELATIONS,
        relation_limit=RECALL_RELATION_LIMIT,
        expansion_limit_default=RECALL_EXPANSION_LIMIT,
    )

    # Emit SSE event for real-time monitoring
    elapsed_ms = int((time.perf_counter() - query_start) * 1000)
    result_count = 0
    try:
        # Response is either a tuple (response, status) or Response object
        resp_data = response[0] if isinstance(response, tuple) else response
        if hasattr(resp_data, "get_json"):
            data = resp_data.get_json(silent=True) or {}
            result_count = len(data.get("memories", []))
    except Exception as e:
        logger.debug("Failed to parse response for result_count", exc_info=e)

    emit_event(
        "memory.recall",
        {
            "query": query_text[:50] if query_text else "(no query)",
            "limit": limit,
            "result_count": result_count,
            "elapsed_ms": elapsed_ms,
            "tags": tag_filters[:3] if tag_filters else [],
        },
        utc_now,
    )

    return response


@memory_bp.route("/associate", methods=["POST"])
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
        abort(400, description=f"Relation type must be one of {sorted(ALLOWED_RELATIONS)}")

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


@consolidation_bp.route("/consolidate", methods=["POST"])
def consolidate_memories() -> Any:
    """Run memory consolidation."""
    data = request.get_json() or {}
    mode = data.get("mode", "full")
    dry_run = data.get("dry_run", True)

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

        return jsonify({"status": "success", "consolidation": results}), 200
    except Exception as e:
        logger.error(f"Consolidation failed: {e}")
        return jsonify({"error": "Consolidation failed", "details": str(e)}), 500


@consolidation_bp.route("/consolidate/status", methods=["GET"])
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

        return (
            jsonify(
                {
                    "status": "success",
                    "next_runs": next_runs,
                    "history": history,
                    "thread_alive": bool(
                        state.consolidation_thread and state.consolidation_thread.is_alive()
                    ),
                    "tick_seconds": CONSOLIDATION_TICK_SECONDS,
                }
            ),
            200,
        )
    except Exception as e:
        logger.error(f"Failed to get consolidation status: {e}")
        return jsonify({"error": "Failed to get status", "details": str(e)}), 500


@recall_bp.route("/startup-recall", methods=["GET"])
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
                lessons.append(
                    {
                        "id": row[0],
                        "content": row[1],
                        "tags": row[2] if row[2] else [],
                        "importance": row[3] if row[3] else 0.5,
                        "type": row[4] if row[4] else "Context",
                        "metadata": json.loads(row[5]) if row[5] else {},
                    }
                )

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
                system_rules.append(
                    {"id": row[0], "content": row[1], "tags": row[2] if row[2] else []}
                )

        response = {
            "status": "success",
            "critical_lessons": lessons,
            "system_rules": system_rules,
            "lesson_count": len(lessons),
            "has_critical": any(l.get("importance", 0) >= 0.9 for l in lessons),
            "summary": f"Recalled {len(lessons)} lesson(s) and {len(system_rules)} system rule(s)",
        }

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Startup recall failed: {e}")
        return jsonify({"error": "Startup recall failed", "details": str(e)}), 500


@recall_bp.route("/analyze", methods=["GET"])
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
            analytics["patterns"].append(
                {
                    "type": p_type,
                    "description": content,
                    "confidence": round(confidence, 3) if confidence else 0,
                    "observations": observations or 0,
                }
            )

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
            analytics["preferences"].append(
                {
                    "prefers": preferred,
                    "over": over,
                    "context": context,
                    "strength": round(strength, 3) if strength else 0,
                }
            )

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
                        "avg_importance": round(data["total_importance"] / data["count"], 3),
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

    return jsonify(
        {
            "status": "success",
            "analytics": analytics,
            "generated_at": utc_now(),
            "query_time_ms": round((time.perf_counter() - query_start) * 1000, 2),
        }
    )


def _normalize_tags(value: Any) -> List[str]:
    try:
        return _normalize_tags_value(value)
    except ValueError as exc:
        abort(400, description=str(exc))


def _coerce_importance(value: Any) -> float:
    try:
        return _coerce_importance_value(value)
    except ValueError as exc:
        abort(400, description=str(exc))


def _coerce_embedding(value: Any) -> Optional[List[float]]:
    return _coerce_embedding_value(value, state.effective_vector_size)


def _generate_placeholder_embedding(content: str) -> List[float]:
    return _generate_placeholder_embedding_value(content, state.effective_vector_size)


def _generate_real_embedding(content: str) -> List[float]:
    return _generate_real_embedding_value(
        content,
        init_embedding_provider=init_embedding_provider,
        state=state,
        logger=logger,
        placeholder_embedding=_generate_placeholder_embedding,
    )


def _generate_real_embeddings_batch(contents: List[str]) -> List[List[float]]:
    return _generate_real_embeddings_batch_value(
        contents,
        init_embedding_provider=init_embedding_provider,
        state=state,
        logger=logger,
        placeholder_embedding=_generate_placeholder_embedding,
    )


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


configure_recall_helpers(
    parse_iso_datetime=_parse_iso_datetime,
    prepare_tag_filters=_prepare_tag_filters,
    build_graph_tag_predicate=_build_graph_tag_predicate,
    build_qdrant_tag_filter=_build_qdrant_tag_filter,
    serialize_node=_serialize_node,
    fetch_relations=_fetch_relations,
    extract_keywords=_extract_keywords,
    coerce_embedding=_coerce_embedding,
    generate_real_embedding=_generate_real_embedding,
    logger=logger,
    collection_name=COLLECTION_NAME,
)


@recall_bp.route("/memories/<memory_id>/related", methods=["GET"])
def get_related_memories(memory_id: str) -> Any:
    """Return related memories by traversing relationship edges.

    Query params:
      - relationship_types: comma-separated list of relationship types to traverse
      - max_depth: traversal depth (default 1, max 3)
      - limit: max number of related memories to return (default RECALL_RELATION_LIMIT)
    """
    graph = get_memory_graph()
    if graph is None:
        abort(503, description="FalkorDB is unavailable")

    # Parse and sanitize relationship types
    rel_types_param = (request.args.get("relationship_types") or "").strip()
    if rel_types_param:
        requested = [part.strip().upper() for part in rel_types_param.split(",") if part.strip()]
        rel_types = [t for t in requested if t in ALLOWED_RELATIONS]
        if not rel_types:
            rel_types = sorted(ALLOWED_RELATIONS)
    else:
        rel_types = sorted(ALLOWED_RELATIONS)

    # Depth and limit
    try:
        max_depth = int(request.args.get("max_depth", 1))
    except (TypeError, ValueError):
        max_depth = 1
    max_depth = max(1, min(max_depth, 3))

    try:
        limit = int(request.args.get("limit", RECALL_RELATION_LIMIT))
    except (TypeError, ValueError):
        limit = RECALL_RELATION_LIMIT
    limit = max(1, min(limit, 200))

    # Build relationship type pattern like :A|B|C
    if rel_types:
        rel_pattern = ":" + "|".join(rel_types)
    else:
        rel_pattern = ""

    # Build Cypher query
    # We prefer full memory nodes (not summaries) for downstream consumers
    query = f"""
        MATCH (m:Memory {{id: $id}}){'-[r' + rel_pattern + f']-' if rel_pattern else '-[r]-'}(related:Memory)
        WHERE m.id <> related.id
        CALL apoc.path.expandConfig(related, {{
            relationshipFilter: '{'|'.join(rel_types)}',
            minLevel: 0,
            maxLevel: $max_depth,
            bfs: true,
            filterStartNode: true
        }}) YIELD path
        WITH DISTINCT related
        RETURN related
        ORDER BY coalesce(related.importance, 0.0) DESC, coalesce(related.timestamp, '') DESC
        LIMIT $limit
    """

    # If APOC is unavailable, fall back to simple 1..depth traversal without apoc
    fallback_query = f"""
        MATCH (m:Memory {{id: $id}}){'-[r' + rel_pattern + f'*1..$max_depth]-' if rel_pattern else '-[r*1..$max_depth]-'}(related:Memory)
        WHERE m.id <> related.id
        RETURN DISTINCT related
        ORDER BY coalesce(related.importance, 0.0) DESC, coalesce(related.timestamp, '') DESC
        LIMIT $limit
    """

    params = {"id": memory_id, "max_depth": max_depth, "limit": limit}

    try:
        result = graph.query(query, params)
    except Exception:
        # Try fallback if APOC or features not available
        try:
            result = graph.query(fallback_query, params)
        except Exception:
            logger.exception("Failed to traverse related memories for %s", memory_id)
            abort(500, description="Failed to fetch related memories")

    related: List[Dict[str, Any]] = []
    for row in getattr(result, "result_set", []) or []:
        node = row[0]
        data = _serialize_node(node)
        if data.get("id") != memory_id:
            related.append(data)

    return jsonify(
        {
            "status": "success",
            "memory_id": memory_id,
            "count": len(related),
            "related_memories": related,
            "relationship_types": rel_types,
            "max_depth": max_depth,
            "limit": limit,
        }
    )


from automem.api.admin import create_admin_blueprint_full
from automem.api.consolidation import create_consolidation_blueprint_full
from automem.api.enrichment import create_enrichment_blueprint
from automem.api.graph import create_graph_blueprint

# Register blueprints after all routes are defined
from automem.api.health import create_health_blueprint
from automem.api.memory import create_memory_blueprint_full
from automem.api.recall import create_recall_blueprint

health_bp = create_health_blueprint(
    get_memory_graph,
    get_qdrant_client,
    state,
    GRAPH_NAME,
    COLLECTION_NAME,
    utc_now,
)

enrichment_bp = create_enrichment_blueprint(
    _require_admin_token,
    state,
    enqueue_enrichment,
    ENRICHMENT_MAX_ATTEMPTS,
)

recall_bp = create_recall_blueprint(
    get_memory_graph,
    get_qdrant_client,
    _normalize_tag_list,
    _normalize_timestamp,
    _parse_time_expression,
    _extract_keywords,
    _compute_metadata_score,
    _result_passes_filters,
    _graph_keyword_search,
    _vector_search,
    _vector_filter_only_tag_search,
    RECALL_MAX_LIMIT,
    logger,
    ALLOWED_RELATIONS,
    RECALL_RELATION_LIMIT,
    _serialize_node,
    _summarize_relation_node,
    update_last_accessed,
    jit_enrich_fn=jit_enrich_lightweight if JIT_ENRICHMENT_ENABLED else None,
)

memory_bp = create_memory_blueprint_full(
    get_memory_graph,
    get_qdrant_client,
    _normalize_tags,
    _normalize_tag_list,
    _compute_tag_prefixes,
    _coerce_importance,
    _coerce_embedding,
    _normalize_timestamp,
    utc_now,
    _serialize_node,
    _parse_metadata_field,
    _generate_real_embedding,
    enqueue_enrichment,
    enqueue_embedding,
    lambda content: memory_classifier.classify(content),
    PointStruct,
    COLLECTION_NAME,
    ALLOWED_RELATIONS,
    RELATIONSHIP_TYPES,
    state,
    logger,
    update_last_accessed,
    get_openai_client,
)

admin_bp = create_admin_blueprint_full(
    _require_admin_token,
    init_openai,
    get_openai_client,
    get_qdrant_client,
    get_memory_graph,
    PointStruct,
    COLLECTION_NAME,
    lambda: state.effective_vector_size,  # Use runtime-detected dimension
    EMBEDDING_MODEL,
    utc_now,
    logger,
)

consolidation_bp = create_consolidation_blueprint_full(
    get_memory_graph,
    get_qdrant_client,
    _build_consolidator_from_config,
    _persist_consolidation_run,
    _build_scheduler_from_graph,
    _load_recent_runs,
    state,
    CONSOLIDATION_TICK_SECONDS,
    CONSOLIDATION_HISTORY_LIMIT,
    logger,
)

graph_bp = create_graph_blueprint(
    get_memory_graph,
    get_qdrant_client,
    _serialize_node,
    COLLECTION_NAME,
    logger,
)

stream_bp = create_stream_blueprint(
    require_api_token=require_api_token,
)

app.register_blueprint(health_bp)
app.register_blueprint(enrichment_bp)
app.register_blueprint(memory_bp)
app.register_blueprint(admin_bp)
app.register_blueprint(recall_bp)
app.register_blueprint(consolidation_bp)
app.register_blueprint(graph_bp)
app.register_blueprint(stream_bp)


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
    init_sync_worker()
    # Use :: for IPv6 dual-stack (Railway internal networking uses IPv6)
    app.run(host="::", port=port, debug=False)
