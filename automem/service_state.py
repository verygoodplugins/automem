from __future__ import annotations

import time
from dataclasses import dataclass, field
from queue import Queue
from threading import Event, Lock, Thread
from typing import Any, Dict, Optional, Set

from falkordb import FalkorDB
from qdrant_client import QdrantClient

from automem.config import ENRICHMENT_CIRCUIT_COOLDOWN_SECONDS, VECTOR_SIZE
from automem.embedding.provider import EmbeddingProvider
from automem.utils.time import utc_now


class EnrichmentCircuit:
    """Fail-soft cooldown for definitive LLM quota failures."""

    def __init__(self, cooldown_seconds: float = 300, clock: Any = time.monotonic) -> None:
        self._cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._opened_until = 0.0
        self._probe_pending = False
        self._lock = Lock()
        self.circuit_open_skips = 0
        self.recoveries = 0

    def begin_request(self) -> tuple[bool, bool]:
        """Reserve an LLM request and report whether it is the recovery probe."""
        with self._lock:
            now = self._clock()
            if now < self._opened_until:
                self.circuit_open_skips += 1
                return False, False
            if self._opened_until:
                if self._probe_pending:
                    self.circuit_open_skips += 1
                    return False, False
                self._probe_pending = True
                return True, True
            return True, False

    def record_failure(self, error: str, was_probe: bool = False) -> bool:
        if "insufficient_quota" not in error.lower() and "quota" not in error.lower():
            with self._lock:
                if was_probe and self._probe_pending:
                    self._opened_until = 0.0
                    self._probe_pending = False
            return False
        with self._lock:
            self._opened_until = self._clock() + self._cooldown_seconds
            self._probe_pending = False
            return True

    def record_success(self, was_probe: bool = False) -> None:
        with self._lock:
            if was_probe and self._probe_pending:
                self.recoveries += 1
                self._opened_until = 0.0
                self._probe_pending = False

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "circuit_open_skips": self.circuit_open_skips,
                "recoveries": self.recoveries,
                "open": self._clock() < self._opened_until,
            }


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
class ClassificationStats:
    llm_attempts: int = 0
    llm_successes: int = 0
    fallbacks: int = 0
    pattern_classifications: int = 0
    last_error: Optional[str] = None
    last_error_at: Optional[str] = None
    _lock: Lock = field(default_factory=Lock, init=False, repr=False, compare=False)

    def record_pattern(self) -> None:
        with self._lock:
            self.pattern_classifications += 1

    def record_llm_attempt(self) -> None:
        with self._lock:
            self.llm_attempts += 1

    def record_llm_success(self) -> None:
        with self._lock:
            self.llm_successes += 1

    def record_fallback(self, error: Optional[str] = None) -> None:
        with self._lock:
            self.fallbacks += 1
            if error:
                self.last_error = error
                self.last_error_at = utc_now()

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "llm_attempts": self.llm_attempts,
                "llm_successes": self.llm_successes,
                "fallbacks": self.fallbacks,
                "pattern_classifications": self.pattern_classifications,
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
    openai_client: Any = None  # Keep for backward compatibility (type classification, etc.)
    embedding_provider: Optional[EmbeddingProvider] = None
    enrichment_queue: Optional[Queue] = None
    enrichment_thread: Optional[Thread] = None
    enrichment_stats: EnrichmentStats = field(default_factory=EnrichmentStats)
    classification_stats: ClassificationStats = field(default_factory=ClassificationStats)
    enrichment_circuit: EnrichmentCircuit = field(
        default_factory=lambda: EnrichmentCircuit(ENRICHMENT_CIRCUIT_COOLDOWN_SECONDS)
    )
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
    # Background sync worker
    sync_thread: Optional[Thread] = None
    sync_stop_event: Optional[Event] = None
    sync_last_run: Optional[str] = None
    sync_last_result: Optional[Dict[str, Any]] = None
    # Effective vector size (auto-detected from existing collection or config default)
    effective_vector_size: int = VECTOR_SIZE
