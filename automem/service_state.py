from __future__ import annotations

from dataclasses import dataclass, field
from queue import Queue
from threading import Event, Lock, Thread
from typing import Any, Dict, Optional, Set

from falkordb import FalkorDB
from qdrant_client import QdrantClient

from automem.config import VECTOR_SIZE
from automem.embedding.provider import EmbeddingProvider
from automem.utils.time import utc_now


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
    openai_client: Any = None  # Keep for backward compatibility (type classification, etc.)
    embedding_provider: Optional[EmbeddingProvider] = None
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
    # Background sync worker
    sync_thread: Optional[Thread] = None
    sync_stop_event: Optional[Event] = None
    sync_last_run: Optional[str] = None
    sync_last_result: Optional[Dict[str, Any]] = None
    # Effective vector size (auto-detected from existing collection or config default)
    effective_vector_size: int = VECTOR_SIZE
