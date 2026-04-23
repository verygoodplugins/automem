"""Backend adapters for benchmarking memory systems with a common interface."""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests


def _sanitize_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._:-]+", "-", value).strip("-")
    return slug or "scope"


def _strip_internal_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in payload.items() if not k.startswith("_")}


@dataclass
class SearchRequest:
    query: str
    scope_id: str
    limit: int = 10
    tags: Optional[List[str]] = None
    tag_match: str = "exact"
    tag_mode: str = "any"
    start: Optional[str] = None
    end: Optional[str] = None
    expand_entities: bool = False
    expand_relations: bool = False
    auto_decompose: bool = False
    speaker: Optional[str] = None


@dataclass
class MemoryRecord:
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    score: float = 0.0
    match_type: str = ""


class BenchmarkBackend(ABC):
    """Common interface for benchmark memory backends."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_token: Optional[str] = None,
        scope_prefix: Optional[str] = None,
        work_dir: Optional[str] = None,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.api_token = api_token or ""
        self.scope_prefix = _sanitize_slug(scope_prefix) if scope_prefix else ""
        self.work_dir = Path(work_dir) if work_dir else None
        self.created_ids_by_scope: Dict[str, set[str]] = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name for results."""

    def physical_scope(self, logical_scope: str) -> str:
        logical_scope = _sanitize_slug(logical_scope)
        if not self.scope_prefix:
            return logical_scope
        return f"{self.scope_prefix}:{logical_scope}"

    def remember_created_id(self, scope_id: str, memory_id: Optional[str]) -> None:
        if not memory_id:
            return
        self.created_ids_by_scope.setdefault(scope_id, set()).add(str(memory_id))

    @abstractmethod
    def health_check(self) -> bool:
        """Check backend availability."""

    @abstractmethod
    def cleanup_scope(self, scope_id: str) -> int:
        """Delete or forget memories belonging to a logical scope."""

    @abstractmethod
    def ingest_memories(
        self,
        memories: List[Dict[str, Any]],
        *,
        scope_id: str,
        batch_size: int = 100,
        pause_between_batches: float = 0.0,
    ) -> Dict[str, str]:
        """Store benchmark memories and return benchmark-id -> backend-id mapping."""

    @abstractmethod
    def search(self, request: SearchRequest) -> List[MemoryRecord]:
        """Search for memories."""

    def related_memories(
        self,
        memory_id: str,
        *,
        limit: int = 8,
        max_depth: int = 2,
        relationship_types: Optional[str] = None,
    ) -> List[MemoryRecord]:
        """Fetch related memories when supported."""
        return []


class AutoMemBackend(BenchmarkBackend):
    """Adapter for AutoMem's REST API."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }
        self._batch_api_available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "automem"

    def health_check(self) -> bool:
        try:
            response = requests.get(urljoin(f"{self.base_url}/", "health"), timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _supports_batch(self) -> bool:
        if self._batch_api_available is not None:
            return self._batch_api_available
        try:
            response = requests.post(
                urljoin(f"{self.base_url}/", "memory/batch"),
                headers=self.headers,
                json={"memories": []},
                timeout=5,
            )
            self._batch_api_available = response.status_code in {201, 400}
        except requests.RequestException:
            self._batch_api_available = False
        return self._batch_api_available

    def _delete_memory(self, memory_id: str) -> bool:
        response = requests.delete(
            urljoin(f"{self.base_url}/", f"memory/{memory_id}"),
            headers=self.headers,
            timeout=5,
        )
        return response.status_code in {200, 204}

    def _build_search_params(
        self,
        request: SearchRequest,
        tags: List[str],
        *,
        tag_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "query": request.query,
            "limit": request.limit,
            "tags": tags,
            "tag_match": request.tag_match,
        }
        if tag_mode is not None:
            params["tag_mode"] = tag_mode
        if request.start:
            params["start"] = request.start
        if request.end:
            params["end"] = request.end
        if request.expand_entities:
            params["expand_entities"] = "true"
        if request.expand_relations:
            params["expand_relations"] = "true"
        if request.auto_decompose:
            params["auto_decompose"] = "true"
        return params

    @staticmethod
    def _flatten_results(results: List[Dict[str, Any]]) -> List[MemoryRecord]:
        flattened: List[MemoryRecord] = []
        for item in results:
            memory = item.get("memory") or item
            memory_id = memory.get("id") or item.get("id")
            if not memory_id:
                continue
            flattened.append(
                MemoryRecord(
                    id=str(memory_id),
                    content=memory.get("content", ""),
                    metadata=dict(memory.get("metadata") or {}),
                    tags=list(memory.get("tags") or []),
                    score=float(item.get("score") or 0.0),
                    match_type=str(item.get("match_type") or ""),
                )
            )
        return flattened

    def _run_search(
        self,
        request: SearchRequest,
        tags: List[str],
        *,
        tag_mode: Optional[str] = None,
    ) -> List[MemoryRecord]:
        response = requests.get(
            urljoin(f"{self.base_url}/", "recall"),
            headers=self.headers,
            params=self._build_search_params(request, tags, tag_mode=tag_mode),
            timeout=30,
        )
        if response.status_code != 200:
            return []
        return self._flatten_results(response.json().get("results", []))

    def cleanup_scope(self, scope_id: str) -> int:
        deleted = 0
        remembered_ids = list(self.created_ids_by_scope.get(scope_id, set()))
        if remembered_ids:
            for memory_id in remembered_ids:
                if self._delete_memory(memory_id):
                    deleted += 1
            self.created_ids_by_scope.pop(scope_id, None)
            return deleted

        scope_tag = self.physical_scope(scope_id)
        while True:
            response = requests.get(
                urljoin(f"{self.base_url}/", "recall"),
                headers=self.headers,
                params={"tags": scope_tag, "tag_match": "exact", "limit": 100},
                timeout=10,
            )
            if response.status_code != 200:
                break
            results = response.json().get("results", [])
            if not results:
                break
            batch_deleted = 0
            for item in results:
                memory_id = item.get("id") or item.get("memory", {}).get("id")
                if memory_id and self._delete_memory(str(memory_id)):
                    batch_deleted += 1
            deleted += batch_deleted
            if batch_deleted == 0 or len(results) < 100:
                break
        self.created_ids_by_scope.pop(scope_id, None)
        return deleted

    def ingest_memories(
        self,
        memories: List[Dict[str, Any]],
        *,
        scope_id: str,
        batch_size: int = 100,
        pause_between_batches: float = 0.0,
    ) -> Dict[str, str]:
        memory_map: Dict[str, str] = {}
        scope_tag = self.physical_scope(scope_id)
        for start in range(0, len(memories), batch_size):
            batch = memories[start : start + batch_size]
            has_more_batches = start + batch_size < len(memories)
            stripped = []
            for memory in batch:
                payload = _strip_internal_fields(memory)
                tags = list(payload.get("tags") or [])
                if scope_tag not in tags:
                    tags.append(scope_tag)
                payload["tags"] = tags
                stripped.append(payload)
            benchmark_ids = [str(m.get("_benchmark_id", "")) for m in batch]
            if self._supports_batch():
                try:
                    response = requests.post(
                        urljoin(f"{self.base_url}/", "memory/batch"),
                        headers=self.headers,
                        json={"memories": stripped},
                        timeout=60,
                    )
                    if response.status_code in {200, 201}:
                        result = response.json()
                        for benchmark_id, memory_id in zip(
                            benchmark_ids, result.get("memory_ids", []), strict=True
                        ):
                            if benchmark_id:
                                memory_map[benchmark_id] = str(memory_id)
                                self.remember_created_id(scope_id, str(memory_id))
                        if has_more_batches and pause_between_batches > 0:
                            time.sleep(pause_between_batches)
                        continue
                except requests.RequestException:
                    pass
            for benchmark_id, payload in zip(benchmark_ids, stripped, strict=True):
                response = requests.post(
                    urljoin(f"{self.base_url}/", "memory"),
                    headers=self.headers,
                    json=payload,
                    timeout=15,
                )
                if response.status_code in {200, 201} and benchmark_id:
                    memory_id = response.json().get("memory_id") or response.json().get("id")
                    if memory_id:
                        memory_map[benchmark_id] = str(memory_id)
                        self.remember_created_id(scope_id, str(memory_id))
            if has_more_batches and pause_between_batches > 0:
                time.sleep(pause_between_batches)
        return memory_map

    def search(self, request: SearchRequest) -> List[MemoryRecord]:
        scope_tag = self.physical_scope(request.scope_id)
        requested_tags = list(dict.fromkeys(request.tags or []))
        if not requested_tags:
            return self._run_search(request, [scope_tag])

        if request.tag_mode == "all":
            return self._run_search(request, [scope_tag, *requested_tags], tag_mode="all")

        merged_results: Dict[str, MemoryRecord] = {}
        for tag in requested_tags:
            scoped_results = self._run_search(request, [scope_tag, tag], tag_mode="all")
            for record in scoped_results:
                existing = merged_results.get(record.id)
                if existing is None or record.score > existing.score:
                    merged_results[record.id] = record

        results = sorted(merged_results.values(), key=lambda record: record.score, reverse=True)
        return results[: request.limit]

    def related_memories(
        self,
        memory_id: str,
        *,
        limit: int = 8,
        max_depth: int = 2,
        relationship_types: Optional[str] = None,
    ) -> List[MemoryRecord]:
        response = requests.get(
            urljoin(f"{self.base_url}/", f"memories/{memory_id}/related"),
            headers=self.headers,
            params={
                "limit": limit,
                "max_depth": max_depth,
                "relationship_types": relationship_types
                or "RELATES_TO,LEADS_TO,PART_OF,DERIVED_FROM,SIMILAR_TO,PRECEDED_BY,EXPLAINS,SHARES_THEME,PARALLEL_CONTEXT",
            },
            timeout=10,
        )
        if response.status_code != 200:
            return []
        related = []
        for item in response.json().get("related_memories", []):
            related_id = item.get("id")
            if not related_id:
                continue
            related.append(
                MemoryRecord(
                    id=str(related_id),
                    content=item.get("content", ""),
                    metadata=dict(item.get("metadata") or {}),
                    tags=list(item.get("tags") or []),
                    score=float(item.get("score") or 0.0),
                    match_type=str(item.get("match_type") or ""),
                )
            )
        return related


def create_backend(
    backend: str,
    *,
    base_url: Optional[str] = None,
    api_token: Optional[str] = None,
    scope_prefix: Optional[str] = None,
    work_dir: Optional[str] = None,
) -> BenchmarkBackend:
    backend_key = backend.strip().lower()
    if backend_key == "automem":
        return AutoMemBackend(
            base_url=base_url,
            api_token=api_token,
            scope_prefix=scope_prefix,
            work_dir=work_dir,
        )
    raise NotImplementedError(
        f"Unsupported benchmark backend: {backend}. "
        "Cross-backend adapters live in the automem-evals repo."
    )
