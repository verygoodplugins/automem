#!/usr/bin/env python3
"""Repair generated entity tags on a local AutoMem clone.

Default mode is a dry-run audit: scan Memory nodes, write a deterministic
plan, and do not mutate data. Execution applies a previously written plan.
Rollback applies the rollback JSONL emitted next to that plan.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - dotenv is a runtime convenience
    load_dotenv = None  # type: ignore[assignment]

from automem.utils.entity_quality import name_from_slug, slugify_entity, validate_entity_tag
from automem.utils.tags import _compute_tag_prefixes, _normalize_tag_list

if load_dotenv is not None:
    load_dotenv()
    load_dotenv(Path.home() / ".config" / "automem" / ".env")


ENTITY_CATEGORIES = ("people", "organizations", "tools", "projects", "concepts")
REPAIR_MODES = ("sync-only", "reject-only", "canonicalize-safe")
DEFAULT_REPAIR_MODE = "canonicalize-safe"

_SAFE_PERSON_NAME_PARTICLES = {
    "da",
    "de",
    "del",
    "der",
    "di",
    "du",
    "la",
    "le",
    "st",
    "van",
    "von",
}


class GraphUpdateTimedOut(TimeoutError):
    """Raised when a FalkorDB write batch exceeds the local repair deadline."""


class QdrantPayloadTimedOut(TimeoutError):
    """Raised when a Qdrant payload update exceeds the local repair deadline."""


@contextmanager
def _graph_update_deadline(seconds: Optional[float], *, rows: int) -> Iterator[None]:
    if seconds is None or seconds <= 0:
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)

    def _raise_timeout(_signum: int, _frame: Any) -> None:
        raise GraphUpdateTimedOut(f"graph update timed out after {seconds:g}s for {rows} rows")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0 or previous_timer[1] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


@contextmanager
def _qdrant_payload_deadline(seconds: Optional[float], *, memory_id: str) -> Iterator[None]:
    if seconds is None or seconds <= 0:
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)

    def _raise_timeout(_signum: int, _frame: Any) -> None:
        raise QdrantPayloadTimedOut(
            f"qdrant payload update timed out after {seconds:g}s for {memory_id}"
        )

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0 or previous_timer[1] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


class RepairPlanItem:
    def __init__(
        self,
        *,
        memory_id: str,
        original_tags: list[str],
        repaired_tags: list[str],
        original_tag_prefixes: list[str],
        repaired_tag_prefixes: list[str],
        original_metadata: dict[str, Any],
        repaired_metadata: dict[str, Any],
        actions: list[dict[str, Any]],
        original_qdrant_payload: Optional[dict[str, Any]] = None,
        repaired_qdrant_payload: Optional[dict[str, Any]] = None,
    ) -> None:
        self.memory_id = memory_id
        self.original_tags = original_tags
        self.repaired_tags = repaired_tags
        self.original_tag_prefixes = original_tag_prefixes
        self.repaired_tag_prefixes = repaired_tag_prefixes
        self.original_metadata = original_metadata
        self.repaired_metadata = repaired_metadata
        self.actions = actions
        self.original_qdrant_payload = original_qdrant_payload
        self.repaired_qdrant_payload = repaired_qdrant_payload

    def to_plan_record(self) -> dict[str, Any]:
        return {
            "id": self.memory_id,
            "original_tags": self.original_tags,
            "repaired_tags": self.repaired_tags,
            "original_tag_prefixes": self.original_tag_prefixes,
            "repaired_tag_prefixes": self.repaired_tag_prefixes,
            "original_metadata": self.original_metadata,
            "repaired_metadata": self.repaired_metadata,
            "original_qdrant_payload": self.original_qdrant_payload,
            "repaired_qdrant_payload": self.repaired_qdrant_payload,
            "actions": self.actions,
        }

    def to_rollback_record(self) -> dict[str, Any]:
        return {
            "id": self.memory_id,
            "original_tags": self.repaired_tags,
            "repaired_tags": self.original_tags,
            "original_tag_prefixes": self.repaired_tag_prefixes,
            "repaired_tag_prefixes": self.original_tag_prefixes,
            "original_metadata": self.repaired_metadata,
            "repaired_metadata": self.original_metadata,
            "original_qdrant_payload": self.repaired_qdrant_payload,
            "repaired_qdrant_payload": self.original_qdrant_payload,
            "rollback_from_tags": self.repaired_tags,
            "rollback_from_tag_prefixes": self.repaired_tag_prefixes,
            "rollback_from_metadata": self.repaired_metadata,
            "rollback_from_qdrant_payload": self.repaired_qdrant_payload,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "RepairPlanItem":
        memory_id = str(record.get("id") or record.get("memory_id") or "")
        original_tags = _normalize_tag_list(record.get("original_tags"))
        if not original_tags:
            original_tags = _normalize_tag_list(record.get("rollback_from_tags"))
        repaired_tags = _normalize_tag_list(record.get("repaired_tags"))
        original_tag_prefixes = _normalize_tag_list(record.get("original_tag_prefixes"))
        if not original_tag_prefixes:
            original_tag_prefixes = _normalize_tag_list(record.get("rollback_from_tag_prefixes"))
        original_metadata = _parse_metadata(record.get("original_metadata"))
        if not original_metadata:
            original_metadata = _parse_metadata(record.get("rollback_from_metadata"))
        repaired_metadata = _parse_metadata(record.get("repaired_metadata"))
        original_qdrant_payload = _parse_qdrant_payload(record.get("original_qdrant_payload"))
        if original_qdrant_payload is None:
            original_qdrant_payload = _parse_qdrant_payload(
                record.get("rollback_from_qdrant_payload")
            )
        repaired_qdrant_payload = _parse_qdrant_payload(record.get("repaired_qdrant_payload"))
        return cls(
            memory_id=memory_id,
            original_tags=original_tags,
            repaired_tags=repaired_tags,
            original_tag_prefixes=original_tag_prefixes,
            repaired_tag_prefixes=_normalize_tag_list(record.get("repaired_tag_prefixes"))
            or _compute_tag_prefixes(repaired_tags),
            original_metadata=original_metadata,
            repaired_metadata=repaired_metadata,
            actions=list(record.get("actions") or []),
            original_qdrant_payload=original_qdrant_payload,
            repaired_qdrant_payload=repaired_qdrant_payload,
        )

    @property
    def tag_changed(self) -> bool:
        return self.original_tags != self.repaired_tags

    @property
    def tag_prefix_changed(self) -> bool:
        return self.original_tag_prefixes != self.repaired_tag_prefixes

    @property
    def metadata_changed(self) -> bool:
        return self.original_metadata != self.repaired_metadata

    @property
    def metadata_entities_changed(self) -> bool:
        return self.original_metadata.get("entities") != self.repaired_metadata.get("entities")

    @property
    def qdrant_payload_changed(self) -> bool:
        if self.original_qdrant_payload is None or self.repaired_qdrant_payload is None:
            return False
        return self.original_qdrant_payload != self.repaired_qdrant_payload


class RepairPlanResult:
    def __init__(
        self,
        *,
        mode: str,
        items: list[RepairPlanItem],
        processed_count: int,
        unchanged_count: int,
        rejected_tags: list[dict[str, Any]],
        canonicalized_tags: list[dict[str, Any]],
        ambiguous_people: list[dict[str, Any]],
    ) -> None:
        self.mode = mode
        self.items = items
        self.processed_count = processed_count
        self.unchanged_count = unchanged_count
        self.rejected_tags = rejected_tags
        self.canonicalized_tags = canonicalized_tags
        self.ambiguous_people = ambiguous_people

    def summary(self) -> dict[str, Any]:
        delta_counts = {
            "entity_tags_removed": 0,
            "entity_tags_added": 0,
            "bare_tags_removed": 0,
            "bare_tags_added": 0,
            "memories_with_bare_tag_changes": 0,
        }
        for item in self.items:
            item_counts = _tag_delta_counts(item)
            for key, value in item_counts.items():
                delta_counts[key] += value
        return {
            "mode": self.mode,
            "processed": self.processed_count,
            "changed": len(self.items),
            "unchanged": self.unchanged_count,
            "tag_changes": sum(1 for item in self.items if item.tag_changed),
            "tag_prefix_changes": sum(1 for item in self.items if item.tag_prefix_changed),
            "metadata_entity_changes": sum(
                1 for item in self.items if item.metadata_entities_changed
            ),
            "metadata_changes": sum(1 for item in self.items if item.metadata_changed),
            "qdrant_payload_changes": sum(1 for item in self.items if item.qdrant_payload_changed),
            "rejected_tags": len(self.rejected_tags),
            "canonicalized_tags": len(self.canonicalized_tags),
            "ambiguous_people": len(self.ambiguous_people),
            "dry_run": True,
            **delta_counts,
        }


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None or not str(value).strip():
        return default
    return str(value).strip()


def _is_local_host(host: Optional[str]) -> bool:
    if not host:
        return True
    normalized = host.strip().strip("[]").lower()
    return normalized in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def _qdrant_host_from_env() -> Optional[str]:
    raw_url = _get_env("QDRANT_URL", "http://localhost:6333")
    if not raw_url:
        return None
    return urlparse(raw_url).hostname


def assert_local_targets(*, allow_non_local: bool, check_qdrant: bool) -> None:
    if allow_non_local:
        return
    falkor_host = _get_env("FALKORDB_HOST", "localhost")
    if not _is_local_host(falkor_host):
        raise SystemExit(
            f"refusing non-local FalkorDB host without --allow-non-local: {falkor_host}"
        )
    if check_qdrant:
        qdrant_host = _qdrant_host_from_env()
        if not _is_local_host(qdrant_host):
            raise SystemExit(
                f"refusing non-local Qdrant host without --allow-non-local: {qdrant_host}"
            )


def _parse_metadata(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except Exception:
            return {"_raw_metadata": value}
        if isinstance(parsed, dict):
            return dict(parsed)
        return {"_raw_metadata": parsed}
    return {"_raw_metadata": value}


def _parse_qdrant_payload(value: Any) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    if not isinstance(value, dict):
        return None
    return {
        "tags": _normalize_tag_list(value.get("tags")),
        "tag_prefixes": _normalize_tag_list(value.get("tag_prefixes")),
        "metadata": _parse_metadata(value.get("metadata")),
    }


def _desired_qdrant_payload(
    *,
    tags: list[str],
    tag_prefixes: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "tags": list(tags),
        "tag_prefixes": list(tag_prefixes),
        "metadata": dict(metadata),
    }


def _entity_name_from_tag(tag: str) -> tuple[str, str]:
    _prefix, category, slug = tag.split(":", 2)
    return category, name_from_slug(slug)


def _entity_parts_from_tag(tag: str) -> tuple[str, str, str]:
    _prefix, category, slug = tag.split(":", 2)
    return category, slug, name_from_slug(slug)


def _entity_display_lookup(metadata: dict[str, Any]) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    entities = metadata.get("entities")
    if not isinstance(entities, dict):
        return lookup
    for category, names in entities.items():
        if category not in ENTITY_CATEGORIES or not isinstance(names, list):
            continue
        for name in names:
            if not isinstance(name, str) or not name.strip():
                continue
            slug = slugify_entity(name)
            lookup.setdefault((category, slug), name.strip())
    return lookup


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _with_repaired_entities(metadata: dict[str, Any], tags: list[str]) -> dict[str, Any]:
    repaired = dict(metadata)
    entities: dict[str, list[str]] = {}
    display_lookup = _entity_display_lookup(metadata)
    for tag in tags:
        if not isinstance(tag, str) or not tag.startswith("entity:"):
            continue
        try:
            category, slug, fallback_name = _entity_parts_from_tag(tag)
        except ValueError:
            continue
        if category not in ENTITY_CATEGORIES:
            continue
        name = display_lookup.get((category, slug), fallback_name)
        entities.setdefault(category, [])
        if name not in entities[category]:
            entities[category].append(name)

    if entities:
        repaired["entities"] = entities
    else:
        repaired.pop("entities", None)
    return repaired


def _tag_set(tags: list[str], *, entity: bool) -> set[str]:
    return {
        tag
        for tag in tags
        if isinstance(tag, str)
        and (tag.startswith("entity:") if entity else not tag.startswith("entity:"))
    }


def _tag_delta_counts(item: RepairPlanItem) -> dict[str, int]:
    original_entities = _tag_set(item.original_tags, entity=True)
    repaired_entities = _tag_set(item.repaired_tags, entity=True)
    original_bare = _tag_set(item.original_tags, entity=False)
    repaired_bare = _tag_set(item.repaired_tags, entity=False)
    return {
        "entity_tags_removed": len(original_entities - repaired_entities),
        "entity_tags_added": len(repaired_entities - original_entities),
        "bare_tags_removed": len(original_bare - repaired_bare),
        "bare_tags_added": len(repaired_bare - original_bare),
        "memories_with_bare_tag_changes": 1 if original_bare != repaired_bare else 0,
    }


def has_bare_tag_mutations(summary: dict[str, Any]) -> bool:
    return bool(
        summary.get("bare_tags_removed")
        or summary.get("bare_tags_added")
        or summary.get("memories_with_bare_tag_changes")
    )


def _normalize_row(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return {
            "id": str(row.get("id") or ""),
            "tags": _normalize_tag_list(row.get("tags")),
            "tag_prefixes": _normalize_tag_list(row.get("tag_prefixes")),
            "metadata": _parse_metadata(row.get("metadata")),
            "qdrant_payload": _parse_qdrant_payload(row.get("qdrant_payload")),
            "content": str(row.get("content") or ""),
        }
    return {
        "id": str(row[0] if len(row) > 0 else ""),
        "tags": _normalize_tag_list(row[1] if len(row) > 1 else None),
        "tag_prefixes": _normalize_tag_list(row[2] if len(row) > 2 else None),
        "metadata": _parse_metadata(row[3] if len(row) > 3 else None),
        "qdrant_payload": None,
        "content": str(row[4] if len(row) > 4 and row[4] is not None else ""),
    }


def _collect_people_targets(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    targets: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for tag in row["tags"]:
            if not isinstance(tag, str) or not tag.startswith("entity:"):
                continue
            validation = validate_entity_tag(tag, context=row.get("content") or None)
            if not validation.accepted or validation.category != "people":
                continue
            tokens = [token for token in validation.canonical_slug.split("-") if token]
            if len(tokens) >= 2:
                targets[tokens[0]].add(validation.canonical_tag)
    return targets


def _content_mentions_slug(content: str, slug: str) -> bool:
    if not content or not slug:
        return False
    haystack = f" {re.sub(r'[^a-z0-9]+', ' ', content.lower()).strip()} "
    needle = " ".join(token for token in slug.split("-") if token)
    return bool(needle and f" {needle} " in haystack)


def _has_safe_person_name_shape(slug: str) -> bool:
    tokens = [token for token in slug.split("-") if token]
    if len(tokens) == 2:
        pass
    elif len(tokens) == 3 and (len(tokens[1]) == 1 or tokens[1] in _SAFE_PERSON_NAME_PARTICLES):
        pass
    else:
        return False

    if any(not re.fullmatch(r"[a-z]+", token) for token in tokens):
        return False
    if tokens[0] == tokens[-1]:
        return False
    return True


def _safe_single_name_people_target(candidate_tag: str, source_content: str) -> bool:
    try:
        category, slug, _name = _entity_parts_from_tag(candidate_tag)
    except ValueError:
        return False
    return (
        category == "people"
        and _has_safe_person_name_shape(slug)
        and _content_mentions_slug(source_content, slug)
    )


def _repair_row(
    row: dict[str, Any],
    *,
    people_targets: dict[str, set[str]],
    mode: str,
) -> tuple[RepairPlanItem, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    memory_id = row["id"]
    original_tags = list(row["tags"])
    repaired_tags: list[str] = []
    actions: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    canonicalized: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []

    for tag in original_tags:
        if not isinstance(tag, str) or not tag.startswith("entity:"):
            repaired_tags.append(tag)
            continue

        if mode == "sync-only":
            repaired_tags.append(tag)
            continue

        validation = validate_entity_tag(tag, context=row.get("content") or None)
        if not validation.accepted:
            event = {
                "memory_id": memory_id,
                "tag": tag,
                "category": validation.category,
                "slug": validation.slug,
                "reason": validation.reason,
            }
            rejected.append(event)
            actions.append({"action": "remove_rejected", **event})
            continue

        canonical_tag = tag if mode == "reject-only" else validation.canonical_tag
        canonical_reason = validation.reason
        tokens = [token for token in validation.canonical_slug.split("-") if token]

        if mode == "canonicalize-safe" and validation.category == "people" and len(tokens) == 1:
            candidates = sorted(people_targets.get(tokens[0], set()))
            if len(candidates) > 1:
                event = {
                    "memory_id": memory_id,
                    "tag": tag,
                    "reason": "ambiguous_single_name_people",
                    "candidates": candidates,
                }
                ambiguous.append(event)
                actions.append({"action": "suppress_ambiguous_people", **event})
                continue
            if len(candidates) == 1 and candidates[0] != canonical_tag:
                candidate = candidates[0]
                if _safe_single_name_people_target(candidate, row.get("content") or ""):
                    canonical_tag = candidate
                    canonical_reason = "single_name_unambiguous"
                else:
                    event = {
                        "memory_id": memory_id,
                        "tag": tag,
                        "reason": "unsafe_single_name_people_target",
                        "candidates": candidates,
                    }
                    ambiguous.append(event)
                    actions.append({"action": "suppress_unsafe_single_name_people", **event})
                    continue

        if canonical_tag != tag:
            event = {
                "memory_id": memory_id,
                "original_tag": tag,
                "canonical_tag": canonical_tag,
                "reason": canonical_reason,
                "confidence": validation.confidence,
            }
            canonicalized.append(event)
            actions.append({"action": "canonicalize", **event})

        repaired_tags.append(canonical_tag)

    repaired_tags = _dedupe_preserve_order(repaired_tags)
    repaired_tag_prefixes = _compute_tag_prefixes(repaired_tags)
    repaired_metadata = _with_repaired_entities(row["metadata"], repaired_tags)
    original_qdrant_payload = row.get("qdrant_payload")
    repaired_qdrant_payload = (
        _desired_qdrant_payload(
            tags=repaired_tags,
            tag_prefixes=repaired_tag_prefixes,
            metadata=repaired_metadata,
        )
        if original_qdrant_payload is not None
        else None
    )
    if (
        original_qdrant_payload is not None
        and repaired_qdrant_payload is not None
        and original_qdrant_payload != repaired_qdrant_payload
    ):
        actions.append({"action": "sync_qdrant_payload", "memory_id": memory_id})

    item = RepairPlanItem(
        memory_id=memory_id,
        original_tags=original_tags,
        repaired_tags=repaired_tags,
        original_tag_prefixes=list(row["tag_prefixes"]),
        repaired_tag_prefixes=repaired_tag_prefixes,
        original_metadata=row["metadata"],
        repaired_metadata=repaired_metadata,
        actions=actions,
        original_qdrant_payload=original_qdrant_payload,
        repaired_qdrant_payload=repaired_qdrant_payload,
    )
    return item, rejected, canonicalized, ambiguous


def build_repair_plan(rows: Iterable[Any], *, mode: str = DEFAULT_REPAIR_MODE) -> RepairPlanResult:
    if mode not in REPAIR_MODES:
        raise ValueError(f"unknown repair mode: {mode}")

    normalized_rows = [_normalize_row(row) for row in rows]
    people_targets = _collect_people_targets(normalized_rows) if mode == "canonicalize-safe" else {}

    items: list[RepairPlanItem] = []
    unchanged = 0
    rejected_tags: list[dict[str, Any]] = []
    canonicalized_tags: list[dict[str, Any]] = []
    ambiguous_people: list[dict[str, Any]] = []

    for row in normalized_rows:
        item, rejected, canonicalized, ambiguous = _repair_row(
            row,
            people_targets=people_targets,
            mode=mode,
        )
        rejected_tags.extend(rejected)
        canonicalized_tags.extend(canonicalized)
        ambiguous_people.extend(ambiguous)

        changed = (
            item.tag_changed
            or item.tag_prefix_changed
            or item.metadata_changed
            or item.qdrant_payload_changed
        )
        if changed:
            items.append(item)
        else:
            unchanged += 1

    return RepairPlanResult(
        mode=mode,
        items=items,
        processed_count=len(normalized_rows),
        unchanged_count=unchanged,
        rejected_tags=rejected_tags,
        canonicalized_tags=canonicalized_tags,
        ambiguous_people=ambiguous_people,
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = dict(row)
            if isinstance(clean.get("candidates"), list):
                clean["candidates"] = ";".join(clean["candidates"])
            writer.writerow(clean)


def write_audit_artifacts(report_dir: Path | str, result: RepairPlanResult) -> None:
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)
    (report_path / "summary.json").write_text(
        json.dumps(result.summary(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(report_path / "plan.jsonl", (item.to_plan_record() for item in result.items))
    _write_jsonl(
        report_path / "rollback.jsonl", (item.to_rollback_record() for item in result.items)
    )
    _write_csv(
        report_path / "rejected-tags.csv",
        ["memory_id", "tag", "category", "slug", "reason"],
        result.rejected_tags,
    )
    _write_csv(
        report_path / "canonicalized-tags.csv",
        ["memory_id", "original_tag", "canonical_tag", "reason", "confidence"],
        result.canonicalized_tags,
    )
    _write_csv(
        report_path / "ambiguous-people.csv",
        ["memory_id", "tag", "reason", "candidates"],
        result.ambiguous_people,
    )


def load_plan_items(path: Path | str) -> list[RepairPlanItem]:
    items: list[RepairPlanItem] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            items.append(RepairPlanItem.from_record(json.loads(line)))
    return items


def apply_repair_plan(
    graph: Any,
    items: list[RepairPlanItem],
    *,
    qdrant_client: Any = None,
    qdrant_collection: Optional[str] = None,
    batch_size: int = 250,
    qdrant_retries: int = 3,
    qdrant_retry_delay_seconds: float = 0.5,
    qdrant_timeout_seconds: float = 60.0,
    graph_timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    graph_updates = 0
    graph_failures = 0
    graph_failure_details: list[dict[str, Any]] = []
    qdrant_updates = 0
    qdrant_failures = 0
    qdrant_failure_details: list[dict[str, str]] = []
    tag_updates = sum(1 for item in items if item.tag_changed)
    tag_prefix_updates = sum(1 for item in items if item.tag_prefix_changed)
    metadata_updates = sum(1 for item in items if item.metadata_changed)

    for offset in range(0, len(items), batch_size):
        batch = items[offset : offset + batch_size]
        rows = [
            {
                "id": item.memory_id,
                "tags": item.repaired_tags,
                "tag_prefixes": item.repaired_tag_prefixes,
                "metadata": json.dumps(item.repaired_metadata, ensure_ascii=False, sort_keys=True),
            }
            for item in batch
            if item.tag_changed or item.tag_prefix_changed or item.metadata_changed
        ]
        if rows:
            try:
                with _graph_update_deadline(graph_timeout_seconds, rows=len(rows)):
                    graph.query(
                        """
                        UNWIND $rows AS row
                        MATCH (m:Memory {id: row.id})
                        SET m.tags = row.tags,
                            m.tag_prefixes = row.tag_prefixes,
                            m.metadata = row.metadata
                        """,
                        {"rows": rows},
                    )
                graph_updates += len(rows)
            except Exception as exc:
                graph_failures += 1
                graph_failure_details.append(
                    {
                        "offset": offset,
                        "rows": len(rows),
                        "error": str(exc) or exc.__class__.__name__,
                        "error_type": exc.__class__.__name__,
                    }
                )
                break

        if qdrant_client is not None and qdrant_collection:
            for item in batch:
                payload = item.repaired_qdrant_payload or _desired_qdrant_payload(
                    tags=item.repaired_tags,
                    tag_prefixes=item.repaired_tag_prefixes,
                    metadata=item.repaired_metadata,
                )
                last_error = ""
                attempts = max(0, qdrant_retries) + 1
                for attempt in range(attempts):
                    try:
                        with _qdrant_payload_deadline(
                            qdrant_timeout_seconds,
                            memory_id=item.memory_id,
                        ):
                            qdrant_client.set_payload(
                                collection_name=qdrant_collection,
                                points=[item.memory_id],
                                payload=payload,
                            )
                        qdrant_updates += 1
                        last_error = ""
                        break
                    except Exception as exc:
                        last_error = str(exc) or exc.__class__.__name__
                        if attempt + 1 < attempts and qdrant_retry_delay_seconds > 0:
                            time.sleep(qdrant_retry_delay_seconds)
                if last_error:
                    qdrant_failures += 1
                    qdrant_failure_details.append(
                        {"memory_id": item.memory_id, "error": last_error}
                    )

    return {
        "graph_updates": graph_updates,
        "graph_failures": graph_failures,
        "graph_failure_details": graph_failure_details,
        "qdrant_updates": qdrant_updates,
        "qdrant_failures": qdrant_failures,
        "qdrant_failure_details": qdrant_failure_details,
        "tag_updates": tag_updates,
        "tag_prefix_updates": tag_prefix_updates,
        "metadata_updates": metadata_updates,
    }


def connect_falkordb() -> Any:
    try:
        from falkordb import FalkorDB  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing falkordb dependency; install requirements.txt") from exc

    host = _get_env("FALKORDB_HOST", "localhost") or "localhost"
    port = int(_get_env("FALKORDB_PORT", "6379") or "6379")
    password = _get_env("FALKORDB_PASSWORD")
    username = _get_env("FALKORDB_USERNAME") or ("default" if password else None)
    graph_name = _get_env("FALKORDB_GRAPH", "memories") or "memories"

    db = FalkorDB(host=host, port=port, username=username, password=password)
    return db.select_graph(graph_name)


def connect_qdrant() -> tuple[Any, Optional[str]]:
    try:
        from qdrant_client import QdrantClient  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing qdrant-client dependency; install requirements.txt") from exc

    url = _get_env("QDRANT_URL", "http://localhost:6333")
    api_key = _get_env("QDRANT_API_KEY")
    collection = _get_env("QDRANT_COLLECTION", "memories") or "memories"
    timeout = float(_get_env("QDRANT_TIMEOUT_SECONDS", "60") or "60")
    if not url:
        return None, None
    return QdrantClient(url=url, api_key=api_key, timeout=timeout), collection


def _point_id(point: Any) -> str:
    if isinstance(point, dict):
        return str(point.get("id") or "")
    return str(getattr(point, "id", "") or "")


def _point_payload(point: Any) -> dict[str, Any]:
    if isinstance(point, dict):
        return point.get("payload") or {}
    return getattr(point, "payload", None) or {}


def _retrieve_qdrant_points(
    qdrant_client: Any,
    *,
    qdrant_collection: str,
    batch_ids: list[str],
    batch_retries: int,
    batch_retry_delay_seconds: float,
) -> list[Any]:
    attempts = max(0, batch_retries) + 1
    last_error: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            return list(
                qdrant_client.retrieve(
                    collection_name=qdrant_collection,
                    ids=batch_ids,
                    with_payload=True,
                    with_vectors=False,
                )
                or []
            )
        except Exception as exc:
            if not _is_query_timeout(exc):
                raise
            last_error = exc
            if attempt + 1 < attempts and batch_retry_delay_seconds > 0:
                time.sleep(batch_retry_delay_seconds)

    if len(batch_ids) <= 1:
        assert last_error is not None
        raise last_error

    midpoint = max(1, len(batch_ids) // 2)
    return [
        *_retrieve_qdrant_points(
            qdrant_client,
            qdrant_collection=qdrant_collection,
            batch_ids=batch_ids[:midpoint],
            batch_retries=batch_retries,
            batch_retry_delay_seconds=batch_retry_delay_seconds,
        ),
        *_retrieve_qdrant_points(
            qdrant_client,
            qdrant_collection=qdrant_collection,
            batch_ids=batch_ids[midpoint:],
            batch_retries=batch_retries,
            batch_retry_delay_seconds=batch_retry_delay_seconds,
        ),
    ]


def attach_qdrant_payloads(
    rows: list[dict[str, Any]],
    *,
    qdrant_client: Any,
    qdrant_collection: str,
    batch_size: int = 250,
    batch_retries: int = 3,
    batch_retry_delay_seconds: float = 0.5,
) -> list[dict[str, Any]]:
    enriched = [dict(row) for row in rows]
    index_by_id = {row["id"]: index for index, row in enumerate(enriched) if row.get("id")}
    ids = sorted(index_by_id)
    for offset in range(0, len(ids), batch_size):
        batch_ids = ids[offset : offset + batch_size]
        points = _retrieve_qdrant_points(
            qdrant_client,
            qdrant_collection=qdrant_collection,
            batch_ids=batch_ids,
            batch_retries=batch_retries,
            batch_retry_delay_seconds=batch_retry_delay_seconds,
        )
        for point in points or []:
            memory_id = _point_id(point)
            if memory_id not in index_by_id:
                continue
            enriched[index_by_id[memory_id]]["qdrant_payload"] = _parse_qdrant_payload(
                _point_payload(point)
            )
    return enriched


def _is_query_timeout(exc: Exception) -> bool:
    text = f"{exc.__class__.__name__}: {exc}".lower()
    transient_markers = (
        "timed out",
        "timeout",
        "server disconnected",
        "remoteprotocolerror",
        "responsehandlingexception",
        "connection reset",
        "connection aborted",
        "connection refused",
    )
    return any(marker in text for marker in transient_markers)


def _fetch_memory_rows(
    graph: Any,
    batch_ids: list[str],
    *,
    batch_retries: int,
    batch_retry_delay_seconds: float,
) -> list[list[Any]]:
    attempts = max(0, batch_retries) + 1
    last_error: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            result = graph.query(
                """
                MATCH (m:Memory)
                WHERE m.id IN $ids
                RETURN m.id, m.tags, m.tag_prefixes, m.metadata, m.content
                """,
                {"ids": batch_ids},
            )
            return list(getattr(result, "result_set", []) or [])
        except Exception as exc:
            if not _is_query_timeout(exc):
                raise
            last_error = exc
            if attempt + 1 < attempts and batch_retry_delay_seconds > 0:
                time.sleep(batch_retry_delay_seconds)

    if len(batch_ids) <= 1:
        assert last_error is not None
        raise last_error

    midpoint = max(1, len(batch_ids) // 2)
    return [
        *_fetch_memory_rows(
            graph,
            batch_ids[:midpoint],
            batch_retries=batch_retries,
            batch_retry_delay_seconds=batch_retry_delay_seconds,
        ),
        *_fetch_memory_rows(
            graph,
            batch_ids[midpoint:],
            batch_retries=batch_retries,
            batch_retry_delay_seconds=batch_retry_delay_seconds,
        ),
    ]


def _fetch_memory_id_page(
    graph: Any,
    *,
    after: str,
    page_size: int,
    batch_retries: int,
    batch_retry_delay_seconds: float,
) -> list[str]:
    attempts = max(0, batch_retries) + 1
    current_page_size = max(1, page_size)
    last_error: Optional[Exception] = None

    while current_page_size >= 1:
        for attempt in range(attempts):
            try:
                result = graph.query(
                    """
                    MATCH (m:Memory)
                    WHERE $after = '' OR m.id > $after
                    RETURN m.id
                    ORDER BY m.id
                    LIMIT $limit
                    """,
                    {"after": after, "limit": current_page_size},
                )
                return [
                    str(row[0]).strip()
                    for row in (getattr(result, "result_set", []) or [])
                    if row and row[0]
                ]
            except Exception as exc:
                if not _is_query_timeout(exc):
                    raise
                last_error = exc
                if attempt + 1 < attempts and batch_retry_delay_seconds > 0:
                    time.sleep(batch_retry_delay_seconds)

        if current_page_size <= 1:
            assert last_error is not None
            raise last_error
        current_page_size = max(1, current_page_size // 2)

    assert last_error is not None
    raise last_error


def _iter_memory_ids(
    graph: Any,
    *,
    id_page_size: int,
    limit: int,
    batch_retries: int,
    batch_retry_delay_seconds: float,
) -> Iterator[str]:
    after = ""
    yielded = 0
    seen: set[str] = set()
    while True:
        remaining = max(0, limit - yielded) if limit else 0
        page_size = min(id_page_size, remaining) if remaining else id_page_size
        ids = _fetch_memory_id_page(
            graph,
            after=after,
            page_size=page_size,
            batch_retries=batch_retries,
            batch_retry_delay_seconds=batch_retry_delay_seconds,
        )
        ids = [memory_id for memory_id in ids if memory_id not in seen]
        if not ids:
            return

        for memory_id in ids:
            seen.add(memory_id)
            yielded += 1
            yield memory_id
            if limit and yielded >= limit:
                return

        after = ids[-1]
        if len(ids) < page_size:
            return


def iter_memory_rows(
    graph: Any,
    *,
    batch_size: int,
    limit: int = 0,
    id_page_size: int = 0,
    batch_retries: int = 2,
    batch_retry_delay_seconds: float = 0.25,
) -> Iterator[dict[str, Any]]:
    batch_size = max(1, batch_size)
    id_page_size = max(1, id_page_size or batch_size * 2)
    memory_ids = list(
        _iter_memory_ids(
            graph,
            id_page_size=id_page_size,
            limit=limit,
            batch_retries=batch_retries,
            batch_retry_delay_seconds=batch_retry_delay_seconds,
        )
    )

    for offset in range(0, len(memory_ids), batch_size):
        batch_ids = memory_ids[offset : offset + batch_size]
        result_rows = _fetch_memory_rows(
            graph,
            batch_ids,
            batch_retries=batch_retries,
            batch_retry_delay_seconds=batch_retry_delay_seconds,
        )
        rows_by_id: dict[str, dict[str, Any]] = {}
        for row in result_rows:
            normalized = _normalize_row(row)
            memory_id = normalized["id"]
            if memory_id:
                rows_by_id[memory_id] = normalized

        for memory_id in batch_ids:
            normalized = rows_by_id.get(memory_id)
            if normalized:
                yield normalized


def _default_report_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return PROJECT_ROOT / "lab" / "results" / "entity-tag-repair" / timestamp


def _write_apply_summary(report_dir: Path, *, mode: str, stats: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": mode,
        "dry_run": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **stats,
    }
    (report_dir / f"{mode}-summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--limit", type=int, default=0, help="Process at most N memories.")
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--plan", default=None, help="Plan JSONL to execute.")
    parser.add_argument(
        "--mode",
        choices=REPAIR_MODES,
        default=None,
        help=f"Dry-run planning mode. Default: {DEFAULT_REPAIR_MODE}.",
    )
    parser.add_argument(
        "--sync-only",
        action="store_const",
        const="sync-only",
        dest="mode",
        help="Only sync tag_prefixes/metadata.entities from existing tags.",
    )
    parser.add_argument(
        "--reject-only",
        action="store_const",
        const="reject-only",
        dest="mode",
        help="Remove rejected entity tags without canonicalizing accepted tags.",
    )
    parser.add_argument(
        "--canonicalize-safe",
        action="store_const",
        const="canonicalize-safe",
        dest="mode",
        help="Reject low-quality tags and apply high-confidence canonicalization.",
    )
    parser.add_argument("--execute", action="store_true", help="Apply a previously written plan.")
    parser.add_argument("--rollback", default=None, help="Rollback JSONL to apply.")
    parser.add_argument("--no-qdrant", action="store_true", help="Do not sync Qdrant payloads.")
    parser.add_argument(
        "--qdrant-retries",
        type=int,
        default=int(os.getenv("QDRANT_PAYLOAD_REPAIR_RETRIES", "5")),
        help="Retries per Qdrant set_payload call during execute/rollback.",
    )
    parser.add_argument(
        "--qdrant-retry-delay-seconds",
        type=float,
        default=float(os.getenv("QDRANT_PAYLOAD_REPAIR_RETRY_DELAY_SECONDS", "0.5")),
        help="Delay between Qdrant set_payload retries.",
    )
    parser.add_argument(
        "--qdrant-timeout-seconds",
        type=float,
        default=float(
            os.getenv(
                "QDRANT_PAYLOAD_REPAIR_TIMEOUT_SECONDS",
                os.getenv("QDRANT_TIMEOUT_SECONDS", "60"),
            )
        ),
        help="Max seconds per Qdrant set_payload call during execute/rollback; 0 disables.",
    )
    parser.add_argument(
        "--graph-scan-retries",
        type=int,
        default=int(os.getenv("FALKORDB_SCAN_RETRIES", "2")),
        help="Retries per Memory property batch before splitting the batch.",
    )
    parser.add_argument(
        "--graph-id-page-size",
        type=int,
        default=int(os.getenv("FALKORDB_SCAN_ID_PAGE_SIZE", "0")),
        help="Memory id page size for local graph scans; 0 uses 2x --batch-size.",
    )
    parser.add_argument(
        "--graph-scan-retry-delay-seconds",
        type=float,
        default=float(os.getenv("FALKORDB_SCAN_RETRY_DELAY_SECONDS", "0.25")),
        help="Delay between Memory property batch retries.",
    )
    parser.add_argument(
        "--graph-update-timeout-seconds",
        type=float,
        default=float(os.getenv("FALKORDB_UPDATE_TIMEOUT_SECONDS", "60")),
        help="Max seconds per FalkorDB write batch during execute/rollback; 0 disables.",
    )
    parser.add_argument(
        "--allow-non-local",
        action="store_true",
        help="Allow connecting to non-local FalkorDB/Qdrant targets.",
    )
    args = parser.parse_args()

    if args.execute and args.rollback:
        parser.error("--execute and --rollback are mutually exclusive")
    if args.execute and not args.plan:
        parser.error("--execute requires --plan")

    assert_local_targets(allow_non_local=args.allow_non_local, check_qdrant=not args.no_qdrant)

    if args.execute or args.rollback:
        plan_path = Path(args.plan if args.execute else args.rollback)
        report_dir = Path(args.report_dir) if args.report_dir else plan_path.parent
        items = load_plan_items(plan_path)
        mode = "execute" if args.execute else "rollback"
        synthetic_result = RepairPlanResult(
            mode=mode,
            items=items,
            processed_count=len(items),
            unchanged_count=0,
            rejected_tags=[],
            canonicalized_tags=[],
            ambiguous_people=[],
        )
        bare_summary = synthetic_result.summary()
        if has_bare_tag_mutations(bare_summary):
            stats = {
                "graph_updates": 0,
                "qdrant_updates": 0,
                "qdrant_failures": 0,
                "bare_tag_mutation_failure": True,
                **{
                    key: bare_summary[key]
                    for key in (
                        "bare_tags_removed",
                        "bare_tags_added",
                        "memories_with_bare_tag_changes",
                    )
                },
            }
            _write_apply_summary(report_dir, mode=mode, stats=stats)
            print(json.dumps({"mode": mode, **stats}, sort_keys=True))
            return 3
        graph = connect_falkordb()
        qdrant_client = None
        qdrant_collection = None
        if not args.no_qdrant:
            qdrant_client, qdrant_collection = connect_qdrant()
        stats = apply_repair_plan(
            graph,
            items,
            qdrant_client=qdrant_client,
            qdrant_collection=qdrant_collection,
            batch_size=args.batch_size,
            qdrant_retries=args.qdrant_retries,
            qdrant_retry_delay_seconds=args.qdrant_retry_delay_seconds,
            qdrant_timeout_seconds=args.qdrant_timeout_seconds,
            graph_timeout_seconds=args.graph_update_timeout_seconds,
        )
        _write_apply_summary(report_dir, mode=mode, stats=stats)
        print(json.dumps({"mode": mode, **stats}, sort_keys=True))
        return 2 if stats.get("qdrant_failures") or stats.get("graph_failures") else 0

    report_dir = Path(args.report_dir) if args.report_dir else _default_report_dir()
    graph = connect_falkordb()
    start = time.time()
    rows = list(
        iter_memory_rows(
            graph,
            batch_size=args.batch_size,
            limit=args.limit,
            id_page_size=args.graph_id_page_size,
            batch_retries=args.graph_scan_retries,
            batch_retry_delay_seconds=args.graph_scan_retry_delay_seconds,
        )
    )
    if not args.no_qdrant:
        qdrant_client, qdrant_collection = connect_qdrant()
        if qdrant_client is not None and qdrant_collection:
            rows = attach_qdrant_payloads(
                rows,
                qdrant_client=qdrant_client,
                qdrant_collection=qdrant_collection,
                batch_size=args.batch_size,
                batch_retries=args.qdrant_retries,
                batch_retry_delay_seconds=args.qdrant_retry_delay_seconds,
            )
    result = build_repair_plan(rows, mode=args.mode or DEFAULT_REPAIR_MODE)
    write_audit_artifacts(report_dir, result)
    summary = {
        **result.summary(),
        "elapsed_seconds": round(time.time() - start, 2),
        "report_dir": str(report_dir),
    }
    print(json.dumps(summary, sort_keys=True))
    if has_bare_tag_mutations(summary):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
