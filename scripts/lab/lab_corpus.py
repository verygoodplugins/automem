"""Corpus-side helpers for the AutoMem Recall Quality Lab.

All HTTP lives here, behind injectable clients (http_get / http_post) so the
logic is unit-testable without a live server. Imported by run_recall_test.py
and the parallel matrix harness.
"""

import time
from typing import Any, Dict, List, Optional

import requests


def recall(
    api_url: str,
    headers: Dict[str, str],
    query: str,
    *,
    limit: int = 20,
    expand_relations: bool = False,
    current_only: bool = True,
    recency_bias: Optional[str] = None,
    context_tags: Optional[List[str]] = None,
    http_get=requests.get,
) -> Dict[str, Any]:
    """GET /recall with explicit recall parameters; returns parsed JSON."""
    params: Dict[str, Any] = {"query": query, "limit": limit}
    if expand_relations:
        params["expand_relations"] = "true"
    params["current_only"] = "true" if current_only else "false"
    if recency_bias is not None:
        params["recency_bias"] = recency_bias
    if context_tags:
        params["context_tags"] = context_tags
    resp = http_get(f"{api_url}/recall", params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def iso_days_ago(days: float) -> str:
    """UTC ISO-8601 timestamp `days` in the past."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - days * 86400))


def make_distractor_memories(
    n: int,
    *,
    age_days: float = 180,
    importance: float = 0.05,
    tag: str = "lab-distractor",
) -> List[Dict[str, Any]]:
    """Build n aged, low-importance, labelled distractor /memory payloads.

    Pre-backdated so the `forget` consolidation mode treats them as stale; the
    `lab-distractor` tag + metadata flag make them unambiguous noise for the
    distractor-precision metric.
    """
    ts = iso_days_ago(age_days)
    payloads: List[Dict[str, Any]] = []
    for i in range(n):
        payloads.append(
            {
                "content": (
                    f"[lab-distractor #{i}] stale unrelated note about "
                    f"miscellaneous topic {i}; safe to forget."
                ),
                "tags": [tag],
                "importance": importance,
                "timestamp": ts,
                "last_accessed": ts,
                "metadata": {"lab_distractor": True},
            }
        )
    return payloads


def inject_distractors(
    api_url: str,
    headers: Dict[str, str],
    payloads: List[Dict[str, Any]],
    *,
    http_post=requests.post,
) -> List[str]:
    """POST distractor memories; return the created memory IDs."""
    ids: List[str] = []
    for payload in payloads:
        resp = http_post(f"{api_url}/memory", json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        mid = str(
            data.get("memory_id") or data.get("id") or (data.get("memory") or {}).get("id") or ""
        )
        if mid:
            ids.append(mid)
    return ids


def extract_ids(recall_json: Dict[str, Any]) -> List[str]:
    """Pull memory IDs from a /recall response (nested or flat shape)."""
    results = recall_json.get("results", recall_json.get("memories", []))
    ids: List[str] = []
    for r in results:
        mem = r.get("memory", r)
        mid = str(mem.get("id", r.get("id", "")))
        if mid:
            ids.append(mid)
    return ids


CONSOLIDATION_ORDER = ["decay", "creative", "cluster", "forget"]


def run_consolidation(
    api_url: str,
    headers: Dict[str, str],
    modes: Optional[List[str]] = None,
    *,
    dry_run: bool = False,
    http_post=requests.post,
) -> Dict[str, Dict[str, Any]]:
    """Run a REAL consolidation pass per mode, in order.

    dry_run defaults to False: the /consolidate endpoint's own default is True,
    which makes creative/cluster/forget silent no-ops. Decay runs first so
    creative/cluster see memories with relevance_score > 0.3.
    """
    if modes is None:
        modes = list(CONSOLIDATION_ORDER)
    out: Dict[str, Dict[str, Any]] = {}
    for mode in modes:
        resp = http_post(
            f"{api_url}/consolidate",
            json={"mode": mode, "dry_run": dry_run},
            headers=headers,
            timeout=120,
        )
        resp.raise_for_status()
        out[mode] = resp.json()
    return out
