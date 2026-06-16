"""Corpus-side helpers for the AutoMem Recall Quality Lab.

All HTTP lives here, behind injectable clients (http_get / http_post) so the
logic is unit-testable without a live server. Imported by run_recall_test.py
and the parallel matrix harness.
"""

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
    http_get=requests.get,
) -> Dict[str, Any]:
    """GET /recall with explicit recall parameters; returns parsed JSON."""
    params: Dict[str, Any] = {"query": query, "limit": limit}
    if expand_relations:
        params["expand_relations"] = "true"
    params["current_only"] = "true" if current_only else "false"
    if recency_bias is not None:
        params["recency_bias"] = recency_bias
    resp = http_get(f"{api_url}/recall", params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


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
