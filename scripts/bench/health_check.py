#!/usr/bin/env python3
"""Recall health check — lightweight diagnostics beyond accuracy benchmarks.

Checks score distribution, entity quality, latency, and precision on curated
queries. Designed to run after restore_and_eval.sh or standalone.

Usage:
    python scripts/bench/health_check.py
    python scripts/bench/health_check.py --base-url http://localhost:8001 --output health.json
"""

import argparse
import json
import logging
import math
import os
import re
import statistics
import sys
import time
import uuid as uuid_mod
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

API_URL = os.getenv("AUTOMEM_TEST_BASE_URL", "http://localhost:8001")
API_TOKEN = os.getenv("AUTOMEM_TEST_API_TOKEN", os.getenv("AUTOMEM_API_TOKEN", "test-token"))

DIVERSE_QUERIES = [
    "What was discussed about authentication and security?",
    "Tell me about the birthday party last weekend",
    "What programming languages does the team use?",
    "Plans for the upcoming vacation trip",
    "Recent decisions about database architecture",
]

GARBAGE_ENTITY_PATTERNS = [
    re.compile(r"^entity:[^:]+:.$"),  # single-char slug
    re.compile(r"^entity:[^:]+:the$", re.IGNORECASE),
    re.compile(r"^entity:[^:]+:a$", re.IGNORECASE),
    re.compile(r"^entity:[^:]+:an$", re.IGNORECASE),
    re.compile(r"^entity:[^:]+:it$", re.IGNORECASE),
    re.compile(r"^entity:[^:]+:is$", re.IGNORECASE),
    re.compile(r"^entity:[^:]+:was$", re.IGNORECASE),
    re.compile(r"\\u[0-9a-fA-F]{4}"),  # unicode escapes in tags
    re.compile(r"^entity:[^:]+:.*'s$"),  # possessives
]


def get_headers(api_token: Optional[str] = None) -> dict:
    token = api_token or API_TOKEN
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def check_score_distribution(base_url: str, api_token: Optional[str] = None) -> Dict[str, Any]:
    """Hit recall with diverse queries and analyze score distributions.

    Catches the "everything scores the same" problem (#78) where importance
    dominates and semantic relevance has no signal.
    """
    all_scores: List[float] = []
    per_query: List[Dict[str, Any]] = []
    latencies: List[float] = []

    for query in DIVERSE_QUERIES:
        start = time.perf_counter()
        try:
            resp = requests.get(
                f"{base_url}/recall",
                params={"query": query, "limit": 10},
                headers=get_headers(api_token),
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as e:
            per_query.append({"query": query, "error": str(e)})
            continue

        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

        results = data.get("results", [])
        scores = [r.get("final_score", 0) for r in results]
        all_scores.extend(scores)

        per_query.append(
            {
                "query": query[:60],
                "count": len(results),
                "scores": {
                    "min": round(min(scores), 4) if scores else 0,
                    "max": round(max(scores), 4) if scores else 0,
                    "mean": round(statistics.mean(scores), 4) if scores else 0,
                    "spread": round(max(scores) - min(scores), 4) if scores else 0,
                },
                "latency_ms": round(latency_ms, 1),
                "query_time_ms": data.get("query_time_ms", 0),
            }
        )

    score_health = "unknown"
    spread = 0
    if all_scores:
        spread = max(all_scores) - min(all_scores)
        stddev = statistics.stdev(all_scores) if len(all_scores) > 1 else 0
        if stddev < 0.02:
            score_health = "BAD: scores are nearly identical — scoring formula is broken"
        elif stddev < 0.05:
            score_health = "WARN: low score variance — weak differentiation"
        else:
            score_health = "OK: scores show meaningful variance"

    return {
        "check": "score_distribution",
        "verdict": score_health,
        "total_scores": len(all_scores),
        "global_stats": {
            "min": round(min(all_scores), 4) if all_scores else 0,
            "max": round(max(all_scores), 4) if all_scores else 0,
            "mean": round(statistics.mean(all_scores), 4) if all_scores else 0,
            "stddev": (round(statistics.stdev(all_scores), 4) if len(all_scores) > 1 else 0),
            "spread": round(spread, 4),
        },
        "latency": {
            "p50_ms": round(statistics.median(latencies), 1) if latencies else 0,
            "p95_ms": (
                round(
                    sorted(latencies)[
                        max(
                            0,
                            min(len(latencies) - 1, math.ceil(0.95 * len(latencies)) - 1),
                        )
                    ],
                    1,
                )
                if latencies
                else 0
            ),
            "mean_ms": round(statistics.mean(latencies), 1) if latencies else 0,
        },
        "per_query": per_query,
    }


def check_entity_quality(
    base_url: str, sample_size: int = 30, api_token: Optional[str] = None
) -> Dict[str, Any]:
    """Sample memories and check entity tag quality.

    Catches #72 (misclassified entities) and #71 (name confusion).
    """
    try:
        resp = requests.get(
            f"{base_url}/recall",
            params={"query": "conversation people places things", "limit": sample_size},
            headers=get_headers(api_token),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        return {"check": "entity_quality", "verdict": f"ERROR: {e}", "sampled": 0}

    results = data.get("results", [])
    total_entity_tags = 0
    garbage_tags: List[str] = []
    memories_with_entities = 0
    memories_without_entities = 0

    for r in results:
        mem = r.get("memory", r)
        tags = mem.get("tags", [])
        entity_tags = [t for t in tags if t.startswith("entity:")]
        total_entity_tags += len(entity_tags)

        if entity_tags:
            memories_with_entities += 1
        else:
            memories_without_entities += 1

        for tag in entity_tags:
            for pattern in GARBAGE_ENTITY_PATTERNS:
                if pattern.search(tag):
                    garbage_tags.append(tag)
                    break

    garbage_pct = (len(garbage_tags) / total_entity_tags * 100) if total_entity_tags > 0 else 0

    if garbage_pct > 20:
        verdict = f"BAD: {garbage_pct:.0f}% garbage entity tags"
    elif garbage_pct > 5:
        verdict = f"WARN: {garbage_pct:.0f}% garbage entity tags"
    elif total_entity_tags == 0:
        verdict = "WARN: no entity tags found in sample"
    else:
        verdict = f"OK: {garbage_pct:.1f}% garbage ({len(garbage_tags)}/{total_entity_tags})"

    return {
        "check": "entity_quality",
        "verdict": verdict,
        "sampled_memories": len(results),
        "memories_with_entities": memories_with_entities,
        "memories_without_entities": memories_without_entities,
        "total_entity_tags": total_entity_tags,
        "garbage_tags": len(garbage_tags),
        "garbage_pct": round(garbage_pct, 1),
        "garbage_examples": garbage_tags[:10],
    }


def check_cross_query_overlap(base_url: str, api_token: Optional[str] = None) -> Dict[str, Any]:
    """Check if different queries return identical results (the original #78 symptom).

    If 3 unrelated queries return the same top-5 IDs, the scoring formula is broken.
    """
    query_results: Dict[str, List[str]] = {}

    for query in DIVERSE_QUERIES[:3]:
        try:
            resp = requests.get(
                f"{base_url}/recall",
                params={"query": query, "limit": 5},
                headers=get_headers(api_token),
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError):
            logger.exception("Cross-query overlap check failed for query: %s", query[:60])
            continue

        ids: List[str] = []
        for r in data.get("results", []):
            raw = r.get("id", r.get("memory", {}).get("id", ""))
            if not raw or not isinstance(raw, str):
                continue
            try:
                ids.append(str(uuid_mod.UUID(raw.strip())))
            except ValueError:
                continue
        query_results[query[:40]] = ids

    if len(query_results) < 2:
        return {
            "check": "cross_query_overlap",
            "verdict": "SKIP: not enough queries succeeded",
        }

    id_lists = list(query_results.values())
    overlap_pairs = 0
    total_pairs = 0
    for i in range(len(id_lists)):
        for j in range(i + 1, len(id_lists)):
            total_pairs += 1
            set_i = set(id_lists[i])
            set_j = set(id_lists[j])
            if set_i and set_j and set_i == set_j:
                overlap_pairs += 1

    if overlap_pairs == total_pairs and total_pairs > 0:
        verdict = "BAD: ALL queries return identical results — scoring is broken"
    elif overlap_pairs > 0:
        verdict = f"WARN: {overlap_pairs}/{total_pairs} query pairs return identical results"
    else:
        verdict = "OK: different queries return different results"

    return {
        "check": "cross_query_overlap",
        "verdict": verdict,
        "overlap_pairs": overlap_pairs,
        "total_pairs": total_pairs,
        "query_results": {k: v[:5] for k, v in query_results.items()},
    }


def run_all_checks(base_url: str, api_token: Optional[str] = None) -> Dict[str, Any]:
    """Run all health checks and return a summary."""
    print(f"Running health checks against {base_url}...")

    checks = []

    print("  [1/3] Score distribution & latency...")
    checks.append(check_score_distribution(base_url, api_token=api_token))

    print("  [2/3] Entity quality...")
    checks.append(check_entity_quality(base_url, api_token=api_token))

    print("  [3/3] Cross-query overlap...")
    checks.append(check_cross_query_overlap(base_url, api_token=api_token))

    failures = sum(1 for c in checks if c.get("verdict", "").startswith("BAD"))
    warnings = sum(1 for c in checks if c.get("verdict", "").startswith("WARN"))

    if failures > 0:
        overall = "UNHEALTHY"
    elif warnings > 0:
        overall = "DEGRADED"
    else:
        overall = "HEALTHY"

    print(f"\n{'=' * 60}")
    print(f"  RECALL HEALTH: {overall}")
    print(f"{'=' * 60}")
    for c in checks:
        status = c.get("verdict", "?")
        name = c.get("check", "?")
        print(f"  {name:30s} {status}")

    latency_info = checks[0].get("latency", {})
    if latency_info:
        print(
            f"\n  Latency: p50={latency_info.get('p50_ms', 0):.0f}ms "
            f"p95={latency_info.get('p95_ms', 0):.0f}ms "
            f"mean={latency_info.get('mean_ms', 0):.0f}ms"
        )

    print(f"{'=' * 60}")

    return {
        "overall": overall,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": base_url,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoMem recall health check")
    parser.add_argument("--base-url", default=API_URL, help="AutoMem API URL")
    parser.add_argument("--api-token", default=None, help="API token")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    report = run_all_checks(args.base_url, api_token=args.api_token)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {args.output}")

    if report["overall"] == "UNHEALTHY":
        sys.exit(2)
    elif report["overall"] == "DEGRADED":
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
