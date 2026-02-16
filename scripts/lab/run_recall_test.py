#!/usr/bin/env python3
"""Recall Quality Test Harness — run recall tests with statistical comparison.

Executes test queries against AutoMem with different scoring configurations,
computes IR metrics (Recall@K, MRR, NDCG), and performs statistical comparison.

Usage:
    # Run baseline test
    python scripts/lab/run_recall_test.py --config baseline

    # Compare two configs
    python scripts/lab/run_recall_test.py --config fix_v1 --compare baseline

    # Sweep a parameter
    python scripts/lab/run_recall_test.py --sweep SEARCH_WEIGHT_VECTOR 0.20,0.30,0.40,0.50

    # Use custom test set
    python scripts/lab/run_recall_test.py --config baseline --test-set lab/test_sets/custom.json
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

API_URL = os.getenv("AUTOMEM_TEST_BASE_URL", "http://localhost:8001")
API_TOKEN = os.getenv("AUTOMEM_API_TOKEN", "test-token")
CONFIGS_DIR = Path(__file__).parent / "configs"
RESULTS_DIR = Path("lab/results")


def get_headers() -> dict:
    return {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}


# ---------- Metrics ----------


def recall_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int) -> float:
    """Fraction of expected IDs found in top-K results."""
    if not expected_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = sum(1 for eid in expected_ids if eid in top_k)
    return hits / len(expected_ids)


def mrr(retrieved_ids: List[str], expected_ids: List[str]) -> float:
    """Mean Reciprocal Rank — position of first relevant result."""
    expected_set = set(expected_ids)
    for i, rid in enumerate(retrieved_ids):
        if rid in expected_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    expected_set = set(expected_ids)
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in expected_set:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1) = 0

    # Ideal DCG: all relevant docs at top
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(expected_ids), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ---------- Statistical Comparison ----------


def paired_ttest(a: List[float], b: List[float]) -> Dict[str, Any]:
    """Paired t-test + Cohen's d effect size. Pure Python, no scipy needed."""
    n = len(a)
    if n < 2 or n != len(b):
        return {"p_value": 1.0, "t_stat": 0.0, "effect_size": 0.0, "significant": False}

    diffs = [b[i] - a[i] for i in range(n)]
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    std_d = var_d**0.5 if var_d > 0 else 1e-10

    t_stat = mean_d / (std_d / n**0.5)

    # Approximate p-value using normal distribution for large n
    # For n >= 30 this is reasonable; for smaller n it's conservative
    z = abs(t_stat)
    # Approximation of 2-tailed p from standard normal
    p_value = 2 * (1 - 0.5 * (1 + math.erf(z / 2**0.5)))

    # Cohen's d
    pooled_std = (
        (sum((ai - sum(a) / n) ** 2 for ai in a) + sum((bi - sum(b) / n) ** 2 for bi in b))
        / (2 * n - 2)
    ) ** 0.5
    cohens_d = (sum(b) / n - sum(a) / n) / pooled_std if pooled_std > 0 else 0.0

    effect_label = "negligible"
    if abs(cohens_d) >= 0.8:
        effect_label = "large"
    elif abs(cohens_d) >= 0.5:
        effect_label = "medium"
    elif abs(cohens_d) >= 0.2:
        effect_label = "small"

    return {
        "t_stat": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "cohens_d": round(cohens_d, 4),
        "effect_size": effect_label,
        "significant": p_value < 0.05,
        "mean_diff": round(mean_d, 4),
    }


# ---------- Config Management ----------


def load_config(name: str) -> Dict[str, str]:
    """Load a named config from configs/ directory."""
    config_path = CONFIGS_DIR / f"{name}.json"
    if not config_path.exists():
        print(f"ERROR: Config '{name}' not found at {config_path}")
        print(f"Available configs: {[f.stem for f in CONFIGS_DIR.glob('*.json')]}")
        sys.exit(1)

    with open(config_path) as f:
        return json.load(f)


def apply_config(config: Dict[str, str]) -> Dict[str, str]:
    """Apply config by setting environment variables. Returns original values for restore."""
    original = {}
    for key, value in config.items():
        original[key] = os.environ.get(key)
        os.environ[key] = str(value)
    return original


def restore_config(original: Dict[str, Optional[str]]):
    """Restore original environment variables."""
    for key, value in original.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def restart_api_with_config(config: Dict[str, str]):
    """Restart the Docker API container with new environment variables.

    For env vars that are read at import time (like scoring weights),
    we need to restart the Flask process.
    """
    env_args = " ".join(f"-e {k}={v}" for k, v in config.items())

    # Check if any scoring/consolidation weights changed — these need restart
    needs_restart = any(
        k.startswith(("SEARCH_WEIGHT_", "CONSOLIDATION_", "RECALL_")) for k in config
    )

    if needs_restart:
        # Update docker compose env and restart
        subprocess.run(
            ["docker", "compose", "up", "-d", "--no-deps", "flask-api"],
            capture_output=True,
            env={**os.environ, **config},
            cwd=str(Path(__file__).parent.parent.parent),
        )
        # Wait for API to be ready
        for _ in range(30):
            try:
                resp = requests.get(f"{API_URL}/health", timeout=2)
                if resp.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(1)


# ---------- Test Runner ----------


@dataclass
class QueryResult:
    query: str
    expected_ids: List[str]
    retrieved_ids: List[str]
    recall_5: float = 0.0
    recall_10: float = 0.0
    recall_20: float = 0.0
    mrr_val: float = 0.0
    ndcg_10: float = 0.0
    latency_ms: float = 0.0
    category: str = ""


@dataclass
class TestRunResult:  # noqa: pytest will skip due to __init__
    __test__ = False  # Tell pytest this is not a test class
    config_name: str
    query_results: List[QueryResult] = field(default_factory=list)
    timestamp: str = ""

    @property
    def mean_recall_5(self) -> float:
        vals = [q.recall_5 for q in self.query_results]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def mean_recall_10(self) -> float:
        vals = [q.recall_10 for q in self.query_results]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def mean_recall_20(self) -> float:
        vals = [q.recall_20 for q in self.query_results]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def mean_mrr(self) -> float:
        vals = [q.mrr_val for q in self.query_results]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def mean_ndcg_10(self) -> float:
        vals = [q.ndcg_10 for q in self.query_results]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def mean_latency(self) -> float:
        vals = [q.latency_ms for q in self.query_results]
        return sum(vals) / len(vals) if vals else 0.0

    def by_category(self) -> Dict[str, Dict[str, float]]:
        from collections import defaultdict

        cats: Dict[str, List[QueryResult]] = defaultdict(list)
        for q in self.query_results:
            cats[q.category].append(q)
        return {
            cat: {
                "recall_10": sum(q.recall_10 for q in qs) / len(qs),
                "mrr": sum(q.mrr_val for q in qs) / len(qs),
                "ndcg_10": sum(q.ndcg_10 for q in qs) / len(qs),
                "count": len(qs),
            }
            for cat, qs in cats.items()
        }


def run_single_query(query_data: Dict[str, Any], api_url: str) -> QueryResult:
    """Execute a single recall query and compute metrics."""
    query = query_data["query"]
    expected_ids = query_data.get("expected_ids", [])
    category = query_data.get("category", "unknown")

    start = time.perf_counter()
    try:
        resp = requests.get(
            f"{api_url}/recall",
            params={"query": query, "limit": 20},
            headers=get_headers(),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  WARNING: Query failed: {query[:50]}... — {e}")
        return QueryResult(
            query=query, expected_ids=expected_ids, retrieved_ids=[], category=category
        )

    latency_ms = (time.perf_counter() - start) * 1000

    results = data.get("results", data.get("memories", []))
    retrieved_ids = []
    for r in results:
        mem = r.get("memory", r)
        mid = str(mem.get("id", r.get("id", "")))
        if mid:
            retrieved_ids.append(mid)

    return QueryResult(
        query=query,
        expected_ids=expected_ids,
        retrieved_ids=retrieved_ids,
        recall_5=recall_at_k(retrieved_ids, expected_ids, 5),
        recall_10=recall_at_k(retrieved_ids, expected_ids, 10),
        recall_20=recall_at_k(retrieved_ids, expected_ids, 20),
        mrr_val=mrr(retrieved_ids, expected_ids),
        ndcg_10=ndcg_at_k(retrieved_ids, expected_ids, 10),
        latency_ms=latency_ms,
        category=category,
    )


def run_test(config_name: str, queries: List[Dict], api_url: str) -> TestRunResult:
    """Run all test queries with a specific config."""
    result = TestRunResult(
        config_name=config_name, timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    for i, q in enumerate(queries):
        qr = run_single_query(q, api_url)
        result.query_results.append(qr)
        # Progress dot
        hit = "." if qr.recall_10 > 0 else "x"
        print(hit, end="", flush=True)
        if (i + 1) % 50 == 0:
            print(f" [{i + 1}/{len(queries)}]")

    print(f" [{len(queries)}/{len(queries)}]")
    return result


def print_summary(result: TestRunResult):
    """Print a summary of test results."""
    print(f"\n--- {result.config_name} ---")
    print(f"  Recall@5:   {result.mean_recall_5:.3f}")
    print(f"  Recall@10:  {result.mean_recall_10:.3f}")
    print(f"  Recall@20:  {result.mean_recall_20:.3f}")
    print(f"  MRR:        {result.mean_mrr:.3f}")
    print(f"  NDCG@10:    {result.mean_ndcg_10:.3f}")
    print(f"  Latency:    {result.mean_latency:.0f}ms avg")

    cats = result.by_category()
    if len(cats) > 1:
        print(f"\n  By category:")
        for cat, metrics in sorted(cats.items()):
            print(
                f"    {cat:12s} R@10={metrics['recall_10']:.3f}  "
                f"MRR={metrics['mrr']:.3f}  n={metrics['count']}"
            )


def print_comparison(a: TestRunResult, b: TestRunResult):
    """Print statistical comparison between two test runs."""
    print(f"\n=== COMPARISON: {a.config_name} -> {b.config_name} ===")

    metrics = [
        ("Recall@5", [q.recall_5 for q in a.query_results], [q.recall_5 for q in b.query_results]),
        (
            "Recall@10",
            [q.recall_10 for q in a.query_results],
            [q.recall_10 for q in b.query_results],
        ),
        (
            "Recall@20",
            [q.recall_20 for q in a.query_results],
            [q.recall_20 for q in b.query_results],
        ),
        ("MRR", [q.mrr_val for q in a.query_results], [q.mrr_val for q in b.query_results]),
        (
            "NDCG@10",
            [q.ndcg_10 for q in a.query_results],
            [q.ndcg_10 for q in b.query_results],
        ),
    ]

    for name, vals_a, vals_b in metrics:
        stats = paired_ttest(vals_a, vals_b)
        mean_a = sum(vals_a) / len(vals_a) if vals_a else 0
        mean_b = sum(vals_b) / len(vals_b) if vals_b else 0
        diff = mean_b - mean_a
        pct = (diff / mean_a * 100) if mean_a > 0 else 0

        sig = "**" if stats["significant"] else "  "
        direction = "+" if diff >= 0 else ""
        print(
            f"  {sig}{name:10s}: {mean_a:.3f} -> {mean_b:.3f}  "
            f"({direction}{diff:.3f}, {direction}{pct:.1f}%)  "
            f"p={stats['p_value']:.4f}  d={stats['cohens_d']:.2f} ({stats['effect_size']})"
        )

    print(f"\n  ** = statistically significant (p < 0.05)")


def save_results(result: TestRunResult, output_dir: Path):
    """Save test results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{result.config_name}_{ts}.json"

    data = {
        "config_name": result.config_name,
        "timestamp": result.timestamp,
        "summary": {
            "recall_5": result.mean_recall_5,
            "recall_10": result.mean_recall_10,
            "recall_20": result.mean_recall_20,
            "mrr": result.mean_mrr,
            "ndcg_10": result.mean_ndcg_10,
            "latency_ms": result.mean_latency,
        },
        "by_category": result.by_category(),
        "queries": [
            {
                "query": q.query,
                "expected_ids": q.expected_ids,
                "retrieved_ids": q.retrieved_ids[:10],
                "recall_10": q.recall_10,
                "mrr": q.mrr_val,
                "ndcg_10": q.ndcg_10,
                "latency_ms": round(q.latency_ms, 1),
                "category": q.category,
            }
            for q in result.query_results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved: {output_path}")
    return output_path


def load_test_set(path: str) -> List[Dict]:
    """Load test queries from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return data.get("queries", data) if isinstance(data, dict) else data


def main():
    parser = argparse.ArgumentParser(description="AutoMem Recall Quality Test Harness")
    parser.add_argument("--config", type=str, required=False, help="Config name to test")
    parser.add_argument("--compare", type=str, help="Config name to compare against")
    parser.add_argument("--sweep", nargs=2, metavar=("PARAM", "VALUES"), help="Sweep a parameter")
    parser.add_argument(
        "--test-set",
        type=str,
        default="lab/test_sets/queries_100.json",
        help="Test query set JSON",
    )
    parser.add_argument("--api-url", type=str, default=API_URL, help="AutoMem API URL")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR), help="Results dir")
    args = parser.parse_args()

    # Load test set
    if not Path(args.test_set).exists():
        # Try to find any test set
        test_sets = (
            sorted(Path("lab/test_sets").glob("*.json")) if Path("lab/test_sets").exists() else []
        )
        if test_sets:
            args.test_set = str(test_sets[0])
            print(f"Using test set: {args.test_set}")
        else:
            print("ERROR: No test set found. Run create_test_queries.py first.")
            sys.exit(1)

    queries = load_test_set(args.test_set)
    print(f"Loaded {len(queries)} test queries from {args.test_set}")

    output_dir = Path(args.output_dir)

    if args.sweep:
        # Parameter sweep mode
        param_name, values_str = args.sweep
        values = [v.strip() for v in values_str.split(",")]
        print(f"\n=== SWEEP: {param_name} = {values} ===\n")

        all_results = []
        for val in values:
            config_name = f"sweep_{param_name}_{val}"
            config = {param_name: val}

            print(f"\n--- Testing {param_name}={val} ---")
            restart_api_with_config(config)
            time.sleep(2)

            result = run_test(config_name, queries, args.api_url)
            all_results.append(result)
            print_summary(result)
            save_results(result, output_dir)

        # Print sweep comparison table
        print(f"\n=== SWEEP RESULTS: {param_name} ===")
        print(f"{'Value':>8s}  {'R@5':>6s}  {'R@10':>6s}  {'R@20':>6s}  {'MRR':>6s}  {'NDCG':>6s}")
        print("-" * 50)
        for val, result in zip(values, all_results):
            print(
                f"{val:>8s}  {result.mean_recall_5:6.3f}  {result.mean_recall_10:6.3f}  "
                f"{result.mean_recall_20:6.3f}  {result.mean_mrr:6.3f}  {result.mean_ndcg_10:6.3f}"
            )

    elif args.compare:
        # Comparison mode
        if not args.config:
            print("ERROR: --compare requires --config to specify the candidate config")
            sys.exit(1)
        config_a = load_config(args.compare)
        config_b = load_config(args.config)

        print(f"\n--- Running baseline: {args.compare} ---")
        restart_api_with_config(config_a)
        time.sleep(2)
        result_a = run_test(args.compare, queries, args.api_url)
        print_summary(result_a)
        save_results(result_a, output_dir)

        print(f"\n--- Running candidate: {args.config} ---")
        restart_api_with_config(config_b)
        time.sleep(2)
        result_b = run_test(args.config, queries, args.api_url)
        print_summary(result_b)
        save_results(result_b, output_dir)

        print_comparison(result_a, result_b)

    elif args.config:
        # Single config mode
        config = load_config(args.config)
        print(f"\n--- Running: {args.config} ---")
        restart_api_with_config(config)
        time.sleep(2)

        result = run_test(args.config, queries, args.api_url)
        print_summary(result)
        save_results(result, output_dir)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
