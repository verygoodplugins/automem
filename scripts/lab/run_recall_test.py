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
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab_corpus import (  # noqa: E402
    extract_ids,
    inject_distractors,
    make_distractor_memories,
    recall,
    run_consolidation,
)
from lab_metrics import (  # noqa: E402
    config_complexity,
    distractor_rate_at_k,
    mrr,
    ndcg_at_k,
    paired_ttest,
    recall_at_k,
)

API_URL = os.getenv("AUTOMEM_TEST_BASE_URL", "http://localhost:8001")
API_TOKEN = os.getenv("AUTOMEM_API_TOKEN", "test-token")
CONFIGS_DIR = Path(__file__).parent / "configs"
RESULTS_DIR = Path("lab/results")


def get_headers() -> dict:
    return {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}


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


def apply_config(config: Dict[str, str]) -> Dict[str, Optional[str]]:
    """Apply config by setting environment variables. Returns original values for restore."""
    original = {}
    for key, value in config.items():
        original[key] = os.environ.get(key)
        os.environ[key] = str(value)
    return original


def restore_config(original: Dict[str, Optional[str]]) -> None:
    """Restore original environment variables."""
    for key, value in original.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def restart_api_with_config(config: Dict[str, str], api_url: str = API_URL) -> None:
    """Restart the Docker API container with new environment variables.

    Writes overrides to .env.bench and uses --env-file to ensure Docker
    actually picks up the new values (not just the subprocess environment).
    """
    # Check if any scoring/consolidation settings changed — these need restart
    needs_restart = any(k.startswith(("SEARCH_", "CONSOLIDATION_", "RECALL_")) for k in config)

    if not needs_restart:
        return

    repo_root = Path(__file__).parent.parent.parent
    env_bench_path = repo_root / ".env.bench"

    # Write override env file so Docker Compose reads it at container start
    with open(env_bench_path, "w") as f:
        for k, v in config.items():
            f.write(f"{k}={v}\n")

    # Restart with override file applied on top of .env
    result = subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            ".env",
            "--env-file",
            ".env.bench",
            "up",
            "-d",
            "--no-deps",
            "--force-recreate",
            "flask-api",
        ],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"docker compose restart failed: {result.stderr.strip() or result.stdout.strip()}"
        )

    # Wait for API to be healthy (fail loudly on timeout)
    last_exc = None
    for attempt in range(60):
        try:
            resp = requests.get(f"{api_url}/health", timeout=2)
            if resp.status_code == 200:
                print(f"API ready after config change ({attempt}s)")
                return
        except Exception as e:
            last_exc = e
            if attempt % 10 == 0 and attempt > 0:
                print(f"  Still waiting for API... ({attempt}s, last error: {e})")
        time.sleep(1)

    raise TimeoutError("API did not become healthy after config change (60s timeout)") from last_exc


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
    distractor_rate_10: float = 0.0
    latency_ms: float = 0.0
    category: str = ""


@dataclass
class TestRunResult:  # noqa: pytest will skip due to __init__
    __test__ = False  # Tell pytest this is not a test class
    config_name: str
    query_results: List[QueryResult] = field(default_factory=list)
    timestamp: str = ""
    complexity: int = 0

    @property
    def mean_distractor_rate_10(self) -> float:
        vals = [q.distractor_rate_10 for q in self.query_results]
        return sum(vals) / len(vals) if vals else 0.0

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


def run_single_query(
    query_data: Dict[str, Any],
    api_url: str,
    *,
    distractor_ids: Optional[set] = None,
    recall_params: Optional[Dict[str, Any]] = None,
) -> QueryResult:
    """Execute a single recall query and compute metrics (incl. distractor rate)."""
    query = query_data["query"]
    expected_ids = query_data.get("expected_ids", [])
    category = query_data.get("category", "unknown")
    distractor_ids = distractor_ids or set()
    recall_params = dict(recall_params or {})
    if query_data.get("context_tags"):
        recall_params["context_tags"] = query_data["context_tags"]

    start = time.perf_counter()
    try:
        data = recall(api_url, get_headers(), query, **recall_params)
    except Exception as e:  # noqa: BLE001
        print(f"  WARNING: Query failed: {query[:50]}... — {e}")
        return QueryResult(
            query=query, expected_ids=expected_ids, retrieved_ids=[], category=category
        )

    latency_ms = (time.perf_counter() - start) * 1000
    retrieved_ids = extract_ids(data)

    return QueryResult(
        query=query,
        expected_ids=expected_ids,
        retrieved_ids=retrieved_ids,
        recall_5=recall_at_k(retrieved_ids, expected_ids, 5),
        recall_10=recall_at_k(retrieved_ids, expected_ids, 10),
        recall_20=recall_at_k(retrieved_ids, expected_ids, 20),
        mrr_val=mrr(retrieved_ids, expected_ids),
        ndcg_10=ndcg_at_k(retrieved_ids, expected_ids, 10),
        distractor_rate_10=distractor_rate_at_k(retrieved_ids, distractor_ids, 10),
        latency_ms=latency_ms,
        category=category,
    )


def run_test(
    config_name: str,
    queries: List[Dict],
    api_url: str,
    *,
    config: Optional[Dict[str, str]] = None,
    distractor_ids: Optional[set] = None,
    recall_params: Optional[Dict[str, Any]] = None,
) -> TestRunResult:
    """Run all test queries with a specific config."""
    result = TestRunResult(
        config_name=config_name, timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    result.complexity = config_complexity(config or {})

    for i, q in enumerate(queries):
        qr = run_single_query(
            q, api_url, distractor_ids=distractor_ids, recall_params=recall_params
        )
        result.query_results.append(qr)
        hit = "." if qr.recall_10 > 0 else "x"
        print(hit, end="", flush=True)
        if (i + 1) % 50 == 0:
            print(f" [{i + 1}/{len(queries)}]")

    print(f" [{len(queries)}/{len(queries)}]")
    return result


def build_scorecard(result: TestRunResult) -> Dict[str, Any]:
    """The legible scorecard: NDCG@10 primary, distractor-rate guardrail,
    latency + complexity tiebreakers."""
    return {
        "name": result.config_name,
        "ndcg_10": result.mean_ndcg_10,
        "distractor_rate_10": result.mean_distractor_rate_10,
        "latency_ms": result.mean_latency,
        "complexity": result.complexity,
    }


def print_summary(result: TestRunResult) -> None:
    """Print a summary of test results."""
    print(f"\n--- {result.config_name} ---")
    print(f"  Recall@5:   {result.mean_recall_5:.3f}")
    print(f"  Recall@10:  {result.mean_recall_10:.3f}")
    print(f"  Recall@20:  {result.mean_recall_20:.3f}")
    print(f"  MRR:        {result.mean_mrr:.3f}")
    print(f"  NDCG@10:    {result.mean_ndcg_10:.3f}")
    print(f"  Distractor@10: {result.mean_distractor_rate_10:.3f} (lower better)")
    print(f"  Complexity:   {result.complexity} active knobs")
    print(f"  Latency:    {result.mean_latency:.0f}ms avg")

    cats = result.by_category()
    if len(cats) > 1:
        print("\n  By category:")
        for cat, metrics in sorted(cats.items()):
            print(
                f"    {cat:12s} R@10={metrics['recall_10']:.3f}  "
                f"MRR={metrics['mrr']:.3f}  n={metrics['count']}"
            )


def print_comparison(a: TestRunResult, b: TestRunResult) -> None:
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

    print("\n  ** = statistically significant (p < 0.05)")


def save_results(result: TestRunResult, output_dir: Path) -> Path:
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
            "distractor_rate_10": result.mean_distractor_rate_10,
            "complexity": result.complexity,
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
                "distractor_rate_10": q.distractor_rate_10,
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


def main() -> None:
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
    parser.add_argument(
        "--distractors",
        type=int,
        default=0,
        help="Inject N aged labelled distractor memories before testing",
    )
    parser.add_argument(
        "--expand-relations", action="store_true", help="Recall with expand_relations"
    )
    parser.add_argument(
        "--no-current-only",
        action="store_true",
        help="Recall with current_only=false (include archived)",
    )
    parser.add_argument(
        "--recency-bias",
        type=str,
        default=None,
        choices=["on", "off", "auto"],
        help="Recall recency_bias override",
    )
    parser.add_argument(
        "--consolidate",
        action="store_true",
        help="Run a real consolidation pass (dry_run=false) before recall",
    )
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

    recall_params = {
        "expand_relations": args.expand_relations,
        "current_only": not args.no_current_only,
    }
    if args.recency_bias is not None:
        recall_params["recency_bias"] = args.recency_bias

    distractor_ids: set = set()
    if args.distractors > 0:
        payloads = make_distractor_memories(args.distractors)
        distractor_ids = set(inject_distractors(args.api_url, get_headers(), payloads))
        print(f"Injected {len(distractor_ids)} distractor memories")
    if args.consolidate:
        steps = run_consolidation(args.api_url, get_headers())
        print(f"Consolidation pass complete: {list(steps.keys())}")

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
            restart_api_with_config(config, api_url=args.api_url)
            time.sleep(2)

            result = run_test(
                config_name,
                queries,
                args.api_url,
                config=config,
                distractor_ids=distractor_ids,
                recall_params=recall_params,
            )
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
        restart_api_with_config(config_a, api_url=args.api_url)
        time.sleep(2)
        result_a = run_test(
            args.compare,
            queries,
            args.api_url,
            config=config_a,
            distractor_ids=distractor_ids,
            recall_params=recall_params,
        )
        print_summary(result_a)
        save_results(result_a, output_dir)

        print(f"\n--- Running candidate: {args.config} ---")
        restart_api_with_config(config_b, api_url=args.api_url)
        time.sleep(2)
        result_b = run_test(
            args.config,
            queries,
            args.api_url,
            config=config_b,
            distractor_ids=distractor_ids,
            recall_params=recall_params,
        )
        print_summary(result_b)
        save_results(result_b, output_dir)

        print_comparison(result_a, result_b)

    elif args.config:
        # Single config mode
        config = load_config(args.config)
        print(f"\n--- Running: {args.config} ---")
        restart_api_with_config(config, api_url=args.api_url)
        time.sleep(2)

        result = run_test(
            args.config,
            queries,
            args.api_url,
            config=config,
            distractor_ids=distractor_ids,
            recall_params=recall_params,
        )
        print_summary(result)
        save_results(result, output_dir)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
