#!/usr/bin/env python3
"""Compare two benchmark result files and print a side-by-side table.

Usage:
    python scripts/bench/compare_results.py --baseline results/a.json --test results/b.json
    python scripts/bench/compare_results.py --baseline results/a.json --test results/b.json --output compare.json
"""

import argparse
import json
import sys
from pathlib import Path


def load_results(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Result file not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in {path}: {e}", file=sys.stderr)
        sys.exit(1)


def format_pct(val: float) -> str:
    return f"{val:.1%}"


def format_delta(delta: float) -> str:
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.1%}"


def compare_locomo(baseline: dict, test: dict) -> dict:
    """Compare two LoCoMo benchmark results."""
    category_names = {
        1: "Single-hop Recall",
        2: "Temporal Understanding",
        3: "Multi-hop Reasoning",
        4: "Open Domain",
        5: "Complex Reasoning",
    }

    b_overall = baseline.get("overall", {})
    t_overall = test.get("overall", {})

    b_acc = b_overall.get("accuracy", 0)
    t_acc = t_overall.get("accuracy", 0)
    delta = t_acc - b_acc

    print("\n" + "=" * 65)
    print(f"  {'Metric':<25} {'Baseline':>10} {'Test':>10} {'Delta':>10}")
    print("-" * 65)
    print(
        f"  {'Overall Accuracy':<25} {format_pct(b_acc):>10} {format_pct(t_acc):>10} {format_delta(delta):>10}"
    )
    print(
        f"  {'Correct / Total':<25} {b_overall.get('correct', '?'):>5}/{b_overall.get('total', '?'):<4} {t_overall.get('correct', '?'):>5}/{t_overall.get('total', '?'):<4}"
    )
    print(
        f"  {'Elapsed Time':<25} {b_overall.get('elapsed_time', 0):>9.0f}s {t_overall.get('elapsed_time', 0):>9.0f}s"
    )
    print("-" * 65)

    b_cats = baseline.get("categories", {})
    t_cats = test.get("categories", {})

    category_deltas = {}
    cat_keys = {str(k) for k in b_cats.keys()} | {str(k) for k in t_cats.keys()}
    for cat_key in sorted(cat_keys, key=lambda k: (not k.isdigit(), int(k) if k.isdigit() else k)):
        b_cat = b_cats.get(cat_key, {})
        t_cat = t_cats.get(cat_key, {})

        b_cat_acc = b_cat.get("accuracy", 0)
        t_cat_acc = t_cat.get("accuracy", 0)
        cat_delta = t_cat_acc - b_cat_acc

        name = (
            category_names.get(int(cat_key), f"Category {cat_key}")
            if cat_key.isdigit()
            else f"Category {cat_key}"
        )

        print(
            f"  {name:<25} {format_pct(b_cat_acc):>10} {format_pct(t_cat_acc):>10} {format_delta(cat_delta):>10}"
        )
        category_deltas[name] = cat_delta

    print("=" * 65)

    # Verdict
    if delta > 0.01:
        print(f"\n  IMPROVEMENT: Test is {format_delta(delta)} better overall")
    elif delta < -0.01:
        print(f"\n  REGRESSION: Test is {format_delta(abs(delta))} worse overall")
    else:
        print(f"\n  NO SIGNIFICANT CHANGE ({format_delta(delta)})")

    return {
        "baseline_accuracy": b_acc,
        "test_accuracy": t_acc,
        "delta": delta,
        "category_deltas": category_deltas,
    }


def compare_longmemeval(baseline: dict, test: dict) -> dict:
    """Compare two LongMemEval benchmark results."""
    b_acc = baseline.get("accuracy", baseline.get("overall", {}).get("accuracy", 0))
    t_acc = test.get("accuracy", test.get("overall", {}).get("accuracy", 0))
    delta = t_acc - b_acc

    b_total = baseline.get("total_questions", baseline.get("overall", {}).get("total", "?"))
    t_total = test.get("total_questions", test.get("overall", {}).get("total", "?"))
    b_retrieval = baseline.get("retrieval", {}).get("recall_any_at_5")
    t_retrieval = test.get("retrieval", {}).get("recall_any_at_5")

    print("\n" + "=" * 65)
    print(f"  {'Metric':<25} {'Baseline':>10} {'Test':>10} {'Delta':>10}")
    print("-" * 65)
    print(
        f"  {'Accuracy':<25} {format_pct(b_acc):>10} {format_pct(t_acc):>10} {format_delta(delta):>10}"
    )
    retrieval_delta = None
    if b_retrieval is not None and t_retrieval is not None:
        retrieval_delta = t_retrieval - b_retrieval
        print(
            "  "
            f"{'Recall@5':<25} {format_pct(b_retrieval):>10} {format_pct(t_retrieval):>10} {format_delta(retrieval_delta):>10}"
        )
    print(f"  {'Questions':<25} {b_total!s:>10} {t_total!s:>10}")
    print("=" * 65)

    return {
        "baseline_accuracy": b_acc,
        "test_accuracy": t_acc,
        "delta": delta,
        "baseline_recall_at_5": b_retrieval,
        "test_recall_at_5": t_retrieval,
        "recall_at_5_delta": retrieval_delta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two benchmark result files")
    parser.add_argument("--baseline", required=True, help="Path to baseline results JSON")
    parser.add_argument("--test", required=True, help="Path to test results JSON")
    parser.add_argument("--output", default=None, help="Save comparison to JSON file")
    args = parser.parse_args()

    baseline = load_results(args.baseline)
    test = load_results(args.test)

    print(f"\nBaseline: {Path(args.baseline).name}")
    print(f"Test:     {Path(args.test).name}")

    # Detect and validate format consistency before comparison
    baseline_is_locomo = "categories" in baseline
    test_is_locomo = "categories" in test
    if baseline_is_locomo != test_is_locomo:
        print("ERROR: Cannot compare results from different benchmark formats")
        print(f"  Baseline: {'LoCoMo' if baseline_is_locomo else 'LongMemEval'}")
        print(f"  Test:     {'LoCoMo' if test_is_locomo else 'LongMemEval'}")
        sys.exit(1)

    if baseline_is_locomo:
        comparison = compare_locomo(baseline, test)
    else:
        comparison = compare_longmemeval(baseline, test)

    if args.output:
        comparison["baseline_file"] = args.baseline
        comparison["test_file"] = args.test
        with open(args.output, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved to: {args.output}")


if __name__ == "__main__":
    main()
