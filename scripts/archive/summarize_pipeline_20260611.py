#!/usr/bin/env python3
"""Tabulate the 2026-06-11 release-verification sweep results.

Pools prod_parity runs as baseline; reports per-config metric deltas with a
paired difference test on per-query recall@5 (computed from retrieved_ids[:5])
against parity run 1, plus per-category R@10 deltas. The p-value uses a normal
(z) approximation of the paired t-statistic — accurate for the n=200 query
sets here, but not a Student's t p-value for small n.
"""

import glob
import json
import math
import os
import sys
from collections import defaultdict

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "lab/results"
RUN_DATE = "20260611"

CONFIG_ORDER = [
    "prod_parity",
    "cap2",
    "cap3",
    "cap4",
    "gate005",
    "gate010",
    "gate015",
    "gate020",
    "gate025",
    "recw90",
    "recw365",
    "recexp",
    "recbias",
]


def load_runs():
    runs = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, f"*_{RUN_DATE}_*.json"))):
        name = os.path.basename(path)
        for cfg in CONFIG_ORDER:
            if name.startswith(cfg + "_" + RUN_DATE):
                with open(path) as fh:
                    runs[cfg].append(json.load(fh))
                break
    return runs


def perq_recall5(run):
    out = {}
    for q in run.get("queries", []):
        exp = q.get("expected_ids") or []
        top5 = set((q.get("retrieved_ids") or [])[:5])
        out[q["query"]] = (sum(1 for e in exp if e in top5) / len(exp)) if exp else None
    return out


def paired_z(a, b):
    # z-approximation: t-statistic against the normal CDF (n=200 here, so t ~= z)
    diffs = [y - x for x, y in zip(a, b)]
    n = len(diffs)
    if n < 2:
        return float("nan")
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    if var == 0:
        return 1.0
    t = mean / math.sqrt(var / n)
    return 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))


def main():
    runs = load_runs()
    if not runs.get("prod_parity"):
        sys.exit(f"no prod_parity runs found in {RESULTS_DIR}")

    base_runs = runs["prod_parity"]
    base_pq = perq_recall5(base_runs[0])
    base_r5 = sum(r["summary"]["recall_5"] for r in base_runs) / len(base_runs)

    print(
        f"{'config':>12} {'runs':>4} {'R@5':>7} {'R@10':>7} {'MRR':>7} {'NDCG':>7} {'dR@5':>7} {'p':>8}"
    )
    print("-" * 66)
    for cfg in CONFIG_ORDER:
        rs = runs.get(cfg, [])
        if not rs:
            continue
        s = [r["summary"] for r in rs]
        r5 = sum(x["recall_5"] for x in s) / len(s)
        r10 = sum(x["recall_10"] for x in s) / len(s)
        mrr = sum(x["mrr"] for x in s) / len(s)
        ndcg = sum(x["ndcg_10"] for x in s) / len(s)
        if cfg == "prod_parity":
            print(
                f"{cfg:>12} {len(rs):>4} {r5:7.3f} {r10:7.3f} {mrr:7.3f} {ndcg:7.3f} {'—':>7} {'—':>8}"
            )
            continue
        cand_pq = perq_recall5(rs[0])
        common = [q for q, v in base_pq.items() if v is not None and cand_pq.get(q) is not None]
        p = paired_z([base_pq[q] for q in common], [cand_pq[q] for q in common])
        print(
            f"{cfg:>12} {len(rs):>4} {r5:7.3f} {r10:7.3f} {mrr:7.3f} {ndcg:7.3f} {r5 - base_r5:+7.3f} {p:8.4f}"
        )

    print("\nPer-category R@10 delta vs parity run 1 (n in parens):")
    base_cat = base_runs[0].get("by_category", {})
    cats = sorted(base_cat)
    print(f"{'config':>12} " + " ".join(f"{c[:9]:>9}" for c in cats))
    print(
        f"{'(n)':>12} "
        + " ".join(f"{'(' + str(base_cat[c].get('count', 0)) + ')':>9}" for c in cats)
    )
    for cfg in CONFIG_ORDER:
        rs = runs.get(cfg, [])
        if not rs:
            continue
        cat = rs[0].get("by_category", {})
        if cfg == "prod_parity":
            row = [f"{base_cat[c].get('recall_10', 0):.3f}" for c in cats]
        else:
            row = [
                f"{cat.get(c, {}).get('recall_10', 0) - base_cat[c].get('recall_10', 0):+.3f}"
                for c in cats
            ]
        print(f"{cfg:>12} " + " ".join(f"{v:>9}" for v in row))


if __name__ == "__main__":
    main()
