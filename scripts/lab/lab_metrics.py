"""Pure scoring functions for the AutoMem Recall Quality Lab.

No I/O lives here — every function is deterministic and unit-testable.
Imported by run_recall_test.py and by the parallel matrix harness.
"""

import math
from typing import Any, Dict, Iterable, List


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
            dcg += 1.0 / math.log2(i + 2)
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(expected_ids), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def distractor_rate_at_k(retrieved_ids: List[str], distractor_ids: Iterable[str], k: int) -> float:
    """Fraction of the top-k results that are known distractors. Lower is better.

    Distractors are memories we injected and labelled as never-relevant, so a
    result that is a distractor is unambiguous noise. This is the precision
    guardrail and the only metric that can see the `forget` consolidation mode
    working (known-item recall is blind to suppression).
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    dset = set(distractor_ids)
    hits = sum(1 for rid in top_k if rid in dset)
    return hits / len(top_k)


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
    z = abs(t_stat)
    p_value = 2 * (1 - 0.5 * (1 + math.erf(z / 2**0.5)))

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


_FLAG_KEYS = {
    "ENRICHMENT_ENABLED",
    "JIT_ENRICHMENT_ENABLED",
    "ENRICHMENT_ENABLE_SUMMARIES",
    "RECALL_RECENCY_BIAS",
}
_OFF_VALUES = {"", "0", "0.0", "off", "false", "no", "none"}


def _is_off(value: Any) -> bool:
    return str(value).strip().lower() in _OFF_VALUES


def config_complexity(config: Dict[str, Any]) -> int:
    """Count 'active' knobs in a config. Lower = simpler/more elegant.

    This is the simplicity tiebreaker: among configs within noise on quality,
    the one with fewer active knobs wins.
    """
    count = 0
    for key, value in config.items():
        ku = str(key).upper()
        if ku.startswith("SEARCH_WEIGHT_"):
            try:
                if float(value) != 0.0:
                    count += 1
            except (TypeError, ValueError):
                continue
        elif ku in _FLAG_KEYS:
            if not _is_off(value):
                count += 1
        elif ku.endswith(("_THRESHOLD", "_CAP", "_GATE")):
            try:
                if float(value) > 0.0:
                    count += 1
            except (TypeError, ValueError):
                if not _is_off(value):
                    count += 1
    return count


def pick_winner(
    cards: List[Dict[str, Any]],
    *,
    baseline_name: str,
    ndcg_tol: float = 0.005,
    distractor_tol: float = 0.01,
) -> Dict[str, Any]:
    """Apply the scorecard decision rule and return the winning card + reason.

    Rule: highest NDCG@10 that does not regress distractor-precision vs the
    baseline; break ties (within ndcg_tol) toward fewer knobs, then lower latency.
    """
    baseline = next((c for c in cards if c["name"] == baseline_name), None)
    if baseline is None:
        raise ValueError(f"baseline_name {baseline_name!r} not found among cards")
    # The baseline is always eligible against itself (distractor_tol >= 0), so
    # `eligible` is never empty and the max() below is always safe.
    ceiling = baseline["distractor_rate_10"] + distractor_tol
    eligible = [c for c in cards if c["distractor_rate_10"] <= ceiling]

    best_ndcg = max(c["ndcg_10"] for c in eligible)
    contenders = [c for c in eligible if c["ndcg_10"] >= best_ndcg - ndcg_tol]
    winner = dict(min(contenders, key=lambda c: (c["complexity"], c["latency_ms"])))

    if winner["name"] == baseline_name:
        winner["reason"] = "no candidate beat baseline NDCG@10 without precision regression"
    else:
        winner["reason"] = (
            f"best NDCG@10 within tolerance, lowest complexity ({winner['complexity']}) "
            f"and latency ({winner['latency_ms']:.0f}ms)"
        )
    return winner
