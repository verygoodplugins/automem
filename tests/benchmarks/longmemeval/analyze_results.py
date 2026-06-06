"""Analyze LongMemEval result JSON artifacts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _pct(correct: int, total: int) -> Optional[float]:
    if total <= 0:
        return None
    return correct / total


def _result_details(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    details = results.get("details")
    if isinstance(details, list):
        return details
    legacy_results = results.get("results")
    if isinstance(legacy_results, list):
        return legacy_results
    return []


def _status_counts(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"hit": 0, "miss": 0, "unknown": 0}
    for row in rows:
        recall_hit = row.get("recall_hit_at_5")
        if recall_hit is True:
            counts["hit"] += 1
        elif recall_hit is False:
            counts["miss"] += 1
        else:
            counts["unknown"] += 1
    return counts


def analyze_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Return compact diagnostics for a LongMemEval result artifact."""
    details = _result_details(results)
    total = len(details)
    correct = sum(1 for row in details if row.get("is_correct") is True)
    recall_total = sum(1 for row in details if row.get("recall_hit_at_5") is not None)
    recall_hits = sum(1 for row in details if row.get("recall_hit_at_5") is True)
    failures = [row for row in details if row.get("is_correct") is not True]

    by_type: Dict[str, Dict[str, Any]] = {}
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in details:
        grouped[str(row.get("question_type", "unknown"))].append(row)

    for question_type in sorted(grouped):
        rows = grouped[question_type]
        type_total = len(rows)
        type_correct = sum(1 for row in rows if row.get("is_correct") is True)
        type_recall_total = sum(1 for row in rows if row.get("recall_hit_at_5") is not None)
        type_recall_hits = sum(1 for row in rows if row.get("recall_hit_at_5") is True)
        by_type[question_type] = {
            "total": type_total,
            "correct": type_correct,
            "accuracy": _pct(type_correct, type_total),
            "recall_hits": type_recall_hits,
            "recall_total": type_recall_total,
            "recall_at_5": _pct(type_recall_hits, type_recall_total),
        }

    failure_rows = []
    for row in failures:
        failure_rows.append(
            {
                "question_id": row.get("question_id"),
                "question_type": row.get("question_type"),
                "question": row.get("question"),
                "reference": row.get("reference"),
                "hypothesis": row.get("hypothesis"),
                "recall_hit_at_5": row.get("recall_hit_at_5"),
                "retrieved_session_ids": row.get("retrieved_session_ids", []),
                "answer_session_ids": row.get("answer_session_ids", []),
                "judge_error": row.get("judge_error"),
                "error": row.get("error"),
            }
        )

    judge_errors = results.get("judge_errors")
    if judge_errors is None:
        judge_errors = sum(1 for row in details if row.get("judge_error"))

    memory_ingest_failures = results.get("memory_ingest_failures")
    if memory_ingest_failures is None:
        memory_ingest_failures = sum(1 for row in details if row.get("error_type"))

    empty_hypotheses = sum(1 for row in details if not row.get("hypothesis"))
    publishable = results.get("publishable")
    if publishable is None:
        publishable = judge_errors == 0

    return {
        "overall": {
            "total": total,
            "correct": correct,
            "accuracy": _pct(correct, total),
            "recall_hits": recall_hits,
            "recall_total": recall_total,
            "recall_at_5": _pct(recall_hits, recall_total),
        },
        "by_type": by_type,
        "failures_by_recall": _status_counts(failures),
        "judge_errors": judge_errors,
        "memory_ingest_failures": memory_ingest_failures,
        "empty_hypotheses": empty_hypotheses,
        "publishable": publishable,
        "failure_rows": failure_rows,
    }


def load_results(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _format_rate(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def _truncate(value: Any, limit: int = 96) -> str:
    text = "" if value is None else str(value).replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def format_analysis(analysis: Dict[str, Any], max_failures: int = 20) -> str:
    """Format diagnostics as compact Markdown."""
    overall = analysis["overall"]
    lines = [
        "# LongMemEval Result Analysis",
        "",
        "## Overall",
        "",
        f"- Accuracy: {_format_rate(overall['accuracy'])} "
        f"({overall['correct']}/{overall['total']})",
        f"- Recall@5: {_format_rate(overall['recall_at_5'])} "
        f"({overall['recall_hits']}/{overall['recall_total']})",
        f"- Judge errors: {analysis['judge_errors']}",
        f"- Memory ingest failures: {analysis['memory_ingest_failures']}",
        f"- Empty hypotheses: {analysis['empty_hypotheses']}",
        f"- Publishable: {analysis['publishable']}",
        "",
        "## By Type",
        "",
        "| Question type | Accuracy | Recall@5 |",
        "|---|---:|---:|",
    ]

    for question_type, stats in analysis["by_type"].items():
        lines.append(
            f"| {question_type} | {_format_rate(stats['accuracy'])} "
            f"({stats['correct']}/{stats['total']}) | "
            f"{_format_rate(stats['recall_at_5'])} "
            f"({stats['recall_hits']}/{stats['recall_total']}) |"
        )

    failure_counts = analysis["failures_by_recall"]
    lines.extend(
        [
            "",
            "## Failures By Recall",
            "",
            f"- Retrieved answer session at @5: {failure_counts['hit']}",
            f"- Retrieval miss at @5: {failure_counts['miss']}",
            f"- Unknown recall status: {failure_counts['unknown']}",
            "",
            f"## Failure Rows (first {max_failures})",
            "",
            "| ID | Type | Recall@5 | Reference | Hypothesis |",
            "|---|---|---:|---|---|",
        ]
    )

    for row in analysis["failure_rows"][:max_failures]:
        lines.append(
            f"| {row.get('question_id')} | {row.get('question_type')} | "
            f"{row.get('recall_hit_at_5')} | {_truncate(row.get('reference'))} | "
            f"{_truncate(row.get('hypothesis'))} |"
        )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", type=Path, help="LongMemEval result JSON path")
    parser.add_argument(
        "--max-failures",
        type=int,
        default=20,
        help="Maximum failure rows to print in text mode",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable analysis JSON",
    )
    args = parser.parse_args()

    analysis = analyze_results(load_results(args.result_json))
    if args.json:
        print(json.dumps(analysis, indent=2, ensure_ascii=False))
    else:
        print(format_analysis(analysis, max_failures=args.max_failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
