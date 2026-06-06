#!/usr/bin/env python3
"""Generate a markdown failure report from a LoCoMo results JSON file.

Usage:
    python scripts/bench/analyze_locomo_results.py benchmarks/results/locomo-mini.json
    python scripts/bench/analyze_locomo_results.py results.json --output report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

CATEGORY_NAMES = {
    1: "Single-hop Recall",
    2: "Temporal Understanding",
    3: "Multi-hop Reasoning",
    4: "Open Domain",
    5: "Complex Reasoning",
}


def load_results(path: str) -> Dict[str, Any]:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Result file not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"ERROR: Invalid JSON in {path}: {exc}", file=sys.stderr)
        sys.exit(1)


def format_pct(value: float) -> str:
    return f"{value:.2%}"


def escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def collect_question_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for conversation in results.get("conversations", []):
        sample_id = conversation.get("sample_id", "unknown")
        for qa in conversation.get("qa_results", []):
            rows.append(
                {
                    "sample_id": sample_id,
                    "category": qa.get("category"),
                    "question": qa.get("question", ""),
                    "expected_answer": qa.get("expected_answer"),
                    "is_correct": qa.get("is_correct"),
                    "confidence": float(qa.get("confidence") or 0.0),
                    "recalled_count": int(qa.get("recalled_count") or 0),
                    "explanation": qa.get("explanation", ""),
                }
            )
    return rows


def render_report(results: Dict[str, Any], source_path: Path) -> str:
    rows = collect_question_rows(results)
    failed = [row for row in rows if row["is_correct"] is False]
    skipped = [row for row in rows if row["is_correct"] is None]

    overall = results.get("overall", {})
    accuracy = float(overall.get("accuracy") or 0.0)
    correct = overall.get("correct", 0)
    total = overall.get("total", 0)

    by_category: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in failed:
        if isinstance(row["category"], int):
            by_category[row["category"]].append(row)

    category_counts = sorted(by_category.items(), key=lambda item: item[0])
    conversation_counts = Counter(row["sample_id"] for row in failed)
    explanation_counts = Counter(row["explanation"] for row in failed)

    lines: List[str] = []
    lines.append(f"# LoCoMo Failure Report: `{source_path.name}`")
    lines.append("")
    lines.append("## Overview")
    lines.append(f"- Overall accuracy: {format_pct(accuracy)} ({correct}/{total})")
    lines.append(f"- Failed questions: {len(failed)}")
    lines.append(f"- Skipped questions: {len(skipped)}")
    lines.append(f"- Conversations with failures: {len(conversation_counts)}")
    lines.append("")

    lines.append("## Failures by Category")
    lines.append("| Category | Name | Failed |")
    lines.append("|---|---|---:|")
    for category, category_rows in category_counts:
        lines.append(
            f"| {category} | {CATEGORY_NAMES.get(category, f'Category {category}')} | {len(category_rows)} |"
        )
    if not category_counts:
        lines.append("| - | None | 0 |")
    lines.append("")

    lines.append("## Failures by Conversation")
    lines.append("| Conversation | Failed |")
    lines.append("|---|---:|")
    for sample_id, count in sorted(
        conversation_counts.items(), key=lambda item: (-item[1], item[0])
    ):
        lines.append(f"| {escape_md(sample_id)} | {count} |")
    if not conversation_counts:
        lines.append("| None | 0 |")
    lines.append("")

    lines.append("## Top Failure Explanations")
    lines.append("| Explanation | Count |")
    lines.append("|---|---:|")
    for explanation, count in explanation_counts.most_common():
        lines.append(f"| {escape_md(explanation)} | {count} |")
    if not explanation_counts:
        lines.append("| None | 0 |")
    lines.append("")

    lines.append("## Failed Questions by Category")
    for category, category_rows in category_counts:
        lines.append(
            f"### Category {category}: {CATEGORY_NAMES.get(category, f'Category {category}')}"
        )
        for row in sorted(category_rows, key=lambda item: (item["sample_id"], item["question"])):
            lines.append(f"- [{escape_md(row['sample_id'])}] {escape_md(row['question'])}")
            lines.append(f"  - Expected: {escape_md(row['expected_answer'])}")
            lines.append(f"  - Confidence: {row['confidence']:.2f}")
            lines.append(f"  - Recalled: {row['recalled_count']}")
            lines.append(f"  - Explanation: {escape_md(row['explanation'])}")
        lines.append("")
    if not category_counts:
        lines.append("No failed questions.")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a LoCoMo benchmark results JSON file")
    parser.add_argument("results", help="Path to results JSON")
    parser.add_argument("--output", default=None, help="Optional path to save the markdown report")
    args = parser.parse_args()

    source_path = Path(args.results)
    report = render_report(load_results(args.results), source_path)
    if args.output:
        Path(args.output).write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
