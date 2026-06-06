import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "bench" / "analyze_locomo_results.py"


def test_analyze_locomo_results_generates_markdown_report(tmp_path: Path) -> None:
    input_path = tmp_path / "results.json"
    output_path = tmp_path / "report.md"
    input_path.write_text(
        json.dumps(
            {
                "overall": {"accuracy": 0.5, "correct": 2, "total": 4},
                "conversations": [
                    {
                        "sample_id": "conv-1",
                        "qa_results": [
                            {
                                "question": "What is Alice's job?",
                                "expected_answer": "Engineer",
                                "category": 1,
                                "is_correct": False,
                                "confidence": 0.0,
                                "recalled_count": 12,
                                "explanation": "No good match (max: 0.00)",
                            },
                            {
                                "question": "When did Alice move?",
                                "expected_answer": "2023",
                                "category": 2,
                                "is_correct": True,
                                "confidence": 1.0,
                                "recalled_count": 10,
                                "explanation": "Found in evidence dialog D1:2",
                            },
                        ],
                    },
                    {
                        "sample_id": "conv-2",
                        "qa_results": [
                            {
                                "question": "Why did Bob leave?",
                                "expected_answer": "Family reasons",
                                "category": 3,
                                "is_correct": False,
                                "confidence": 0.25,
                                "recalled_count": 20,
                                "explanation": "No good match (max: 0.25)",
                            },
                            {
                                "question": "What did Bob buy?",
                                "expected_answer": "A bike",
                                "category": 1,
                                "is_correct": False,
                                "confidence": 0.0,
                                "recalled_count": 8,
                                "explanation": "No good match (max: 0.00)",
                            },
                        ],
                    },
                ],
            }
        )
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(input_path), "--output", str(output_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    stdout = result.stdout
    assert "## Overview" in stdout
    assert "Failed questions: 3" in stdout
    assert "## Failures by Category" in stdout
    assert "| 1 | Single-hop Recall | 2 |" in stdout
    assert "## Failures by Conversation" in stdout
    assert "| conv-2 | 2 |" in stdout
    assert "## Top Failure Explanations" in stdout
    assert "| No good match (max: 0.00) | 2 |" in stdout
    assert "### Category 1: Single-hop Recall" in stdout
    assert "What is Alice's job?" in stdout
    assert output_path.read_text() == stdout
