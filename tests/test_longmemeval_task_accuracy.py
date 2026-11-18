import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.benchmarks.longmemeval.analyze_results import analyze_results, format_analysis
from tests.benchmarks.longmemeval.evaluator import LongMemEvalScorer

QUESTION_TYPES = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "knowledge-update",
    "temporal-reasoning",
)


@pytest.fixture
def imbalanced_results():
    return [{"question_type": "single-session-user", "is_correct": True} for _ in range(10)] + [
        {"question_type": question_type, "is_correct": False}
        for question_type in QUESTION_TYPES[1:]
    ]


@pytest.mark.parametrize("consumer", ["scorer", "analyzer"])
def test_task_accuracy_weights_each_canonical_type_equally(imbalanced_results, consumer, capsys):
    if consumer == "scorer":
        scorer = LongMemEvalScorer()
        for row in imbalanced_results:
            scorer.add_result(row)
        scores = scorer.print_report()
        text = capsys.readouterr().out
    else:
        scores = analyze_results({"details": imbalanced_results})
        text = format_analysis(scores)

    assert scores["overall"]["accuracy"] == pytest.approx(2 / 3)
    assert scores["task_averaged"]["accuracy"] == pytest.approx(1 / 6)
    assert "Task-averaged Accuracy: 16.67%" in text
    json.dumps(scores, allow_nan=False)


@pytest.mark.parametrize("missing_type", QUESTION_TYPES)
def test_missing_canonical_type_does_not_become_zero_or_use_unknown(
    imbalanced_results, missing_type
):
    rows = [row for row in imbalanced_results if row["question_type"] != missing_type]
    rows.append({"question_type": "unknown", "is_correct": True})
    scorer = LongMemEvalScorer()
    for row in rows:
        scorer.add_result(row)
    for scores in (scorer.compute_scores(), analyze_results({"details": rows})):
        assert scores["task_averaged"]["accuracy"] is None
        assert "unknown" in scores["by_type"]
        json.dumps(scores, allow_nan=False)


@pytest.mark.parametrize("rows", [[], [{"question_type": "multi-session", "is_correct": True}]])
def test_incomplete_task_accuracy_is_unavailable(rows, capsys):
    scorer = LongMemEvalScorer()
    for row in rows:
        scorer.add_result(row)
    scores = scorer.print_report()
    analysis = analyze_results({"details": rows})
    assert scores["task_averaged"]["accuracy"] is None
    assert analysis["task_averaged"]["accuracy"] is None
    assert "Task-averaged Accuracy: n/a" in capsys.readouterr().out
    assert "Task-averaged Accuracy: n/a" in format_analysis(analysis)


def test_unknown_type_does_not_change_six_task_average(imbalanced_results):
    rows = imbalanced_results + [{"question_type": "unknown", "is_correct": True}]
    scorer = LongMemEvalScorer()
    for row in rows:
        scorer.add_result(row)
    for scores in (scorer.compute_scores(), analyze_results({"details": rows})):
        assert scores["overall"]["accuracy"] == 11 / 16
        assert scores["task_averaged"]["accuracy"] == pytest.approx(1 / 6)


def test_all_incorrect_tasks_report_zero(capsys):
    rows = [
        {"question_type": question_type, "is_correct": False} for question_type in QUESTION_TYPES
    ]
    scorer = LongMemEvalScorer()
    for row in rows:
        scorer.add_result(row)
    scores = scorer.print_report()
    analysis = analyze_results({"results": rows})
    assert scores["task_averaged"]["accuracy"] == 0.0
    assert analysis["task_averaged"]["accuracy"] == 0.0
    assert "Task-averaged Accuracy: 0.00%" in capsys.readouterr().out
    assert "Task-averaged Accuracy: 0.00%" in format_analysis(analysis)


def test_abstention_contributes_to_its_base_type(imbalanced_results):
    rows = imbalanced_results + [
        {"question_type": "multi-session", "is_correct": True, "is_abstention": True}
    ]
    scorer = LongMemEvalScorer()
    for row in rows:
        scorer.add_result(row)
    scores = scorer.compute_scores()
    assert scores["abstention"] == {"total": 1, "correct": 1, "accuracy": 1.0}
    for report in (scores, analyze_results({"details": rows})):
        assert report["by_type"]["multi-session"]["accuracy"] == 0.5
        assert report["task_averaged"]["accuracy"] == 0.25


def test_task_accuracy_preserves_each_consumers_correctness_rule():
    rows = [{"question_type": question_type, "is_correct": 1} for question_type in QUESTION_TYPES]
    scorer = LongMemEvalScorer()
    for row in rows:
        scorer.add_result(row)
    scores = scorer.compute_scores()
    analysis = analyze_results({"details": rows})
    assert scores["overall"]["accuracy"] == 1.0
    assert analysis["overall"]["accuracy"] == 0.0
    assert scores["task_averaged"]["accuracy"] == 1.0
    assert analysis["task_averaged"]["accuracy"] == 0.0


def test_existing_scorer_metrics_are_preserved():
    scorer = LongMemEvalScorer()
    scorer.add_result(
        {"question_type": "multi-session", "is_correct": True, "recall_hit_at_5": False}
    )
    scorer.add_result(
        {
            "question_type": "multi-session",
            "is_correct": False,
            "recall_hit_at_5": True,
            "is_abstention": True,
        }
    )
    scores = scorer.compute_scores()
    assert {key: value for key, value in scores.items() if key != "task_averaged"} == {
        "overall": {"accuracy": 0.5, "correct": 1, "total": 2},
        "retrieval": {"recall_any_at_5": 0.5, "hits": 1, "total": 2},
        "by_type": {
            "multi-session": {"correct": 1, "total": 2, "accuracy": 0.5, "name": "Multi-Session"}
        },
        "retrieval_by_type": {
            "multi-session": {
                "hits": 1,
                "total": 2,
                "recall_any_at_5": 0.5,
                "name": "Multi-Session",
            }
        },
        "abstention": {"total": 1, "correct": 0, "accuracy": 0.0},
    }


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("as_json", [False, True])
def test_analyzer_script_reports_task_accuracy_from_another_directory(
    imbalanced_results, tmp_path, legacy, as_json
):
    result_file = tmp_path / "results.json"
    result_file.write_text(json.dumps({"results" if legacy else "details": imbalanced_results}))
    script = Path(__file__).parent / "benchmarks" / "longmemeval" / "analyze_results.py"
    command = [sys.executable, "-S", str(script.resolve()), str(result_file)]
    if as_json:
        command.append("--json")
    result = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True, check=True)
    if as_json:
        assert json.loads(result.stdout)["task_averaged"]["accuracy"] == pytest.approx(1 / 6)
    else:
        assert "Task-averaged Accuracy: 16.67%" in result.stdout


def test_analyzer_module_reports_task_accuracy(imbalanced_results, tmp_path):
    result_file = tmp_path / "results.json"
    result_file.write_text(json.dumps({"details": imbalanced_results}))
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "tests.benchmarks.longmemeval.analyze_results",
            str(result_file),
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout)["task_averaged"]["accuracy"] == pytest.approx(1 / 6)
