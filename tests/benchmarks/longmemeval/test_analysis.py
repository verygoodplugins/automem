from tests.benchmarks.longmemeval.analyze_results import analyze_results, format_analysis


def test_analyze_results_reports_type_recall_failures_and_publishability() -> None:
    analysis = analyze_results(
        {
            "judge_errors": 1,
            "memory_ingest_failures": 0,
            "publishable": False,
            "details": [
                {
                    "question_id": "q1",
                    "question_type": "single-session-preference",
                    "reference": "tea",
                    "hypothesis": "tea",
                    "is_correct": True,
                    "recall_hit_at_5": True,
                    "retrieved_session_ids": ["s1"],
                    "answer_session_ids": ["s1"],
                },
                {
                    "question_id": "q2",
                    "question_type": "single-session-preference",
                    "reference": "coffee",
                    "hypothesis": "",
                    "is_correct": False,
                    "recall_hit_at_5": False,
                    "judge_error": "timeout",
                    "retrieved_session_ids": ["s2"],
                    "answer_session_ids": ["s3"],
                },
                {
                    "question_id": "q3",
                    "question_type": "multi-session",
                    "reference": "Paris",
                    "hypothesis": "Lyon",
                    "is_correct": False,
                    "recall_hit_at_5": True,
                    "retrieved_session_ids": ["s4"],
                    "answer_session_ids": ["s4"],
                },
            ],
        }
    )

    assert analysis["overall"]["correct"] == 1
    assert analysis["overall"]["total"] == 3
    assert analysis["overall"]["recall_hits"] == 2
    assert analysis["by_type"]["single-session-preference"]["correct"] == 1
    assert analysis["by_type"]["single-session-preference"]["recall_hits"] == 1
    assert analysis["failures_by_recall"] == {"hit": 1, "miss": 1, "unknown": 0}
    assert analysis["judge_errors"] == 1
    assert analysis["memory_ingest_failures"] == 0
    assert analysis["empty_hypotheses"] == 1
    assert analysis["publishable"] is False
    assert analysis["failure_rows"][0]["question_id"] == "q2"


def test_format_analysis_includes_compact_failure_rows() -> None:
    text = format_analysis(
        {
            "overall": {
                "total": 1,
                "correct": 0,
                "accuracy": 0.0,
                "recall_hits": 1,
                "recall_total": 1,
                "recall_at_5": 1.0,
            },
            "by_type": {
                "multi-session": {
                    "total": 1,
                    "correct": 0,
                    "accuracy": 0.0,
                    "recall_hits": 1,
                    "recall_total": 1,
                    "recall_at_5": 1.0,
                }
            },
            "failures_by_recall": {"hit": 1, "miss": 0, "unknown": 0},
            "judge_errors": 0,
            "memory_ingest_failures": 0,
            "empty_hypotheses": 0,
            "publishable": True,
            "failure_rows": [
                {
                    "question_id": "q1",
                    "question_type": "multi-session",
                    "reference": "Paris",
                    "hypothesis": "Lyon",
                    "recall_hit_at_5": True,
                }
            ],
        },
        max_failures=1,
    )

    assert "Accuracy: 0.00% (0/1)" in text
    assert "| multi-session | 0.00% (0/1) | 100.00% (1/1) |" in text
    assert "| q1 | multi-session | True | Paris | Lyon |" in text
