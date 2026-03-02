from tests.benchmarks.longmemeval.evaluator import normalize_text, quick_score


def test_normalize_text_coerces_non_string_inputs() -> None:
    assert normalize_text(500) == "500"
    assert normalize_text(None) == ""


def test_quick_score_handles_numeric_reference() -> None:
    result = quick_score("The answer is 500.", 500, "q_1")
    assert isinstance(result, dict)
    assert result["is_correct"] is True
