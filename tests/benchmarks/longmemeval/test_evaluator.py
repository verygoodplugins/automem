from types import SimpleNamespace

from tests.benchmarks.longmemeval.evaluator import llm_evaluate, normalize_text, quick_score


def test_normalize_text_coerces_non_string_inputs() -> None:
    assert normalize_text(500) == "500"
    assert normalize_text(None) == ""


def test_quick_score_handles_numeric_reference() -> None:
    result = quick_score("The answer is 500.", 500, "q_1")
    assert isinstance(result, dict)
    assert result["is_correct"] is True


def test_llm_evaluate_uses_gpt5_compatible_structured_request() -> None:
    captured = {}

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            content = '{"correct": true, "confidence": 0.9, "explanation": "matches"}'
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))

    result = llm_evaluate(
        question="What is the name?",
        hypothesis="Ada",
        reference="Ada",
        question_id="q_1",
        model="gpt-5.4-mini-2026-03-17",
        client=client,
    )

    assert result["is_correct"] is True
    assert captured["model"] == "gpt-5.4-mini-2026-03-17"
    assert captured["max_completion_tokens"] == 200
    assert "max_tokens" not in captured
    assert "temperature" not in captured
    assert captured["response_format"]["type"] == "json_schema"
    assert captured["response_format"]["json_schema"]["strict"] is True


def test_llm_evaluate_keeps_json_mode_for_non_gpt5_models() -> None:
    captured = {}

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            content = '{"correct": false, "confidence": 0.2, "explanation": "miss"}'
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))

    result = llm_evaluate(
        question="What is the name?",
        hypothesis="Grace",
        reference="Ada",
        question_id="q_1",
        model="gpt-4o-mini",
        client=client,
    )

    assert result["is_correct"] is False
    assert captured["max_tokens"] == 200
    assert captured["temperature"] == 0
    assert "max_completion_tokens" not in captured
    assert captured["response_format"] == {"type": "json_object"}
