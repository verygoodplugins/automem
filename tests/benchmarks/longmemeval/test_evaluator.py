from types import SimpleNamespace

import tests.benchmarks.longmemeval.test_longmemeval as harness
from tests.benchmarks.longmemeval.evaluator import llm_evaluate, normalize_text, quick_score


class _DummyBackend:
    name = "automem"

    def health_check(self) -> bool:
        return True

    def cleanup_scope(self, scope_id: str) -> int:
        return 0

    def ingest_memories(self, *args, **kwargs):
        return {}

    def search(self, request):
        return []


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


def test_stratified_selection_preserves_dataset_order_and_limits_per_type(
    monkeypatch,
) -> None:
    monkeypatch.setattr(harness, "create_backend", lambda *args, **kwargs: _DummyBackend())
    benchmark = harness.LongMemEvalBenchmark(harness.get_config("baseline", per_type=1))
    dataset = [
        {"question_id": "u1", "question_type": "single-session-user"},
        {"question_id": "u2", "question_type": "single-session-user"},
        {"question_id": "m1", "question_type": "multi-session"},
        {"question_id": "t1", "question_type": "temporal-reasoning"},
    ]

    selected, metadata = benchmark.select_dataset(dataset)

    assert [item["question_id"] for item in selected] == ["u1", "m1", "t1"]
    assert metadata["strategy"] == "stratified_per_type"
    assert metadata["per_type"] == 1
    assert metadata["selected_type_distribution"] == {
        "multi-session": 1,
        "single-session-user": 1,
        "temporal-reasoning": 1,
    }


def test_generate_answer_uses_gpt5_compatible_chat_completion_args(monkeypatch) -> None:
    monkeypatch.setattr(harness, "create_backend", lambda *args, **kwargs: _DummyBackend())
    benchmark = harness.LongMemEvalBenchmark(harness.get_config("baseline", llm_model="gpt-5-mini"))
    captured = {}

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Step 3 - Answer: Ada"))]
            )

    benchmark.openai_client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))

    answer = benchmark.generate_answer(
        "What is the name?",
        [{"content": "User: Her name is Ada.", "metadata": {}}],
        "2023/05/30 (Tue) 20:42",
    )

    assert answer == "Ada"
    assert captured["model"] == "gpt-5-mini"
    assert captured["max_completion_tokens"] == benchmark.GPT5_ANSWER_MAX_COMPLETION_TOKENS[0]
    assert "max_tokens" not in captured
    assert "temperature" not in captured


def test_generate_answer_retries_empty_gpt5_content_with_larger_budget(
    monkeypatch,
) -> None:
    monkeypatch.setattr(harness, "create_backend", lambda *args, **kwargs: _DummyBackend())
    benchmark = harness.LongMemEvalBenchmark(harness.get_config("baseline", llm_model="gpt-5-mini"))
    budgets = []

    class _Completions:
        def create(self, **kwargs):
            budgets.append(kwargs["max_completion_tokens"])
            content = "" if len(budgets) == 1 else "Step 3 - Answer: Ada"
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

    benchmark.openai_client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))

    answer = benchmark.generate_answer(
        "What is the name?",
        [{"content": "User: Her name is Ada.", "metadata": {}}],
        "2023/05/30 (Tue) 20:42",
    )

    assert answer == "Ada"
    assert budgets == list(benchmark.GPT5_ANSWER_MAX_COMPLETION_TOKENS)


def test_judge_failure_records_error_after_retries(monkeypatch) -> None:
    monkeypatch.setattr(harness, "create_backend", lambda *args, **kwargs: _DummyBackend())
    monkeypatch.setattr(harness.time, "sleep", lambda seconds: None)
    benchmark = harness.LongMemEvalBenchmark(harness.get_config("baseline", use_llm_eval=True))

    class _Completions:
        def create(self, **kwargs):
            raise RuntimeError("judge unavailable")

    benchmark.openai_client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))

    result = benchmark._llm_evaluate_with_retries(
        question="What is the name?",
        hypothesis="Ada",
        reference="Ada",
        question_id="q_1",
        model="gpt-5.4-mini-2026-03-17",
    )

    assert result["is_correct"] is False
    assert result["judge_attempts"] == 3
    assert "RuntimeError: judge unavailable" in result["judge_error"]
