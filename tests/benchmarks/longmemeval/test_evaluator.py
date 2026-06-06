import json
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


def _sample_item(question_id: str = "q1") -> dict:
    return {
        "question_id": question_id,
        "question_type": "multi-session",
        "question": "Where did I go?",
        "answer": "Paris",
        "question_date": "2023/05/30 (Tue) 20:42",
        "answer_session_ids": ["s1"],
        "haystack_sessions": [[{"role": "user", "content": "I went to Paris."}]],
        "haystack_session_ids": ["s1"],
        "haystack_dates": ["2023/05/30 (Tue) 20:42"],
    }


def _result(question_id: str, *, judge_error=None, is_correct=True) -> dict:
    return {
        "question_id": question_id,
        "question_type": "multi-session",
        "question": "Where did I go?",
        "reference": "Paris",
        "hypothesis": "Paris" if is_correct else "Lyon",
        "is_correct": is_correct,
        "confidence": 1.0 if is_correct else 0.0,
        "explanation": "test result",
        "recalled_count": 1,
        "memories_stored": 1,
        "is_abstention": False,
        "question_date": "2023/05/30 (Tue) 20:42",
        "answer_session_ids": ["s1"],
        "retrieved_session_ids": ["s1"],
        "recall_hit_at_5": True,
        "judge_attempts": 1,
        "judge_error": judge_error,
    }


def _benchmark(monkeypatch, tmp_path):
    monkeypatch.setattr(harness, "create_backend", lambda *args, **kwargs: _DummyBackend())
    config = harness.get_config(
        "baseline",
        results_dir=str(tmp_path),
        data_file=str(tmp_path / "unused.json"),
    )
    return harness.LongMemEvalBenchmark(config)


def test_resume_loads_completed_partial_rows_and_skips_completed_question(
    monkeypatch,
    tmp_path,
) -> None:
    benchmark = _benchmark(monkeypatch, tmp_path)
    output_base = str(tmp_path / "resume-run")
    partial_path = benchmark.artifact_paths(output_base)["partial"]
    completed = _result("q1")
    with open(partial_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(completed) + "\n")

    monkeypatch.setattr(benchmark, "load_dataset", lambda: [_sample_item("q1"), _sample_item("q2")])
    monkeypatch.setattr(benchmark, "ingest_sessions", lambda item: 1)
    monkeypatch.setattr(
        benchmark,
        "evaluate_question",
        lambda item, stored: _result(item["question_id"]),
    )
    cleaned = []
    monkeypatch.setattr(
        benchmark,
        "cleanup_test_data",
        lambda question_id=None: cleaned.append(question_id),
    )

    scores = benchmark.run_benchmark(output_path=output_base, resume=True)

    assert [row["question_id"] for row in scores["details"]] == ["q1", "q2"]
    assert cleaned == ["q2"]
    assert scores["overall"]["total"] == 2
    assert len(open(partial_path, encoding="utf-8").read().splitlines()) == 2


def test_load_partial_results_ignores_malformed_trailing_jsonl_line(
    monkeypatch,
    tmp_path,
) -> None:
    benchmark = _benchmark(monkeypatch, tmp_path)
    output_base = str(tmp_path / "partial-run")
    partial_path = benchmark.artifact_paths(output_base)["partial"]
    with open(partial_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(_result("q1")) + "\n")
        f.write("{not-json")

    results = benchmark._load_partial_results(output_base)

    assert [row["question_id"] for row in results] == ["q1"]


def test_status_json_records_progress_and_artifact_metadata(
    monkeypatch,
    tmp_path,
) -> None:
    benchmark = _benchmark(monkeypatch, tmp_path)
    output_base = str(tmp_path / "status-run")
    monkeypatch.setattr(benchmark, "load_dataset", lambda: [_sample_item("q1")])
    monkeypatch.setattr(benchmark, "ingest_sessions", lambda item: 1)
    monkeypatch.setattr(
        benchmark,
        "evaluate_question",
        lambda item, stored: _result(item["question_id"]),
    )
    monkeypatch.setattr(benchmark, "cleanup_test_data", lambda question_id=None: None)

    benchmark.run_benchmark(output_path=output_base)
    status_path = benchmark.artifact_paths(output_base)["status"]

    with open(status_path, encoding="utf-8") as f:
        status = json.load(f)

    assert status["status"] == "completed"
    assert status["completed"] == 1
    assert status["remaining"] == 0
    assert status["artifacts"]["partial"].endswith(".partial.jsonl")
    assert status["config"]["answerer_model"] == "gpt-5-mini"


def test_publishable_false_when_any_judge_errors_are_recorded(
    monkeypatch,
    tmp_path,
) -> None:
    benchmark = _benchmark(monkeypatch, tmp_path)
    output_base = str(tmp_path / "judge-error-run")
    monkeypatch.setattr(benchmark, "load_dataset", lambda: [_sample_item("q1")])
    monkeypatch.setattr(benchmark, "ingest_sessions", lambda item: 1)
    monkeypatch.setattr(
        benchmark,
        "evaluate_question",
        lambda item, stored: _result(item["question_id"], judge_error="timeout", is_correct=False),
    )
    monkeypatch.setattr(benchmark, "cleanup_test_data", lambda question_id=None: None)

    scores = benchmark.run_benchmark(output_path=output_base)

    assert scores["judge_errors"] == 1
    assert scores["publishable"] is False
    assert scores["publishable_reason"] == "judge_errors_present"
