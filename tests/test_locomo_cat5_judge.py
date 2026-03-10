import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

_MODULE_NAME = "locomo_benchmark_module_cat5"


def _load_locomo_module() -> Any:
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]
    module_path = Path(__file__).resolve().parent / "benchmarks" / "test_locomo.py"
    spec = spec_from_file_location(_MODULE_NAME, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def locomo_module() -> Any:
    return _load_locomo_module()


@pytest.fixture()
def locomo_evaluator(locomo_module: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.delenv("BENCH_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = locomo_module.LoCoMoConfig()
    config.judge_model = None
    evaluator = locomo_module.LoCoMoEvaluator(config)
    evaluator.openai_client = None
    evaluator.has_openai_api_key = False
    evaluator.use_embedding_similarity = False
    return evaluator


def _fake_memory(dialog_id: str, content: str) -> Dict[str, Any]:
    return {
        "content": content,
        "metadata": {
            "dialog_id": dialog_id,
            "session_datetime": "2023-05-08T13:56:00+00:00",
        },
    }


def _cat5_qa(**overrides: Any) -> Dict[str, Any]:
    qa = {
        "question": "How does Jolene plan to pursue her dream of climbing mountains?",
        "answer": "",
        "adversarial_answer": "By ignoring training and winging it.",
        "category": 5,
        "evidence": ["D10:20"],
    }
    qa.update(overrides)
    return qa


def test_cat5_without_judge_still_skips(locomo_evaluator: Any) -> None:
    result = locomo_evaluator._evaluate_question(_cat5_qa(), "conv-1")

    assert result["is_correct"] is None
    assert result["recalled_count"] == 0
    assert result["explanation"] == "Skipped: requires LLM judge"


def test_cat5_with_judge_counts_toward_category_results(
    locomo_evaluator: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    locomo_evaluator.config.judge_model = "gpt-4o"
    monkeypatch.setattr(
        locomo_evaluator,
        "_recall_memories_for_qa",
        lambda question, sample_id, evidence: [_fake_memory("D10:20", "Jolene watches videos.")],
    )
    monkeypatch.setattr(
        locomo_evaluator,
        "judge_complex_reasoning",
        lambda question, adversarial_answer, recalled_memories, evidence_dialog_ids, sample_id: (
            True,
            0.92,
            "LLM judge: supported by evidence",
            "She plans to study and start with beginner climbs.",
            "supported by evidence",
        ),
    )

    conversation = {"qa": [_cat5_qa()]}
    result = locomo_evaluator._evaluate_only(conversation, "conv-1")

    assert result["correct"] == 1
    assert result["total_questions"] == 1
    assert locomo_evaluator.results[5] == [True]
    assert result["qa_results"][0]["judge_generated_answer"] is not None


def test_fetch_evidence_memories_uses_local_conversation_cache(
    locomo_evaluator: Any,
) -> None:
    locomo_evaluator.local_conversation_memories["conv-26"] = {
        "D2:3": _fake_memory("D2:3", "Target evidence"),
        "D5:8": _fake_memory("D5:8", "Secondary evidence"),
    }

    evidence = locomo_evaluator.fetch_evidence_memories(
        ["D2:3", "D5:8"],
        "conv-26",
        use_local_cache=True,
    )

    assert [memory["metadata"]["dialog_id"] for memory in evidence] == ["D2:3", "D5:8"]


def test_cat5_judge_uses_cache(locomo_evaluator: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    locomo_evaluator.config.judge_model = "gpt-4o"
    calls = {"count": 0}

    class FakeCompletions:
        def create(self, **kwargs: Any) -> Any:
            calls["count"] += 1
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                '{"generated_answer": "She will research and train.", '
                                '"verdict": "supported", '
                                '"correct": true, "confidence": 0.88, '
                                '"reasoning": "Matches the evidence dialog."}'
                            )
                        )
                    )
                ]
            )

    locomo_evaluator.openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    monkeypatch.setattr(
        locomo_evaluator,
        "fetch_evidence_memories",
        lambda evidence_dialog_ids, sample_id, use_local_cache=False: [
            _fake_memory("D10:20", "Jolene is gathering information and watching videos.")
        ],
    )

    recalled = [_fake_memory("R1", "Jolene is gathering information and watching videos.")]
    first = locomo_evaluator.judge_complex_reasoning(
        _cat5_qa()["question"],
        _cat5_qa()["adversarial_answer"],
        recalled,
        ["D10:20"],
        "conv-1",
    )
    second = locomo_evaluator.judge_complex_reasoning(
        _cat5_qa()["question"],
        _cat5_qa()["adversarial_answer"],
        recalled,
        ["D10:20"],
        "conv-1",
    )

    assert calls["count"] == 1
    assert first == second
    assert first[0] is True


def test_cat5_judge_sets_request_timeout(
    locomo_evaluator: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    locomo_evaluator.config.judge_model = "gpt-4o"
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs: Any) -> Any:
            captured["timeout"] = kwargs.get("timeout")
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                '{"generated_answer": "She will research and train.", '
                                '"verdict": "supported", '
                                '"correct": true, "confidence": 0.88, '
                                '"reasoning": "Matches the evidence dialog."}'
                            )
                        )
                    )
                ]
            )

    locomo_evaluator.openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    monkeypatch.setattr(
        locomo_evaluator,
        "fetch_evidence_memories",
        lambda evidence_dialog_ids, sample_id, use_local_cache=False: [
            _fake_memory("D10:20", "Jolene is gathering information and watching videos.")
        ],
    )

    result = locomo_evaluator.judge_complex_reasoning(
        _cat5_qa()["question"],
        _cat5_qa()["adversarial_answer"],
        [_fake_memory("R1", "Jolene is gathering information and watching videos.")],
        ["D10:20"],
        "conv-1",
    )

    assert result[0] is True
    assert captured["timeout"] == locomo_evaluator.OPENAI_REQUEST_TIMEOUT_SECONDS


@pytest.mark.parametrize(
    ("evidence_memories", "response_content", "expected_message"),
    [
        (
            [],
            '{"generated_answer": "", "verdict": "unsupported", "correct": true, "confidence": 0.1, "reasoning": ""}',
            "Skipped: no evidence memories available for LLM judge",
        ),
        (
            [{"content": "evidence", "metadata": {"dialog_id": "D10:20"}}],
            "not-json",
            "Skipped: LLM judge error:",
        ),
    ],
)
def test_cat5_judge_failures_skip_instead_of_marking_wrong(
    locomo_evaluator: Any,
    monkeypatch: pytest.MonkeyPatch,
    evidence_memories: List[Dict[str, Any]],
    response_content: str,
    expected_message: str,
) -> None:
    locomo_evaluator.config.judge_model = "gpt-4o"

    class FakeCompletions:
        def create(self, **kwargs: Any) -> Any:
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=response_content))]
            )

    locomo_evaluator.openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    monkeypatch.setattr(
        locomo_evaluator,
        "fetch_evidence_memories",
        lambda evidence_dialog_ids, sample_id, use_local_cache=False: evidence_memories,
    )
    monkeypatch.setattr(
        locomo_evaluator,
        "_recall_memories_for_qa",
        lambda question, sample_id, evidence: [
            _fake_memory("R1", "Jolene is gathering information and watching videos.")
        ],
    )

    result = locomo_evaluator._evaluate_question(_cat5_qa(), "conv-1")

    assert result["is_correct"] is None
    assert result["confidence"] == 0.0
    assert expected_message in result["explanation"]


def test_cat5_judge_prompt_allows_abstention_and_wrong_premise(
    locomo_evaluator: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    locomo_evaluator.config.judge_model = "gpt-4o"
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs: Any) -> Any:
            captured["messages"] = kwargs["messages"]
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                '{"generated_answer": "I don\'t know; the premise seems wrong.", '
                                '"verdict": "contradiction", '
                                '"correct": true, "confidence": 0.91, '
                                '"reasoning": "The evidence shows the question names the wrong person."}'
                            )
                        )
                    )
                ]
            )

    locomo_evaluator.openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    monkeypatch.setattr(
        locomo_evaluator,
        "fetch_evidence_memories",
        lambda evidence_dialog_ids, sample_id, use_local_cache=False: [
            _fake_memory("D2:3", "The realization belonged to Melanie, not Caroline.")
        ],
    )

    result = locomo_evaluator.judge_complex_reasoning(
        "What did Caroline realize after her charity race?",
        "self-care is important",
        [_fake_memory("R1", "Melanie talked about self-care after the race.")],
        ["D2:3"],
        "conv-26",
    )

    prompt = captured["messages"][1]["content"]
    assert "Do NOT assume the adversarial answer is always false" in prompt
    assert (
        "abstaining, correcting the person/entity, or stating that the premise is unsupported"
        in prompt
    )
    assert result[0] is True


def test_cat5_judge_prompt_includes_more_than_old_top_12_memories(
    locomo_evaluator: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    locomo_evaluator.config.judge_model = "gpt-4o"
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs: Any) -> Any:
            captured["messages"] = kwargs["messages"]
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                '{"generated_answer": "Answer", '
                                '"verdict": "supported", '
                                '"correct": true, "confidence": 0.8, '
                                '"reasoning": "ok"}'
                            )
                        )
                    )
                ]
            )

    locomo_evaluator.openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    monkeypatch.setattr(
        locomo_evaluator,
        "fetch_evidence_memories",
        lambda evidence_dialog_ids, sample_id, use_local_cache=False: [
            _fake_memory("D10:20", "Target evidence")
        ],
    )
    recalled = [_fake_memory(f"R{i}", f"Memory {i}") for i in range(1, 16)]

    locomo_evaluator.judge_complex_reasoning(
        _cat5_qa()["question"],
        _cat5_qa()["adversarial_answer"],
        recalled,
        ["D10:20"],
        "conv-1",
    )

    prompt = captured["messages"][1]["content"]
    assert "[R15] Memory 15" in prompt


def test_cat5_with_canonical_answer_stays_deterministic(
    locomo_evaluator: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    locomo_evaluator.config.judge_model = "gpt-4o"
    monkeypatch.setattr(
        locomo_evaluator,
        "_recall_memories_for_qa",
        lambda question, sample_id, evidence: [_fake_memory("D5:8", "Caroline did not make it.")],
    )
    monkeypatch.setattr(
        locomo_evaluator,
        "check_answer_in_memories",
        lambda question, expected_answer, recalled_memories, evidence_dialog_ids, sample_id: (
            True,
            1.0,
            "Found answer",
        ),
    )
    monkeypatch.setattr(
        locomo_evaluator,
        "judge_complex_reasoning",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("judge should not be used")),
    )

    result = locomo_evaluator._evaluate_question(
        _cat5_qa(
            question="Did Caroline make the black and white bowl in the photo?",
            answer="No",
            adversarial_answer="Yes",
            evidence=["D5:8"],
        ),
        "conv-26",
    )

    assert result["is_correct"] is True
    assert result["expected_answer"] == "No"
    assert result["judge_generated_answer"] is None
    assert result["explanation"].startswith("Deterministic cat-5 scoring:")
