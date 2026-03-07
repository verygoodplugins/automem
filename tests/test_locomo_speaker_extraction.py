from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_locomo_module():
    module_path = Path(__file__).resolve().parent / "benchmarks" / "test_locomo.py"
    spec = spec_from_file_location("locomo_benchmark_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_speaker_from_question_handles_ascii_possessive_name() -> None:
    locomo_module = _load_locomo_module()
    evaluator = locomo_module.LoCoMoEvaluator(locomo_module.LoCoMoConfig())

    speaker = evaluator._extract_speaker_from_question(
        "Would Caroline's sister pursue writing as a career?"
    )

    assert speaker == "Caroline"


def test_extract_speaker_from_question_handles_curly_possessive_name() -> None:
    locomo_module = _load_locomo_module()
    evaluator = locomo_module.LoCoMoEvaluator(locomo_module.LoCoMoConfig())

    speaker = evaluator._extract_speaker_from_question(
        "Would Caroline’s sister pursue writing as a career?"
    )

    assert speaker == "Caroline"
