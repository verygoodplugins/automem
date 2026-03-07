from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest


def _load_locomo_module() -> Any:
    module_path = Path(__file__).resolve().parent / "benchmarks" / "test_locomo.py"
    spec = spec_from_file_location("locomo_benchmark_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def locomo_evaluator() -> Any:
    locomo_module = _load_locomo_module()
    return locomo_module.LoCoMoEvaluator(locomo_module.LoCoMoConfig())


def test_extract_speaker_from_question_handles_ascii_possessive_name(
    locomo_evaluator: Any,
) -> None:
    speaker = locomo_evaluator._extract_speaker_from_question(
        "Would Caroline's sister pursue writing as a career?"
    )

    assert speaker == "Caroline"


def test_extract_speaker_from_question_handles_curly_possessive_name(
    locomo_evaluator: Any,
) -> None:
    speaker = locomo_evaluator._extract_speaker_from_question(
        "Would Caroline’s sister pursue writing as a career?"
    )

    assert speaker == "Caroline"
