import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any


class _DummyBackend:
    name = "automem"

    def health_check(self) -> bool:
        return True


def _load_module(module_name: str, path: Path) -> Any:
    spec = spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_global_env(tmp_path: Path) -> None:
    env_dir = tmp_path / ".config" / "automem"
    env_dir.mkdir(parents=True)
    (env_dir / ".env").write_text("OPENAI_API_KEY=file-openai-key\n", encoding="utf-8")


def test_locomo_benchmark_loads_openai_api_key_from_global_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_global_env(tmp_path)

    module_path = Path(__file__).resolve().parent / "benchmarks" / "test_locomo.py"
    module = _load_module("locomo_benchmark_env_loading", module_path)
    monkeypatch.setattr(module, "create_backend", lambda *args, **kwargs: _DummyBackend())

    evaluator = module.LoCoMoEvaluator(module.LoCoMoConfig())

    assert evaluator.has_openai_api_key is True
    assert evaluator.openai_client is not None


def test_longmemeval_benchmark_loads_openai_api_key_from_global_env(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_global_env(tmp_path)

    module_path = (
        Path(__file__).resolve().parent / "benchmarks" / "longmemeval" / "test_longmemeval.py"
    )
    module = _load_module("longmemeval_benchmark_env_loading", module_path)
    monkeypatch.setattr(module, "create_backend", lambda *args, **kwargs: _DummyBackend())

    benchmark = module.LongMemEvalBenchmark(module.get_config("baseline"))

    assert benchmark.openai_client is not None


def test_longmemeval_llm_eval_defaults_to_canonical_judge(monkeypatch) -> None:
    monkeypatch.delenv("LONGMEMEVAL_EVAL_LLM_MODEL", raising=False)
    module_path = (
        Path(__file__).resolve().parent / "benchmarks" / "longmemeval" / "test_longmemeval.py"
    )
    module = _load_module("longmemeval_canonical_judge", module_path)
    monkeypatch.setattr(module, "create_backend", lambda *args, **kwargs: _DummyBackend())

    config = module.get_config("baseline", use_llm_eval=True)
    benchmark = module.LongMemEvalBenchmark(config)

    assert benchmark.effective_judge_model() == "gpt-5.4-mini-2026-03-17"


def test_longmemeval_answerer_defaults_to_gpt5_mini(monkeypatch) -> None:
    monkeypatch.delenv("LONGMEMEVAL_LLM_MODEL", raising=False)
    monkeypatch.delitem(sys.modules, "tests.benchmarks.longmemeval.configs", raising=False)
    module_path = (
        Path(__file__).resolve().parent / "benchmarks" / "longmemeval" / "test_longmemeval.py"
    )
    module = _load_module("longmemeval_default_answerer", module_path)

    assert module.get_config("baseline").llm_model == "gpt-5-mini"
