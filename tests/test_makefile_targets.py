"""Tests for developer-facing Makefile benchmark targets."""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_bench_current_state_target_is_no_llm_pytest_gate() -> None:
    result = subprocess.run(
        ["make", "-n", "bench-current-state"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    output = result.stdout
    assert "pytest" in output
    assert "tests/test_api_endpoints.py" in output
    assert "recall_state_mode" in output
    assert "recall_current_only" in output
    assert "--llm-eval" not in output
    assert "OPENAI_API_KEY" not in output
