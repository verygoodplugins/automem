import os
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def test_runner_forwards_stratified_resume_judge_and_existing_token(
    tmp_path: Path,
) -> None:
    sandbox = tmp_path / "repo"
    (sandbox / "scripts" / "lib").mkdir(parents=True)
    (sandbox / "tests" / "benchmarks" / "longmemeval" / "data").mkdir(parents=True)

    (sandbox / "test-longmemeval-benchmark.sh").write_text(
        (REPO_ROOT / "test-longmemeval-benchmark.sh").read_text()
    )
    (sandbox / "scripts" / "lib" / "common.sh").write_text(
        (REPO_ROOT / "scripts" / "lib" / "common.sh").read_text()
    )
    (
        sandbox / "tests" / "benchmarks" / "longmemeval" / "data" / "longmemeval_s_cleaned.json"
    ).write_text("[]")
    (sandbox / ".env").write_text("AUTOMEM_API_TOKEN=env-token\n")
    (sandbox / "test-longmemeval-benchmark.sh").chmod(0o755)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_args = tmp_path / "python-args.txt"

    _write_executable(bin_dir / "curl", "#!/bin/sh\nexit 0\n")
    _write_executable(
        bin_dir / "docker",
        "#!/bin/sh\n"
        'if [ "$1" = "info" ]; then exit 0; fi\n'
        'if [ "$1" = "compose" ]; then exit 0; fi\n'
        "exit 0\n",
    )
    _write_executable(
        bin_dir / "python3",
        "#!/bin/sh\n" f"printf '%s\\n' \"$@\" > '{python_args}'\n" "exit 0\n",
    )

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["AUTOMEM_TEST_API_TOKEN"] = "existing-token"

    result = subprocess.run(
        [
            str(sandbox / "test-longmemeval-benchmark.sh"),
            "--per-type",
            "5",
            "--llm-eval",
            "--llm-model",
            "gpt-5-mini",
            "--eval-llm-model",
            "custom-judge",
            "--output",
            "benchmarks/results/run-base",
            "--resume",
        ],
        cwd=sandbox,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )

    args = python_args.read_text().splitlines()

    assert result.returncode == 0, result.stdout + result.stderr
    assert "--api-token" in args
    assert args[args.index("--api-token") + 1] == "existing-token"
    assert "--per-type" in args
    assert args[args.index("--per-type") + 1] == "5"
    assert "--llm-eval" in args
    assert args[args.index("--llm-model") + 1] == "gpt-5-mini"
    assert args[args.index("--eval-llm-model") + 1] == "custom-judge"
    assert args[args.index("--output") + 1] == "benchmarks/results/run-base"
    assert "--resume" in args


def test_runner_uses_env_file_token_when_test_token_missing(tmp_path: Path) -> None:
    sandbox = tmp_path / "repo"
    (sandbox / "scripts" / "lib").mkdir(parents=True)
    (sandbox / "tests" / "benchmarks" / "longmemeval" / "data").mkdir(parents=True)

    (sandbox / "test-longmemeval-benchmark.sh").write_text(
        (REPO_ROOT / "test-longmemeval-benchmark.sh").read_text()
    )
    (sandbox / "scripts" / "lib" / "common.sh").write_text(
        (REPO_ROOT / "scripts" / "lib" / "common.sh").read_text()
    )
    (
        sandbox / "tests" / "benchmarks" / "longmemeval" / "data" / "longmemeval_s_cleaned.json"
    ).write_text("[]")
    (sandbox / ".env").write_text("AUTOMEM_API_TOKEN=env-token\n")
    (sandbox / "test-longmemeval-benchmark.sh").chmod(0o755)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_args = tmp_path / "python-args-env.txt"

    _write_executable(bin_dir / "curl", "#!/bin/sh\nexit 0\n")
    _write_executable(
        bin_dir / "docker",
        "#!/bin/sh\n"
        'if [ "$1" = "info" ]; then exit 0; fi\n'
        'if [ "$1" = "compose" ]; then exit 0; fi\n'
        "exit 0\n",
    )
    _write_executable(
        bin_dir / "python3",
        "#!/bin/sh\n" f"printf '%s\\n' \"$@\" > '{python_args}'\n" "exit 0\n",
    )

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env.pop("AUTOMEM_TEST_API_TOKEN", None)

    result = subprocess.run(
        [str(sandbox / "test-longmemeval-benchmark.sh"), "--max-questions", "1"],
        cwd=sandbox,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )

    args = python_args.read_text().splitlines()

    assert result.returncode == 0, result.stdout + result.stderr
    assert args[args.index("--api-token") + 1] == "env-token"
