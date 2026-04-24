import os
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def test_runner_prints_resolved_judge_and_output_summary(tmp_path: Path) -> None:
    sandbox = tmp_path / "repo"
    (sandbox / "scripts" / "lib").mkdir(parents=True)
    (sandbox / "tests" / "benchmarks" / "locomo" / "data").mkdir(parents=True)

    (sandbox / "test-locomo-benchmark.sh").write_text(
        (REPO_ROOT / "test-locomo-benchmark.sh").read_text()
    )
    (sandbox / "scripts" / "lib" / "common.sh").write_text(
        (REPO_ROOT / "scripts" / "lib" / "common.sh").read_text()
    )
    (sandbox / "tests" / "benchmarks" / "locomo" / "data" / "locomo10.json").write_text("[]")
    (sandbox / "test-locomo-benchmark.sh").chmod(0o755)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_args = tmp_path / "python-args.txt"

    _write_executable(
        bin_dir / "curl",
        "#!/bin/sh\nexit 0\n",
    )
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
    env["BENCH_JUDGE_MODEL"] = "custom-judge-model"

    result = subprocess.run(
        [
            str(sandbox / "test-locomo-benchmark.sh"),
            "--conversations",
            "0",
            "--judge",
            "--output",
            "benchmarks/results/test.json",
        ],
        cwd=sandbox,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )

    stdout = result.stdout + result.stderr

    assert result.returncode == 0
    assert "Judge:" in stdout
    assert "enabled" in stdout
    assert "custom-judge-model" in stdout
    assert "Conversations:" in stdout
    assert "0" in stdout
    assert "Output:" in stdout
    assert "benchmarks/results/test.json" in stdout
    assert "--judge" in python_args.read_text()
