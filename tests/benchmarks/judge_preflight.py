"""Judge quota/auth preflight for AutoMem benchmark runs.

Makes one minimal chat completion against the pinned benchmark judge model
(see tests/benchmarks/judge_policy.py) before a benchmark's question loop
starts, so quota or auth failures abort the run before any ingestion work.

Exit codes:
    0  preflight passed (judge reachable, quota available)
    1  other failure (network, unexpected error, openai missing)
    2  quota / rate-limit failure (HTTP 429, insufficient_quota)
    3  auth failure (HTTP 401, missing/invalid OPENAI_API_KEY)

CLI:
    python -m tests.benchmarks.judge_preflight [--model MODEL]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

# Allow running as a script as well as a module.
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is a benchmark dependency
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised via classify paths
    OpenAI = None

from tests.benchmarks.judge_policy import CANONICAL_BENCHMARK_JUDGE_MODEL, is_gpt5_family

EXIT_OK = 0
EXIT_OTHER = 1
EXIT_QUOTA = 2
EXIT_AUTH = 3

_QUOTA_MARKERS = (
    "insufficient_quota",
    "rate limit",
    "rate_limit",
    "ratelimit",
    "429",
    "quota",
)
_AUTH_MARKERS = (
    "invalid_api_key",
    "incorrect api key",
    "authentication",
    "unauthorized",
    "401",
    "api key",
)


def classify_judge_error(exc: BaseException) -> str:
    """Classify a judge call failure into 'quota', 'auth', or 'other'.

    Matches on exception class name + message text rather than openai
    exception classes so it stays import-safe and easy to test.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(marker in text for marker in _QUOTA_MARKERS):
        return "quota"
    if any(marker in text for marker in _AUTH_MARKERS):
        return "auth"
    return "other"


def _build_client() -> Tuple[Optional[Any], Optional[Tuple[int, str]]]:
    """Construct the same OpenAI client the benchmark harness uses."""
    if load_dotenv is not None:
        load_dotenv()
        load_dotenv(Path.home() / ".config" / "automem" / ".env")
    if OpenAI is None:
        return None, (EXIT_OTHER, "judge preflight FAILED: openai package not installed")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, (
            EXIT_AUTH,
            "judge preflight FAILED: OPENAI_API_KEY is not set "
            "(export it or add it to .env / ~/.config/automem/.env)",
        )
    return OpenAI(api_key=api_key), None


def run_preflight(model: Optional[str] = None, client: Optional[Any] = None) -> Tuple[int, str]:
    """Make one minimal judge completion; return (exit_code, one-line message)."""
    model = model or CANONICAL_BENCHMARK_JUDGE_MODEL
    if client is None:
        client, error = _build_client()
        if error is not None:
            return error

    request_kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with OK."}],
    }
    if is_gpt5_family(model):
        request_kwargs["max_completion_tokens"] = 16
    else:
        request_kwargs["temperature"] = 0
        request_kwargs["max_tokens"] = 16

    try:
        client.chat.completions.create(**request_kwargs)
    except Exception as exc:  # noqa: BLE001 - classify every failure mode
        category = classify_judge_error(exc)
        if category == "quota":
            return (
                EXIT_QUOTA,
                f"judge preflight FAILED: quota/rate-limit error for {model} "
                f"({exc}) — top up OpenAI billing or wait and retry before "
                "starting the benchmark",
            )
        if category == "auth":
            return (
                EXIT_AUTH,
                f"judge preflight FAILED: auth error for {model} ({exc}) — " "check OPENAI_API_KEY",
            )
        return (EXIT_OTHER, f"judge preflight FAILED: {model} unreachable ({exc})")

    return (EXIT_OK, f"judge preflight OK: {model} reachable with available quota")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=CANONICAL_BENCHMARK_JUDGE_MODEL,
        help=f"Judge model to probe (default: {CANONICAL_BENCHMARK_JUDGE_MODEL})",
    )
    args = parser.parse_args(argv)
    code, message = run_preflight(model=args.model)
    print(message)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
