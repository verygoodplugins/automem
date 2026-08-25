"""OpenAI-compatible chat completion parameter helpers."""

from __future__ import annotations

from typing import Any

REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def chat_completion_token_params(model: str, max_output_tokens: int) -> dict[str, Any]:
    """Return output-budget parameters compatible with an OpenAI chat model.

    OpenAI reasoning models reject ``max_tokens`` and non-default temperature;
    regular chat models use ``max_tokens``. Model names may be provider-prefixed
    (for example, ``openai/gpt-5-mini``), so inspect the final name segment.
    """
    model_name = model.rsplit("/", 1)[-1].strip().lower()
    if model_name.startswith(REASONING_MODEL_PREFIXES):
        return {"max_completion_tokens": max_output_tokens}
    return {"max_tokens": max_output_tokens, "temperature": 0.3}
