from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional


class MemoryClassifier:
    """Classifies memories into specific types based on content patterns."""

    MAX_COMPLETION_PREFIXES = ("o1", "o3", "o4", "gpt-4o", "gpt-4.1", "gpt-5")

    PATTERNS = {
        "Decision": [
            r"decided to",
            r"chose (\w+) over",
            r"going with",
            r"picked",
            r"selected",
            r"will use",
            r"choosing",
            r"opted for",
        ],
        "Pattern": [
            r"usually",
            r"typically",
            r"tend to",
            r"pattern i noticed",
            r"often",
            r"frequently",
            r"regularly",
            r"consistently",
        ],
        "Preference": [
            r"prefer",
            r"like.*better",
            r"favorite",
            r"always use",
            r"rather than",
            r"instead of",
            r"favor",
        ],
        "Style": [
            r"wrote.*in.*style",
            r"communicated",
            r"responded to",
            r"formatted as",
            r"using.*tone",
            r"expressed as",
        ],
        "Habit": [
            r"\balways\b(?!\s+use\b)",
            r"every time",
            r"habitually",
            r"routine",
            r"daily",
            r"weekly",
            r"monthly",
        ],
        "Insight": [
            r"realized",
            r"discovered",
            r"learned that",
            r"understood",
            r"figured out",
            r"insight",
            r"revelation",
        ],
        "Context": [
            r"during",
            r"while working on",
            r"in the context of",
            r"when",
            r"at the time",
            r"situation was",
        ],
    }

    SYSTEM_PROMPT = """You are a memory classification system. Classify each memory into exactly ONE of these types:

- **Decision**: Choices made, selected options, what was decided
- **Pattern**: Recurring behaviors, typical approaches, consistent tendencies
- **Preference**: Likes/dislikes, favorites, personal tastes
- **Style**: Communication approach, formatting, tone used
- **Habit**: Regular routines, repeated actions, schedules
- **Insight**: Discoveries, learnings, realizations, key findings
- **Context**: Situational background, what was happening, circumstances

Return JSON with: {"type": "<type>", "confidence": <0.0-1.0>}"""

    def __init__(
        self,
        *,
        normalize_memory_type: Callable[[str], tuple[str, bool]],
        ensure_openai_client: Callable[[], None],
        get_openai_client: Callable[[], Any],
        classification_model: str,
        logger: Any,
    ) -> None:
        self._normalize_memory_type = normalize_memory_type
        self._ensure_openai_client = ensure_openai_client
        self._get_openai_client = get_openai_client
        self._classification_model = classification_model
        self._logger = logger

    def classify(self, content: str, *, use_llm: bool = True) -> tuple[str, float]:
        """Classify memory type and return confidence score."""
        content_lower = content.lower()

        for memory_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    confidence = 0.6
                    matches = sum(1 for p in patterns if re.search(p, content_lower))
                    if matches > 1:
                        confidence = min(0.95, confidence + (matches * 0.1))
                    return memory_type, confidence

        if use_llm:
            try:
                result = self._classify_with_llm(content)
                if result:
                    return result
            except Exception:
                self._logger.exception("LLM classification failed, using fallback")

        return "Memory", 0.3

    def _classify_with_llm(self, content: str) -> Optional[tuple[str, float]]:
        client = self._get_openai_client()
        if client is None:
            self._ensure_openai_client()
            client = self._get_openai_client()
        if client is None:
            return None

        try:
            extra_params: dict[str, Any] = {}
            uses_max_completion_tokens = self._classification_model.startswith(
                self.MAX_COMPLETION_PREFIXES
            )
            if uses_max_completion_tokens:
                extra_params["max_completion_tokens"] = 50
            else:
                extra_params["max_tokens"] = 50
                extra_params["temperature"] = 0.3

            response = client.chat.completions.create(
                model=self._classification_model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": content[:1000]},
                ],
                response_format={"type": "json_object"},
                **extra_params,
            )

            raw_content = response.choices[0].message.content
            if not raw_content:
                self._logger.warning("LLM returned empty classification response")
                return None

            try:
                result = json.loads(raw_content)
            except (json.JSONDecodeError, TypeError):
                self._logger.warning("LLM returned invalid JSON classification response")
                return None

            raw_type = result.get("type", "Memory")
            confidence = float(result.get("confidence", 0.7))

            memory_type, was_normalized = self._normalize_memory_type(raw_type)
            if not memory_type:
                self._logger.warning("LLM returned unmappable type '%s', using Context", raw_type)
                return "Context", 0.5

            if was_normalized and memory_type != raw_type:
                self._logger.debug("LLM type normalized '%s' -> '%s'", raw_type, memory_type)

            self._logger.info("LLM classified as %s (confidence: %.2f)", memory_type, confidence)
            return memory_type, confidence
        except Exception as exc:
            self._logger.warning("LLM classification failed: %s", exc)
            return None
