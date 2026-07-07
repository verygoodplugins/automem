"""Tests for the answer-generation prompt builder (temporal_answer_hint flag)."""

from typing import Any, Dict, List

import tests.benchmarks.longmemeval.test_longmemeval as harness


class _DummyBackend:
    name = "automem"

    def health_check(self) -> bool:
        return True

    def cleanup_scope(self, scope_id: str) -> int:
        return 0

    def ingest_memories(self, *args, **kwargs):
        return {}

    def search(self, request):
        return []


def _make_benchmark(monkeypatch, **overrides) -> "harness.LongMemEvalBenchmark":
    monkeypatch.setattr(harness, "create_backend", lambda *args, **kwargs: _DummyBackend())
    return harness.LongMemEvalBenchmark(harness.get_config("baseline", **overrides))


QUESTION = "What pets do I have?"
QUESTION_DATE = "2023/06/01 (Thu) 12:00"

# Deliberately ordered newest-first (score order) so chronological re-ordering is observable.
MEMORIES: List[Dict[str, Any]] = [
    {
        "id": "m1",
        "content": "I adopted a cat named Miso.",
        "metadata": {"session_date": "2023/05/20 (Sat) 10:00"},
        "tags": [],
        "score": 0.91,
        "match_type": "vector",
    },
    {
        "id": "m2",
        "content": "I adopted a dog named Rex.",
        "metadata": {"session_date": "2023/03/01 (Wed) 09:00"},
        "tags": [],
        "score": 0.85,
        "match_type": "vector",
    },
]


def _legacy_chain_of_note_prompt(question: str, memories, question_date: str) -> str:
    """Byte-for-byte copy of the pre-flag chain-of-note prompt.

    Byte identity is load-bearing: this baseline prompt is the
    experimental control for the temporal_answer_hint flag. If it
    silently drifts from the historical prompt, new benchmark scores
    are no longer comparable to previously recorded ones.
    """
    lines = []
    for i, mem in enumerate(memories, 1):
        content = mem.get("content", "")
        metadata = mem.get("metadata", {})
        date = metadata.get("session_date", "")
        if date:
            lines.append(f"[Memory {i} - {date}]\n{content}")
        else:
            lines.append(f"[Memory {i}]\n{content}")
    context = "\n\n".join(lines) if memories else "(No relevant memories found)"

    return f"""You are answering a question based on recalled conversation memories.

First, extract relevant information from each memory excerpt.
Then, reason about the answer based on the extracted information.
Finally, provide a concise answer.

If the information is not available in the provided memories, respond with "I don't know."

Memories:
{context}

Question (asked on {question_date}): {question}

Step 1 - Extract relevant information:
Step 2 - Reasoning:
Step 3 - Answer:"""


def test_config_defaults_temporal_answer_hint_off():
    config = harness.get_config("baseline")
    assert config.temporal_answer_hint is False


def test_temporal_answer_config_preset_enables_flag():
    config = harness.get_config("temporal-answer")
    assert config.temporal_answer_hint is True


def test_evidence_answer_config_preset_enables_answer_focus_prompt():
    config = harness.get_config("evidence-answer")
    assert config.answer_focus_prompt is True
    assert config.temporal_answer_hint is False


def test_prompt_flag_off_is_byte_identical_to_legacy(monkeypatch):
    benchmark = _make_benchmark(monkeypatch)
    prompt = benchmark._build_answer_prompt(QUESTION, MEMORIES, QUESTION_DATE)
    assert prompt == _legacy_chain_of_note_prompt(QUESTION, MEMORIES, QUESTION_DATE)


def test_prompt_flag_off_no_memories_is_byte_identical(monkeypatch):
    benchmark = _make_benchmark(monkeypatch)
    prompt = benchmark._build_answer_prompt(QUESTION, [], QUESTION_DATE)
    assert prompt == _legacy_chain_of_note_prompt(QUESTION, [], QUESTION_DATE)


def test_prompt_flag_on_adds_conflict_and_abstention_guidance(monkeypatch):
    benchmark = _make_benchmark(monkeypatch, temporal_answer_hint=True)
    prompt = benchmark._build_answer_prompt(QUESTION, MEMORIES, QUESTION_DATE)

    # Conflict-recency guidance
    assert "prefer the most recent" in prompt
    # Anti-overabstention guidance without forbidding abstention
    assert "re-check each memory" in prompt
    assert 'respond with "I don\'t know."' in prompt


def test_prompt_flag_on_renders_memories_chronologically_with_scores(monkeypatch):
    benchmark = _make_benchmark(monkeypatch, temporal_answer_hint=True)
    prompt = benchmark._build_answer_prompt(QUESTION, MEMORIES, QUESTION_DATE)

    # m2 (March) must come before m1 (May) despite lower retrieval score
    assert prompt.index("dog named Rex") < prompt.index("cat named Miso")
    # Scores are noted in the memory headers
    assert "0.85" in prompt
    assert "0.91" in prompt


def test_evidence_prompt_instructs_per_memory_evidence_extraction(monkeypatch):
    benchmark = _make_benchmark(monkeypatch, answer_focus_prompt=True)
    prompt = benchmark._build_answer_prompt(QUESTION, MEMORIES, QUESTION_DATE)

    assert "For each memory, decide whether it contains exact supporting evidence" in prompt
    assert "The evidence may be paraphrased or implied by a stated preference" in prompt
    assert "For count or multi-session questions, combine all supporting memories" in prompt
    assert "For recommendation or preference questions, use stated preferences" in prompt
    assert "For temporal questions, use the memory dates" in prompt
    assert 'Do not answer "I don\'t know" until every memory has been checked' in prompt
    assert "Step 1 - Evidence by memory:" in prompt
    assert "Step 2 - Use the strongest supporting evidence:" in prompt
    assert "Step 3 - Answer:" in prompt


def test_evidence_prompt_keeps_rank_order_with_dates_and_scores(monkeypatch):
    benchmark = _make_benchmark(monkeypatch, answer_focus_prompt=True)
    prompt = benchmark._build_answer_prompt(QUESTION, MEMORIES, QUESTION_DATE)

    assert prompt.index("cat named Miso") < prompt.index("dog named Rex")
    assert "[Memory 1 - 2023/05/20 (Sat) 10:00 - retrieval score 0.91]" in prompt
    assert "[Memory 2 - 2023/03/01 (Wed) 09:00 - retrieval score 0.85]" in prompt


def test_evidence_prompt_no_memories_still_allows_dont_know(monkeypatch):
    benchmark = _make_benchmark(monkeypatch, answer_focus_prompt=True)
    prompt = benchmark._build_answer_prompt(QUESTION, [], QUESTION_DATE)

    assert "(No relevant memories found)" in prompt
    assert 'respond with "I don\'t know."' in prompt


def test_prompt_flag_on_plain_variant_keeps_guidance(monkeypatch):
    benchmark = _make_benchmark(monkeypatch, temporal_answer_hint=True, use_chain_of_note=False)
    prompt = benchmark._build_answer_prompt(QUESTION, MEMORIES, QUESTION_DATE)
    assert "prefer the most recent" in prompt
    assert prompt.index("dog named Rex") < prompt.index("cat named Miso")


def test_prompt_flag_off_plain_variant_is_byte_identical(monkeypatch):
    benchmark = _make_benchmark(monkeypatch, use_chain_of_note=False)
    prompt = benchmark._build_answer_prompt(QUESTION, MEMORIES, QUESTION_DATE)

    lines = []
    for i, mem in enumerate(MEMORIES, 1):
        date = mem["metadata"]["session_date"]
        lines.append(f"[Memory {i} - {date}]\n{mem['content']}")
    context = "\n\n".join(lines)
    expected = f"""Based on the following conversation history excerpts, answer the question.
If the answer cannot be determined from the provided context, say "I don't know."

Context:
{context}

Question: {QUESTION}
Answer:"""
    assert prompt == expected
