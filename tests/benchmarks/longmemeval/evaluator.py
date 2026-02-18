"""
LongMemEval Evaluation Module

Provides scoring utilities:
- Quick local scoring (exact match, substring, F1 token overlap)
- GPT-4o-based evaluation (matches LongMemEval paper's methodology)
"""

import json
import os
import re
from collections import Counter
from typing import Any, ClassVar, Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip, remove articles."""
    text = text.lower().strip()
    # Remove common articles and filler
    text = re.sub(r"\b(the|a|an|is|was|were|are|am)\b", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(hypothesis: str, reference: str) -> bool:
    """Check if normalized hypothesis matches reference."""
    return normalize_text(hypothesis) == normalize_text(reference)


def substring_match(hypothesis: str, reference: str) -> bool:
    """Check if reference appears as substring in hypothesis."""
    h = normalize_text(hypothesis)
    r = normalize_text(reference)
    return r in h or h in r


def f1_token_overlap(hypothesis: str, reference: str) -> float:
    """Compute F1 score based on token overlap."""
    h_tokens = normalize_text(hypothesis).split()
    r_tokens = normalize_text(reference).split()

    if not h_tokens or not r_tokens:
        return 0.0

    h_counts = Counter(h_tokens)
    r_counts = Counter(r_tokens)
    common_count = sum((h_counts & r_counts).values())
    if common_count == 0:
        return 0.0

    precision = common_count / len(h_tokens)
    recall = common_count / len(r_tokens)
    return 2 * precision * recall / (precision + recall)


def is_abstention_question(question_id: str) -> bool:
    """Check if this is an unanswerable (abstention) question."""
    return question_id.endswith("_abs")


def check_abstention_response(hypothesis: str) -> bool:
    """Check if the model correctly abstained from answering."""
    h = hypothesis.lower().strip()
    abstention_phrases = [
        "i don't know",
        "i do not know",
        "cannot determine",
        "not enough information",
        "no information",
        "not mentioned",
        "unable to determine",
        "no relevant",
        "cannot be determined",
        "isn't mentioned",
        "not available",
        "don't have enough",
        "do not have enough",
        "no memory",
        "no record",
    ]
    return any(phrase in h for phrase in abstention_phrases)


def quick_score(hypothesis: str, reference: str, question_id: str) -> Dict[str, Any]:
    """
    Quick local scoring without LLM calls.

    Returns dict with:
      - exact_match: bool
      - substring_match: bool
      - f1: float
      - is_correct: bool (heuristic based on above)
      - is_abstention: bool
      - abstention_correct: bool (if applicable)
    """
    is_abs = is_abstention_question(question_id)

    if is_abs:
        abstained = check_abstention_response(hypothesis)
        return {
            "exact_match": False,
            "substring_match": False,
            "f1": 0.0,
            "is_correct": abstained,
            "is_abstention": True,
            "abstention_correct": abstained,
        }

    em = exact_match(hypothesis, reference)
    sm = substring_match(hypothesis, reference)
    f1 = f1_token_overlap(hypothesis, reference)

    # Heuristic: correct if exact match, substring match, or high F1
    is_correct = em or sm or f1 >= 0.5

    return {
        "exact_match": em,
        "substring_match": sm,
        "f1": f1,
        "is_correct": is_correct,
        "is_abstention": False,
        "abstention_correct": None,
    }


def llm_evaluate(
    question: str,
    hypothesis: str,
    reference: str,
    question_id: str,
    model: str = "gpt-4o",
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    GPT-4o-based evaluation matching LongMemEval paper methodology.

    The evaluator asks the LLM to judge whether the hypothesis correctly
    answers the question given the reference answer.

    Returns dict with:
      - is_correct: bool
      - confidence: float
      - explanation: str
    """
    if client is None:
        if OpenAI is None:
            raise ImportError("openai package required for LLM evaluation")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required for LLM evaluation")
        client = OpenAI(api_key=api_key)

    is_abs = is_abstention_question(question_id)

    if is_abs:
        # For abstention questions, check if model correctly said "I don't know"
        prompt = f"""You are evaluating whether an AI assistant correctly identified that a question cannot be answered from the available information.

Question: {question}

The assistant's response: {hypothesis}

The correct behavior is to indicate that the information is not available or that it doesn't know the answer.

Did the assistant correctly abstain from answering (i.e., indicate it doesn't know or can't determine the answer)?

Respond with ONLY a JSON object:
{{"correct": true/false, "confidence": 0.0-1.0, "explanation": "brief reason"}}"""
    else:
        prompt = f"""You are evaluating whether an AI assistant's answer to a question is correct.

Question: {question}

Reference answer: {reference}

Assistant's answer: {hypothesis}

Judge whether the assistant's answer is correct. The answer doesn't need to be word-for-word identical, but it should convey the same key information as the reference answer. Minor variations in phrasing, additional context, or slightly different formatting are acceptable as long as the core answer is correct.

Respond with ONLY a JSON object:
{{"correct": true/false, "confidence": 0.0-1.0, "explanation": "brief reason"}}"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        content = response.choices[0].message.content.strip()

        # Parse JSON response (handle fenced markdown like ```json ... ```)
        fence_match = re.match(r"^\s*```[a-zA-Z0-9_-]*\s*\n(?P<body>.*)\n\s*```\s*$", content, re.S)
        if fence_match:
            content = fence_match.group("body").strip()

        result = json.loads(content)
        return {
            "is_correct": result.get("correct", False),
            "confidence": result.get("confidence", 0.0),
            "explanation": result.get("explanation", ""),
        }

    except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as e:
        # Fall back to quick scoring on LLM failure
        qs = quick_score(hypothesis, reference, question_id)
        return {
            "is_correct": qs["is_correct"],
            "confidence": 0.5,
            "explanation": f"LLM eval failed ({e}), used quick score fallback",
        }


class LongMemEvalScorer:
    """Aggregates scores across questions and produces reports."""

    # Question type display names
    TYPE_NAMES: ClassVar[Dict[str, str]] = {
        "single-session-user": "Single-Session (User)",
        "single-session-assistant": "Single-Session (Assistant)",
        "single-session-preference": "Single-Session (Preference)",
        "multi-session": "Multi-Session",
        "knowledge-update": "Knowledge Update",
        "temporal-reasoning": "Temporal Reasoning",
    }

    # Competitive landscape for comparison
    COMPETITORS: ClassVar[Dict[str, float]] = {
        "Mastra OM (gpt-5-mini)": 94.87,
        "Emergence Internal": 86.0,
        "Mastra OM (gpt-4o)": 84.23,
        "Oracle gpt-4o": 82.4,
        "Supermemory": 81.6,
        "Mastra RAG (gpt-4o)": 80.0,
        "LongMemEval paper best RAG": 72.0,
        "Zep": 71.2,
        "Naive RAG": 52.0,
        "Best Guess (no memory)": 18.8,
    }

    def __init__(self) -> None:
        self.results: List[Dict[str, Any]] = []

    def add_result(self, result: Dict[str, Any]) -> None:
        """Add a single question result."""
        self.results.append(result)

    def compute_scores(self) -> Dict[str, Any]:
        """Compute aggregated scores."""
        if not self.results:
            return {
                "overall": {"accuracy": 0.0, "correct": 0, "total": 0},
                "by_type": {},
                "abstention": {"total": 0, "correct": 0, "accuracy": 0.0},
            }

        total = len(self.results)
        correct = sum(1 for r in self.results if r.get("is_correct", False))
        overall_accuracy = correct / total if total > 0 else 0.0

        # By question type
        by_type = {}
        for r in self.results:
            qtype = r.get("question_type", "unknown")
            if qtype not in by_type:
                by_type[qtype] = {"correct": 0, "total": 0}
            by_type[qtype]["total"] += 1
            if r.get("is_correct", False):
                by_type[qtype]["correct"] += 1

        for qtype in by_type:
            t = by_type[qtype]
            t["accuracy"] = t["correct"] / t["total"] if t["total"] > 0 else 0.0
            t["name"] = self.TYPE_NAMES.get(qtype, qtype)

        # Abstention stats
        abs_results = [r for r in self.results if r.get("is_abstention", False)]
        abs_correct = sum(1 for r in abs_results if r.get("is_correct", False))

        return {
            "overall": {
                "accuracy": overall_accuracy,
                "correct": correct,
                "total": total,
            },
            "by_type": by_type,
            "abstention": {
                "total": len(abs_results),
                "correct": abs_correct,
                "accuracy": abs_correct / len(abs_results) if abs_results else 0.0,
            },
        }

    def print_report(
        self, config_name: str = "baseline", elapsed_time: float = 0.0
    ) -> Dict[str, Any]:
        """Print a formatted benchmark report."""
        scores = self.compute_scores()
        overall = scores["overall"]

        print(f"\n{'='*60}")
        print("LongMemEval Benchmark Results")
        print(f"Config: {config_name}")
        if elapsed_time > 0:
            print(f"Elapsed: {elapsed_time:.1f}s")
        print(f"{'='*60}")

        print(
            f"\nOverall Accuracy: {overall['accuracy']:.2%} ({overall['correct']}/{overall['total']})"
        )

        print("\nAccuracy by Question Type:")
        print(f"  {'Type':<35s} {'Accuracy':>8s} {'Count':>8s}")
        print(f"  {'-'*35} {'-'*8} {'-'*8}")

        for qtype, data in sorted(scores["by_type"].items()):
            name = data.get("name", qtype)
            acc = data["accuracy"]
            count = f"{data['correct']}/{data['total']}"
            print(f"  {name:<35s} {acc:>7.1%} {count:>8s}")

        # Abstention
        abs_data = scores["abstention"]
        if abs_data["total"] > 0:
            print(
                f"\nAbstention Questions: {abs_data['accuracy']:.1%} ({abs_data['correct']}/{abs_data['total']})"
            )

        # Competitive comparison
        print("\nCompetitive Landscape:")
        automem_pct = overall["accuracy"] * 100
        for name, score in sorted(self.COMPETITORS.items(), key=lambda x: -x[1]):
            marker = " <-- AutoMem" if abs(score - automem_pct) < 0.5 else ""
            print(f"  {name:<35s} {score:>6.1f}%{marker}")
        print(f"  {'AutoMem (' + config_name + ')':<35s} {automem_pct:>6.1f}%  <<<")

        print(f"\n{'='*60}")
        return scores
