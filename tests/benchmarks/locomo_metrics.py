"""
Official LoCoMo Evaluation Metrics

This module implements the exact evaluation metrics from the LoCoMo paper
(ACL 2024: "Evaluating Very Long-Term Conversational Memory of LLM Agents").

The official metrics use:
- Porter Stemmer for word normalization
- Token-level F1 score (not exact match)
- Special handling for each category

Source: https://github.com/snap-research/locomo/blob/main/task_eval/evaluation.py
"""

import re
import string
from collections import Counter
from typing import Any, List, Tuple

import nltk
from nltk.stem import PorterStemmer

# Initialize stemmer
try:
    ps = PorterStemmer()
except LookupError:
    nltk.download("punkt")
    ps = PorterStemmer()


def normalize_answer(s: str) -> str:
    """
    Official LoCoMo answer normalization.

    1. Remove commas
    2. Remove articles (a, an, the, and)
    3. Remove punctuation
    4. Lowercase
    5. Normalize whitespace
    """
    s = s.replace(",", "")

    def remove_articles(text):
        return re.sub(r"\b(a|an|the|and)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    """
    Official LoCoMo F1 score with Porter stemming.

    Computes token-level F1 between prediction and ground truth,
    using Porter stemmer for normalization.

    Args:
        prediction: Generated/predicted answer
        ground_truth: Expected answer

    Returns:
        F1 score between 0.0 and 1.0
    """
    prediction_tokens = [ps.stem(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [ps.stem(w) for w in normalize_answer(ground_truth).split()]

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return 0.0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def f1_multi_answer(prediction: str, ground_truth: str) -> float:
    """
    Official LoCoMo F1 for multi-hop questions (category 1).

    Handles comma-separated sub-answers by computing F1 for each
    ground truth sub-answer against all prediction sub-answers,
    then averaging.

    Args:
        prediction: Generated answer (may contain multiple answers separated by comma)
        ground_truth: Expected answer (may contain multiple answers separated by comma)

    Returns:
        Average F1 across all ground truth sub-answers
    """
    predictions = [p.strip() for p in prediction.split(",")]
    ground_truths = [g.strip() for g in ground_truth.split(",")]

    # For each ground truth, find the best matching prediction
    scores = []
    for gt in ground_truths:
        best_score = max([f1_score(pred, gt) for pred in predictions])
        scores.append(best_score)

    return sum(scores) / len(scores) if scores else 0.0


def evaluate_category_5_adversarial(output: str) -> float:
    """
    Official LoCoMo adversarial evaluation (category 5).

    Category 5 questions test if the model correctly identifies
    when information is NOT in the conversation.

    Args:
        output: Model's generated answer

    Returns:
        1.0 if correctly identified as "no information", 0.0 otherwise
    """
    output_lower = output.lower()
    if "no information available" in output_lower or "not mentioned" in output_lower:
        return 1.0
    return 0.0


def evaluate_qa_official(prediction: str, ground_truth: str, category: int) -> Tuple[float, str]:
    """
    Official LoCoMo evaluation with category-specific handling.

    Args:
        prediction: Generated/predicted answer
        ground_truth: Expected answer
        category: Question category (1-5)

    Returns:
        (score, method) tuple where:
        - score: 0.0-1.0 evaluation score
        - method: String describing evaluation method used
    """
    # Category 5: Adversarial (check if model says "no information")
    if category == 5:
        score = evaluate_category_5_adversarial(prediction)
        return score, "adversarial_phrase_match"

    # Category 3 (multi-hop): Use first answer only (before semicolon)
    if category == 3:
        ground_truth = ground_truth.split(";")[0].strip()

    # Category 1 (single-hop with multiple sub-answers): Use multi-answer F1
    if category == 1:
        score = f1_multi_answer(prediction, ground_truth)
        return score, "f1_multi_answer"

    # Categories 2, 3, 4: Standard F1
    score = f1_score(prediction, ground_truth)
    return score, "f1_standard"


def extract_answer_from_context(memories: List[str], question: str, answer_hint: str = None) -> str:
    """
    Extract the most likely answer from retrieved memory context.

    This is a simple heuristic extraction. For full E2E QA evaluation,
    use an LLM to generate the answer (see generate_answer_llm).

    Args:
        memories: List of retrieved memory contents
        question: The question being asked
        answer_hint: Optional hint about expected answer format

    Returns:
        Extracted answer string
    """
    if not memories:
        return "no information available"

    # Simple extraction: concatenate all memories
    # This is a baseline - full E2E uses LLM generation
    context = " ".join(memories)

    # If we're looking for specific patterns, try to extract them
    if answer_hint:
        # Look for the hint in context
        hint_lower = answer_hint.lower()
        context_lower = context.lower()

        if hint_lower in context_lower:
            return answer_hint

    # Return truncated context as "answer" for retrieval-only eval
    # Note: This is NOT how E2E QA works - it's just for retrieval debugging
    return context[:500]


class OfficialLoCoMoEvaluator:
    """
    Evaluator using official LoCoMo metrics.

    Tracks both retrieval-based and E2E QA metrics.
    """

    def __init__(self):
        self.results_by_category = {i: [] for i in range(1, 6)}
        self.results_overall = []
        self.methods_used = []

    def evaluate(self, prediction: str, ground_truth: str, category: int) -> Tuple[float, str]:
        """
        Evaluate a single QA pair.

        Args:
            prediction: Generated answer
            ground_truth: Expected answer
            category: Question category (1-5)

        Returns:
            (score, method) tuple
        """
        score, method = evaluate_qa_official(prediction, ground_truth, category)

        self.results_by_category[category].append(score)
        self.results_overall.append(score)
        self.methods_used.append(method)

        return score, method

    def get_category_accuracy(self, category: int, threshold: float = 0.5) -> float:
        """
        Get accuracy for a category using threshold.

        Args:
            category: Category number (1-5)
            threshold: F1 threshold for "correct" (default 0.5)

        Returns:
            Accuracy as fraction
        """
        scores = self.results_by_category[category]
        if not scores:
            return 0.0
        correct = sum(1 for s in scores if s >= threshold)
        return correct / len(scores)

    def get_category_mean_f1(self, category: int) -> float:
        """
        Get mean F1 score for a category.

        Args:
            category: Category number (1-5)

        Returns:
            Mean F1 score
        """
        scores = self.results_by_category[category]
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def get_overall_accuracy(self, threshold: float = 0.5) -> float:
        """
        Get overall accuracy using threshold.

        Args:
            threshold: F1 threshold for "correct" (default 0.5)

        Returns:
            Overall accuracy as fraction
        """
        if not self.results_overall:
            return 0.0
        correct = sum(1 for s in self.results_overall if s >= threshold)
        return correct / len(self.results_overall)

    def get_overall_mean_f1(self) -> float:
        """
        Get overall mean F1 score.

        Returns:
            Mean F1 across all questions
        """
        if not self.results_overall:
            return 0.0
        return sum(self.results_overall) / len(self.results_overall)

    def get_summary(self, threshold: float = 0.5) -> dict:
        """
        Get comprehensive summary of evaluation results.

        Args:
            threshold: F1 threshold for accuracy calculation

        Returns:
            Dictionary with all metrics
        """
        category_names = {
            1: "Single-hop Recall",
            2: "Temporal Understanding",
            3: "Multi-hop Reasoning",
            4: "Open Domain",
            5: "Complex Reasoning (Adversarial)",
        }

        summary = {
            "overall": {
                "accuracy": self.get_overall_accuracy(threshold),
                "mean_f1": self.get_overall_mean_f1(),
                "total_questions": len(self.results_overall),
            },
            "categories": {},
            "threshold_used": threshold,
        }

        for cat in range(1, 6):
            scores = self.results_by_category[cat]
            if scores:
                summary["categories"][cat] = {
                    "name": category_names[cat],
                    "accuracy": self.get_category_accuracy(cat, threshold),
                    "mean_f1": self.get_category_mean_f1(cat),
                    "total": len(scores),
                    "correct": sum(1 for s in scores if s >= threshold),
                }

        return summary

    def print_summary(self, threshold: float = 0.5):
        """Print formatted summary to console."""
        summary = self.get_summary(threshold)

        print("\n" + "=" * 60)
        print("📊 OFFICIAL LOCOMO METRICS (F1 with Porter Stemmer)")
        print("=" * 60)

        print(f"\n🎯 Overall Accuracy (F1 >= {threshold}): {summary['overall']['accuracy']:.2%}")
        print(f"📈 Overall Mean F1: {summary['overall']['mean_f1']:.4f}")
        print(f"📝 Total Questions: {summary['overall']['total_questions']}")

        print("\n📈 Category Breakdown:")
        print("-" * 60)
        for cat, data in summary["categories"].items():
            print(
                f"  {data['name']:35s}: "
                f"Acc={data['accuracy']:6.2%} "
                f"F1={data['mean_f1']:.4f} "
                f"({data['correct']}/{data['total']})"
            )

        print("=" * 60)
