#!/usr/bin/env python3
"""
Quick Multi-hop Iteration Test for AutoMem

Runs ONLY multi-hop questions (Category 3) to enable fast iteration.
Full benchmark: ~20 minutes, 1986 questions
This script: ~3-5 minutes, 96 multi-hop questions

Usage:
    python tests/benchmarks/test_multihop_quick.py

    # Or with options:
    python tests/benchmarks/test_multihop_quick.py --conversations conv-26,conv-43
    python tests/benchmarks/test_multihop_quick.py --limit 2  # First 2 conversations only
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.benchmarks.test_locomo import LoCoMoConfig, LoCoMoEvaluator


def run_multihop_quick(conversations: list = None, limit: int = None, verbose: bool = False):
    """Run only multi-hop questions for fast iteration."""

    config = LoCoMoConfig()
    evaluator = LoCoMoEvaluator(config)

    # Health check
    if not evaluator.health_check():
        print("❌ AutoMem is not available. Start with: make dev")
        return None

    # Load dataset
    with open(config.data_file) as f:
        data = json.load(f)

    # Filter conversations if specified
    if conversations:
        data = [c for c in data if c["sample_id"] in conversations]
    if limit:
        data = data[:limit]

    # Filter to only conversations with multi-hop questions
    data = [c for c in data if any(q.get("category") == 3 for q in c["qa"])]

    print("=" * 60)
    print("🎯 Quick Multi-hop Iteration Test")
    print("=" * 60)
    print(f"Conversations: {len(data)}")
    total_multihop = sum(len([q for q in c["qa"] if q.get("category") == 3]) for c in data)
    print(f"Multi-hop questions: {total_multihop}")
    print()

    # Cleanup any previous test data
    evaluator.cleanup_test_data()

    results = {"correct": 0, "total": 0, "by_conversation": {}, "failures": []}

    start_time = time.time()

    for conv in data:
        sample_id = conv["sample_id"]

        # Load conversation into AutoMem
        print(f"\n📥 Loading {sample_id}...")
        memory_map = evaluator.load_conversation_into_automem(conv, sample_id)

        # Wait for enrichment
        time.sleep(1)

        # Get only multi-hop questions
        multihop_questions = [q for q in conv["qa"] if q.get("category") == 3]

        print(f"❓ Evaluating {len(multihop_questions)} multi-hop questions...")

        conv_correct = 0
        conv_total = len(multihop_questions)

        for i, qa in enumerate(multihop_questions):
            question = qa["question"]
            answer = qa["answer"]
            evidence = qa.get("evidence", [])

            # Use multi-hop recall with graph traversal
            recalled_memories = evaluator.multi_hop_recall_with_graph(
                question, sample_id, initial_limit=20, max_connected=50
            )

            # Check answer
            is_correct, confidence, explanation = evaluator.check_answer_in_memories(
                question,
                answer,
                recalled_memories,
                evidence_dialog_ids=evidence,
                sample_id=sample_id,
            )

            if is_correct:
                conv_correct += 1
                results["correct"] += 1
            else:
                results["failures"].append(
                    {
                        "conversation": sample_id,
                        "question": question,
                        "expected": answer,
                        "evidence_count": len(evidence),
                        "memories_recalled": len(recalled_memories),
                        "explanation": explanation,
                    }
                )

                if verbose:
                    print(f"  ❌ Q: {question[:50]}...")
                    print(f"     A: {answer}")
                    print(f"     {explanation}")

            results["total"] += 1

            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{conv_total}...")

        accuracy = conv_correct / conv_total * 100 if conv_total > 0 else 0
        results["by_conversation"][sample_id] = {
            "correct": conv_correct,
            "total": conv_total,
            "accuracy": accuracy,
        }
        print(f"📊 {sample_id}: {accuracy:.1f}% ({conv_correct}/{conv_total})")

    # Cleanup
    evaluator.cleanup_test_data()

    elapsed = time.time() - start_time
    overall_accuracy = results["correct"] / results["total"] * 100 if results["total"] > 0 else 0

    print("\n" + "=" * 60)
    print("📊 MULTI-HOP RESULTS")
    print("=" * 60)
    print(f"🎯 Accuracy: {overall_accuracy:.2f}% ({results['correct']}/{results['total']})")
    print(f"⏱️  Time: {elapsed:.1f}s")
    print()

    if results["failures"] and verbose:
        print("\n📝 Sample failures:")
        for f in results["failures"][:5]:
            print(f"  Q: {f['question'][:60]}...")
            print(f"  Expected: {f['expected']}")
            print(f"  {f['explanation']}")
            print()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick multi-hop iteration test")
    parser.add_argument("--conversations", type=str, help="Comma-separated conversation IDs")
    parser.add_argument("--limit", type=int, help="Limit to first N conversations")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show failure details")

    args = parser.parse_args()

    conversations = args.conversations.split(",") if args.conversations else None

    run_multihop_quick(conversations=conversations, limit=args.limit, verbose=args.verbose)
