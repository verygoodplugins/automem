#!/usr/bin/env python3
"""
Minimal test to verify experiment framework integration.
Runs one config against one conversation subset.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tests.benchmarks.experiments.experiment_config import ExperimentConfig
from tests.benchmarks.test_locomo import LoCoMoEvaluator, LoCoMoConfig


async def run_minimal_test():
    """Run a minimal test to verify integration"""
    
    print("=" * 60)
    print("🧪 MINIMAL INTEGRATION TEST")
    print("=" * 60)
    
    # Create a simple test config
    exp_config = ExperimentConfig(
        name="minimal_test",
        embedding_model="text-embedding-3-small",
        enrichment_model="gpt-4o-mini",
        recall_limit=10,
    )
    
    print(f"\n📋 Test Config:")
    print(f"   Name: {exp_config.name}")
    print(f"   Recall Limit: {exp_config.recall_limit}")
    
    # Create locomo config
    locomo_config = LoCoMoConfig(
        base_url=os.getenv("AUTOMEM_BASE_URL", "http://localhost:8001"),
        api_token=os.getenv("AUTOMEM_API_TOKEN", "test-token"),
        recall_limit=exp_config.recall_limit,
        eval_mode="e2e",
        use_official_f1=True,
        use_lenient_eval=True,
        e2e_model="gpt-4o-mini",  # Use cheaper model for test
        eval_judge_model="gpt-4o-mini",
    )
    
    print(f"\n🔧 LoCoMo Config:")
    print(f"   Base URL: {locomo_config.base_url}")
    print(f"   Eval Mode: {locomo_config.eval_mode}")
    print(f"   Lenient: {locomo_config.use_lenient_eval}")
    
    # Create evaluator
    evaluator = LoCoMoEvaluator(locomo_config)
    
    print(f"\n📂 Loading dataset...")
    evaluator.load_dataset()
    print(f"   Loaded {len(evaluator.conversations)} conversations")
    
    # Run on JUST the first 5 questions of the first conversation
    conv_id = list(evaluator.conversations.keys())[0]
    conv_data = evaluator.conversations[conv_id]
    
    print(f"\n🎯 Testing with conversation: {conv_id}")
    print(f"   Total questions in conv: {len(conv_data['qa_pairs'])}")
    print(f"   Testing first 5 questions only")
    
    # Manually run a mini evaluation
    results = await run_mini_eval(evaluator, conv_id, max_questions=5)
    
    print(f"\n📊 RESULTS:")
    print(f"   Questions evaluated: {results['total']}")
    print(f"   Correct: {results['correct']}")
    print(f"   Accuracy: {results['accuracy']:.2%}")
    
    print(f"\n✅ Integration test complete!")
    return results


async def run_mini_eval(evaluator, conv_id: str, max_questions: int = 5):
    """Run evaluation on a subset of questions"""
    conv_data = evaluator.conversations[conv_id]
    
    # Load memories
    print(f"\n📥 Loading memories...")
    await evaluator.load_conversation_memories(conv_id, conv_data)
    
    # Wait for enrichment
    print(f"⏳ Waiting for enrichment...")
    await asyncio.sleep(3)
    
    # Evaluate questions
    qa_pairs = conv_data["qa_pairs"][:max_questions]
    correct = 0
    total = 0
    
    for i, qa in enumerate(qa_pairs):
        question = qa["question"]
        category = qa.get("category", 1)
        
        if category == 5:
            expected = qa.get("adversarial_answer", "No information")
        else:
            expected = qa.get("answer", "")
        
        print(f"\n   Q{i+1}: {question[:50]}...")
        
        # Recall memories
        recalled = await evaluator.recall_memories(question)
        print(f"   Retrieved: {len(recalled)} memories")
        
        # Generate answer in E2E mode
        if evaluator.config.eval_mode == "e2e":
            generated = evaluator.generate_answer_e2e(question, recalled, category)
            print(f"   Generated: {generated[:50]}...")
            
            # Evaluate
            is_correct, confidence, explanation = evaluator._evaluate_lenient_semantic(
                question, expected, generated, category
            )
        else:
            is_correct, confidence, explanation = evaluator.check_answer_in_memories(
                question, expected, recalled, category
            )
        
        if is_correct:
            correct += 1
            print(f"   ✅ Correct (conf: {confidence:.2f})")
        else:
            print(f"   ❌ Incorrect (conf: {confidence:.2f})")
        
        total += 1
    
    # Cleanup
    print(f"\n🧹 Cleaning up test memories...")
    await evaluator.cleanup_test_memories()
    
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total > 0 else 0,
    }


if __name__ == "__main__":
    asyncio.run(run_minimal_test())
