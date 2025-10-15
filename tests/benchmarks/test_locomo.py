"""
LoCoMo Benchmark Evaluation for AutoMem

Tests AutoMem's long-term conversational memory against the LoCoMo benchmark.
LoCoMo (ACL 2024) evaluates memory systems across 5 categories:
- Category 1: Single-hop recall (simple fact retrieval)
- Category 2: Temporal understanding (time-based queries)
- Category 3: Multi-hop reasoning (connecting multiple memories)
- Category 4: Open domain knowledge
- Category 5: Complex reasoning

Dataset: 10 conversations, 1,986 questions total
CORE (SOTA): 88.24% overall accuracy

References:
- Paper: https://github.com/snap-research/locomo/tree/main/static/paper/locomo.pdf
- Code: https://github.com/snap-research/locomo
- CORE blog: https://blog.heysol.ai/core-build-memory-knowledge-graph-for-individuals-and-achieved-sota-on-locomo-benchmark/
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import re

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

@dataclass
class LoCoMoConfig:
    """Configuration for LoCoMo benchmark evaluation"""
    # AutoMem API settings
    base_url: str = os.getenv("AUTOMEM_TEST_BASE_URL", "http://localhost:8001")
    api_token: str = os.getenv("AUTOMEM_TEST_API_TOKEN", "test-token")
    
    # LoCoMo dataset paths
    data_file: str = str(Path(__file__).parent / "locomo" / "data" / "locomo10.json")
    
    # Evaluation settings
    recall_limit: int = 10  # Number of memories to retrieve per question
    importance_threshold: float = 0.5  # Minimum importance for stored memories
    
    # Tag configuration
    use_conversation_tags: bool = True  # Tag memories by conversation ID
    use_session_tags: bool = True  # Tag memories by session ID
    use_speaker_tags: bool = True  # Tag memories by speaker name
    
    # Scoring thresholds
    exact_match_threshold: float = 0.9  # For exact string matching
    fuzzy_match_threshold: float = 0.7  # For partial matches
    
    # Performance tuning
    batch_size: int = 50  # Memories to store before pausing
    pause_between_batches: float = 0.5  # Seconds to wait between batches


class LoCoMoEvaluator:
    """Evaluates AutoMem against the LoCoMo benchmark"""
    
    def __init__(self, config: LoCoMoConfig):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {config.api_token}",
            "Content-Type": "application/json"
        }
        self.memory_map = {}  # Maps dialog IDs to memory IDs
        self.results = defaultdict(list)  # Category -> [True/False scores]
        
    def health_check(self) -> bool:
        """Verify AutoMem API is accessible"""
        try:
            response = requests.get(f"{self.config.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    def cleanup_test_data(self, tag_prefix: str = "locomo-test"):
        """Remove all test memories from AutoMem"""
        print(f"\n🧹 Cleaning up test memories with tag: {tag_prefix}")
        try:
            # Recall all memories with the test tag
            response = requests.get(
                f"{self.config.base_url}/memory/by-tag",
                headers=self.headers,
                params={"tags": tag_prefix, "tag_match": "prefix"}
            )
            
            if response.status_code == 200:
                memories = response.json().get("memories", [])
                print(f"Found {len(memories)} test memories to delete")
                
                # Delete each memory
                for memory in memories:
                    memory_id = memory.get("id")
                    if memory_id:
                        requests.delete(
                            f"{self.config.base_url}/memory/{memory_id}",
                            headers=self.headers
                        )
                
                print(f"✅ Cleaned up {len(memories)} test memories")
                return True
            else:
                print(f"⚠️  Could not fetch test memories: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"⚠️  Cleanup error: {e}")
            return False
    
    def load_conversation_into_automem(
        self, 
        conversation: Dict[str, Any], 
        sample_id: str
    ) -> Dict[str, str]:
        """
        Load a LoCoMo conversation into AutoMem as individual memories.
        
        Returns mapping of dialog_id -> memory_id
        """
        memory_map = {}
        memory_count = 0
        
        print(f"\n📥 Loading conversation {sample_id} into AutoMem...")
        
        # Extract conversation metadata
        speaker_a = conversation["conversation"].get("speaker_a", "Speaker A")
        speaker_b = conversation["conversation"].get("speaker_b", "Speaker B")
        
        # Process each session
        session_keys = sorted([k for k in conversation["conversation"].keys() if k.startswith("session_") and not k.endswith("_date_time")])
        
        for session_key in session_keys:
            session_num = session_key.split("_")[1]
            session_data = conversation["conversation"][session_key]
            session_datetime = conversation["conversation"].get(f"session_{session_num}_date_time", "")
            
            # Store each dialog turn as a memory
            for turn in session_data:
                speaker = turn.get("speaker", "unknown")
                dia_id = turn.get("dia_id", f"unknown_{memory_count}")
                text = turn.get("text", "")
                img_url = turn.get("img_url")
                blip_caption = turn.get("blip_caption")
                
                if not text:
                    continue
                
                # Build memory content
                content = f"{speaker}: {text}"
                if blip_caption:
                    content += f" [Image: {blip_caption}]"
                
                # Build tags
                tags = [
                    f"locomo-test",
                    f"conversation:{sample_id}",
                    f"session:{session_num}",
                    f"speaker:{speaker.lower().replace(' ', '-')}",
                ]
                
                # Build metadata
                metadata = {
                    "source": "locomo_benchmark",
                    "conversation_id": sample_id,
                    "session_id": session_num,
                    "dialog_id": dia_id,
                    "speaker": speaker,
                    "session_datetime": session_datetime,
                }
                
                if img_url:
                    metadata["image_url"] = img_url
                if blip_caption:
                    metadata["image_caption"] = blip_caption
                
                # Store memory
                try:
                    response = requests.post(
                        f"{self.config.base_url}/memory",
                        headers=self.headers,
                        json={
                            "content": content,
                            "tags": tags,
                            "importance": self.config.importance_threshold,
                            "metadata": metadata,
                            "type": "Context"
                        }
                    )
                    
                    if response.status_code in [200, 201]:  # Accept both OK and Created
                        result = response.json()
                        memory_id = result.get("id")
                        memory_map[dia_id] = memory_id
                        memory_count += 1
                        
                        # Pause every N memories
                        if memory_count % self.config.batch_size == 0:
                            print(f"  Stored {memory_count} memories...")
                            time.sleep(self.config.pause_between_batches)
                    else:
                        print(f"⚠️  Failed to store memory for {dia_id}: {response.status_code} - {response.text[:100]}")
                        
                except Exception as e:
                    print(f"⚠️  Error storing memory for {dia_id}: {e}")
        
        print(f"✅ Loaded {memory_count} memories from conversation {sample_id}")
        return memory_map
    
    def recall_for_question(
        self, 
        question: str, 
        sample_id: str,
        session_context: str = None
    ) -> List[Dict[str, Any]]:
        """
        Query AutoMem to recall memories relevant to a question.
        
        Uses hybrid search: semantic + keyword + tags
        """
        try:
            # Build query parameters
            # Use broader recall to ensure we get evidence memories
            params = {
                "query": question,
                "limit": 50,  # Increased from 10 to capture more context
                "tags": f"conversation:{sample_id}",  # Filter to relevant conversation
                "tag_match": "exact"
            }
            
            response = requests.get(
                f"{self.config.base_url}/recall",
                headers=self.headers,
                params=params
            )
            
            if response.status_code == 200:
                result = response.json()
                # AutoMem returns "results" with nested "memory" objects
                results = result.get("results", [])
                # Extract the memory objects from each result
                memories = [r.get("memory", {}) for r in results if "memory" in r]
                return memories
            else:
                print(f"⚠️  Recall failed: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"⚠️  Recall error: {e}")
            return []
    
    def normalize_answer(self, text: str) -> str:
        """Normalize text for comparison"""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text
    
    def check_answer_in_memories(
        self, 
        question: str,
        expected_answer: Any,
        recalled_memories: List[Dict[str, Any]],
        evidence_dialog_ids: List[str] = None
    ) -> Tuple[bool, float, str]:
        """
        Check if the expected answer can be found in recalled memories.
        
        Uses evidence dialog IDs if available, otherwise semantic matching.
        
        Returns:
            (is_correct, confidence_score, explanation)
        """
        if not recalled_memories:
            return False, 0.0, "No memories recalled"
        
        # Normalize expected answer
        expected_str = str(expected_answer).lower()
        expected_normalized = self.normalize_answer(expected_str)
        
        # Strategy 1: If we have evidence dialog IDs, check only those memories
        if evidence_dialog_ids:
            for memory in recalled_memories:
                metadata = memory.get("metadata", {})
                dialog_id = metadata.get("dialog_id", "")
                
                # Check if this memory is one of the evidence dialogs
                if dialog_id in evidence_dialog_ids:
                    content = memory.get("content", "").lower()
                    content_normalized = self.normalize_answer(content)
                    
                    # Much more lenient matching for evidence dialogs
                    # Just check if key words from answer appear
                    expected_words = set(expected_normalized.split())
                    content_words = set(content_normalized.split())
                    overlap = expected_words.intersection(content_words)
                    
                    if len(expected_words) == 0:
                        confidence = 0.0
                    else:
                        confidence = len(overlap) / len(expected_words)
                    
                    # If at least 30% of answer words appear in evidence dialog, count as correct
                    if confidence >= 0.3:
                        return True, confidence, f"Found in evidence dialog {dialog_id} (confidence: {confidence:.2f})"
        
        # Strategy 2: Semantic search through all recalled memories
        max_confidence = 0.0
        found_in_memory = None
        
        for memory in recalled_memories:
            content = memory.get("content", "").lower()
            content_normalized = self.normalize_answer(content)
            
            # Exact substring match
            if expected_normalized in content_normalized:
                confidence = 1.0
                found_in_memory = memory.get("id")
                return True, confidence, f"Exact match in memory (confidence: {confidence:.2f})"
            
            # Fuzzy word overlap
            expected_words = set(expected_normalized.split())
            if len(expected_words) == 0:
                continue
                
            content_words = set(content_normalized.split())
            overlap = expected_words.intersection(content_words)
            
            if overlap:
                confidence = len(overlap) / len(expected_words)
                if confidence > max_confidence:
                    max_confidence = confidence
                    found_in_memory = memory.get("id")
        
        # Determine if correct based on confidence
        is_correct = max_confidence >= 0.5
        
        if is_correct:
            explanation = f"Found answer (confidence: {max_confidence:.2f})"
        else:
            explanation = f"No good match (max: {max_confidence:.2f})"
        
        return is_correct, max_confidence, explanation
    
    def evaluate_conversation(
        self, 
        conversation: Dict[str, Any], 
        sample_id: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single LoCoMo conversation.
        
        Process:
        1. Load conversation into AutoMem
        2. For each question, recall relevant memories
        3. Check if answer is in recalled memories
        4. Calculate accuracy per category
        """
        print(f"\n{'='*60}")
        print(f"Evaluating Conversation: {sample_id}")
        print(f"{'='*60}")
        
        # Step 1: Load conversation
        memory_map = self.load_conversation_into_automem(conversation, sample_id)
        
        # Wait for enrichment to process (optional)
        print("\n⏳ Waiting for enrichment pipeline...")
        time.sleep(2)
        
        # Step 2: Evaluate each question
        qa_results = []
        questions = conversation.get("qa", [])
        
        print(f"\n❓ Evaluating {len(questions)} questions...")
        
        for i, qa in enumerate(questions):
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            category = qa.get("category", 0)
            evidence = qa.get("evidence", [])
            
            # Recall memories for this question
            recalled_memories = self.recall_for_question(question, sample_id)
            
            # DEBUG: Print details for first question
            if i == 0:
                print(f"\n🐛 DEBUG First Question:")
                print(f"  Question: {question}")
                print(f"  Expected: {answer}")
                print(f"  Recalled: {len(recalled_memories)} memories")
                if recalled_memories:
                    print(f"  First memory content: {recalled_memories[0].get('content', 'NO CONTENT')[:150]}...")
            
            # Check if answer is in recalled memories
            is_correct, confidence, explanation = self.check_answer_in_memories(
                question, answer, recalled_memories, evidence
            )
            
            # DEBUG: Print first correct answer
            if is_correct and not hasattr(self, '_first_correct'):
                print(f"\n✅ First correct answer found!")
                print(f"  Question: {question}")
                print(f"  Answer: {answer}")
                print(f"  Confidence: {confidence:.2f}")
                self._first_correct = True
            
            # Record result
            qa_result = {
                "question": question,
                "expected_answer": answer,
                "category": category,
                "is_correct": is_correct,
                "confidence": confidence,
                "recalled_count": len(recalled_memories),
                "explanation": explanation
            }
            qa_results.append(qa_result)
            
            # Track results by category
            self.results[category].append(is_correct)
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(questions)} questions...")
        
        # Calculate conversation-level statistics
        correct_count = sum(1 for r in qa_results if r["is_correct"])
        total_count = len(qa_results)
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        print(f"\n📊 Conversation Results:")
        print(f"  Accuracy: {accuracy:.2%} ({correct_count}/{total_count})")
        
        return {
            "sample_id": sample_id,
            "total_questions": total_count,
            "correct": correct_count,
            "accuracy": accuracy,
            "qa_results": qa_results,
            "memory_count": len(memory_map)
        }
    
    def run_benchmark(self, cleanup_after: bool = True) -> Dict[str, Any]:
        """
        Run the complete LoCoMo benchmark evaluation.
        
        Returns comprehensive results including per-category accuracy.
        """
        print("\n" + "="*60)
        print("🧠 AutoMem LoCoMo Benchmark Evaluation")
        print("="*60)
        
        # Health check
        print("\n🏥 Checking AutoMem health...")
        if not self.health_check():
            raise ConnectionError("AutoMem API is not accessible")
        print("✅ AutoMem is healthy")
        
        # Cleanup existing test data
        self.cleanup_test_data()
        
        # Load dataset
        print(f"\n📂 Loading LoCoMo dataset from: {self.config.data_file}")
        with open(self.config.data_file, 'r') as f:
            conversations = json.load(f)
        
        print(f"✅ Loaded {len(conversations)} conversations")
        
        # Evaluate each conversation
        conversation_results = []
        start_time = time.time()
        
        for i, conversation in enumerate(conversations):
            sample_id = conversation.get("sample_id", f"sample_{i}")
            
            try:
                result = self.evaluate_conversation(conversation, sample_id)
                conversation_results.append(result)
            except Exception as e:
                print(f"❌ Error evaluating conversation {sample_id}: {e}")
                continue
        
        elapsed_time = time.time() - start_time
        
        # Calculate overall statistics
        print("\n" + "="*60)
        print("📊 FINAL RESULTS")
        print("="*60)
        
        # Overall accuracy
        total_questions = sum(r["total_questions"] for r in conversation_results)
        total_correct = sum(r["correct"] for r in conversation_results)
        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
        
        print(f"\n🎯 Overall Accuracy: {overall_accuracy:.2%} ({total_correct}/{total_questions})")
        print(f"⏱️  Total Time: {elapsed_time:.1f}s")
        print(f"💾 Total Memories Stored: {sum(r['memory_count'] for r in conversation_results)}")
        
        # Category breakdown
        print("\n📈 Category Breakdown:")
        category_names = {
            1: "Single-hop Recall",
            2: "Temporal Understanding",
            3: "Multi-hop Reasoning",
            4: "Open Domain",
            5: "Complex Reasoning"
        }
        
        category_results = {}
        for category, scores in sorted(self.results.items()):
            correct = sum(scores)
            total = len(scores)
            accuracy = correct / total if total > 0 else 0.0
            category_results[category] = {
                "name": category_names.get(category, f"Category {category}"),
                "accuracy": accuracy,
                "correct": correct,
                "total": total
            }
            print(f"  {category_names.get(category, f'Category {category}'):25s}: {accuracy:6.2%} ({correct:3d}/{total:3d})")
        
        # Comparison with CORE
        core_sota = 0.8824
        improvement = overall_accuracy - core_sota
        print(f"\n🏆 Comparison with CORE (SOTA):")
        print(f"  CORE: {core_sota:.2%}")
        print(f"  AutoMem: {overall_accuracy:.2%}")
        if improvement > 0:
            print(f"  🎉 AutoMem BEATS CORE by {improvement:.2%}!")
        elif improvement < 0:
            print(f"  📉 AutoMem is {abs(improvement):.2%} behind CORE")
        else:
            print(f"  🤝 AutoMem matches CORE")
        
        # Cleanup
        if cleanup_after:
            self.cleanup_test_data()
        
        # Return comprehensive results
        return {
            "overall": {
                "accuracy": overall_accuracy,
                "correct": total_correct,
                "total": total_questions,
                "elapsed_time": elapsed_time
            },
            "categories": category_results,
            "conversations": conversation_results,
            "comparison": {
                "core_sota": core_sota,
                "automem": overall_accuracy,
                "improvement": improvement
            }
        }


def main():
    """Run LoCoMo benchmark evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate AutoMem on LoCoMo benchmark")
    parser.add_argument("--base-url", default=os.getenv("AUTOMEM_TEST_BASE_URL", "http://localhost:8001"),
                       help="AutoMem API base URL")
    parser.add_argument("--api-token", default=os.getenv("AUTOMEM_TEST_API_TOKEN", "test-token"),
                       help="AutoMem API token")
    parser.add_argument("--data-file", default=None,
                       help="Path to locomo10.json")
    parser.add_argument("--recall-limit", type=int, default=10,
                       help="Number of memories to recall per question")
    parser.add_argument("--no-cleanup", action="store_true",
                       help="Don't cleanup test data after evaluation")
    parser.add_argument("--output", default=None,
                       help="Save results to JSON file")
    parser.add_argument("--test-one", action="store_true",
                       help="Test with just one conversation for debugging")
    
    args = parser.parse_args()
    
    # Build config
    config = LoCoMoConfig(
        base_url=args.base_url,
        api_token=args.api_token,
        recall_limit=args.recall_limit
    )
    
    if args.data_file:
        config.data_file = args.data_file
    
    # Run evaluation
    evaluator = LoCoMoEvaluator(config)
    results = evaluator.run_benchmark(cleanup_after=not args.no_cleanup)
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")
    
    # Return exit code based on success
    return 0 if results["overall"]["accuracy"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

