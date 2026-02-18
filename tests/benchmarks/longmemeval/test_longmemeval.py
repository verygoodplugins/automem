"""
LongMemEval Benchmark Evaluation for AutoMem

Tests AutoMem's long-term conversational memory against the LongMemEval
benchmark (ICLR 2025). LongMemEval evaluates memory systems across 6 question
types testing 5 core abilities:

- Single-session user: Recall user-stated info
- Single-session assistant: Recall assistant-stated info
- Single-session preference: Extract implicit preferences
- Multi-session: Synthesize across sessions
- Knowledge-update: Handle info that changed over time
- Temporal-reasoning: Time-based reasoning

Dataset: 500 questions over ~40 conversation sessions (~115k tokens)

Competitive landscape (LongMemEval_S, gpt-4o):
  Mastra OM (gpt-5-mini): 94.87%
  Mastra OM (gpt-4o):     84.23%
  Oracle gpt-4o:           82.4%
  Zep:                     71.2%
  Best Guess (no memory):  18.8%

References:
- Paper: https://arxiv.org/abs/2410.10813
- Dataset: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
- Code: https://github.com/xiaowu0162/LongMemEval
"""

import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# Add project root to path for imports (file -> longmemeval -> benchmarks -> tests -> project root)
_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from tests.benchmarks.longmemeval.configs import LongMemEvalConfig, get_config
from tests.benchmarks.longmemeval.evaluator import (
    LongMemEvalScorer,
    check_abstention_response,
    is_abstention_question,
    llm_evaluate,
    quick_score,
)


class LongMemEvalBenchmark:
    """Evaluates AutoMem against the LongMemEval benchmark."""

    def __init__(self, config: LongMemEvalConfig):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {config.api_token}",
            "Content-Type": "application/json",
        }
        self.scorer = LongMemEvalScorer()
        self.openai_client = None
        if OpenAI and os.getenv("OPENAI_API_KEY"):
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def health_check(self) -> bool:
        """Verify AutoMem API is accessible."""
        try:
            response = requests.get(
                f"{self.config.base_url}/health",
                timeout=self.config.request_timeout,
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load LongMemEval dataset from JSON file."""
        data_file = self.config.data_file
        if not os.path.exists(data_file):
            raise FileNotFoundError(
                f"Dataset not found: {data_file}\n"
                f"Download from: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned"
            )

        with open(data_file) as f:
            data = json.load(f)

        print(f"Loaded {len(data)} questions from {Path(data_file).name}")
        return data

    def cleanup_test_data(self, question_id: Optional[str] = None):
        """Remove test memories from AutoMem."""
        tag = f"{self.config.tag_prefix}:{question_id}" if question_id else self.config.tag_prefix
        match_mode = "exact" if question_id else "prefix"

        total_deleted = 0
        while True:
            try:
                response = requests.get(
                    f"{self.config.base_url}/recall",
                    headers=self.headers,
                    params={"tags": tag, "tag_match": match_mode, "limit": 100},
                    timeout=self.config.request_timeout,
                )

                if response.status_code != 200:
                    break

                results = response.json().get("results", [])
                if not results:
                    break

                for r in results:
                    memory_id = r.get("id")
                    if memory_id:
                        requests.delete(
                            f"{self.config.base_url}/memory/{memory_id}",
                            headers=self.headers,
                            timeout=self.config.request_timeout,
                        )
                        total_deleted += 1

                if len(results) < 100:
                    break

            except Exception as e:
                print(f"Cleanup error: {e}")
                break

        if total_deleted > 0:
            print(f"  Cleaned up {total_deleted} memories")

    def _parse_session_date(self, date_str: str) -> Optional[str]:
        """Parse LongMemEval date format to ISO format.

        Input format: '2023/05/20 (Sat) 02:21'
        Output: '2023-05-20T02:21:00'
        """
        try:
            # Strip day name in parens
            cleaned = re.sub(r"\s*\([^)]+\)\s*", " ", date_str).strip()
            dt = datetime.strptime(cleaned, "%Y/%m/%d %H:%M")
            return dt.isoformat()
        except (ValueError, AttributeError):
            return None

    def _format_session_as_memory(
        self, session_turns: List[Dict], session_id: str, session_date: str
    ) -> str:
        """Format a session's turns into a single memory content block."""
        lines = []
        for turn in session_turns:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            prefix = "User" if role == "user" else "Assistant"
            lines.append(f"{prefix}: {content}")
        return "\n".join(lines)

    def ingest_sessions(self, item: Dict[str, Any]) -> int:
        """
        Ingest all haystack sessions for one question into AutoMem.

        Returns the number of memories stored.
        """
        question_id = item["question_id"]
        sessions = item["haystack_sessions"]
        session_ids = item["haystack_session_ids"]
        session_dates = item["haystack_dates"]
        stored = 0

        for i, (session_turns, sid, date_str) in enumerate(
            zip(sessions, session_ids, session_dates)
        ):
            if not session_turns:
                continue

            timestamp = self._parse_session_date(date_str)

            if self.config.storage_strategy == "per-session":
                # One memory per session
                content = self._format_session_as_memory(session_turns, sid, date_str)
                tags = [
                    self.config.tag_prefix,
                    f"{self.config.tag_prefix}:{question_id}",
                    f"session:{sid}",
                ]
                metadata = {
                    "source": "longmemeval_benchmark",
                    "question_id": question_id,
                    "session_id": sid,
                    "session_date": date_str,
                    "turn_count": len(session_turns),
                }

                payload = {
                    "content": content,
                    "tags": tags,
                    "importance": self.config.importance,
                    "metadata": metadata,
                    "type": "Context",
                }
                if timestamp:
                    payload["timestamp"] = timestamp

                try:
                    response = requests.post(
                        f"{self.config.base_url}/memory",
                        headers=self.headers,
                        json=payload,
                        timeout=self.config.request_timeout,
                    )
                    if response.status_code in [200, 201]:
                        stored += 1
                    else:
                        print(f"  Failed to store session {sid}: " f"{response.status_code}")
                except Exception as e:
                    print(f"  Error storing session {sid}: {e}")

            elif self.config.storage_strategy == "per-turn":
                # One memory per turn
                for j, turn in enumerate(session_turns):
                    content = turn.get("content", "")
                    role = turn.get("role", "unknown")
                    if not content:
                        continue

                    prefix = "User" if role == "user" else "Assistant"
                    tags = [
                        self.config.tag_prefix,
                        f"{self.config.tag_prefix}:{question_id}",
                        f"session:{sid}",
                        f"role:{role}",
                    ]
                    metadata = {
                        "source": "longmemeval_benchmark",
                        "question_id": question_id,
                        "session_id": sid,
                        "session_date": date_str,
                        "turn_index": j,
                        "role": role,
                    }

                    payload = {
                        "content": f"{prefix}: {content}",
                        "tags": tags,
                        "importance": self.config.importance,
                        "metadata": metadata,
                        "type": "Context",
                    }
                    if timestamp:
                        payload["timestamp"] = timestamp

                    try:
                        response = requests.post(
                            f"{self.config.base_url}/memory",
                            headers=self.headers,
                            json=payload,
                            timeout=self.config.request_timeout,
                        )
                        if response.status_code in [200, 201]:
                            stored += 1
                    except Exception:
                        pass

            # Pause between batches
            if (i + 1) % self.config.batch_size == 0:
                time.sleep(self.config.pause_between_batches)

        return stored

    def _extract_temporal_bounds(
        self, question: str, question_date: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract temporal bounds from the question for time-bounded recall.

        Returns (start_iso, end_iso) or (None, None) if no temporal hints.
        """
        if not self.config.use_temporal_hints:
            return None, None

        q_lower = question.lower()

        # Parse question date as upper bound
        end_dt = self._parse_session_date(question_date)

        # Look for temporal keywords that suggest time bounds
        temporal_keywords = [
            "before",
            "after",
            "when",
            "last time",
            "first time",
            "recently",
            "earlier",
            "previously",
            "initially",
            "most recent",
            "latest",
            "originally",
            "changed",
            "updated",
            "used to",
        ]

        has_temporal = any(kw in q_lower for kw in temporal_keywords)

        if has_temporal and end_dt:
            return None, end_dt

        return None, None

    def recall_for_question(
        self, question: str, question_id: str, question_date: str
    ) -> List[Dict[str, Any]]:
        """
        Recall relevant memories for a question.

        Uses the question text as semantic query, with optional
        temporal hints and graph expansion.
        """
        params = {
            "query": question,
            "limit": self.config.recall_limit,
            "tags": f"{self.config.tag_prefix}:{question_id}",
            "tag_match": "exact",
        }

        if self.config.expand_entities:
            params["expand_entities"] = "true"
        if self.config.expand_relations:
            params["expand_relations"] = "true"
        if self.config.auto_decompose:
            params["auto_decompose"] = "true"

        # Temporal bounds
        start_iso, end_iso = self._extract_temporal_bounds(question, question_date)
        if start_iso:
            params["start"] = start_iso
        if end_iso:
            params["end"] = end_iso

        try:
            response = requests.get(
                f"{self.config.base_url}/recall",
                headers=self.headers,
                params=params,
                timeout=self.config.request_timeout,
            )
            if response.status_code == 200:
                results = response.json().get("results", [])
                # Flatten: AutoMem nests memory data under "memory" key
                flattened = []
                for r in results:
                    mem = r.get("memory", {})
                    mem["score"] = r.get("score", 0)
                    mem["match_type"] = r.get("match_type", "")
                    flattened.append(mem)
                return flattened
            else:
                print(f"  Recall failed for {question_id}: {response.status_code}")
                return []
        except Exception as e:
            print(f"  Recall error for {question_id}: {e}")
            return []

    def _format_memories_for_prompt(self, memories: List[Dict[str, Any]]) -> str:
        """Format recalled memories for the answer generation prompt."""
        if not memories:
            return "(No relevant memories found)"

        lines = []
        for i, mem in enumerate(memories, 1):
            content = mem.get("content", "")
            # Include timestamp if available
            metadata = mem.get("metadata", {})
            date = metadata.get("session_date", "")
            if date:
                lines.append(f"[Memory {i} - {date}]\n{content}")
            else:
                lines.append(f"[Memory {i}]\n{content}")
        return "\n\n".join(lines)

    def generate_answer(
        self,
        question: str,
        memories: List[Dict[str, Any]],
        question_date: str,
    ) -> str:
        """
        Generate an answer using LLM with recalled memories as context.

        Uses chain-of-note prompting as recommended by LongMemEval paper.
        """
        if not self.openai_client:
            # Without LLM, return the most relevant memory content as-is
            if memories:
                return memories[0].get("content", "I don't know.")
            return "I don't know."

        context = self._format_memories_for_prompt(memories)

        if self.config.use_chain_of_note:
            prompt = f"""You are answering a question based on recalled conversation memories.

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
        else:
            prompt = f"""Based on the following conversation history excerpts, answer the question.
If the answer cannot be determined from the provided context, say "I don't know."

Context:
{context}

Question: {question}
Answer:"""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500,
            )
            full_answer = response.choices[0].message.content.strip()

            # Extract final answer from chain-of-note format
            if self.config.use_chain_of_note and "Step 3" in full_answer:
                # Get everything after "Step 3 - Answer:"
                parts = full_answer.split("Step 3")
                if len(parts) > 1:
                    answer_part = parts[-1]
                    # Strip the label
                    answer_part = re.sub(
                        r"^\s*[-:]\s*(Answer\s*[-:]?\s*)?", "", answer_part
                    ).strip()
                    if answer_part:
                        return answer_part

            return full_answer

        except Exception as e:
            print(f"  LLM generation error: {e}")
            if memories:
                return memories[0].get("content", "I don't know.")
            return "I don't know."

    def evaluate_question(self, item: Dict[str, Any], memories_stored: int) -> Dict[str, Any]:
        """
        Evaluate a single question: recall, generate answer, score.

        Returns a result dict.
        """
        question_id = item["question_id"]
        question = item["question"]
        reference = item.get("answer", "")
        question_type = item["question_type"]
        question_date = item.get("question_date", "")

        # Recall
        memories = self.recall_for_question(question, question_id, question_date)

        # Generate answer
        hypothesis = self.generate_answer(question, memories, question_date)

        # Score
        if self.config.use_llm_eval and self.openai_client:
            eval_result = llm_evaluate(
                question=question,
                hypothesis=hypothesis,
                reference=reference,
                question_id=question_id,
                model=self.config.llm_model,
                client=self.openai_client,
            )
            is_correct = eval_result["is_correct"]
            confidence = eval_result["confidence"]
            explanation = eval_result["explanation"]
        else:
            qs = quick_score(hypothesis, reference, question_id)
            is_correct = qs["is_correct"]
            confidence = qs["f1"]
            explanation = (
                f"exact={qs['exact_match']}, sub={qs['substring_match']}, " f"f1={qs['f1']:.3f}"
            )

        result = {
            "question_id": question_id,
            "question_type": question_type,
            "question": question,
            "reference": reference,
            "hypothesis": hypothesis,
            "is_correct": is_correct,
            "confidence": confidence,
            "explanation": explanation,
            "recalled_count": len(memories),
            "memories_stored": memories_stored,
            "is_abstention": is_abstention_question(question_id),
            "question_date": question_date,
        }

        return result

    def run_benchmark(self, cleanup_after: bool = True) -> Dict[str, Any]:
        """
        Run the full LongMemEval benchmark.

        For each question:
        1. Ingest all haystack sessions
        2. Recall relevant memories
        3. Generate answer via LLM
        4. Score the answer
        5. Clean up memories for this question

        Returns comprehensive results dict.
        """
        # Health check
        if not self.health_check():
            print("AutoMem API is not accessible. Aborting benchmark.")
            return {"overall": {"accuracy": 0.0, "correct": 0, "total": 0}}

        # Load dataset
        dataset = self.load_dataset()

        # Apply question limit
        if self.config.max_questions > 0:
            dataset = dataset[: self.config.max_questions]

        print(f"\nRunning LongMemEval benchmark:")
        print(f"  Config: {self.config.name}")
        print(f"  Questions: {len(dataset)}")
        print(f"  Storage: {self.config.storage_strategy}")
        print(f"  Recall limit: {self.config.recall_limit}")
        print(f"  Expand entities: {self.config.expand_entities}")
        print(f"  Expand relations: {self.config.expand_relations}")
        print(f"  Temporal hints: {self.config.use_temporal_hints}")
        print(f"  LLM model: {self.config.llm_model}")
        print(f"  LLM eval: {self.config.use_llm_eval}")
        has_llm = self.openai_client is not None
        print(f"  OpenAI available: {has_llm}")
        print()

        start_time = time.time()
        all_results = []

        for idx, item in enumerate(dataset):
            question_id = item["question_id"]
            question_type = item["question_type"]
            num_sessions = len(item.get("haystack_sessions", []))

            print(f"[{idx + 1}/{len(dataset)}] {question_type}: " f"{item['question'][:60]}...")

            # 1. Ingest sessions
            stored = self.ingest_sessions(item)
            print(f"  Ingested {stored} memories from {num_sessions} sessions")

            # 2. Evaluate (recall + generate + score)
            result = self.evaluate_question(item, stored)
            all_results.append(result)
            self.scorer.add_result(result)

            status = "CORRECT" if result["is_correct"] else "WRONG"
            print(f"  {status} | recalled={result['recalled_count']} | " f"{result['explanation']}")
            if not result["is_correct"]:
                print(f"    Expected: {result['reference'][:80]}")
                print(f"    Got:      {result['hypothesis'][:80]}")

            # 3. Clean up this question's data
            if cleanup_after:
                self.cleanup_test_data(question_id)

        elapsed_time = time.time() - start_time

        # Print report
        scores = self.scorer.print_report(config_name=self.config.name, elapsed_time=elapsed_time)
        scores["elapsed_time"] = elapsed_time
        scores["config"] = {
            "name": self.config.name,
            "storage_strategy": self.config.storage_strategy,
            "recall_limit": self.config.recall_limit,
            "expand_entities": self.config.expand_entities,
            "expand_relations": self.config.expand_relations,
            "auto_decompose": self.config.auto_decompose,
            "use_temporal_hints": self.config.use_temporal_hints,
            "llm_model": self.config.llm_model,
            "use_llm_eval": self.config.use_llm_eval,
        }
        scores["details"] = all_results

        return scores

    def save_results(self, scores: Dict[str, Any], output_path: Optional[str] = None):
        """Save benchmark results to JSONL and JSON files."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                self.config.results_dir,
                f"{self.config.name}_{timestamp}",
            )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Save full results as JSON
        json_path = f"{output_path}.json"
        with open(json_path, "w") as f:
            json.dump(scores, f, indent=2)
        print(f"\nResults saved to: {json_path}")

        # Save hypotheses as JSONL (for LongMemEval's evaluator)
        jsonl_path = f"{output_path}.jsonl"
        with open(jsonl_path, "w") as f:
            for detail in scores.get("details", []):
                line = {
                    "question_id": detail["question_id"],
                    "hypothesis": detail["hypothesis"],
                }
                f.write(json.dumps(line) + "\n")
        print(f"Hypotheses saved to: {jsonl_path}")

        return json_path, jsonl_path


def main():
    """Run LongMemEval benchmark evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate AutoMem on LongMemEval benchmark")
    parser.add_argument(
        "--base-url",
        default=os.getenv("AUTOMEM_TEST_BASE_URL", "http://localhost:8001"),
        help="AutoMem API base URL",
    )
    parser.add_argument(
        "--api-token",
        default=os.getenv("AUTOMEM_TEST_API_TOKEN", "test-token"),
        help="AutoMem API token",
    )
    parser.add_argument(
        "--config",
        default="baseline",
        choices=[
            "baseline",
            "per-turn",
            "expand-entities",
            "expand-relations",
            "high-k",
            "temporal",
            "full-graph",
        ],
        help="Benchmark configuration preset (default: baseline)",
    )
    parser.add_argument(
        "--data-file",
        default=None,
        help="Path to longmemeval_s_cleaned.json",
    )
    parser.add_argument(
        "--recall-limit",
        type=int,
        default=None,
        help="Override recall limit from config",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="LLM model for answer generation (default: gpt-4o)",
    )
    parser.add_argument(
        "--llm-eval",
        action="store_true",
        help="Use GPT-4o for evaluation (costs money, more accurate)",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't cleanup test data after evaluation",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=0,
        help="Limit number of questions (0 = all, useful for debugging)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (without extension)",
    )

    args = parser.parse_args()

    # Build config from preset + overrides
    overrides = {
        "base_url": args.base_url,
        "api_token": args.api_token,
    }
    if args.data_file:
        overrides["data_file"] = args.data_file
    if args.recall_limit is not None:
        overrides["recall_limit"] = args.recall_limit
    if args.llm_model:
        overrides["llm_model"] = args.llm_model
    if args.llm_eval:
        overrides["use_llm_eval"] = True
    if args.max_questions > 0:
        overrides["max_questions"] = args.max_questions

    config = get_config(args.config, **overrides)

    # Run benchmark
    benchmark = LongMemEvalBenchmark(config)
    scores = benchmark.run_benchmark(cleanup_after=not args.no_cleanup)

    # Save results
    benchmark.save_results(scores, args.output)

    # Return exit code
    return 0 if scores.get("overall", {}).get("total", 0) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
