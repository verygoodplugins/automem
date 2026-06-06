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
import logging
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from dotenv import load_dotenv

# Add project root to path for imports (file -> longmemeval -> benchmarks -> tests -> project root)
_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

load_dotenv()
load_dotenv(Path.home() / ".config" / "automem" / ".env")

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from tests.benchmarks.backends import MemoryRecord, SearchRequest, create_backend
from tests.benchmarks.judge_policy import (
    CANONICAL_BENCHMARK_JUDGE_MODEL,
    DEFAULT_JUDGE_PROVIDER,
    is_gpt5_family,
    judge_metadata,
)
from tests.benchmarks.longmemeval.configs import LongMemEvalConfig, get_config
from tests.benchmarks.longmemeval.evaluator import (
    LongMemEvalScorer,
    check_abstention_response,
    is_abstention_question,
    llm_evaluate,
    quick_score,
)

logger = logging.getLogger(__name__)


class LongMemEvalResult(TypedDict, total=False):
    question_id: str
    question_type: str
    question: str
    reference: str
    hypothesis: str
    is_correct: bool
    confidence: float
    explanation: str
    recalled_count: int
    memories_stored: int
    is_abstention: bool
    question_date: str
    answer_session_ids: List[str]
    retrieved_session_ids: List[str]
    recall_hit_at_5: bool
    judge_attempts: int
    judge_error: Optional[str]
    error: Optional[str]
    error_type: Optional[str]


class LongMemEvalBenchmark:
    """Evaluates AutoMem against the LongMemEval benchmark."""

    JUDGE_MAX_ATTEMPTS = 3
    JUDGE_RETRY_BASE_SECONDS = 0.5
    GPT5_ANSWER_MAX_COMPLETION_TOKENS = (2000, 4000)
    LEGACY_ANSWER_MAX_TOKENS = 500

    def __init__(self, config: LongMemEvalConfig) -> None:
        self.config = config
        scope_prefix = f"longmemeval-{config.backend}-{int(time.time())}"
        self.backend = create_backend(
            config.backend,
            base_url=config.base_url,
            api_token=config.api_token,
            scope_prefix=scope_prefix,
            work_dir=config.work_dir,
        )
        self.scorer = LongMemEvalScorer()
        self.memory_ingest_failures = 0
        self.openai_client = None
        if OpenAI and os.getenv("OPENAI_API_KEY"):
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._last_output_base: Optional[str] = None

    def effective_judge_model(self) -> Optional[str]:
        if not self.config.use_llm_eval:
            return None
        return self.config.eval_llm_model or CANONICAL_BENCHMARK_JUDGE_MODEL

    def health_check(self) -> bool:
        """Verify the configured backend is accessible."""
        return self.backend.health_check()

    @staticmethod
    def _record_to_memory(record: MemoryRecord) -> Dict[str, Any]:
        return {
            "id": record.id,
            "content": record.content,
            "metadata": dict(record.metadata),
            "tags": list(record.tags),
            "score": record.score,
            "match_type": record.match_type,
        }

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load LongMemEval dataset from JSON file."""
        data_file = self.config.data_file
        if not os.path.exists(data_file):
            raise FileNotFoundError(
                f"Dataset not found: {data_file}\n"
                f"Download from: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned"
            )

        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"Loaded {len(data)} questions from {Path(data_file).name}")
        return data

    @staticmethod
    def question_type_distribution(dataset: List[Dict[str, Any]]) -> Dict[str, int]:
        """Return a stable question-type distribution for result metadata."""
        return dict(
            sorted(Counter(item.get("question_type", "unknown") for item in dataset).items())
        )

    def select_dataset(
        self, dataset: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Apply configured question selection and return metadata about the slice."""
        full_distribution = self.question_type_distribution(dataset)

        if self.config.per_type > 0:
            selected: List[Dict[str, Any]] = []
            selected_counts: Counter[str] = Counter()
            for item in dataset:
                question_type = item.get("question_type", "unknown")
                if selected_counts[question_type] >= self.config.per_type:
                    continue
                selected.append(item)
                selected_counts[question_type] += 1
            strategy = "stratified_per_type"
            strategy_params: Dict[str, Any] = {"per_type": self.config.per_type}
        elif self.config.max_questions > 0:
            selected = dataset[: self.config.max_questions]
            strategy = "prefix"
            strategy_params = {
                "max_questions": self.config.max_questions,
                "debug_prefix": True,
            }
        else:
            selected = list(dataset)
            strategy = "all"
            strategy_params = {}

        selected_distribution = self.question_type_distribution(selected)
        return selected, {
            "strategy": strategy,
            **strategy_params,
            "full_total": len(dataset),
            "selected_total": len(selected),
            "full_type_distribution": full_distribution,
            "selected_type_distribution": selected_distribution,
        }

    def cleanup_test_data(self, question_id: Optional[str] = None) -> None:
        """Remove test memories from AutoMem."""
        if self.config.backend == "automem" and question_id:
            scope_id = question_id
        elif question_id:
            scope_id = question_id
        else:
            return

        try:
            total_deleted = self.backend.cleanup_scope(scope_id)
        except Exception as e:
            logger.exception("Cleanup error: %s", e)
            return

        if total_deleted > 0:
            print(f"  Cleaned up {total_deleted} memories")

    def resolve_output_base(self, output_path: Optional[str] = None) -> str:
        """Resolve the output base used for final, partial, and status artifacts."""
        if output_path is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                self.config.results_dir,
                f"{self.config.name}_{timestamp}",
            )

        for suffix in (".partial.jsonl", ".status.json", ".jsonl", ".json"):
            if output_path.endswith(suffix):
                output_path = output_path[: -len(suffix)]
                break
        self._last_output_base = output_path
        return output_path

    @staticmethod
    def artifact_paths(output_base: str) -> Dict[str, str]:
        return {
            "json": f"{output_base}.json",
            "jsonl": f"{output_base}.jsonl",
            "partial": f"{output_base}.partial.jsonl",
            "status": f"{output_base}.status.json",
        }

    def _append_partial_result(self, output_base: str, result: Dict[str, Any]) -> None:
        paths = self.artifact_paths(output_base)
        os.makedirs(os.path.dirname(paths["partial"]) or ".", exist_ok=True)
        with open(paths["partial"], "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

    def _write_status(
        self,
        output_base: str,
        *,
        started_at: str,
        selection: Dict[str, Any],
        completed: int,
        total: int,
        skipped: int,
        elapsed_time: float,
        status: str,
        judge_errors: int,
    ) -> None:
        paths = self.artifact_paths(output_base)
        os.makedirs(os.path.dirname(paths["status"]) or ".", exist_ok=True)
        payload = {
            "status": status,
            "started_at": started_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_time": elapsed_time,
            "output_base": output_base,
            "artifacts": paths,
            "completed": completed,
            "total": total,
            "skipped": skipped,
            "remaining": max(total - completed, 0),
            "memory_ingest_failures": self.memory_ingest_failures,
            "judge_errors": judge_errors,
            "selection": selection,
            "config": {
                "backend": self.config.backend,
                "name": self.config.name,
                "answerer_model": self.config.llm_model,
                "llm_model": self.config.llm_model,
                "use_llm_eval": self.config.use_llm_eval,
                "judge_model": self.effective_judge_model(),
            },
        }
        with open(paths["status"], "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _load_partial_results(self, output_base: str) -> List[Dict[str, Any]]:
        paths = self.artifact_paths(output_base)
        partial_path = paths["partial"]
        if not os.path.exists(partial_path):
            return []

        with open(partial_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        results: List[Dict[str, Any]] = []
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    print(f"Warning: ignoring malformed trailing line in {partial_path}")
                    break
                raise
            if result.get("question_id"):
                results.append(result)
        return results

    def _parse_session_date(self, date_str: str) -> Optional[str]:
        """Parse LongMemEval date format to ISO format.

        Input format: '2023/05/20 (Sat) 02:21'
        Output: '2023-05-20T02:21:00'
        """
        try:
            # Strip day name in parens
            cleaned = re.sub(r"\s*\([^)]+\)\s*", " ", date_str).strip()
            dt = datetime.strptime(cleaned, "%Y/%m/%d %H:%M")
            dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat().replace("+00:00", "Z")
        except (ValueError, AttributeError):
            return None

    def _format_session_as_memory(self, session_turns: List[Dict[str, Any]]) -> str:
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
        payloads: List[Dict[str, Any]] = []

        for session_turns, sid, date_str in zip(sessions, session_ids, session_dates, strict=True):
            if not session_turns:
                continue

            timestamp = self._parse_session_date(date_str)

            if self.config.storage_strategy == "per-session":
                # One memory per session
                content = self._format_session_as_memory(session_turns)
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
                    "_benchmark_id": sid,
                }
                if timestamp:
                    payload["timestamp"] = timestamp
                payloads.append(payload)

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
                        "_benchmark_id": f"{sid}:{j}",
                    }
                    if timestamp:
                        payload["timestamp"] = timestamp
                    payloads.append(payload)

        try:
            memory_map = self.backend.ingest_memories(
                payloads,
                scope_id=question_id,
                batch_size=self.config.batch_size,
                pause_between_batches=self.config.pause_between_batches,
            )
        except Exception:
            self.memory_ingest_failures += len(payloads)
            logger.exception("Failed to ingest memories for question %s", question_id)
            return 0

        return len(memory_map)

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
        # Temporal bounds
        start_iso, end_iso = self._extract_temporal_bounds(question, question_date)

        try:
            records = self.backend.search(
                SearchRequest(
                    query=question,
                    scope_id=question_id,
                    limit=self.config.recall_limit,
                    tags=[f"{self.config.tag_prefix}:{question_id}"],
                    tag_match="exact",
                    start=start_iso,
                    end=end_iso,
                    expand_entities=self.config.expand_entities,
                    expand_relations=self.config.expand_relations,
                    auto_decompose=self.config.auto_decompose,
                )
            )
            return [self._record_to_memory(record) for record in records]
        except Exception as e:
            logger.exception("Recall error for %s: %s", question_id, e)
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

    @staticmethod
    def _top_unique_session_ids(memories: List[Dict[str, Any]], limit: int = 5) -> List[str]:
        session_ids: List[str] = []
        seen = set()
        for memory in memories:
            metadata = memory.get("metadata") or {}
            session_id = metadata.get("session_id")
            if not session_id or session_id in seen:
                continue
            seen.add(session_id)
            session_ids.append(session_id)
            if len(session_ids) >= limit:
                break
        return session_ids

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
            if is_gpt5_family(self.config.llm_model):
                token_budgets = self.GPT5_ANSWER_MAX_COMPLETION_TOKENS
            else:
                token_budgets = (self.LEGACY_ANSWER_MAX_TOKENS,)

            full_answer = ""
            for token_budget in token_budgets:
                request_kwargs: Dict[str, Any] = {
                    "model": self.config.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if is_gpt5_family(self.config.llm_model):
                    request_kwargs["max_completion_tokens"] = token_budget
                else:
                    request_kwargs["temperature"] = 0
                    request_kwargs["max_tokens"] = token_budget

                response = self.openai_client.chat.completions.create(**request_kwargs)
                full_answer = response.choices[0].message.content.strip()
                if full_answer:
                    break
                logger.warning(
                    "LLM answer generation returned empty content for %s at token budget %s",
                    self.config.llm_model,
                    token_budget,
                )
            if not full_answer:
                raise ValueError("LLM answer generation returned empty content")

            # Extract final answer from chain-of-note format
            if self.config.use_chain_of_note:
                step3_match = re.search(
                    r"Step\s*3(?:\s*[\.\-:]|\s)*\s*(?:Answer|Final Answer)?\s*[:\-]?\s*",
                    full_answer,
                    flags=re.IGNORECASE,
                )
                if step3_match:
                    trailing = full_answer[step3_match.end() :]
                    next_step_match = re.search(
                        r"\n\s*Step\s*\d+\s*[\.\-:]", trailing, flags=re.IGNORECASE
                    )
                    answer_part = (
                        trailing[: next_step_match.start()] if next_step_match else trailing
                    )
                    answer_part = re.sub(
                        r"^\s*(?:Answer|Final Answer)\s*[:\-]?\s*",
                        "",
                        answer_part,
                        flags=re.IGNORECASE,
                    ).strip()
                    if answer_part and (
                        len(answer_part) < len(full_answer) or re.search(r"[A-Za-z]", answer_part)
                    ):
                        return answer_part

            return full_answer

        except Exception as e:
            logger.exception("LLM generation error: %s", e)
            if memories:
                return memories[0].get("content", "I don't know.")
            return "I don't know."

    def _llm_evaluate_with_retries(
        self,
        *,
        question: str,
        hypothesis: str,
        reference: str,
        question_id: str,
        model: str,
    ) -> Dict[str, Any]:
        """Run the judge with retries and return a non-crashing error result on failure."""
        last_error: Optional[Exception] = None
        for attempt in range(1, self.JUDGE_MAX_ATTEMPTS + 1):
            try:
                result = llm_evaluate(
                    question=question,
                    hypothesis=hypothesis,
                    reference=reference,
                    question_id=question_id,
                    model=model,
                    client=self.openai_client,
                    fallback_on_error=False,
                )
                result["judge_attempts"] = attempt
                result["judge_error"] = None
                return result
            except Exception as exc:
                last_error = exc
                if attempt < self.JUDGE_MAX_ATTEMPTS:
                    sleep_seconds = self.JUDGE_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                    logger.warning(
                        "Judge attempt %s/%s failed for %s: %s",
                        attempt,
                        self.JUDGE_MAX_ATTEMPTS,
                        question_id,
                        exc,
                    )
                    time.sleep(sleep_seconds)

        error = f"{type(last_error).__name__}: {last_error}" if last_error else "unknown error"
        return {
            "is_correct": False,
            "confidence": 0.0,
            "explanation": f"Judge failed after {self.JUDGE_MAX_ATTEMPTS} attempts: {error}",
            "judge_attempts": self.JUDGE_MAX_ATTEMPTS,
            "judge_error": error,
        }

    def evaluate_question(self, item: Dict[str, Any], memories_stored: int) -> LongMemEvalResult:
        """
        Evaluate a single question: recall, generate answer, score.

        Returns a result dict.
        """
        question_id = item["question_id"]
        question = item["question"]
        reference = item.get("answer", "")
        question_type = item["question_type"]
        question_date = item.get("question_date", "")
        answer_session_ids = list(item.get("answer_session_ids") or [])

        # Recall
        memories = self.recall_for_question(question, question_id, question_date)
        retrieved_session_ids = self._top_unique_session_ids(memories, limit=5)
        recall_hit_at_5 = any(
            session_id in retrieved_session_ids for session_id in answer_session_ids
        )

        # Generate answer
        hypothesis = self.generate_answer(question, memories, question_date)

        # Score
        judge_attempts = 0
        judge_error = None
        if self.config.use_llm_eval:
            eval_model = self.effective_judge_model()
            if self.openai_client and eval_model:
                eval_result = self._llm_evaluate_with_retries(
                    question=question,
                    hypothesis=hypothesis,
                    reference=reference,
                    question_id=question_id,
                    model=eval_model,
                )
            else:
                eval_result = {
                    "is_correct": False,
                    "confidence": 0.0,
                    "explanation": "LLM eval unavailable: OPENAI_API_KEY required",
                    "judge_attempts": 0,
                    "judge_error": "OPENAI_API_KEY required",
                }
            is_correct = eval_result["is_correct"]
            confidence = eval_result["confidence"]
            explanation = eval_result["explanation"]
            judge_attempts = int(eval_result.get("judge_attempts") or 0)
            judge_error = eval_result.get("judge_error")
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
            "answer_session_ids": answer_session_ids,
            "retrieved_session_ids": retrieved_session_ids,
            "recall_hit_at_5": recall_hit_at_5,
            "judge_attempts": judge_attempts,
            "judge_error": judge_error,
        }

        return result

    def _build_error_result(
        self, item: Dict[str, Any], memories_stored: int, exc: Exception
    ) -> LongMemEvalResult:
        """Build a persisted result for unexpected per-question failures."""
        question_id = item.get("question_id", "unknown")
        error = f"{type(exc).__name__}: {exc}"
        return {
            "question_id": question_id,
            "question_type": item.get("question_type", "unknown"),
            "question": item.get("question", ""),
            "reference": item.get("answer", ""),
            "hypothesis": "",
            "is_correct": False,
            "confidence": 0.0,
            "explanation": f"Benchmark question failed: {error}",
            "recalled_count": 0,
            "memories_stored": memories_stored,
            "is_abstention": is_abstention_question(question_id),
            "question_date": item.get("question_date", ""),
            "answer_session_ids": list(item.get("answer_session_ids") or []),
            "retrieved_session_ids": [],
            "recall_hit_at_5": False,
            "judge_attempts": 0,
            "judge_error": None,
            "error": error,
            "error_type": type(exc).__name__,
        }

    def run_benchmark(
        self,
        cleanup_after: bool = True,
        output_path: Optional[str] = None,
        resume: bool = False,
    ) -> Dict[str, Any]:
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
        self.memory_ingest_failures = 0
        self.scorer = LongMemEvalScorer()
        output_base = self.resolve_output_base(output_path)
        paths = self.artifact_paths(output_base)
        os.makedirs(os.path.dirname(paths["partial"]) or ".", exist_ok=True)

        if not resume:
            with open(paths["partial"], "w", encoding="utf-8"):
                pass

        # Health check
        if not self.health_check():
            print(f"{self.backend.name} backend is not accessible. Aborting benchmark.")
            return {"overall": {"accuracy": 0.0, "correct": 0, "total": 0}}

        # Fail fast if LLM eval is requested but no OpenAI client is available.
        if self.config.use_llm_eval and not self.openai_client:
            raise RuntimeError(
                "--llm-eval requires OPENAI_API_KEY to be set, but no OpenAI client is "
                "configured. Set OPENAI_API_KEY and retry, or omit --llm-eval to use "
                "quick_score instead."
            )

        # Load dataset
        full_dataset = self.load_dataset()
        dataset, selection_metadata = self.select_dataset(full_dataset)

        resumed_results: List[Dict[str, Any]] = []
        completed_ids = set()
        if resume:
            raw_results = self._load_partial_results(output_base)
            # De-duplicate by question_id (last occurrence wins, preserving order).
            seen: Dict[str, Dict[str, Any]] = {}
            for result in raw_results:
                qid = result.get("question_id")
                if qid:
                    seen[qid] = result
                else:
                    print(f"WARNING: resumed result missing question_id; skipping: {result!r:.200}")
            resumed_results = list(seen.values())
            for result in resumed_results:
                completed_ids.add(result["question_id"])
                self.scorer.add_result(result)

        print("\nRunning LongMemEval benchmark:")
        print(f"  Backend: {self.backend.name}")
        print(f"  Config: {self.config.name}")
        print(f"  Questions: {len(dataset)}")
        print(f"  Selection: {selection_metadata['strategy']}")
        if self.config.per_type > 0:
            print(f"  Per type: {self.config.per_type}")
        print(f"  Storage: {self.config.storage_strategy}")
        print(f"  Recall limit: {self.config.recall_limit}")
        print(f"  Expand entities: {self.config.expand_entities}")
        print(f"  Expand relations: {self.config.expand_relations}")
        print(f"  Temporal hints: {self.config.use_temporal_hints}")
        print(f"  Answerer model: {self.config.llm_model}")
        print(f"  LLM eval: {self.config.use_llm_eval}")
        if self.config.use_llm_eval:
            print(f"  Judge model: {self.effective_judge_model()}")
        has_llm = self.openai_client is not None
        print(f"  OpenAI available: {has_llm}")
        print(f"  Output base: {output_base}")
        if resume:
            print(f"  Resumed completed questions: {len(completed_ids)}")
        print()

        start_time = time.time()
        started_at = datetime.now(timezone.utc).isoformat()
        all_results = resumed_results[:]
        skipped = 0

        for idx, item in enumerate(dataset):
            question_id = item["question_id"]
            question_type = item["question_type"]
            num_sessions = len(item.get("haystack_sessions", []))

            if question_id in completed_ids:
                skipped += 1
                print(f"[{idx + 1}/{len(dataset)}] SKIP completed {question_id}")
                continue

            print(f"[{idx + 1}/{len(dataset)}] {question_type}: " f"{item['question'][:60]}...")

            stored = 0
            try:
                # 1. Ingest sessions
                stored = self.ingest_sessions(item)
                print(f"  Ingested {stored} memories from {num_sessions} sessions")

                # 2. Evaluate (recall + generate + score)
                result = self.evaluate_question(item, stored)
            except Exception as exc:
                logger.exception("Question failed for %s: %s", question_id, exc)
                result = self._build_error_result(item, stored, exc)
            finally:
                # 3. Clean up this question's data even when recall/generation/judge fails
                if cleanup_after:
                    self.cleanup_test_data(question_id)

            all_results.append(result)
            self.scorer.add_result(result)
            completed_ids.add(question_id)
            self._append_partial_result(output_base, result)

            status = "CORRECT" if result["is_correct"] else "WRONG"
            print(f"  {status} | recalled={result['recalled_count']} | " f"{result['explanation']}")
            if result.get("judge_error"):
                print(f"    Judge error: {result['judge_error']}")
            if not result["is_correct"]:
                expected = result.get("reference")
                hypothesis = result.get("hypothesis")
                expected_preview = ("" if expected is None else str(expected))[:80]
                hypothesis_preview = ("" if hypothesis is None else str(hypothesis))[:80]
                print(f"    Expected: {expected_preview}")
                print(f"    Got:      {hypothesis_preview}")

            elapsed_so_far = time.time() - start_time
            self._write_status(
                output_base,
                started_at=started_at,
                selection=selection_metadata,
                completed=len(completed_ids),
                total=len(dataset),
                skipped=skipped,
                elapsed_time=elapsed_so_far,
                status="running",
                judge_errors=sum(1 for row in all_results if row.get("judge_error")),
            )

        elapsed_time = time.time() - start_time

        # Print report
        scores = self.scorer.print_report(config_name=self.config.name, elapsed_time=elapsed_time)
        scores["backend"] = self.backend.name
        scores["elapsed_time"] = elapsed_time
        scores["selection"] = selection_metadata
        effective_judge_model = self.effective_judge_model()
        scores["config"] = {
            "backend": self.config.backend,
            "name": self.config.name,
            "storage_strategy": self.config.storage_strategy,
            "recall_limit": self.config.recall_limit,
            "expand_entities": self.config.expand_entities,
            "expand_relations": self.config.expand_relations,
            "auto_decompose": self.config.auto_decompose,
            "use_temporal_hints": self.config.use_temporal_hints,
            "answerer_model": self.config.llm_model,
            "llm_model": self.config.llm_model,
            "eval_llm_model": self.config.eval_llm_model,
            "use_llm_eval": self.config.use_llm_eval,
            "max_questions": self.config.max_questions,
            "per_type": self.config.per_type,
            **judge_metadata(effective_judge_model, provider=DEFAULT_JUDGE_PROVIDER),
        }
        judge_errors = sum(1 for row in all_results if row.get("judge_error"))
        scores["memory_ingest_failures"] = self.memory_ingest_failures
        scores["judge_errors"] = judge_errors
        scores["publishable"] = judge_errors == 0
        scores["publishable_reason"] = None if judge_errors == 0 else "judge_errors_present"
        scores["artifacts"] = paths
        scores["details"] = all_results
        self._write_status(
            output_base,
            started_at=started_at,
            selection=selection_metadata,
            completed=len(completed_ids),
            total=len(dataset),
            skipped=skipped,
            elapsed_time=elapsed_time,
            status="completed",
            judge_errors=judge_errors,
        )

        return scores

    def save_results(
        self, scores: Dict[str, Any], output_path: Optional[str] = None
    ) -> Tuple[str, str]:
        """Save benchmark results to JSONL and JSON files."""
        if output_path is None and self._last_output_base:
            output_path = self._last_output_base
        output_path = self.resolve_output_base(output_path)
        paths = self.artifact_paths(output_path)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Save full results as JSON
        json_path = paths["json"]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2)
        print(f"\nResults saved to: {json_path}")

        # Save hypotheses as JSONL (for LongMemEval's evaluator)
        jsonl_path = paths["jsonl"]
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for detail in scores.get("details", []):
                line = {
                    "question_id": detail["question_id"],
                    "hypothesis": detail["hypothesis"],
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(f"Hypotheses saved to: {jsonl_path}")

        return json_path, jsonl_path


def main() -> int:
    """Run LongMemEval benchmark evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a memory backend on LongMemEval")
    parser.add_argument(
        "--base-url",
        default=os.getenv("AUTOMEM_TEST_BASE_URL", "http://localhost:8001"),
        help="Backend API base URL",
    )
    parser.add_argument(
        "--api-token",
        default=os.getenv("AUTOMEM_TEST_API_TOKEN", "test-token"),
        help="Backend API token",
    )
    parser.add_argument(
        "--backend",
        default="automem",
        choices=["automem"],
        help="Memory backend to benchmark (cross-backend runs live in automem-evals)",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Optional working directory for local backends like mempalace",
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
        help="LLM model for answer generation (default: gpt-5-mini)",
    )
    parser.add_argument(
        "--eval-llm-model",
        default=None,
        help=f"LLM model for evaluation (default: {CANONICAL_BENCHMARK_JUDGE_MODEL})",
    )
    parser.add_argument(
        "--llm-eval",
        action="store_true",
        help="Use canonical OpenAI judge for evaluation (costs money, more accurate)",
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
        help="Prefix-limit questions (0 = all; debug smoke only, biased by dataset order)",
    )
    parser.add_argument(
        "--per-type",
        type=int,
        default=0,
        help="Select up to N questions per LongMemEval question_type",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (without extension)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from <output>.partial.jsonl; requires --output",
    )

    args = parser.parse_args()
    if args.max_questions > 0 and args.per_type > 0:
        parser.error("--max-questions and --per-type are mutually exclusive")
    if args.resume and not args.output:
        parser.error("--resume requires --output so the partial artifact is explicit")

    # Build config from preset + overrides
    overrides = {
        "backend": args.backend,
        "base_url": args.base_url,
        "api_token": args.api_token,
        "work_dir": args.work_dir,
    }
    if args.data_file:
        overrides["data_file"] = args.data_file
    if args.recall_limit is not None:
        overrides["recall_limit"] = args.recall_limit
    if args.llm_model:
        overrides["llm_model"] = args.llm_model
    if args.eval_llm_model:
        overrides["eval_llm_model"] = args.eval_llm_model
    if args.llm_eval:
        overrides["use_llm_eval"] = True
    if args.max_questions > 0:
        overrides["max_questions"] = args.max_questions
    if args.per_type > 0:
        overrides["per_type"] = args.per_type

    config = get_config(args.config, **overrides)

    # Run benchmark
    benchmark = LongMemEvalBenchmark(config)
    output_base = benchmark.resolve_output_base(args.output)
    scores = benchmark.run_benchmark(
        cleanup_after=not args.no_cleanup,
        output_path=output_base,
        resume=args.resume,
    )

    # Save results
    benchmark.save_results(scores, output_base)

    # Return exit code
    return 0 if scores.get("overall", {}).get("total", 0) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
