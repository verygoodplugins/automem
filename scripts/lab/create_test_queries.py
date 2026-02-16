#!/usr/bin/env python3
"""Generate real-world test queries from production AutoMem data.

Samples diverse memories and uses GPT-4o-mini to generate natural questions
that a user would ask to retrieve each memory. Outputs a JSON test set
for use with run_recall_test.py.

Usage:
    # Generate 50 test queries from local AutoMem
    python scripts/lab/create_test_queries.py

    # Custom count and output
    python scripts/lab/create_test_queries.py --count 100 --output lab/test_sets/custom.json

    # Use a specific API endpoint
    python scripts/lab/create_test_queries.py --api-url http://localhost:8001
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

API_URL = os.getenv("AUTOMEM_TEST_BASE_URL", "http://localhost:8001")
API_TOKEN = os.getenv("AUTOMEM_API_TOKEN", "test-token")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


def get_headers() -> dict:
    return {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}


def fetch_all_memories(api_url: str) -> List[Dict[str, Any]]:
    """Fetch all memories from AutoMem via the analyze endpoint or tag-based recall."""
    print("Fetching memories from AutoMem...")

    # Use recall with high limit to get a broad sample
    # We'll do multiple passes with different sort orders
    memories = {}

    for sort_by in ["score", "time_desc", "updated_desc"]:
        resp = requests.get(
            f"{api_url}/recall",
            params={"query": "*", "limit": 200, "sort": sort_by},
            headers=get_headers(),
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", data.get("memories", []))
            for r in results:
                mem = r.get("memory", r)
                mid = str(mem.get("id", ""))
                if mid and mid not in memories:
                    memories[mid] = mem

    # Also fetch by common types
    for mem_type in ["Decision", "Pattern", "Preference", "Insight", "Context", "Habit", "Style"]:
        resp = requests.get(
            f"{api_url}/recall",
            params={"query": mem_type, "limit": 50},
            headers=get_headers(),
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", data.get("memories", []))
            for r in results:
                mem = r.get("memory", r)
                mid = str(mem.get("id", ""))
                if mid and mid not in memories:
                    memories[mid] = mem

    all_mems = list(memories.values())
    print(f"  Fetched {len(all_mems)} unique memories")
    return all_mems


def stratified_sample(memories: List[Dict], count: int) -> List[Dict]:
    """Sample memories stratified by type, importance, and recency."""
    by_type: Dict[str, List[Dict]] = defaultdict(list)
    for m in memories:
        mem_type = m.get("type", "Context") or "Context"
        by_type[mem_type].append(m)

    selected = []
    remaining = count

    # 1. High importance (>= 0.7) — 25%
    high_imp = [m for m in memories if (m.get("importance") or 0) >= 0.7]
    n_high = min(len(high_imp), count // 4)
    if high_imp:
        selected.extend(random.sample(high_imp, n_high))
        remaining -= n_high

    # 2. One from each type — ensure diversity
    selected_ids = {m.get("id") for m in selected}
    for mem_type, type_mems in by_type.items():
        available = [m for m in type_mems if m.get("id") not in selected_ids]
        if available and remaining > 0:
            pick = random.choice(available)
            selected.append(pick)
            selected_ids.add(pick.get("id"))
            remaining -= 1

    # 3. Fill remaining randomly
    available = [m for m in memories if m.get("id") not in selected_ids]
    if available and remaining > 0:
        extra = random.sample(available, min(len(available), remaining))
        selected.extend(extra)

    return selected[:count]


def generate_questions_batch(memories: List[Dict]) -> List[Dict[str, Any]]:
    """Use GPT-4o-mini to generate natural questions for a batch of memories."""
    if not OPENAI_API_KEY:
        print("WARNING: OPENAI_API_KEY not set. Generating template questions instead.")
        return generate_template_questions(memories)

    results = []
    batch_size = 10  # Process 10 at a time to stay within token limits

    for i in range(0, len(memories), batch_size):
        batch = memories[i : i + batch_size]
        print(f"  Generating questions for memories {i + 1}-{i + len(batch)}...")

        memory_descriptions = []
        for j, mem in enumerate(batch):
            content = mem.get("content", "")[:300]
            mem_type = mem.get("type", "Context")
            tags = ", ".join((mem.get("tags") or [])[:5])
            memory_descriptions.append(
                f"Memory {j + 1} (type: {mem_type}, tags: {tags}):\n{content}"
            )

        prompt = f"""For each memory below, generate 2 natural questions that a user would ask
an AI assistant to retrieve this specific memory. Questions should be:
- Natural and conversational (how a real person would ask)
- Specific enough to uniquely identify this memory
- Varied in style (some direct, some contextual)

Return JSON array of objects with: memory_index (0-based), questions (array of 2 strings)

Memories:
{chr(10).join(memory_descriptions)}"""

        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "response_format": {"type": "json_object"},
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)

            # Handle both {"questions": [...]} and direct array
            question_list = (
                parsed
                if isinstance(parsed, list)
                else parsed.get("questions", parsed.get("results", []))
            )

            for item in question_list:
                idx = item.get("memory_index", 0)
                if 0 <= idx < len(batch):
                    mem = batch[idx]
                    for q in item.get("questions", []):
                        results.append(
                            {
                                "query": q,
                                "expected_ids": [str(mem.get("id", ""))],
                                "category": mem.get("type", "Context"),
                                "importance": mem.get("importance", 0.5),
                                "memory_preview": mem.get("content", "")[:150],
                            }
                        )
        except Exception as e:
            print(f"  WARNING: GPT-4o-mini failed for batch: {e}")
            results.extend(generate_template_questions(batch))

        time.sleep(0.5)  # Rate limit courtesy

    return results


def generate_template_questions(memories: List[Dict]) -> List[Dict[str, Any]]:
    """Fallback: generate simple template-based questions without LLM."""
    results = []
    for mem in memories:
        content = mem.get("content", "")
        mem_type = mem.get("type", "Context")
        tags = mem.get("tags") or []

        # Extract first sentence or first 100 chars as a reference
        first_sentence = content.split(".")[0][:100] if content else "this memory"

        questions = [
            f"What do I know about {first_sentence}?",
            f"Find my {mem_type.lower()} about {' '.join(tags[:3]) if tags else first_sentence[:50]}",
        ]

        for q in questions:
            results.append(
                {
                    "query": q,
                    "expected_ids": [str(mem.get("id", ""))],
                    "category": mem_type,
                    "importance": mem.get("importance", 0.5),
                    "memory_preview": content[:150],
                }
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate test queries from AutoMem data")
    parser.add_argument("--count", type=int, default=50, help="Number of memories to sample")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    parser.add_argument("--api-url", type=str, default=API_URL, help="AutoMem API URL")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    # Fetch memories
    memories = fetch_all_memories(args.api_url)
    if not memories:
        print("ERROR: No memories found. Is AutoMem running?")
        sys.exit(1)

    # Sample
    sampled = stratified_sample(memories, args.count)
    print(f"Sampled {len(sampled)} memories (stratified by type/importance)")

    # Show distribution
    type_counts = defaultdict(int)
    for m in sampled:
        type_counts[m.get("type", "Context")] += 1
    print(f"  Distribution: {dict(type_counts)}")

    # Generate questions
    print("Generating test questions...")
    queries = generate_questions_batch(sampled)
    print(f"Generated {len(queries)} test queries")

    # Output
    output_path = args.output or f"lab/test_sets/queries_{len(queries)}.json"
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    test_set = {
        "metadata": {
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_count": len(memories),
            "sample_count": len(sampled),
            "query_count": len(queries),
            "seed": args.seed,
            "type_distribution": dict(type_counts),
        },
        "queries": queries,
    }

    with open(output_path, "w") as f:
        json.dump(test_set, f, indent=2)

    print(f"\nSaved to: {output_path}")
    print(f"Review and curate the queries, then run:")
    print(f"  python scripts/lab/run_recall_test.py --test-set {output_path}")


if __name__ == "__main__":
    main()
