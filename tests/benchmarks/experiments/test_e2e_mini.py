#!/usr/bin/env python3
"""
Micro E2E test: Run 3 questions to verify the full pipeline.
"""

import os
import sys
import json
import time
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tests.benchmarks.experiments.experiment_config import ExperimentConfig
from tests.benchmarks.locomo_metrics import f1_score
from openai import OpenAI


def run_micro_e2e():
    print("=" * 60)
    print("MICRO E2E PIPELINE TEST")
    print("=" * 60)
    
    BASE_URL = os.getenv("AUTOMEM_BASE_URL", "http://localhost:8001")
    API_TOKEN = os.getenv("AUTOMEM_API_TOKEN", "test-token")
    
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Load dataset
    data_file = Path(__file__).parent.parent / "locomo" / "data" / "locomo10.json"
    with open(data_file, 'r') as f:
        conversations = json.load(f)
    
    conv = conversations[0]
    sample_id = conv.get('sample_id', 'unknown')
    
    print(f"\n📂 Using conversation: {sample_id}")
    
    # Step 1: Store memories from conversation sessions
    print(f"\n[1/4] Storing memories...")
    test_tag = f"micro-test-{int(time.time())}"
    
    convo = conv.get('conversation', {})
    memories_stored = 0
    
    # Iterate through sessions (keys like 'session_1', 'session_2', etc.)
    for key in sorted(convo.keys()):
        val = convo.get(key)
        # Only process list values (these are dialog sessions)
        if not isinstance(val, list):
            continue
        
        for turn in val[:5]:  # First 5 turns per session
            speaker = turn.get('speaker', 'Unknown')
            text = turn.get('text', '')
            if not text:
                continue
            content = f"[{speaker}]: {text}"
            
            try:
                resp = requests.post(
                    f"{BASE_URL}/memory",
                    headers=headers,
                    json={
                        "content": content,
                        "tags": [test_tag, "locomo-micro"],
                        "metadata": {"dialog_id": turn.get('dia_id', '')}
                    },
                    timeout=10
                )
                if resp.status_code in (200, 201):
                    memories_stored += 1
            except Exception as e:
                print(f"  Error storing: {e}")
        
        if memories_stored >= 20:
            break
    
    print(f"  Stored {memories_stored} memories")
    if memories_stored == 0:
        print("  ⚠️  No memories stored! Check data format.")
        return False
    
    time.sleep(2)  # Wait for enrichment
    
    # Step 2: Test recall
    print(f"\n[2/4] Testing recall and E2E answer generation...")
    qa_pairs = conv.get('qa', [])[:3]  # First 3 questions
    
    client = OpenAI()
    results = []
    
    for i, qa in enumerate(qa_pairs):
        question = qa['question']
        expected = str(qa.get('answer', ''))  # Convert to string
        category = qa.get('category', 1)
        
        print(f"\n  Q{i+1}: {question[:50]}...")
        print(f"      Expected: {expected[:40]}...")
        
        # Recall memories
        resp = requests.get(
            f"{BASE_URL}/recall",
            headers=headers,
            params={"query": question, "tags": test_tag, "limit": 5}
        )
        
        if resp.status_code != 200:
            print(f"      ❌ Recall failed: {resp.status_code}")
            continue
            
        recalled = resp.json().get('results', [])
        print(f"      Retrieved: {len(recalled)} memories")
        
        # Build context
        context = "\n".join([r.get('memory', {}).get('content', '') for r in recalled[:5]])
        
        if not context.strip():
            context = "No relevant memories found."
        
        # Step 3: Generate answer
        try:
            llm_response = client.chat.completions.create(
                model="gpt-4o-mini",  # Fast and cheap for test
                messages=[
                    {"role": "system", "content": "Answer based ONLY on the context. If not in context, say 'no information available'."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                ],
                temperature=0.0,
                max_tokens=100
            )
            generated = llm_response.choices[0].message.content.strip()
        except Exception as e:
            generated = f"Error: {e}"
        
        print(f"      Generated: {generated[:40]}...")
        
        # Step 4: Score with F1
        f1 = f1_score(generated, expected)
        is_correct = f1 >= 0.5
        
        results.append({
            "question": question,
            "expected": expected,
            "generated": generated,
            "f1": f1,
            "correct": is_correct
        })
        
        status = "✅" if is_correct else "❌"
        print(f"      {status} F1: {f1:.3f}")
    
    # Cleanup
    print(f"\n[4/4] Cleaning up...")
    resp = requests.delete(
        f"{BASE_URL}/admin/memories",
        headers={"Authorization": f"Bearer admin"},
        params={"tag": test_tag, "confirm": "yes"}
    )
    print(f"  Cleanup: {resp.status_code}")
    
    # Summary
    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    avg_f1 = sum(r['f1'] for r in results) / total if total > 0 else 0
    
    print("\n" + "=" * 60)
    print("MICRO E2E RESULTS")
    print("=" * 60)
    print(f"  Questions: {total}")
    print(f"  Correct: {correct}/{total} ({100*correct/total:.0f}%)")
    print(f"  Average F1: {avg_f1:.3f}")
    print("=" * 60)
    
    if total == 0:
        print("  ⚠️  No questions evaluated!")
        return False
    
    print("\n✅ E2E pipeline is working!")
    return True


if __name__ == "__main__":
    success = run_micro_e2e()
    sys.exit(0 if success else 1)
