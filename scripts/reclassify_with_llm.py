#!/usr/bin/env python3
"""Reclassify 'Memory' fallback types using LLM classification.

This script finds all memories with type='Memory' (the fallback) and reclassifies
them using the configured CLASSIFICATION_MODEL for more accurate type assignment.

Environment:
    CLASSIFICATION_MODEL: LLM model for classification (default: gpt-4o-mini)
    CLASSIFICATION_BASE_URL: Optional OpenAI-compatible endpoint
        (e.g. https://openrouter.ai/api/v1 for OpenRouter).
    CLASSIFICATION_API_KEY: API key paired with CLASSIFICATION_BASE_URL.
    OPENROUTER_API_KEY: Fallback key when --provider openrouter is selected.
    OPENAI_API_KEY: Fallback key when --provider openai (or default) is selected.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from falkordb import FalkorDB
from openai import OpenAI
from qdrant_client import QdrantClient

# Load environment
load_dotenv()
load_dotenv(Path.home() / ".config" / "automem" / ".env")

FALKORDB_HOST = os.getenv("FALKORDB_HOST", "localhost")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))
FALKORDB_PASSWORD = os.getenv("FALKORDB_PASSWORD")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "memories")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# CLASSIFICATION_BASE_URL lets you point at any OpenAI-compatible endpoint
# (e.g. https://openrouter.ai/api/v1 for OpenRouter).
CLASSIFICATION_BASE_URL = os.getenv("CLASSIFICATION_BASE_URL")
CLASSIFICATION_API_KEY = os.getenv("CLASSIFICATION_API_KEY")
CLASSIFICATION_MODEL = os.getenv("CLASSIFICATION_MODEL", "gpt-4o-mini")

# Valid memory types
VALID_TYPES = {"Decision", "Pattern", "Preference", "Style", "Habit", "Insight", "Context"}

SYSTEM_PROMPT = """You classify a single memory into exactly ONE type. Read the content carefully and pick the most specific type that fits. When multiple fit, use the priority rules at the bottom.

TYPES (with strict definitions):

- **Decision**: A choice was actively made between alternatives, or a commitment was made. Keywords: "chose", "decided", "will use", "picked X over Y", "going with". NOT every recommendation or protocol is a Decision — only if *someone made a choice*.

- **Pattern**: A reusable approach, template, or recurring way of doing something. Keywords: "pattern", "approach", "template", "workflow", "always do X when Y". Repeatable and generalizable.

- **Preference**: A stated like/dislike, favorite, or taste. Someone prefers X. Keywords: "prefers", "likes", "dislikes", "favorite", "wants X over Y" (as a taste, not a decision).

- **Style**: Formatting, tone, naming convention, or communication approach. Keywords: "tabs not spaces", "short commit messages", "formal tone", "snake_case".

- **Habit**: A regular routine or repeated behavior. Keywords: "always X", "every morning", "routinely", "every Friday". Time-regularity is the marker.

- **Insight**: A genuine *realization* or *learning* from experience — something the author *discovered* or *figured out*. Keywords: "turns out", "learned that", "root cause", "realized", "the trick is", "gotcha". NOT textbook facts or product descriptions.

- **Context**: Background facts about a person, place, product, tool, or situation. Keywords: "X is a Y", "X offers Y", "X released", "X is located at", biographical facts, product descriptions, tool capabilities, session logs, conversation fragments. This is the correct type for factual statements that aren't discoveries.

PRIORITY RULES (apply in order):

1. If the memory starts with "Fact:", "Concept:", "[X in #channel]", "Session:", or is biographical/descriptive about an entity → **Context**, not Insight.
2. If the memory describes a gotcha, failure mode, root cause, unexpected behavior, or something the author discovered through experience → **Insight**.
3. A security protocol, best practice, or recommendation is NOT a Decision unless someone explicitly chose it over an alternative → usually **Insight** or **Context**.
4. A tool description ("X does Y", "X is a library that...") is **Context**, not Insight, even if interesting.
5. A DM fragment or conversation excerpt is **Context**, not Decision.
6. A "how I set up X" or "how X works" explanation is **Context** unless it's framed as a reusable approach → then **Pattern**.
7. Only return **Preference** / **Style** / **Habit** when the memory is unambiguously one of those — these are narrow categories, don't stretch them.

Be conservative with Insight — it should mean genuine experiential learning, not any statement that sounds smart. If you're unsure between Insight and Context, pick Context.

Confidence: 0.95+ if the type is obvious, 0.75-0.9 if you had to apply a priority rule, 0.5-0.7 if genuinely ambiguous.

Return JSON with: {"type": "<type>", "confidence": <0.0-1.0>}"""


def get_fallback_memories(client, db_limit: int | None = None) -> list[Dict[str, Any]]:
    """Fetch memories with type='Memory' (fallback).

    When `db_limit` is set, the cap is applied in Cypher so we don't materialize
    the whole corpus on head-sampled partial runs. Random sampling still requires
    the full set, so this stays None in that path.
    """
    print("📥 Fetching memories with fallback type='Memory'...")
    g = client.select_graph("memories")

    cypher = (
        "MATCH (m:Memory) "
        "WHERE m.type = 'Memory' "
        "RETURN m.id as id, m.content as content, m.confidence as confidence"
    )
    params: Dict[str, Any] = {}
    if db_limit is not None:
        cypher += " LIMIT $limit"
        params["limit"] = db_limit

    result = g.query(cypher, params) if params else g.query(cypher)

    memories = []
    for row in result.result_set:
        memories.append(
            {
                "id": row[0],
                "content": row[1],
                "old_confidence": row[2],
            }
        )

    print(f"✅ Found {len(memories)} memories with fallback type\n")
    return memories


def _extract_json(text: str) -> dict:
    """Parse JSON from model output, tolerating ```json fences and prose."""
    text = text.strip()
    # Strip ```json ... ``` fencing
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
        # Trim trailing fence leftover
        if text.endswith("```"):
            text = text[:-3].strip()
    # First try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fallback: find first { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    # Avoid echoing model output (may include memory content) into the exception
    # message — the caller logs exceptions, which would leak corpus data.
    raise ValueError("Model returned non-JSON output (no parseable object)")


def classify_with_llm(
    openai_client: OpenAI, content: str, *, supports_json_mode: bool = False
) -> tuple[str, float]:
    """Classify memory type via OpenAI-compatible chat completion."""
    kwargs = dict(
        model=CLASSIFICATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content[:1000]},
        ],
        temperature=0.3,
        max_tokens=50,
    )
    # response_format=json_object is OpenAI-specific. Gate on the selected endpoint
    # (not the model name) — OpenRouter etc. may reject it even for gpt-* models.
    if supports_json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = openai_client.chat.completions.create(**kwargs)
        raw = response.choices[0].message.content or ""
        result = _extract_json(raw)
        memory_type = result.get("type", "Context")
        confidence = float(result.get("confidence", 0.7))

        # Validate type
        if memory_type not in VALID_TYPES:
            memory_type = "Context"
            confidence = 0.6

        return memory_type, confidence

    except Exception as e:
        print(f"   ⚠️  Classification failed: {e}")
        return "Context", 0.5


def update_memory_type(
    falkor_client, qdrant_client, memory_id: str, new_type: str, new_confidence: float
) -> bool:
    """Update memory type in both FalkorDB and Qdrant."""
    try:
        # Update FalkorDB
        g = falkor_client.select_graph("memories")
        g.query(
            """
            MATCH (m:Memory {id: $id})
            SET m.type = $type, m.confidence = $confidence
            """,
            {"id": memory_id, "type": new_type, "confidence": new_confidence},
        )

        # Update Qdrant
        if qdrant_client:
            try:
                qdrant_client.set_payload(
                    collection_name=QDRANT_COLLECTION,
                    points=[memory_id],
                    payload={"type": new_type, "confidence": new_confidence},
                )
            except Exception as e:
                print(f"   ⚠️  Qdrant update failed: {e}")

        return True
    except Exception as e:
        print(f"   ❌ Update failed: {e}")
        return False


def _positive_int(raw: str) -> int:
    """argparse type for a positive integer (rejects 0 and negatives)."""
    try:
        value = int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected integer, got {raw!r}")
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def main():
    """Main reclassification process."""
    parser = argparse.ArgumentParser(description="Reclassify fallback 'Memory' types via LLM.")
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="Process at most N memories (must be >= 1). Useful for dry-runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Classify and print proposed changes without writing to FalkorDB or Qdrant.",
    )
    parser.add_argument(
        "--sample",
        choices=["head", "random"],
        default="head",
        help="How to pick memories when --limit is set (default: head).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for --sample random (for reproducible slices).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "openrouter"],
        default=None,
        help="Shortcut for setting base URL + key. 'openrouter' uses OPENROUTER_API_KEY.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override CLASSIFICATION_MODEL (e.g. google/gemini-3.1-flash-lite-preview).",
    )
    args = parser.parse_args()

    # Resolve provider → base_url + api_key.
    #
    # Explicit --provider FORCES its canonical base URL so a stale
    # CLASSIFICATION_BASE_URL in the env can't pair an OpenRouter key with the
    # wrong endpoint (or vice versa). Provider-specific keys are only honored
    # for that provider — we never silently fall back across providers (which
    # would leak e.g. OPENAI_API_KEY to a third-party endpoint).
    if args.provider == "openrouter":
        base_url = "https://openrouter.ai/api/v1"
        api_key = CLASSIFICATION_API_KEY or OPENROUTER_API_KEY
    elif args.provider == "openai":
        base_url = None  # OpenAI default endpoint
        api_key = CLASSIFICATION_API_KEY or OPENAI_API_KEY
    else:
        # No explicit provider. Honor CLASSIFICATION_BASE_URL if present
        # (with its paired CLASSIFICATION_API_KEY only), else default to OpenAI.
        base_url = CLASSIFICATION_BASE_URL
        if base_url:
            api_key = CLASSIFICATION_API_KEY
        else:
            api_key = CLASSIFICATION_API_KEY or OPENAI_API_KEY

    # Override model if passed on CLI
    model = args.model or CLASSIFICATION_MODEL
    # Share model with classify_with_llm via a module-level rebind
    globals()["CLASSIFICATION_MODEL"] = model

    # response_format=json_object is OpenAI-only. Gate on the endpoint, not the
    # model name — a gpt-* model name routed through OpenRouter would otherwise
    # request a parameter the gateway may reject.
    supports_json_mode = base_url is None and model.startswith(
        ("gpt-", "o1-", "o3-", "o4-")
    )

    print("=" * 70)
    print("🤖 AutoMem LLM Reclassification Tool")
    if args.dry_run:
        print("   [DRY-RUN MODE — no writes]")
    if args.limit is not None:
        print(f"   [LIMIT: {args.limit} memories, sample={args.sample}]")
    print("=" * 70)
    print()

    if not api_key:
        if args.provider == "openrouter":
            print("❌ No API key for OpenRouter. Set OPENROUTER_API_KEY (or CLASSIFICATION_API_KEY).")
        elif base_url:
            print(f"❌ No API key for {base_url}. Set CLASSIFICATION_API_KEY.")
        else:
            print("❌ No API key for OpenAI. Set OPENAI_API_KEY,")
            print("   or pass --provider openrouter (with OPENROUTER_API_KEY) to use OpenRouter.")
        sys.exit(1)

    # Connect to FalkorDB
    print(f"🔌 Connecting to FalkorDB at {FALKORDB_HOST}:{FALKORDB_PORT}")
    try:
        falkor_client = FalkorDB(
            host=FALKORDB_HOST,
            port=FALKORDB_PORT,
            password=FALKORDB_PASSWORD,
            username="default" if FALKORDB_PASSWORD else None,
        )
        print("✅ Connected to FalkorDB\n")
    except Exception as e:
        print(f"❌ Failed to connect to FalkorDB: {e}")
        sys.exit(1)

    # Connect to Qdrant (optional)
    qdrant_client = None
    if QDRANT_URL:
        print(f"🔌 Connecting to Qdrant at {QDRANT_URL}")
        try:
            qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            print("✅ Connected to Qdrant\n")
        except Exception as e:
            print(f"⚠️  Qdrant connection failed: {e}")
            print("   (Will update FalkorDB only)\n")

    # Initialize OpenAI-compatible client (OpenAI or OpenRouter)
    endpoint_label = base_url or "https://api.openai.com/v1 (OpenAI default)"
    print(f"🤖 Initializing client (model: {model}, endpoint: {endpoint_label})")
    openai_client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    print("✅ Client ready\n")

    # Push --limit into Cypher for head sampling so we don't materialize the
    # entire fallback corpus on partial runs. Random sampling still requires
    # the full set, so we only cap at the DB for head mode.
    db_limit = args.limit if (args.limit is not None and args.sample == "head") else None
    memories = get_fallback_memories(falkor_client, db_limit=db_limit)

    if not memories:
        print("✅ No memories need reclassification!")
        return

    total_available = len(memories)

    # Apply --limit slice (random sampling still needs the full set).
    if args.limit is not None and args.sample == "random" and args.limit < len(memories):
        if args.seed is not None:
            random.seed(args.seed)
        memories = random.sample(memories, args.limit)
        print(f"🎯 Sliced to {len(memories)} memories (of {total_available} available)\n")
    elif args.limit is not None and args.sample == "head":
        print(f"🎯 Capped at {len(memories)} memories (head, db-side LIMIT)\n")

    # Estimate cost — the per-token figure below is OpenAI gpt-4o-mini pricing.
    # For other models / providers (OpenRouter, Azure, vLLM, …) actual cost
    # varies; surface the assumption so the confirmation prompt isn't misleading.
    tokens_per_memory = 370  # ~350 input + 20 output
    total_tokens = len(memories) * tokens_per_memory
    estimated_cost = (total_tokens / 1_000_000) * 0.20  # OpenAI gpt-4o-mini blended rate

    pricing_basis = (
        "OpenAI gpt-4o-mini pricing"
        if base_url is None and model.startswith("gpt-4o-mini")
        else f"OpenAI gpt-4o-mini pricing — actual cost depends on {model}"
    )
    print(f"💰 Estimated cost: ${estimated_cost:.4f} (~{estimated_cost * 100:.1f} cents)")
    print(f"   ({pricing_basis})")
    print(f"📊 Tokens: ~{total_tokens:,}")
    print()

    # Confirm (unless --yes or --dry-run)
    if not args.yes and not args.dry_run:
        response = input(f"🔄 Reclassify {len(memories)} memories with LLM? [y/N]: ")
        if response.lower() != "y":
            print("❌ Reclassification cancelled")
            sys.exit(0)

    print()
    print("🔄 Starting reclassification...")
    print()

    success_count = 0
    failed_count = 0
    type_counts = {}

    for i, memory in enumerate(memories, 1):
        memory_id = memory["id"]
        content = memory["content"] or ""

        content_preview = content[:60] + "..." if len(content) > 60 else content
        print(f"[{i}/{len(memories)}] {content_preview}")

        # Classify with LLM
        new_type, new_confidence = classify_with_llm(
            openai_client, content, supports_json_mode=supports_json_mode
        )
        type_counts[new_type] = type_counts.get(new_type, 0) + 1

        print(f"   → {new_type} (confidence: {new_confidence:.2f})")

        if args.dry_run:
            success_count += 1
            print(f"   🧪 (dry-run, not written)")
        elif update_memory_type(
            falkor_client, qdrant_client, memory_id, new_type, new_confidence
        ):
            success_count += 1
            print(f"   ✅ Updated")
        else:
            failed_count += 1

        # Progress update every 10
        if i % 10 == 0:
            print(f"\n💤 Progress: {success_count} ✅ / {failed_count} ❌\n")
            time.sleep(0.5)  # Rate limiting

    print()
    print("=" * 70)
    print(f"✅ Reclassification complete!")
    print(f"   Success: {success_count}")
    print(f"   Failed: {failed_count}")
    print()
    print("📊 Type Distribution:")
    for mem_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {mem_type}: {count}")
    print("=" * 70)


if __name__ == "__main__":
    main()
