#!/usr/bin/env python3
"""Clean up polluted memory types in FalkorDB and Qdrant.

This script reclassifies memories with invalid types (e.g., session_start, interaction)
back to valid types (Decision, Pattern, Preference, Style, Habit, Insight, Context).
"""

import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Set

from dotenv import load_dotenv
from falkordb import FalkorDB
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

# Valid memory types
VALID_TYPES = {"Decision", "Pattern", "Preference", "Style", "Habit", "Insight", "Context"}

# Classification patterns (from app.py)
PATTERNS = {
    "Decision": [
        r"decided to",
        r"chose (\w+) over",
        r"going with",
        r"picked",
        r"selected",
        r"will use",
        r"choosing",
        r"opted for",
    ],
    "Pattern": [
        r"usually",
        r"typically",
        r"tend to",
        r"pattern i noticed",
        r"often",
        r"frequently",
        r"regularly",
        r"consistently",
    ],
    "Preference": [
        r"prefer",
        r"like.*better",
        r"favorite",
        r"always use",
        r"rather than",
        r"instead of",
        r"favor",
    ],
    "Style": [
        r"wrote.*in.*style",
        r"communicated",
        r"responded to",
        r"formatted as",
        r"using.*tone",
        r"expressed as",
    ],
    "Habit": [r"always", r"every time", r"habitually", r"routine", r"daily", r"weekly", r"monthly"],
    "Insight": [
        r"realized",
        r"discovered",
        r"learned that",
        r"understood",
        r"figured out",
        r"insight",
        r"revelation",
    ],
    "Context": [r"when", r"at the time", r"situation was"],
}


def classify_memory(content: str) -> tuple[str, float]:
    """
    Classify memory type and return confidence score.
    Returns: (type, confidence)
    """
    content_lower = content.lower()

    for memory_type, patterns in PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, content_lower):
                # Start with base confidence
                confidence = 0.6

                # Boost confidence for multiple pattern matches
                matches = sum(1 for p in patterns if re.search(p, content_lower))
                if matches > 1:
                    confidence = min(0.95, confidence + (matches * 0.1))

                return memory_type, confidence

    # Default to Memory type with lower confidence
    return "Memory", 0.3


def get_all_memories(client) -> list[Dict[str, Any]]:
    """Fetch all memories from FalkorDB."""
    print("📥 Fetching all memories from FalkorDB...")
    g = client.select_graph("memories")

    result = g.query(
        """
        MATCH (m:Memory)
        RETURN m.id as id, m.type as type, m.content as content, m.confidence as confidence
    """
    )

    memories = []
    for row in result.result_set:
        memories.append(
            {
                "id": row[0],
                "type": row[1],
                "content": row[2],
                "confidence": row[3],
            }
        )

    print(f"✅ Found {len(memories)} memories\n")
    return memories


def update_memory_type(
    client, qdrant_client, memory_id: str, new_type: str, new_confidence: float
) -> bool:
    """Update memory type in both FalkorDB and Qdrant."""
    try:
        # Update FalkorDB
        g = client.select_graph("memories")
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


def main():
    """Main cleanup process."""
    print("=" * 70)
    print("🧹 AutoMem Memory Type Cleanup Tool")
    print("=" * 70)
    print()
    print("Valid types:", ", ".join(sorted(VALID_TYPES)))
    print()

    # Connect to FalkorDB
    print(f"🔌 Connecting to FalkorDB at {FALKORDB_HOST}:{FALKORDB_PORT}")
    try:
        client = FalkorDB(
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

    # Get all memories
    memories = get_all_memories(client)

    # Analyze type distribution
    type_counts: Dict[str, int] = {}
    invalid_memories = []

    for memory in memories:
        mem_type = memory["type"]
        type_counts[mem_type] = type_counts.get(mem_type, 0) + 1

        if mem_type not in VALID_TYPES and mem_type != "Memory":
            invalid_memories.append(memory)

    print(f"📊 Type Distribution:")
    valid_count = sum(type_counts.get(t, 0) for t in VALID_TYPES)
    invalid_count = len(invalid_memories)
    print(f"   ✅ Valid types: {valid_count}")
    print(f"   ❌ Invalid types: {invalid_count}")
    print(f"   ℹ️  Fallback (Memory): {type_counts.get('Memory', 0)}")
    print()

    if invalid_count > 0:
        print(f"🔍 Found {len(invalid_memories)} memories with invalid types:")
        invalid_type_counts: Dict[str, int] = {}
        for mem in invalid_memories:
            invalid_type_counts[mem["type"]] = invalid_type_counts.get(mem["type"], 0) + 1

        for mem_type, count in sorted(
            invalid_type_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]:
            print(f"   - {mem_type}: {count}")

        if len(invalid_type_counts) > 10:
            print(f"   ... and {len(invalid_type_counts) - 10} more")
        print()

        # Confirm cleanup
        response = input(f"🧹 Reclassify {invalid_count} invalid memories? [y/N]: ")
        if response.lower() != "y":
            print("❌ Cleanup cancelled")
            sys.exit(0)

        print()
        print("🔄 Reclassifying memories...")
        print()

        success_count = 0
        failed_count = 0

        for i, memory in enumerate(invalid_memories, 1):
            memory_id = memory["id"]
            content = memory["content"] or ""
            old_type = memory["type"]

            # Classify
            new_type, new_confidence = classify_memory(content)

            content_preview = content[:50] + "..." if len(content) > 50 else content
            print(f"[{i}/{invalid_count}] {old_type} → {new_type}")
            print(f"   {content_preview}")

            if update_memory_type(client, qdrant_client, memory_id, new_type, new_confidence):
                success_count += 1
                print(f"   ✅ Updated")
            else:
                failed_count += 1

            # Progress update
            if i % 10 == 0:
                print(f"\n💤 Progress: {success_count} ✅ / {failed_count} ❌\n")
                time.sleep(0.5)  # Rate limiting

        print()
        print("=" * 70)
        print(f"✅ Cleanup complete!")
        print(f"   Reclassified: {success_count}")
        print(f"   Failed: {failed_count}")
        print("=" * 70)
    else:
        print("✅ All memory types are valid! No cleanup needed.")


if __name__ == "__main__":
    main()
