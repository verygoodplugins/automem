#!/usr/bin/env python3
"""Remove duplicate memories from Qdrant based on content similarity.

After accidentally running recovery that duplicated memories in Qdrant,
this script will identify and remove duplicates, keeping only the original.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load environment
load_dotenv()
load_dotenv(Path.home() / ".config" / "automem" / ".env")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "memories")


def deduplicate_memories(dry_run: bool = False, auto_confirm: bool = False):
    """Remove duplicate memories from Qdrant."""
    print("=" * 60)
    if dry_run:
        print("🔧 Qdrant Deduplication Tool (DRY RUN - No Changes)")
    else:
        print("🔧 Qdrant Deduplication Tool")
    print("=" * 60)
    print()

    # Connect to Qdrant
    print(f"🔌 Connecting to Qdrant at {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # Get collection info
    try:
        collection = client.get_collection(QDRANT_COLLECTION)
        total_count = collection.points_count
        print(f"📊 Current memory count: {total_count}\n")
    except Exception as e:
        print(f"❌ Error accessing collection: {e}")
        sys.exit(1)

    # Fetch all memories
    print("🔍 Fetching all memories...")
    memories = []
    offset = None

    while True:
        result = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        points, next_offset = result
        memories.extend(points)

        if next_offset is None:
            break
        offset = next_offset

    print(f"✅ Fetched {len(memories)} memories\n")

    # Find duplicates by content hash
    print("🔎 Identifying duplicates...")
    seen_content: Dict[str, str] = {}  # content -> first memory_id
    duplicates: Set[str] = set()

    for memory in memories:
        content = memory.payload.get("content", "")
        timestamp = memory.payload.get("timestamp", "")

        # Create a unique key based on content
        key = f"{content}|{timestamp}"

        if key in seen_content:
            # This is a duplicate - mark for deletion
            duplicates.add(memory.id)
        else:
            # First occurrence - keep this one
            seen_content[key] = memory.id

    print(f"Found {len(duplicates)} duplicates to remove\n")

    if not duplicates:
        print("✅ No duplicates found!")
        return

    # Show what will be deleted
    print(f"📋 Summary:")
    print(f"   Total memories: {len(memories)}")
    print(f"   Duplicates: {len(duplicates)}")
    print(f"   Will keep: {len(memories) - len(duplicates)}")
    print()

    if dry_run:
        print("🔍 DRY RUN - No changes will be made")
        print("   Run without --dry-run to actually delete duplicates")
        return

    # Confirm deletion
    if not auto_confirm:
        print(f"⚠️  This will DELETE {len(duplicates)} duplicate memories from Qdrant")
        print(f"   Keeping {len(memories) - len(duplicates)} unique memories")
        response = input("\nContinue? (yes/no): ")

        if response.lower() not in ("yes", "y"):
            print("❌ Cancelled")
            sys.exit(0)

    # Delete duplicates
    print("\n🗑️  Deleting duplicates...")
    batch_size = 100
    duplicate_list = list(duplicates)

    for i in range(0, len(duplicate_list), batch_size):
        batch = duplicate_list[i : i + batch_size]
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=batch,
        )
        print(
            f"   Deleted batch {i // batch_size + 1}/{(len(duplicate_list) + batch_size - 1) // batch_size}"
        )

    print()
    print("=" * 60)
    print(f"✅ Deduplication Complete!")
    print(f"   Removed: {len(duplicates)} duplicates")
    print(f"   Remaining: {len(memories) - len(duplicates)} unique memories")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove duplicate memories from Qdrant")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt and delete automatically",
    )

    args = parser.parse_args()
    deduplicate_memories(dry_run=args.dry_run, auto_confirm=args.yes)
