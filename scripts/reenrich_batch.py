#!/usr/bin/env python3
"""Re-enrich a batch of memories with updated classification logic."""

import os
import sys
from pathlib import Path
from typing import List

import requests
from dotenv import load_dotenv
from falkordb import FalkorDB

# Load environment
load_dotenv()
load_dotenv(Path.home() / ".config" / "automem" / ".env")

FALKORDB_HOST = os.getenv("FALKORDB_HOST", "localhost")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))
FALKORDB_PASSWORD = os.getenv("FALKORDB_PASSWORD")
AUTOMEM_API_URL = os.getenv("AUTOMEM_API_URL", "http://localhost:8001")
API_TOKEN = os.getenv("AUTOMEM_API_TOKEN")
ADMIN_TOKEN = os.getenv("ADMIN_API_TOKEN")


def get_memory_ids(limit: int = 10) -> List[str]:
    """Get memory IDs from FalkorDB."""
    print(f"🔌 Connecting to FalkorDB at {FALKORDB_HOST}:{FALKORDB_PORT}")

    client = FalkorDB(
        host=FALKORDB_HOST,
        port=FALKORDB_PORT,
        password=FALKORDB_PASSWORD,
        username="default" if FALKORDB_PASSWORD else None,
    )

    g = client.select_graph("memories")
    result = g.query(f"MATCH (m:Memory) RETURN m.id LIMIT {limit}")

    ids = [record[0] for record in result.result_set]
    print(f"✅ Found {len(ids)} memories\n")
    return ids


def trigger_reprocess(ids: List[str]) -> None:
    """Trigger re-enrichment for a batch of memory IDs.

    Note: Admin endpoints require BOTH tokens:
    - Authorization: Bearer <AUTOMEM_API_TOKEN> (for general auth)
    - X-Admin-Token: <ADMIN_API_TOKEN> (for admin access)
    """
    if not API_TOKEN:
        print("❌ ERROR: AUTOMEM_API_TOKEN not set")
        sys.exit(1)

    if not ADMIN_TOKEN:
        print("❌ ERROR: ADMIN_API_TOKEN not set")
        sys.exit(1)

    print(f"🔄 Triggering re-enrichment for {len(ids)} memories...")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_TOKEN}",  # Required for all API calls
        "X-Admin-Token": ADMIN_TOKEN,  # Required for admin endpoints
    }

    payload = {"ids": ids}

    response = requests.post(
        f"{AUTOMEM_API_URL}/enrichment/reprocess",
        json=payload,
        headers=headers,
        timeout=30,
    )

    if response.status_code == 202:
        data = response.json()
        print(f"✅ Queued {data['count']} memories for re-enrichment")
        print(f"   IDs: {', '.join(data['ids'][:5])}{'...' if len(data['ids']) > 5 else ''}")
    else:
        print(f"❌ Failed: {response.status_code}")
        print(f"   {response.text}")
        sys.exit(1)


def main():
    """Main process."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Re-enrich memories with updated classification logic"
    )
    parser.add_argument("--limit", type=int, default=10, help="Number of memories to re-enrich")
    args = parser.parse_args()

    print("=" * 60)
    print(f"🔧 AutoMem Re-Enrichment Tool")
    print("=" * 60)
    print()

    # Get memory IDs
    ids = get_memory_ids(limit=args.limit)

    if not ids:
        print("❌ No memories found!")
        sys.exit(1)

    # Trigger reprocess
    trigger_reprocess(ids)

    print()
    print("=" * 60)
    print("✅ Re-enrichment queued!")
    print("   Check /enrichment/status to monitor progress")
    print("=" * 60)


if __name__ == "__main__":
    main()
