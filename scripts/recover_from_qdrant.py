#!/usr/bin/env python3
"""Recover FalkorDB graph from Qdrant after data loss.

This script reads all memories from Qdrant and re-inserts them into FalkorDB
using the AutoMem API, which will rebuild all graph relationships.
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load environment
load_dotenv()
load_dotenv(Path.home() / ".config" / "automem" / ".env")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "memories")
AUTOMEM_API_URL = os.getenv("AUTOMEM_API_URL") or os.getenv("MCP_MEMORY_HTTP_ENDPOINT", "http://localhost:8001")
API_TOKEN = os.getenv("AUTOMEM_API_TOKEN")
BATCH_SIZE = 50


def get_all_memories() -> List[Dict[str, Any]]:
    """Fetch all memories from Qdrant."""
    print(f"🔍 Connecting to Qdrant at {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    memories = []
    offset = None
    
    while True:
        print(f"📥 Fetching batch (offset: {offset})...")
        result = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=BATCH_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        
        points, next_offset = result
        
        if not points:
            break
            
        for point in points:
            memory = {
                "id": point.id,
                "payload": point.payload,
                "vector": point.vector,
            }
            memories.append(memory)
        
        print(f"   Got {len(points)} memories (total: {len(memories)})")
        
        if next_offset is None:
            break
            
        offset = next_offset
        time.sleep(0.1)  # Rate limiting
    
    print(f"✅ Fetched {len(memories)} total memories from Qdrant\n")
    return memories


def restore_memory(memory: Dict[str, Any]) -> bool:
    """Restore a single memory to FalkorDB via API."""
    payload = memory["payload"]
    
    # Build request payload for /store endpoint
    store_request = {
        "content": payload.get("content", ""),
        "tags": payload.get("tags", []),
        "importance": payload.get("importance"),
        "metadata": payload.get("metadata", {}),
        "timestamp": payload.get("timestamp"),
        "embedding": memory.get("vector"),  # Use existing embedding
    }
    
    # Remove None values
    store_request = {k: v for k, v in store_request.items() if v is not None}
    
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
    }
    
    try:
        response = requests.post(
            f"{AUTOMEM_API_URL}/memory",
            json=store_request,
            headers=headers,
            timeout=30,
        )
        
        if response.status_code in (200, 201):
            return True
        else:
            print(f"   ⚠️  Failed: {response.status_code} - {response.text[:100]}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def main():
    """Main recovery process."""
    print("=" * 60)
    print("🔧 AutoMem Recovery Tool - Rebuild FalkorDB from Qdrant")
    print("=" * 60)
    print()
    
    if not API_TOKEN:
        print("❌ ERROR: AUTOMEM_API_TOKEN not set")
        sys.exit(1)
    
    # Fetch all memories from Qdrant
    memories = get_all_memories()
    
    if not memories:
        print("❌ No memories found in Qdrant!")
        sys.exit(1)
    
    # Restore to FalkorDB
    print(f"🔄 Restoring {len(memories)} memories to FalkorDB...")
    print()
    
    success_count = 0
    failed_count = 0
    
    for i, memory in enumerate(memories, 1):
        content_preview = memory["payload"].get("content", "")[:60]
        print(f"[{i}/{len(memories)}] {content_preview}...")
        
        if restore_memory(memory):
            success_count += 1
            print(f"   ✅ Restored")
        else:
            failed_count += 1
        
        # Rate limiting
        if i % 10 == 0:
            print(f"\n💤 Progress: {success_count} ✅ / {failed_count} ❌\n")
            time.sleep(1)
    
    print()
    print("=" * 60)
    print(f"✅ Recovery Complete!")
    print(f"   Success: {success_count}")
    print(f"   Failed:  {failed_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
