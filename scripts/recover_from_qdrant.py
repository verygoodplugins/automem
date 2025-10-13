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
from falkordb import FalkorDB

# Load environment
load_dotenv()
load_dotenv(Path.home() / ".config" / "automem" / ".env")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "memories")
FALKORDB_HOST = os.getenv("FALKORDB_HOST", "localhost")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))
FALKORDB_PASSWORD = os.getenv("FALKORDB_PASSWORD")
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


def restore_memory_to_graph_only(memory: Dict[str, Any], client) -> bool:
    """Restore a single memory directly to FalkorDB (skip Qdrant to avoid duplicates)."""
    payload = memory["payload"]
    memory_id = memory["id"]
    
    try:
        # Store directly to FalkorDB graph
        g = client.select_graph("memories")
        
        # Build metadata string (exclude reserved fields to prevent overwriting)
        RESERVED_FIELDS = {"type", "confidence", "content", "timestamp", "importance", "tags", "id"}
        metadata_items = []
        metadata_dict = payload.get("metadata", {})
        if metadata_dict:
            for key, value in metadata_dict.items():
                # Skip reserved fields that would overwrite actual memory properties
                if key in RESERVED_FIELDS:
                    continue
                if isinstance(value, (list, dict)):
                    value_str = str(value).replace("'", "\\'")
                else:
                    value_str = str(value).replace("'", "\\'")
                metadata_items.append(f"{key}: '{value_str}'")
        
        metadata_str = ", ".join(metadata_items) if metadata_items else ""
        
        # Build tags string
        tags = payload.get("tags", [])
        tags_str = ", ".join([f"'{tag}'" for tag in tags]) if tags else ""
        
        # Create memory node
        query = f"""
        CREATE (m:Memory {{
            id: '{memory_id}',
            content: $content,
            timestamp: '{payload.get("timestamp", "")}',
            importance: {payload.get("importance", 0.5)},
            type: '{payload.get("type", "Context")}',
            confidence: {payload.get("confidence", 0.6)},
            tags: [{tags_str}]
            {', ' + metadata_str if metadata_str else ''}
        }})
        """
        
        g.query(query, {"content": payload.get("content", "")})
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def main():
    """Main recovery process."""
    print("=" * 60)
    print("🔧 AutoMem Recovery Tool - Rebuild FalkorDB from Qdrant")
    print("=" * 60)
    print()
    
    # Initialize FalkorDB client
    print(f"🔌 Connecting to FalkorDB at {FALKORDB_HOST}:{FALKORDB_PORT}")
    try:
        client = FalkorDB(
            host=FALKORDB_HOST,
            port=FALKORDB_PORT,
            password=FALKORDB_PASSWORD,
            username="default" if FALKORDB_PASSWORD else None
        )
        print("✅ Connected to FalkorDB\n")
    except Exception as e:
        print(f"❌ Failed to connect to FalkorDB: {e}")
        sys.exit(1)
    
    # Clear existing graph
    print("🗑️  Clearing existing graph data...")
    try:
        g = client.select_graph("memories")
        g.query("MATCH (n) DETACH DELETE n")
        print("✅ Graph cleared\n")
    except Exception as e:
        print(f"⚠️  Could not clear graph: {e}\n")
    
    # Fetch all memories from Qdrant
    memories = get_all_memories()
    
    if not memories:
        print("❌ No memories found in Qdrant!")
        sys.exit(1)
    
    # Restore to FalkorDB (skip Qdrant to avoid duplicates)
    print(f"🔄 Restoring {len(memories)} memories to FalkorDB (without duplicating in Qdrant)...")
    print()
    
    success_count = 0
    failed_count = 0
    
    for i, memory in enumerate(memories, 1):
        content_preview = memory["payload"].get("content", "")[:60]
        print(f"[{i}/{len(memories)}] {content_preview}...")
        
        if restore_memory_to_graph_only(memory, client):
            success_count += 1
            print(f"   ✅ Restored")
        else:
            failed_count += 1
        
        # Progress update
        if i % 10 == 0:
            print(f"\n💤 Progress: {success_count} ✅ / {failed_count} ❌\n")
    
    print()
    print("=" * 60)
    print(f"✅ Recovery Complete!")
    print(f"   Success: {success_count}")
    print(f"   Failed:  {failed_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
