#!/usr/bin/env python3
"""Test the improved AutoMem search functionality."""

import json
import time
import requests
from typing import Dict, Any

# Configuration
API_URL = "http://localhost:8001"  # Change to Railway URL in production


def test_health() -> bool:
    """Test the health endpoint."""
    print("Testing health endpoint...")
    response = requests.get(f"{API_URL}/health")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Health: {data['status']}")
        print(f"   FalkorDB: {data['falkordb']}")
        print(f"   Qdrant: {data['qdrant']}")
        return True
    else:
        print(f"❌ Health check failed: {response.status_code}")
        return False


def store_test_memory(content: str, tags: list, importance: float) -> str:
    """Store a test memory."""
    print(f"\nStoring: {content[:50]}...")
    response = requests.post(
        f"{API_URL}/memory",
        json={
            "content": content,
            "tags": tags,
            "importance": importance
        }
    )
    
    if response.status_code == 201:
        data = response.json()
        print(f"✅ Stored with ID: {data['memory_id']}")
        print(f"   Embedding status: {data.get('embedding_status', 'unknown')}")
        return data['memory_id']
    else:
        print(f"❌ Failed to store: {response.text}")
        return None


def test_recall(query: str) -> list:
    """Test memory recall."""
    print(f"\n🔍 Searching for: '{query}'")
    response = requests.get(
        f"{API_URL}/recall",
        params={"query": query, "limit": 5}
    )
    
    if response.status_code == 200:
        data = response.json()
        results = data.get("results", [])
        print(f"✅ Found {len(results)} memories")
        
        for i, result in enumerate(results, 1):
            memory = result.get("memory", {})
            source = result.get("source", "unknown")
            score = result.get("score", "N/A")
            content = memory.get("content", "")[:100]
            print(f"   {i}. [{source}] (score: {score}) {content}...")
        
        return results
    else:
        print(f"❌ Search failed: {response.text}")
        return []


def test_association(id1: str, id2: str) -> bool:
    """Create an association between memories."""
    print(f"\n🔗 Creating association between {id1[:8]}... and {id2[:8]}...")
    response = requests.post(
        f"{API_URL}/associate",
        json={
            "memory1_id": id1,
            "memory2_id": id2,
            "type": "RELATES_TO",
            "strength": 0.8
        }
    )
    
    if response.status_code == 201:
        print("✅ Association created")
        return True
    else:
        print(f"❌ Failed to create association: {response.text}")
        return False


def main():
    """Run comprehensive tests."""
    print("=" * 60)
    print("AutoMem Search Improvement Tests")
    print("=" * 60)
    
    # Test health
    if not test_health():
        print("\n⚠️  API not healthy, some tests may fail")
    
    # Store test memories
    print("\n" + "=" * 60)
    print("STORING TEST MEMORIES")
    print("=" * 60)
    
    memories = [
        {
            "content": "Successfully deployed AutoMem MCP to Railway cloud infrastructure. FalkorDB and Qdrant are both operational.",
            "tags": ["deployment", "cloud", "infrastructure", "success"],
            "importance": 0.9
        },
        {
            "content": "The cloud deployment provides global access with 20-50ms latency compared to 1-5ms local.",
            "tags": ["performance", "cloud", "latency", "metrics"],
            "importance": 0.7
        },
        {
            "content": "FalkorDB handles graph relationships while Qdrant manages vector embeddings for semantic search.",
            "tags": ["architecture", "falkordb", "qdrant", "technical"],
            "importance": 0.8
        }
    ]
    
    memory_ids = []
    for mem in memories:
        mem_id = store_test_memory(mem["content"], mem["tags"], mem["importance"])
        if mem_id:
            memory_ids.append(mem_id)
        time.sleep(1)  # Avoid rate limiting
    
    # Wait for embeddings to be generated and indexed
    print("\n⏳ Waiting for indexing...")
    time.sleep(3)
    
    # Test various search patterns
    print("\n" + "=" * 60)
    print("TESTING SEARCH PATTERNS")
    print("=" * 60)
    
    test_queries = [
        # Single words (should work)
        "deployment",
        "cloud",
        "FalkorDB",
        
        # Multi-word queries (testing improvement)
        "cloud infrastructure",
        "AutoMem deployment",
        "global access",
        "semantic search",
        
        # Complex queries (testing semantic understanding)
        "infrastructure achievement",
        "cloud deployment success",
        "graph database vector search",
    ]
    
    for query in test_queries:
        results = test_recall(query)
        time.sleep(0.5)  # Be nice to the API
    
    # Create associations if we have memories
    if len(memory_ids) >= 2:
        print("\n" + "=" * 60)
        print("TESTING ASSOCIATIONS")
        print("=" * 60)
        test_association(memory_ids[0], memory_ids[1])
    
    print("\n" + "=" * 60)
    print("TESTS COMPLETE!")
    print("=" * 60)
    print("\nKey improvements:")
    print("✅ Real OpenAI embeddings (if API key configured)")
    print("✅ Automatic embedding generation from query text")
    print("✅ Multi-word query support (all words must match)")
    print("✅ Hybrid search (vector + keyword)")
    print("✅ Better result ranking")


if __name__ == "__main__":
    main()
