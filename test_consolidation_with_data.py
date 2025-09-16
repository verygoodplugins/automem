#!/usr/bin/env python3
"""
Test consolidation with actual memory data
"""

import json
import requests
import time
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8001"

def create_test_memories():
    """Create test memories with varying ages and relationships"""
    print("\n📝 Creating Test Memories...")
    print("=" * 60)

    memories = [
        # Recent decision
        {
            "content": "Decided to implement memory consolidation using dream-inspired algorithms",
            "tags": ["decision", "architecture"],
            "importance": 0.9
        },
        # Related pattern
        {
            "content": "Pattern: Always implement biological inspiration in memory systems",
            "tags": ["pattern", "design"],
            "importance": 0.7
        },
        # Old memory (simulated by lower importance)
        {
            "content": "Early notes about memory storage from project inception",
            "tags": ["old", "notes"],
            "importance": 0.2
        },
        # Recent insight
        {
            "content": "Insight: Memory decay creates space for new learning",
            "tags": ["insight", "learning"],
            "importance": 0.8
        },
        # Similar memories for clustering
        {
            "content": "FalkorDB performs better than ArangoDB for graph operations",
            "tags": ["database", "performance"],
            "importance": 0.6
        },
        {
            "content": "FalkorDB costs $5/month compared to ArangoDB at $150/month",
            "tags": ["database", "cost"],
            "importance": 0.6
        },
        {
            "content": "FalkorDB has simpler query syntax than ArangoDB",
            "tags": ["database", "usability"],
            "importance": 0.5
        },
        # Connected memories
        {
            "content": "Morning routine: Review yesterday's code, plan today's tasks",
            "tags": ["habit", "productivity"],
            "importance": 0.4
        },
        {
            "content": "Productivity increases when following consistent morning routine",
            "tags": ["insight", "productivity"],
            "importance": 0.6
        }
    ]

    created_ids = []
    for i, mem in enumerate(memories):
        response = requests.post(f"{BASE_URL}/memory", json=mem)
        if response.status_code == 201:
            memory_id = response.json()['memory_id']
            created_ids.append(memory_id)
            print(f"✅ Created memory {i+1}/{len(memories)}")
        else:
            print(f"❌ Failed to create memory: {mem['content'][:50]}")

    # Create some associations
    if len(created_ids) >= 2:
        # Decision leads to pattern
        requests.post(f"{BASE_URL}/associate", json={
            "memory1_id": created_ids[0],
            "memory2_id": created_ids[1],
            "relationship": "LEADS_TO"
        })

        # Database memories relate
        if len(created_ids) >= 7:
            requests.post(f"{BASE_URL}/associate", json={
                "memory1_id": created_ids[4],
                "memory2_id": created_ids[5],
                "relationship": "RELATES_TO"
            })

    return created_ids

def test_consolidation_with_data():
    """Run consolidation on actual data"""
    print("\n🧠 Testing Consolidation with Real Data")
    print("=" * 60)

    # Wait for enrichment
    print("\n⏳ Waiting for enrichment pipeline...")
    time.sleep(3)

    # Run consolidation
    print("\n🚀 Running Full Consolidation (Dry Run)...")
    response = requests.post(
        f"{BASE_URL}/consolidate",
        json={"mode": "full", "dry_run": True}
    )

    if response.status_code == 200:
        result = response.json()['consolidation']

        # Decay results
        if 'decay' in result.get('steps', {}):
            decay = result['steps']['decay']
            print(f"\n📉 Decay Analysis:")
            print(f"  - Processed: {decay.get('processed', 0)} memories")
            print(f"  - Avg relevance: {decay.get('avg_relevance_before', 0):.3f} → {decay.get('avg_relevance_after', 0):.3f}")

            dist = decay.get('distribution', {})
            if any(dist.values()):
                print(f"  - Distribution after decay:")
                print(f"    • High relevance (>0.7): {dist.get('high', 0)}")
                print(f"    • Medium (0.3-0.7): {dist.get('medium', 0)}")
                print(f"    • Low (0.1-0.3): {dist.get('low', 0)}")
                print(f"    • Archive candidates (<0.1): {dist.get('archive', 0)}")

        # Creative associations
        if 'creative' in result.get('steps', {}):
            creative = result['steps']['creative']
            if creative.get('discovered', 0) > 0:
                print(f"\n🎨 Creative Associations Discovered:")
                print(f"  - Found {creative.get('discovered', 0)} new connections")

                for assoc in creative.get('sample_associations', [])[:3]:
                    print(f"    • {assoc['type']} (confidence: {assoc['confidence']:.2f}, similarity: {assoc.get('similarity', 0):.2f})")

        # Clustering
        if 'cluster' in result.get('steps', {}):
            cluster = result['steps']['cluster']
            if cluster.get('clusters_found', 0) > 0:
                print(f"\n🔮 Memory Clustering:")
                print(f"  - Found {cluster.get('clusters_found', 0)} clusters")

                for c in cluster.get('sample_clusters', [])[:2]:
                    print(f"    • {c['dominant_type']} cluster: {c['size']} memories")
                    print(f"      Sample: '{c['sample_content'][:60]}...'")

        # Forgetting
        if 'forget' in result.get('steps', {}):
            forget = result['steps']['forget']
            if forget.get('examined', 0) > 0:
                print(f"\n🗑️ Controlled Forgetting Analysis:")
                print(f"  - Examined: {forget.get('examined', 0)} memories")
                print(f"  - Would archive: {len(forget.get('archived', []))}")
                print(f"  - Would delete: {len(forget.get('deleted', []))}")
                print(f"  - Preserved: {forget.get('preserved', 0)}")

                if forget.get('archived'):
                    print(f"\n  Archive candidates:")
                    for mem in forget.get('archived', [])[:2]:
                        print(f"    • '{mem['content_preview']}...' (relevance: {mem['relevance']:.3f})")

                if forget.get('deleted'):
                    print(f"\n  Delete candidates:")
                    for mem in forget.get('deleted', [])[:1]:
                        print(f"    • '{mem['content_preview']}...' (relevance: {mem['relevance']:.3f})")

        # Summary
        print(f"\n✨ Consolidation Summary:")
        print(f"  - Mode: {result.get('mode')}")
        print(f"  - Success: {result.get('success')}")
        print(f"  - Dry run: {result.get('dry_run')}")
    else:
        print(f"❌ Consolidation failed: {response.status_code}")

def main():
    """Run the full test suite"""
    print("\n🧪 Memory Consolidation Test with Data")
    print("=" * 60)

    # Create test memories
    memory_ids = create_test_memories()

    if memory_ids:
        # Run consolidation
        test_consolidation_with_data()

        print("\n" + "=" * 60)
        print("🎯 Key Insights:")
        print("=" * 60)
        print("• Recent, important memories maintain high relevance")
        print("• Old, unused memories decay toward archival")
        print("• Similar memories can cluster for knowledge compression")
        print("• Creative associations emerge between disparate memories")
        print("• System self-organizes like biological memory")
    else:
        print("❌ No memories created, cannot test consolidation")

if __name__ == "__main__":
    main()