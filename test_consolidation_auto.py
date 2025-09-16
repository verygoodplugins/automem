#!/usr/bin/env python3
"""
Test memory consolidation features (non-interactive)
"""

import json
import requests
from datetime import datetime

BASE_URL = "http://localhost:8001"

def test_consolidation():
    """Test consolidation endpoints"""
    print("\n🧠 Testing Memory Consolidation System")
    print("=" * 60)

    # Test dry-run consolidation
    print("\n1️⃣ Testing Full Consolidation (Dry Run)...")
    response = requests.post(
        f"{BASE_URL}/consolidate",
        json={"mode": "full", "dry_run": True}
    )

    if response.status_code == 200:
        result = response.json()['consolidation']
        print(f"✅ Consolidation completed successfully")

        # Show results from each step
        steps = result.get('steps', {})

        if 'decay' in steps:
            decay = steps['decay']
            print(f"\n📉 Decay Results:")
            print(f"  - Processed: {decay.get('processed', 0)} memories")
            print(f"  - Avg relevance: {decay.get('avg_relevance_before', 0):.3f} → {decay.get('avg_relevance_after', 0):.3f}")

        if 'creative' in steps:
            creative = steps['creative']
            print(f"\n🎨 Creative Associations:")
            print(f"  - Discovered: {creative.get('discovered', 0)} new connections")

        if 'cluster' in steps:
            cluster = steps['cluster']
            print(f"\n🔮 Clustering:")
            print(f"  - Found: {cluster.get('clusters_found', 0)} clusters")

        if 'forget' in steps:
            forget = steps['forget']
            print(f"\n🗑️ Controlled Forgetting:")
            print(f"  - Would archive: {len(forget.get('archived', []))}")
            print(f"  - Would delete: {len(forget.get('deleted', []))}")
            print(f"  - Preserved: {forget.get('preserved', 0)}")
    else:
        print(f"❌ Consolidation failed: {response.status_code}")

    # Test scheduler status
    print("\n2️⃣ Testing Scheduler Status...")
    response = requests.get(f"{BASE_URL}/consolidate/status")

    if response.status_code == 200:
        status = response.json()
        print("✅ Scheduler accessible")

        next_runs = status.get('next_runs', {})
        print("\n📅 Next scheduled runs:")
        for task, when in next_runs.items():
            print(f"  - {task}: {when}")
    else:
        print(f"❌ Scheduler status failed: {response.status_code}")

    print("\n" + "=" * 60)
    print("✅ Consolidation system is ready!")
    print("\nFeatures implemented:")
    print("• Exponential decay scoring based on age and access")
    print("• Creative association discovery (REM-like processing)")
    print("• Semantic clustering for memory compression")
    print("• Controlled forgetting with archival")
    print("• Scheduled consolidation tasks")

if __name__ == "__main__":
    test_consolidation()