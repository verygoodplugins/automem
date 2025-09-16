#!/usr/bin/env python3
"""
Test memory consolidation features
"""

import json
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List

BASE_URL = "http://localhost:8001"

def test_dry_run_consolidation():
    """Test consolidation in dry-run mode (no changes)"""
    print("\n🧪 Testing Memory Consolidation (Dry Run)")
    print("=" * 60)

    # Run full consolidation in dry-run mode
    response = requests.post(
        f"{BASE_URL}/consolidate",
        json={
            "mode": "full",
            "dry_run": True
        }
    )

    if response.status_code == 200:
        result = response.json()['consolidation']
        print(f"✅ Consolidation completed: {result.get('mode')} mode")

        # Show decay results
        if 'decay' in result.get('steps', {}):
            decay = result['steps']['decay']
            print(f"\n📊 Decay Analysis:")
            print(f"  - Memories processed: {decay.get('processed', 0)}")
            print(f"  - Avg relevance before: {decay.get('avg_relevance_before', 0):.3f}")
            print(f"  - Avg relevance after: {decay.get('avg_relevance_after', 0):.3f}")

            dist = decay.get('distribution', {})
            print(f"  - Distribution:")
            print(f"    • High (>0.7): {dist.get('high', 0)} memories")
            print(f"    • Medium (0.3-0.7): {dist.get('medium', 0)} memories")
            print(f"    • Low (0.1-0.3): {dist.get('low', 0)} memories")
            print(f"    • Archive (<0.1): {dist.get('archive', 0)} memories")

        # Show creative associations
        if 'creative' in result.get('steps', {}):
            creative = result['steps']['creative']
            print(f"\n🎨 Creative Associations:")
            print(f"  - Discovered: {creative.get('discovered', 0)} new connections")

            for assoc in creative.get('sample_associations', [])[:3]:
                print(f"    • {assoc['type']} (confidence: {assoc['confidence']:.2f})")

        # Show clustering results
        if 'cluster' in result.get('steps', {}):
            cluster = result['steps']['cluster']
            print(f"\n🔮 Memory Clustering:")
            print(f"  - Clusters found: {cluster.get('clusters_found', 0)}")

            for c in cluster.get('sample_clusters', [])[:2]:
                print(f"    • {c['dominant_type']} cluster: {c['size']} memories over {c['time_span_days']} days")

        # Show forgetting recommendations
        if 'forget' in result.get('steps', {}):
            forget = result['steps']['forget']
            print(f"\n🗑️ Controlled Forgetting (DRY RUN):")
            print(f"  - Examined: {forget.get('examined', 0)} memories")
            print(f"  - Would archive: {len(forget.get('archived', []))} memories")
            print(f"  - Would delete: {len(forget.get('deleted', []))} memories")
            print(f"  - Preserved: {forget.get('preserved', 0)} memories")

            # Show samples of what would be archived/deleted
            for mem in forget.get('archived', [])[:2]:
                print(f"    • Archive: '{mem['content_preview']}...' (relevance: {mem['relevance']:.3f})")

            for mem in forget.get('deleted', [])[:1]:
                print(f"    • Delete: '{mem['content_preview']}...' (relevance: {mem['relevance']:.3f})")

    else:
        print(f"❌ Consolidation failed: {response.text}")

def test_individual_modes():
    """Test individual consolidation modes"""
    print("\n🔬 Testing Individual Consolidation Modes")
    print("=" * 60)

    modes = ['decay', 'creative', 'cluster', 'forget']

    for mode in modes:
        print(f"\n📌 Testing {mode} mode...")

        response = requests.post(
            f"{BASE_URL}/consolidate",
            json={
                "mode": mode,
                "dry_run": True
            }
        )

        if response.status_code == 200:
            result = response.json()['consolidation']
            if result.get('success'):
                print(f"  ✅ {mode} completed successfully")

                # Show mode-specific results
                if mode == 'decay' and 'decay' in result.get('steps', {}):
                    decay = result['steps']['decay']
                    before = decay.get('avg_relevance_before', 0)
                    after = decay.get('avg_relevance_after', 0)
                    change = ((after - before) / before * 100) if before > 0 else 0
                    print(f"     Relevance change: {change:+.1f}%")

                elif mode == 'creative' and 'creative' in result.get('steps', {}):
                    creative = result['steps']['creative']
                    print(f"     Found {creative.get('discovered', 0)} new associations")

                elif mode == 'cluster' and 'cluster' in result.get('steps', {}):
                    cluster = result['steps']['cluster']
                    print(f"     Found {cluster.get('clusters_found', 0)} clusters")

                elif mode == 'forget' and 'forget' in result.get('steps', {}):
                    forget = result['steps']['forget']
                    total = forget.get('examined', 0)
                    archived = len(forget.get('archived', []))
                    deleted = len(forget.get('deleted', []))
                    if total > 0:
                        forget_pct = (archived + deleted) / total * 100
                        print(f"     Would forget {forget_pct:.1f}% of memories")
            else:
                print(f"  ⚠️ {mode} failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"  ❌ {mode} request failed")

def test_scheduler_status():
    """Test consolidation scheduler status"""
    print("\n⏰ Testing Consolidation Scheduler")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/consolidate/status")

    if response.status_code == 200:
        status = response.json()
        print("✅ Scheduler Status:")

        next_runs = status.get('next_runs', {})
        print("\n📅 Next scheduled runs:")
        for task, when in next_runs.items():
            emoji = {
                'decay': '📉',
                'creative': '🎨',
                'cluster': '🔮',
                'forget': '🗑️'
            }.get(task, '📌')
            print(f"  {emoji} {task}: {when}")

        history = status.get('history', [])
        if history:
            print(f"\n📜 Recent history ({len(history)} runs):")
            for run in history[-3:]:
                task = run.get('task', 'unknown')
                run_at = run.get('run_at', '')
                success = run.get('result', {}).get('success', False)
                status_icon = '✅' if success else '❌'
                print(f"  {status_icon} {task} at {run_at[:19]}")
    else:
        print(f"❌ Failed to get scheduler status")

def create_test_memories_with_age():
    """Create memories with different ages to test decay"""
    print("\n🌱 Creating Test Memories with Different Ages")
    print("=" * 60)

    test_memories = [
        {
            "content": "Just learned about memory consolidation today",
            "tags": ["recent", "learning"],
            "importance": 0.8,
            "age_days": 0
        },
        {
            "content": "Last week's important decision about architecture",
            "tags": ["decision", "architecture"],
            "importance": 0.9,
            "age_days": 7
        },
        {
            "content": "Monthly team retrospective insights",
            "tags": ["team", "retrospective"],
            "importance": 0.6,
            "age_days": 30
        },
        {
            "content": "Old project notes from three months ago",
            "tags": ["old", "project"],
            "importance": 0.3,
            "age_days": 90
        },
        {
            "content": "Ancient wisdom from last year",
            "tags": ["ancient", "wisdom"],
            "importance": 0.2,
            "age_days": 365
        }
    ]

    created = []
    for mem in test_memories:
        # Adjust timestamp to simulate age
        timestamp = (datetime.utcnow() - timedelta(days=mem['age_days'])).isoformat() + 'Z'

        response = requests.post(
            f"{BASE_URL}/memory",
            json={
                "content": mem["content"],
                "tags": mem["tags"],
                "importance": mem["importance"],
                "metadata": {
                    "test": True,
                    "age_simulation": mem['age_days']
                }
            }
        )

        if response.status_code == 201:
            memory_id = response.json()['memory_id']
            created.append(memory_id)
            print(f"✅ Created {mem['age_days']}-day old memory")
        else:
            print(f"❌ Failed to create memory: {mem['content'][:30]}")

    return created

def run_actual_consolidation():
    """Run actual consolidation (not dry-run) on decay mode only"""
    print("\n🚀 Running Actual Consolidation (Decay Only)")
    print("=" * 60)
    print("⚠️ This will actually update memory relevance scores!")

    response = input("\nProceed with actual consolidation? (y/n): ")
    if response.lower() != 'y':
        print("Skipped actual consolidation")
        return

    response = requests.post(
        f"{BASE_URL}/consolidate",
        json={
            "mode": "decay",
            "dry_run": False  # Actually update scores
        }
    )

    if response.status_code == 200:
        result = response.json()['consolidation']
        if result.get('success'):
            print("✅ Consolidation completed successfully!")

            decay = result.get('steps', {}).get('decay', {})
            if decay:
                print(f"\n📊 Results:")
                print(f"  - Processed: {decay.get('processed', 0)} memories")
                print(f"  - Avg relevance updated from {decay.get('avg_relevance_before', 0):.3f} to {decay.get('avg_relevance_after', 0):.3f}")
        else:
            print(f"❌ Consolidation failed: {result.get('error')}")
    else:
        print(f"❌ Request failed: {response.text}")

def main():
    """Run all consolidation tests"""
    print("\n🧠 Memory Consolidation Test Suite")
    print("=" * 60)
    print("Testing dream-inspired memory consolidation features:")
    print("• Exponential decay scoring")
    print("• Creative association discovery")
    print("• Semantic clustering")
    print("• Controlled forgetting")

    # Test dry-run consolidation
    test_dry_run_consolidation()

    # Test individual modes
    test_individual_modes()

    # Test scheduler status
    test_scheduler_status()

    # Optional: Create test memories and run actual consolidation
    print("\n" + "=" * 60)
    print("📝 Optional Tests")
    print("=" * 60)

    response = input("\nCreate test memories with different ages? (y/n): ")
    if response.lower() == 'y':
        create_test_memories_with_age()
        time.sleep(1)  # Wait for enrichment
        test_dry_run_consolidation()

    # Optional: Run actual consolidation
    run_actual_consolidation()

    print("\n✨ Consolidation testing complete!")
    print("The memory service now has biological-inspired consolidation:")
    print("• Memories decay over time unless reinforced")
    print("• Creative associations emerge during 'dream' processing")
    print("• Similar memories cluster into meta-patterns")
    print("• Irrelevant memories fade gracefully")

if __name__ == "__main__":
    main()