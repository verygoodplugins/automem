#!/usr/bin/env python3
"""
Demonstration script for AutoMem Personal Knowledge Graph features.
Shows memory classification, pattern detection, preferences, and analytics.
"""

import json
import time
import requests

BASE_URL = "http://localhost:8001"


def create_memory(content, tags=None, **kwargs):
    """Helper to create a memory."""
    payload = {"content": content}
    if tags:
        payload["tags"] = tags
    payload.update(kwargs)

    response = requests.post(f"{BASE_URL}/memory", json=payload)
    data = response.json()
    print(f"✅ Created {data.get('type', 'Memory')} memory: {data.get('memory_id')[:8]}... "
          f"(confidence: {data.get('confidence', 0):.2f})")
    return data


def create_association(mem1_id, mem2_id, rel_type, **props):
    """Helper to create an association."""
    payload = {
        "memory1_id": mem1_id,
        "memory2_id": mem2_id,
        "type": rel_type,
        **props
    }
    response = requests.post(f"{BASE_URL}/associate", json=payload)
    data = response.json()
    print(f"✅ Created {data['relation_type']} relationship")
    return data


def main():
    print("🧠 AutoMem Personal Knowledge Graph Demo")
    print("=" * 50)

    # 1. Decision memories
    print("\n📊 Creating Decision memories...")
    decision1 = create_memory(
        "I decided to use FalkorDB over ArangoDB for cost reasons - $5/mo vs $150/mo",
        ["infrastructure", "database"]
    )
    time.sleep(0.1)

    decision2 = create_memory(
        "Chose Railway over Heroku for deployment",
        ["deployment", "platform"]
    )
    time.sleep(0.1)

    # 2. Pattern memories
    print("\n🔄 Creating Pattern memories...")
    pattern1 = create_memory(
        "I usually code late at night when it's quiet",
        ["habit", "productivity"]
    )
    time.sleep(0.1)

    pattern2 = create_memory(
        "I typically write tests before implementation",
        ["development", "testing"]
    )
    time.sleep(0.1)

    # 3. Preference memories
    print("\n❤️ Creating Preference memories...")
    pref1 = create_memory(
        "I prefer dark mode for all my development tools",
        ["preferences", "ui"]
    )
    time.sleep(0.1)

    pref2 = create_memory(
        "I like Python better than JavaScript for backend development",
        ["language", "backend"]
    )
    time.sleep(0.1)

    # 4. Style memories
    print("\n✍️ Creating Style memories...")
    style1 = create_memory(
        "I write documentation in a concise style with emojis for clarity",
        ["documentation", "style"]
    )
    time.sleep(0.1)

    # 5. Insight memories
    print("\n💡 Creating Insight memories...")
    insight1 = create_memory(
        "I realized that smaller, focused functions are easier to test and maintain",
        ["programming", "insight"]
    )
    time.sleep(0.1)

    # 6. Create preference relationships
    print("\n🔗 Creating PREFERS_OVER relationships...")

    # Create tool memories for preferences
    falkordb = create_memory("FalkorDB", ["database", "tool"])
    arangodb = create_memory("ArangoDB", ["database", "tool"])
    railway = create_memory("Railway", ["deployment", "tool"])
    heroku = create_memory("Heroku", ["deployment", "tool"])

    time.sleep(0.1)

    # Create preference relationships
    create_association(
        falkordb["memory_id"],
        arangodb["memory_id"],
        "PREFERS_OVER",
        strength=0.95,
        context="cost-effectiveness",
        reason="30x cost difference"
    )

    create_association(
        railway["memory_id"],
        heroku["memory_id"],
        "PREFERS_OVER",
        strength=0.8,
        context="ease-of-use",
        reason="simpler deployment process"
    )

    # 7. Create temporal validity example
    print("\n⏰ Creating temporally-bounded memory...")
    old_pref = create_memory(
        "Used AWS for all deployments",
        ["deployment", "historical"],
        t_valid="2020-01-01T00:00:00Z",
        t_invalid="2023-01-01T00:00:00Z"
    )

    # Wait for enrichment to process
    print("\n⏳ Waiting for enrichment pipeline to process...")
    time.sleep(3)

    # 8. Get analytics
    print("\n📈 Fetching analytics...")
    response = requests.get(f"{BASE_URL}/analyze")
    if response.status_code == 200:
        analytics = response.json()["analytics"]

        print("\n📊 Memory Type Distribution:")
        for mem_type, stats in analytics["memory_types"].items():
            print(f"  - {mem_type}: {stats['count']} memories "
                  f"(avg confidence: {stats['average_confidence']:.2f})")

        if analytics["patterns"]:
            print("\n🔄 Detected Patterns:")
            for pattern in analytics["patterns"]:
                print(f"  - {pattern['description']} "
                      f"(confidence: {pattern['confidence']:.2f})")

        if analytics["preferences"]:
            print("\n❤️ Preferences:")
            for pref in analytics["preferences"]:
                print(f"  - Prefers '{pref['prefers']}' over '{pref['over']}' "
                      f"in context: {pref.get('context', 'general')}")

        if analytics["confidence_distribution"]:
            print("\n🎯 Confidence Distribution:")
            for level, count in analytics["confidence_distribution"].items():
                print(f"  - {level}: {count} memories")

        if analytics["entity_frequency"].get("tools"):
            print("\n🔧 Most Mentioned Tools:")
            for tool, count in analytics["entity_frequency"]["tools"]:
                print(f"  - {tool}: {count} mentions")

    # 9. Test recall with new fields
    print("\n🔍 Testing recall...")
    response = requests.get(f"{BASE_URL}/recall", params={"query": "decided", "limit": 5})
    if response.status_code == 200:
        results = response.json()["results"]
        print(f"Found {len(results)} decision-related memories")

    print("\n✨ Demo complete! The PKG features are working:")
    print("  ✓ Automatic memory classification")
    print("  ✓ Confidence scoring")
    print("  ✓ Enhanced relationships with properties")
    print("  ✓ Temporal validity tracking")
    print("  ✓ Pattern detection (via enrichment)")
    print("  ✓ Analytics endpoint")
    print("  ✓ Entity extraction")


if __name__ == "__main__":
    try:
        # Check if service is running
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            main()
        else:
            print("❌ AutoMem service not responding correctly")
    except requests.ConnectionError:
        print("❌ AutoMem service is not running. Start it with 'make dev'")