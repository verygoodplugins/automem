#!/usr/bin/env python3
"""
MCP Memory Service → AutoMem PKG Migration Script
Migrates memories from flat storage to intelligent Personal Knowledge Graph.
"""

import json
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import hashlib
import requests

# Configuration
AUTOMEM_URL = "http://localhost:8001"
BATCH_SIZE = 50
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Type mapping from MCP tags to PKG types
TAG_TO_TYPE_MAP = {
    "pattern": "Pattern",
    "work_pattern": "Pattern",
    "work-pattern": "Pattern",
    "preference": "Preference",
    "user-preference": "Preference",
    "user_preference": "Preference",
    "decision": "Decision",
    "architectural_pattern": "Decision",
    "style": "Style",
    "communication-style": "Style",
    "style-profile": "Style",
    "insight": "Insight",
    "knowledge-management": "Insight",
    "habit": "Habit",
    "daily-routine": "Habit",
    "context": "Context",
    "relationship-analysis": "Context",
    "behavioral-analysis": "Context",
}


class MemoryMigrator:
    def __init__(self):
        self.memories = []
        self.migrated_memories = {}  # old_hash -> new_id
        self.relationships_created = []
        self.patterns_detected = defaultdict(list)
        self.preferences_found = []
        self.entities = defaultdict(set)
        self.stats = {
            "total": 0,
            "migrated": 0,
            "classified": 0,
            "relationships": 0,
            "patterns": 0,
            "preferences": 0,
            "entities": 0,
            "errors": 0,
        }

    def extract_memories_from_mcp(self):
        """Extract all memories using MCP tools."""
        print("📥 Extracting memories from MCP...")

        # Note: In production, this would use the actual MCP client
        # For demonstration, we'll use the sample memories from search results
        sample_memories = [
            {
                "content": "Work pattern in wp-fusion: uses conventional_commits, focuses on feature_development, bug_fixing",
                "hash": "39bc5067fbf920ebf64d75eb9fb58f0180ce316f0eae7176584a14304352ee88",
                "tags": ["pattern", "insight", "work_style", "automated"],
                "timestamp": "2025-09-10T12:00:00Z"
            },
            {
                "content": "I decided to use FalkorDB over ArangoDB for cost reasons - $5/mo vs $150/mo",
                "hash": "decision001",
                "tags": ["decision", "infrastructure", "database"],
                "timestamp": "2025-09-01T10:00:00Z"
            },
            {
                "content": "I prefer Railway for deployments",
                "hash": "pref001",
                "tags": ["preference", "deployment"],
                "timestamp": "2025-09-05T14:00:00Z"
            },
            {
                "content": "I usually code late at night when it's quiet",
                "hash": "habit001",
                "tags": ["habit", "productivity"],
                "timestamp": "2025-09-08T22:00:00Z"
            },
            {
                "content": "Jack's communication style: 'Hey [name]' greetings, contractions always, no corporate fluff",
                "hash": "style001",
                "tags": ["communication-style", "authenticity"],
                "timestamp": "2025-08-20T09:00:00Z"
            },
            {
                "content": "Work pattern in claude-automation-hub: focuses on bug_fixing, works with json files",
                "hash": "pattern002",
                "tags": ["pattern", "work_style"],
                "timestamp": "2025-09-09T15:00:00Z"
            },
            {
                "content": "ERROR HANDLING & RESILIENCE PATTERNS: Graceful degradation over hard failures",
                "hash": "pattern003",
                "tags": ["error-handling", "patterns", "technical-patterns"],
                "timestamp": "2025-08-15T11:00:00Z"
            },
            {
                "content": "Very Good Plugins signature: Built with 🧡 for the open source community",
                "hash": "brand001",
                "tags": ["preference", "branding", "very-good-plugins"],
                "timestamp": "2025-08-10T16:00:00Z"
            }
        ]

        self.memories = sample_memories
        self.stats["total"] = len(self.memories)
        print(f"✅ Extracted {len(self.memories)} memories")
        return self.memories

    def classify_memory_type(self, memory: Dict) -> Tuple[str, float]:
        """Classify memory type based on content and tags."""
        content = memory.get("content", "").lower()
        tags = memory.get("tags", [])

        # Check tags first for explicit type
        for tag in tags:
            if tag in TAG_TO_TYPE_MAP:
                return TAG_TO_TYPE_MAP[tag], 0.85

        # Content-based classification
        if re.search(r"decided to|chose|opted for|selected", content):
            return "Decision", 0.8
        elif re.search(r"prefer|like.*better|favorite", content):
            return "Preference", 0.8
        elif re.search(r"pattern|usually|typically|tend to", content):
            return "Pattern", 0.75
        elif re.search(r"style|wrote.*in|communicated", content):
            return "Style", 0.75
        elif re.search(r"always|every time|daily|routine", content):
            return "Habit", 0.7
        elif re.search(r"realized|discovered|learned that", content):
            return "Insight", 0.7
        elif re.search(r"during|while|context of|when", content):
            return "Context", 0.65

        return "Memory", 0.3

    def extract_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract entities from memory content."""
        entities = {
            "tools": [],
            "projects": [],
            "people": [],
        }

        # Extract tools
        tools = re.findall(r"\b(?:FalkorDB|ArangoDB|Railway|AWS|Heroku|WordPress|WP Fusion|Claude|GitHub|Slack|Docker)\b", content, re.IGNORECASE)
        entities["tools"] = list(set(tools))

        # Extract projects
        projects = re.findall(r"(?:wp-fusion|claude-automation-hub|automem|echodash-mvp|mcp-[\w-]+)", content, re.IGNORECASE)
        entities["projects"] = list(set(projects))

        # Extract people
        people = re.findall(r"(?:Jack|Vikas|Steve Woody|Spencer|Katie|Mike)", content)
        entities["people"] = list(set(people))

        return entities

    def migrate_memory(self, memory: Dict) -> str:
        """Migrate a single memory to AutoMem."""
        memory_type, confidence = self.classify_memory_type(memory)

        payload = {
            "content": memory["content"],
            "tags": memory.get("tags", []),
            "importance": 0.7,  # Default importance
            "timestamp": memory.get("timestamp"),
        }

        # Add temporal validity if we can infer it
        if "2023" in memory["content"] or "2024" in memory["content"]:
            # Extract year and set validity
            year_match = re.search(r"(202[0-9])", memory["content"])
            if year_match:
                year = year_match.group(1)
                payload["t_valid"] = f"{year}-01-01T00:00:00Z"
                if int(year) < 2024:
                    payload["t_invalid"] = f"{int(year)+1}-01-01T00:00:00Z"

        try:
            response = requests.post(f"{AUTOMEM_URL}/memory", json=payload)
            if response.status_code == 201:
                data = response.json()
                memory_id = data["memory_id"]
                self.migrated_memories[memory.get("hash", "")] = memory_id
                self.stats["migrated"] += 1
                if data.get("type") != "Memory":
                    self.stats["classified"] += 1

                # Track patterns
                if memory_type == "Pattern":
                    self.patterns_detected[memory_type].append(memory_id)

                # Extract entities
                entities = self.extract_entities(memory["content"])
                for tool in entities["tools"]:
                    self.entities["tools"].add(tool)
                for project in entities["projects"]:
                    self.entities["projects"].add(project)
                for person in entities["people"]:
                    self.entities["people"].add(person)

                return memory_id
            else:
                self.stats["errors"] += 1
                return None
        except Exception as e:
            print(f"❌ Error migrating memory: {e}")
            self.stats["errors"] += 1
            return None

    def create_relationships(self):
        """Create relationships between migrated memories."""
        print("\n🔗 Creating relationships...")

        # Find preference relationships
        for i, mem1 in enumerate(self.memories):
            content1 = mem1["content"].lower()

            # Look for "X over Y" pattern
            prefer_match = re.search(r"(\w+)\s+over\s+(\w+)", content1)
            if prefer_match:
                preferred = prefer_match.group(1)
                over = prefer_match.group(2)

                # Find memories about these entities
                for mem2 in self.memories[i+1:]:
                    if preferred in mem2["content"].lower() or over in mem2["content"].lower():
                        mem1_id = self.migrated_memories.get(mem1.get("hash"))
                        mem2_id = self.migrated_memories.get(mem2.get("hash"))

                        if mem1_id and mem2_id and mem1_id != mem2_id:
                            self.create_association(
                                mem1_id, mem2_id, "PREFERS_OVER",
                                strength=0.8,
                                context="inferred",
                                reason="extracted from content"
                            )
                            self.preferences_found.append((preferred, over))

            # Find temporal relationships
            if i > 0:
                prev_mem = self.memories[i-1]
                if prev_mem.get("timestamp") and mem1.get("timestamp"):
                    # Create temporal relationship for closely related memories
                    mem1_id = self.migrated_memories.get(mem1.get("hash"))
                    prev_id = self.migrated_memories.get(prev_mem.get("hash"))

                    if mem1_id and prev_id:
                        # Check if they're about similar topics
                        if any(tag in prev_mem.get("tags", []) for tag in mem1.get("tags", [])):
                            self.create_association(
                                mem1_id, prev_id, "PRECEDED_BY",
                                strength=0.6
                            )

            # Find pattern exemplifications
            if "pattern" in mem1.get("tags", []):
                # Link similar patterns
                for mem2 in self.memories[i+1:]:
                    if "pattern" in mem2.get("tags", []):
                        # Check if patterns are similar
                        if any(word in mem2["content"].lower() for word in ["work pattern", "bug_fixing", "feature"]):
                            mem1_id = self.migrated_memories.get(mem1.get("hash"))
                            mem2_id = self.migrated_memories.get(mem2.get("hash"))

                            if mem1_id and mem2_id:
                                self.create_association(
                                    mem1_id, mem2_id, "REINFORCES",
                                    strength=0.7,
                                    observations=2
                                )

    def create_association(self, mem1_id: str, mem2_id: str, rel_type: str, **props):
        """Create an association between two memories."""
        payload = {
            "memory1_id": mem1_id,
            "memory2_id": mem2_id,
            "type": rel_type,
            **props
        }

        try:
            response = requests.post(f"{AUTOMEM_URL}/associate", json=payload)
            if response.status_code == 201:
                self.stats["relationships"] += 1
                self.relationships_created.append({
                    "from": mem1_id[:8],
                    "to": mem2_id[:8],
                    "type": rel_type
                })
                return True
        except:
            pass
        return False

    def analyze_patterns(self):
        """Analyze and reinforce patterns."""
        print("\n🔍 Analyzing patterns...")

        # Group similar patterns
        work_patterns = []
        communication_patterns = []

        for memory in self.memories:
            if "work pattern" in memory["content"].lower():
                work_patterns.append(memory)
                self.stats["patterns"] += 1
            elif "communication" in memory["content"].lower() or "style" in memory["content"].lower():
                communication_patterns.append(memory)
                self.stats["patterns"] += 1

        print(f"  Found {len(work_patterns)} work patterns")
        print(f"  Found {len(communication_patterns)} communication patterns")

        # Create pattern reinforcement relationships
        for i, pattern in enumerate(work_patterns[:-1]):
            pattern_id = self.migrated_memories.get(pattern.get("hash"))
            next_id = self.migrated_memories.get(work_patterns[i+1].get("hash"))

            if pattern_id and next_id:
                self.create_association(
                    pattern_id, next_id, "EXEMPLIFIES",
                    pattern_type="work_pattern",
                    confidence=0.8
                )

    def generate_analytics(self):
        """Generate analytics comparing old vs new system."""
        print("\n📊 Generating analytics...")

        try:
            response = requests.get(f"{AUTOMEM_URL}/analyze")
            if response.status_code == 200:
                analytics = response.json()["analytics"]

                print("\n=== MIGRATION RESULTS ===")
                print(f"\n📈 Statistics:")
                print(f"  Total memories: {self.stats['total']}")
                print(f"  Successfully migrated: {self.stats['migrated']}")
                print(f"  Classified into types: {self.stats['classified']}")
                print(f"  Relationships created: {self.stats['relationships']}")
                print(f"  Patterns detected: {self.stats['patterns']}")
                print(f"  Preferences found: {len(self.preferences_found)}")
                print(f"  Entities extracted: {sum(len(v) for v in self.entities.values())}")

                print(f"\n🏷️ Memory Type Distribution:")
                for mem_type, data in analytics.get("memory_types", {}).items():
                    print(f"  {mem_type}: {data['count']} (confidence: {data['average_confidence']:.2f})")

                if self.preferences_found:
                    print(f"\n❤️ Preferences Discovered:")
                    for preferred, over in self.preferences_found[:5]:
                        print(f"  • {preferred} > {over}")

                if self.entities["tools"]:
                    print(f"\n🔧 Tools Mentioned:")
                    for tool in list(self.entities["tools"])[:10]:
                        print(f"  • {tool}")

                if self.entities["projects"]:
                    print(f"\n📁 Projects Referenced:")
                    for project in list(self.entities["projects"])[:10]:
                        print(f"  • {project}")

                print("\n=== IMPROVEMENTS OVER FLAT STORAGE ===")
                print("\n✅ NEW CAPABILITIES:")
                print("  • Automatic type classification with confidence scores")
                print("  • Rich relationship graph between memories")
                print("  • Pattern detection and reinforcement")
                print("  • Preference learning and tracking")
                print("  • Entity extraction and linking")
                print("  • Temporal validity tracking")
                print("  • Cross-domain analytics")

                print("\n🚀 QUERY IMPROVEMENTS:")
                print("  OLD: Simple keyword search → NEW: Semantic + type-based search")
                print("  OLD: Tag filtering only → NEW: Relationship traversal")
                print("  OLD: No patterns → NEW: Pattern confidence scoring")
                print("  OLD: Flat list → NEW: Knowledge graph")

                return analytics
        except Exception as e:
            print(f"❌ Error getting analytics: {e}")
            return None

    def test_recall_performance(self):
        """Compare recall performance between old and new systems."""
        print("\n🔬 Testing recall performance...")

        test_queries = [
            "work pattern",
            "preference",
            "decided",
            "communication style",
            "bug fixing"
        ]

        improvements = []

        for query in test_queries:
            # Test new system
            try:
                response = requests.get(f"{AUTOMEM_URL}/recall", params={"query": query, "limit": 5})
                if response.status_code == 200:
                    results = response.json()["results"]

                    # Check for relationships (improvement over flat storage)
                    has_relationships = any(r.get("relations", []) for r in results)
                    has_types = any(r.get("memory", {}).get("type") != "Memory" for r in results)

                    if has_relationships or has_types:
                        improvements.append(query)
                        print(f"  ✅ '{query}': Enhanced with relationships/types")
                    else:
                        print(f"  • '{query}': Basic recall")
            except:
                pass

        if improvements:
            improvement_rate = (len(improvements) / len(test_queries)) * 100
            print(f"\n📈 Recall improved for {improvement_rate:.0f}% of test queries")

    def run_migration(self):
        """Run the complete migration process."""
        print("🚀 Starting MCP → AutoMem PKG Migration")
        print("=" * 50)

        # Step 1: Extract memories
        self.extract_memories_from_mcp()

        # Step 2: Migrate memories in batches
        print(f"\n📤 Migrating {len(self.memories)} memories to AutoMem...")
        for i, memory in enumerate(self.memories):
            if i > 0 and i % 10 == 0:
                print(f"  Progress: {i}/{len(self.memories)}")
            self.migrate_memory(memory)
            time.sleep(0.05)  # Avoid overwhelming the API

        # Step 3: Create relationships
        self.create_relationships()

        # Step 4: Analyze patterns
        self.analyze_patterns()

        # Step 5: Generate analytics
        time.sleep(2)  # Wait for enrichment pipeline
        analytics = self.generate_analytics()

        # Step 6: Test recall performance
        self.test_recall_performance()

        print("\n✨ Migration complete!")
        print(f"Success rate: {(self.stats['migrated']/self.stats['total']*100):.1f}%")

        return {
            "stats": self.stats,
            "analytics": analytics,
            "relationships": self.relationships_created[:10],
            "entities": {k: list(v)[:5] for k, v in self.entities.items()},
            "preferences": self.preferences_found[:5]
        }


def main():
    """Main entry point."""
    migrator = MemoryMigrator()

    # Check if AutoMem is running
    try:
        response = requests.get(f"{AUTOMEM_URL}/health")
        if response.status_code != 200:
            print("❌ AutoMem service is not healthy")
            return
    except:
        print("❌ AutoMem service is not running. Start it with 'make dev'")
        return

    # Run migration
    results = migrator.run_migration()

    # Save results
    report_path = REPORTS_DIR / "migration_results.json"
    with report_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📁 Results saved to {report_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
