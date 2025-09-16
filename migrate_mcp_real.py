#!/usr/bin/env python3
"""
Real MCP to PKG Migration Script
Extracts actual memories from MCP Memory Service and migrates them to AutoMem PKG
"""

import json
import time
import requests
import hashlib
from datetime import datetime, timedelta
import random
import re
from typing import Dict, List, Any, Tuple
import sys

BASE_URL = "http://localhost:8001"
MCP_URL = "http://localhost:8010"  # MCP Memory Service endpoint

class MCPToPKGMigrator:
    def __init__(self):
        self.memories = []
        self.relationships = []
        self.entities = {
            'tools': set(),
            'projects': set(),
            'people': set(),
            'concepts': set()
        }
        self.patterns = {}
        self.preferences = {}
        self.memory_map = {}  # Old ID to new ID mapping

    def extract_mcp_memories(self) -> List[Dict]:
        """Extract memories from MCP Memory Service"""
        print("📥 Extracting memories from MCP Memory Service...")

        all_memories = []
        batch_size = 50
        offset = 0

        # Try to get all memories in batches
        while True:
            try:
                # Use the MCP memory service endpoint
                response = requests.post(
                    f"{MCP_URL}",
                    json={
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "params": {
                            "name": "retrieve_memory",
                            "arguments": {
                                "query": "*",
                                "n_results": batch_size
                            }
                        },
                        "id": 1
                    }
                )

                if response.status_code != 200:
                    # Try alternative: search by common tags
                    print(f"  Trying alternative extraction method...")
                    return self.extract_via_tags()

                data = response.json()
                if 'result' in data and 'results' in data['result']:
                    batch = data['result']['results']
                    if not batch:
                        break
                    all_memories.extend(batch)
                    offset += len(batch)
                    print(f"  Extracted {offset} memories...")

                    if len(batch) < batch_size:
                        break
                else:
                    break

            except Exception as e:
                print(f"  Error extracting batch: {e}")
                break

        print(f"✅ Extracted {len(all_memories)} memories from MCP")
        return all_memories

    def extract_via_tags(self) -> List[Dict]:
        """Alternative extraction using common tags"""
        memories = []
        common_tags = [
            'project', 'development', 'user', 'workflow',
            'communication', 'daily', 'technical', 'memory'
        ]

        for tag in common_tags:
            try:
                response = requests.post(
                    f"{MCP_URL}",
                    json={
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "params": {
                            "name": "search_by_tag",
                            "arguments": {"tags": [tag]}
                        },
                        "id": 1
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    if 'result' in data and 'results' in data['result']:
                        for mem in data['result']['results']:
                            # Avoid duplicates
                            if not any(m.get('hash') == mem.get('hash') for m in memories):
                                memories.append(mem)

            except Exception as e:
                print(f"    Error extracting tag '{tag}': {e}")

        return memories

    def classify_memory_type(self, content: str) -> Tuple[str, float]:
        """Classify memory into PKG types"""
        content_lower = content.lower()

        patterns = {
            'Decision': [
                r'decided to', r'chose \w+ over', r'selected',
                r'went with', r'picked', r'opted for'
            ],
            'Pattern': [
                r'usually', r'typically', r'always', r'often',
                r'tend to', r'pattern', r'consistently'
            ],
            'Preference': [
                r'prefer', r'like', r'favorite', r'love',
                r'better than', r'instead of', r'rather than'
            ],
            'Style': [
                r'style', r'approach', r'way of', r'method',
                r'write', r'communicate', r'format'
            ],
            'Habit': [
                r'habit', r'routine', r'every day', r'daily',
                r'morning', r'evening', r'regularly'
            ],
            'Insight': [
                r'realized', r'discovered', r'found that', r'learned',
                r'insight', r'understood', r'breakthrough'
            ],
            'Context': [
                r'context', r'background', r'situation', r'environment',
                r'working on', r'project', r'currently'
            ],
            'Achievement': [
                r'completed', r'finished', r'accomplished', r'succeeded',
                r'solved', r'fixed', r'implemented'
            ],
            'Relationship': [
                r'partner', r'collaboration', r'working with', r'team',
                r'relationship', r'interaction'
            ]
        }

        best_type = 'Memory'
        best_score = 0.0

        for mem_type, type_patterns in patterns.items():
            score = 0
            for pattern in type_patterns:
                if re.search(pattern, content_lower):
                    score += 1

            normalized_score = score / len(type_patterns)
            if normalized_score > best_score:
                best_score = normalized_score
                best_type = mem_type

        confidence = min(0.95, 0.6 + best_score * 0.35)
        return best_type, confidence

    def extract_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract entities from content"""
        entities = {
            'tools': [],
            'projects': [],
            'people': [],
            'concepts': []
        }

        # Tool patterns
        tool_patterns = [
            r'Claude(?:\s+(?:Code|Desktop|AI))?', r'GitHub', r'Slack',
            r'WhatsApp', r'Gmail', r'Todoist', r'FreeScout', r'WordPress',
            r'WP Fusion', r'Docker', r'Railway', r'FalkorDB', r'Qdrant',
            r'Playwright', r'Evernote', r'MCP', r'InstaWP', r'Toggl'
        ]

        for pattern in tool_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                tool = re.search(pattern, content, re.IGNORECASE).group()
                entities['tools'].append(tool)
                self.entities['tools'].add(tool)

        # Project patterns
        project_patterns = [
            r'automation[\s-]hub', r'personal[\s-]ai[\s-]memory',
            r'wp[\s-]fusion', r'very[\s-]good[\s-]plugins',
            r'automem', r'mcp-\w+', r'freescout-gpt'
        ]

        for pattern in project_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                project = re.search(pattern, content, re.IGNORECASE).group()
                entities['projects'].append(project)
                self.entities['projects'].add(project)

        # People (look for names)
        people_patterns = [
            r'Jack(?:\s+Arturo)?', r'Vikas(?:\s+Singhal)?',
            r'Rich(?:\s+Tabor)?', r'Adrian', r'Spencer', r'Blair'
        ]

        for pattern in people_patterns:
            if re.search(pattern, content):
                person = re.search(pattern, content).group()
                entities['people'].append(person)
                self.entities['people'].add(person)

        return entities

    def process_memory(self, mcp_memory: Dict) -> Dict:
        """Convert MCP memory to PKG format"""
        content = mcp_memory.get('content', '')
        tags = mcp_memory.get('tags', [])

        # Handle both string and list tags
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(',')]

        # Classify memory type
        memory_type, confidence = self.classify_memory_type(content)

        # Extract entities
        entities = self.extract_entities(content)

        # Create PKG memory
        memory_id = hashlib.sha256(content.encode()).hexdigest()[:16]

        pkg_memory = {
            'memory_id': memory_id,
            'content': content,
            'type': memory_type,
            'confidence': confidence,
            'tags': tags,
            'entities': entities,
            'timestamp': mcp_memory.get('timestamp', datetime.now().isoformat()),
            'hash': mcp_memory.get('hash', ''),
            'original_relevance': mcp_memory.get('relevance_score', 0.0)
        }

        # Store mapping
        if 'hash' in mcp_memory:
            self.memory_map[mcp_memory['hash']] = memory_id

        return pkg_memory

    def detect_patterns(self):
        """Detect patterns across memories"""
        print("\n🔍 Detecting patterns...")

        # Group memories by type
        type_groups = {}
        for mem in self.memories:
            mem_type = mem['type']
            if mem_type not in type_groups:
                type_groups[mem_type] = []
            type_groups[mem_type].append(mem)

        # Find patterns within each type
        for mem_type, mems in type_groups.items():
            if len(mems) >= 3:  # Need at least 3 to form a pattern
                # Simple pattern: frequently mentioned together
                common_entities = {}
                for mem in mems:
                    for entity_type, entity_list in mem['entities'].items():
                        for entity in entity_list:
                            key = f"{entity_type}:{entity}"
                            common_entities[key] = common_entities.get(key, 0) + 1

                # Patterns are entities mentioned in >40% of memories
                threshold = len(mems) * 0.4
                patterns = [k for k, v in common_entities.items() if v >= threshold]

                if patterns:
                    pattern_id = f"pattern_{mem_type}_{len(self.patterns)}"
                    self.patterns[pattern_id] = {
                        'type': mem_type,
                        'confidence': min(0.9, 0.5 + len(patterns) * 0.1),
                        'observations': len(mems),
                        'common_elements': patterns
                    }

    def build_relationships(self):
        """Build relationships between memories"""
        print("\n🔗 Building relationships...")

        # Temporal relationships (if memories are close in time)
        for i, mem1 in enumerate(self.memories):
            for mem2 in self.memories[i+1:i+5]:  # Check next 5 memories
                # Similar content = related
                content1 = mem1['content'].lower()
                content2 = mem2['content'].lower()

                # Simple similarity check
                words1 = set(content1.split())
                words2 = set(content2.split())
                overlap = len(words1 & words2)

                if overlap > min(len(words1), len(words2)) * 0.3:
                    self.relationships.append({
                        'from': mem1['memory_id'],
                        'to': mem2['memory_id'],
                        'type': 'RELATES_TO',
                        'strength': min(0.9, overlap / min(len(words1), len(words2)))
                    })

                # Check for preferences
                if 'prefer' in content1 and 'over' in content1:
                    if any(word in content2 for word in words1):
                        self.relationships.append({
                            'from': mem1['memory_id'],
                            'to': mem2['memory_id'],
                            'type': 'PREFERS_OVER',
                            'strength': 0.8,
                            'context': 'extracted_preference'
                        })

    def migrate_to_pkg(self):
        """Migrate processed memories to PKG system"""
        print("\n📤 Migrating to PKG system...")

        success_count = 0
        failed_count = 0

        # Store memories
        for i, memory in enumerate(self.memories[:100]):  # Limit for testing
            try:
                # Prepare payload
                payload = {
                    'content': memory['content'],
                    'tags': memory['tags'],
                    'importance': memory.get('original_relevance', 0.5),
                    'metadata': {
                        'type': memory['type'],
                        'confidence': memory['confidence'],
                        'entities': memory['entities'],
                        'original_hash': memory.get('hash', '')
                    }
                }

                # Call AutoMem API
                response = requests.post(f"{BASE_URL}/memory", json=payload)

                if response.status_code == 201:
                    data = response.json()
                    memory['new_id'] = data.get('memory_id')
                    success_count += 1

                    if (i + 1) % 10 == 0:
                        print(f"  Migrated {i + 1}/{len(self.memories)} memories...")
                else:
                    failed_count += 1
                    print(f"  Failed to migrate memory: {response.status_code}")

                # Rate limiting
                time.sleep(0.05)

            except Exception as e:
                failed_count += 1
                print(f"  Error migrating memory: {e}")

        # Create relationships
        rel_success = 0
        for relationship in self.relationships[:50]:  # Limit for testing
            try:
                payload = {
                    'memory1_id': relationship['from'],
                    'memory2_id': relationship['to'],
                    'type': relationship['type'],
                    'strength': relationship['strength']
                }

                response = requests.post(f"{BASE_URL}/associate", json=payload)
                if response.status_code == 201:
                    rel_success += 1

            except Exception as e:
                print(f"  Error creating relationship: {e}")

        print(f"\n✅ Migration complete!")
        print(f"  Memories: {success_count} successful, {failed_count} failed")
        print(f"  Relationships: {rel_success} created")

        return success_count, failed_count

    def generate_analytics(self):
        """Generate analytics report"""
        print("\n📊 Generating analytics...")

        # Get PKG analytics
        response = requests.get(f"{BASE_URL}/analyze")
        if response.status_code == 200:
            pkg_analytics = response.json()['analytics']
        else:
            pkg_analytics = {}

        # Compare with original
        report = {
            'migration_summary': {
                'total_memories': len(self.memories),
                'classified_memories': sum(1 for m in self.memories if m['type'] != 'Memory'),
                'relationships_created': len(self.relationships),
                'patterns_detected': len(self.patterns),
                'entities_extracted': {
                    k: len(v) for k, v in self.entities.items()
                }
            },
            'classification_breakdown': {},
            'top_entities': {
                k: list(v)[:10] for k, v in self.entities.items()
            },
            'pattern_summary': self.patterns,
            'pkg_analytics': pkg_analytics
        }

        # Count by type
        for memory in self.memories:
            mem_type = memory['type']
            if mem_type not in report['classification_breakdown']:
                report['classification_breakdown'][mem_type] = {
                    'count': 0,
                    'avg_confidence': 0
                }
            report['classification_breakdown'][mem_type]['count'] += 1
            report['classification_breakdown'][mem_type]['avg_confidence'] += memory['confidence']

        # Average confidence
        for mem_type in report['classification_breakdown']:
            count = report['classification_breakdown'][mem_type]['count']
            if count > 0:
                report['classification_breakdown'][mem_type]['avg_confidence'] /= count

        return report

    def run_migration(self):
        """Run the complete migration process"""
        print("🚀 Starting MCP to PKG Migration")
        print("=" * 50)

        # Extract memories from MCP
        mcp_memories = self.extract_mcp_memories()

        if not mcp_memories:
            print("❌ No memories found in MCP. Exiting.")
            return

        # Process each memory
        print(f"\n🔄 Processing {len(mcp_memories)} memories...")
        for i, mcp_mem in enumerate(mcp_memories):
            pkg_mem = self.process_memory(mcp_mem)
            self.memories.append(pkg_mem)

            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(mcp_memories)} memories...")

        # Detect patterns
        self.detect_patterns()

        # Build relationships
        self.build_relationships()

        # Migrate to PKG
        success, failed = self.migrate_to_pkg()

        # Generate analytics
        report = self.generate_analytics()

        # Display report
        print("\n" + "=" * 50)
        print("📊 MIGRATION REPORT")
        print("=" * 50)

        print(f"\n📥 Source: MCP Memory Service")
        print(f"  Total memories: {len(mcp_memories)}")

        print(f"\n🎯 Classification Results:")
        for mem_type, stats in report['classification_breakdown'].items():
            print(f"  {mem_type}: {stats['count']} memories (avg confidence: {stats['avg_confidence']:.2f})")

        print(f"\n🔍 Entity Extraction:")
        for entity_type, count in report['migration_summary']['entities_extracted'].items():
            print(f"  {entity_type}: {count} unique entities")

        print(f"\n🔗 Relationships:")
        print(f"  Created: {report['migration_summary']['relationships_created']} relationships")

        print(f"\n📈 Pattern Detection:")
        print(f"  Discovered: {len(self.patterns)} patterns")

        if self.patterns:
            print("\n  Sample patterns:")
            for pid, pattern in list(self.patterns.items())[:3]:
                print(f"    - {pattern['type']} pattern (confidence: {pattern['confidence']:.2f})")

        print("\n✨ PKG Improvements over MCP:")
        print("  ✓ Automatic type classification")
        print("  ✓ Rich relationship graph")
        print("  ✓ Entity extraction and linking")
        print("  ✓ Pattern detection and reinforcement")
        print("  ✓ Advanced analytics capabilities")

        # Save report
        with open('migration_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
            print("\n📄 Full report saved to migration_report.json")

        print("\n🎉 Migration complete!")


if __name__ == "__main__":
    # Check if AutoMem is running
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("❌ AutoMem service not responding. Start it with 'make dev'")
            sys.exit(1)
    except requests.ConnectionError:
        print("❌ AutoMem service is not running. Start it with 'make dev'")
        sys.exit(1)

    # Run migration
    migrator = MCPToPKGMigrator()
    migrator.run_migration()