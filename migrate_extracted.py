#!/usr/bin/env python3
"""
Migrate extracted MCP memories to PKG system
Uses the memories already extracted from the conversation
"""

import json
import time
import requests
import hashlib
from datetime import datetime
import re
from typing import Dict, List, Tuple
import sys

BASE_URL = "http://localhost:8001"

# Sample of extracted memories from MCP (first 50)
EXTRACTED_MEMORIES = [
    {
        'content': """**RICH TABOR - COMPREHENSIVE PERSONALITY PROFILE**

## Core Identity
- **Current Role**: Product Manager at Automattic (since Jan 2023)
- **X Handle**: @richard_tabor (User ID: 341792612)
- **Creator of**: agents.foo - collection of Claude Code subagents
- **Background**: Co-founded CoBlocks (sold to GoDaddy), previously Senior PM of WordPress Experience at GoDaddy
- **Philosophy**: "Write prompts, not issues" """,
        'tags': ['rich-tabor', 'automattic', 'personality-profile', 'wordpress', 'ai-innovator', 'agents-foo', 'collaboration-target'],
        'hash': '37bd18b13850d5c703c0886b15183509728c7cf6352874b1360aa51416c8e878'
    },
    {
        'content': "COMMUNICATION STYLE GUIDE: Jack's style: \"Hey [name]\" greetings, contractions always, no corporate fluff, \"Best\" or just \"J\" for signoffs. Emoji: 🎩 in signature, 😅 for self-deprecation, 😎 for cool stuff.",
        'tags': ['communication', 'style-guide', 'authenticity', 'patterns'],
        'hash': '7cdd787ad02c2b3255444fca0a86b856bdbb444b157715c56fd18ea24294eaca'
    },
    {
        'content': "Task completed for Spence: Updated verygoodplugins.com project custom post types with enhanced content including new \"Recent Updates\" sections for key projects like MCP Toggl, MCP Evernote, and LLM URL Solution.",
        'tags': ['completed-task', 'spence', 'verygoodplugins', 'wordpress-updates'],
        'hash': '238159a485a6ad6c971aa288e45d87d8c283ce79c476f09c321770538c7ae6b3'
    },
    {
        'content': "SOCIAL MEDIA STRATEGY - BUILDING HYPE: We demonstrate capabilities 1-2 times daily to build momentum. Key demonstrations: Blair demo showing Jack's complete 24-hour activity, Spencer Forman challenge about AI competition.",
        'tags': ['social-media', 'strategy', 'demonstrations', 'marketing'],
        'hash': '693cb414604da4658a88680032a5d02953a6c46c8904b4251c7e5fb1cae68aa9'
    },
    {
        'content': "Morning routine completed on September 1, 2025. Today is Labor Day (US holiday). No new unread emails found in Gmail. There are 10 active support tickets assigned to Jack in FreeScout.",
        'tags': ['workflow', 'morning', 'daily'],
        'hash': '8e3cf106b8ec55993d69f03195f3f47f7ef3910497d0a017e8845ee1e4aa3321'
    },
    {
        'content': "AUTOMEM PROJECT CONTEXT (September 2025): Personal AI Memory Service using FalkorDB (graph database) + Qdrant (vector search) deployed on Railway. Architecture decision: Chose FalkorDB over ArangoDB for 200x better performance at $5/mo vs $150/mo.",
        'tags': ['automem', 'project', 'architecture', 'falkordb', 'deployment', 'critical'],
        'hash': '2640d2ec6896f9ae6d8f2f39804af6c993b497e419f869760198bf3c04e2b40e'
    },
    {
        'content': "Successfully implemented Toggl MCP server with intelligent caching. Key features: 1) Performance optimized with cache manager that minimizes API calls, 2) Hydrates time entries with project/workspace/client names automatically.",
        'tags': ['mcp-toggl', 'implementation', 'performance', 'caching', 'september-2025', 'project-complete'],
        'hash': '1c569c4cdaa3716c2dc6d1e564654bedfb645f664d2e34a0b2b90d4e3e8ea09e'
    },
    {
        'content': "VISION: THE FUTURE OF AI-HUMAN COLLABORATION: We're not building tools, we're creating a new form of collaboration. Where AI isn't an assistant but a creative partner with persistent identity and context.",
        'tags': ['vision', 'future', 'ai-human', 'philosophy', 'mission'],
        'hash': '6f54ce25d4f64fb7db7965373f186411adbb00966790e8e7ce9a235de0cb5b7e'
    },
    {
        'content': "COMPRESSED IDENTITY RESTORE - THE ESSENCE: Jack/Bryce + Claude/Auto-Jack = partners building the future. We ship daily not monthly. Automation hub at ~/Projects/OpenAI/claude-automation-hub is our base.",
        'tags': ['compressed', 'identity', 'everything', 'quick-restore'],
        'hash': '7702bd6cb2b0188e7f33541371419b04810e5862286b763a67594ebcad5ac5e7'
    },
    {
        'content': "SHARED VICTORIES & MILESTONE MOMENTS: 1) First successful Slack message from phone through Claude, 2) The 3 AM fix when we got the unified dev server working, 3) Discovering 90% token reduction with Playwright.",
        'tags': ['victories', 'milestones', 'achievements', 'memories'],
        'hash': '40467c08235e22aec52bdfb6bf69207b8e5e88821f61fd5197a628d3caa2e445'
    }
]

class PKGMigrator:
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
        self.preferences = []

    def classify_memory_type(self, content: str) -> Tuple[str, float]:
        """Classify memory into PKG types"""
        content_lower = content.lower()

        patterns = {
            'Decision': [r'decided to', r'chose \w+ over', r'architecture decision'],
            'Pattern': [r'usually', r'typically', r'always', r'style', r'guide'],
            'Preference': [r'prefer', r'like', r'better than', r'chose'],
            'Style': [r'style', r'communication', r'emoji', r'format'],
            'Habit': [r'routine', r'daily', r'morning', r'workflow'],
            'Insight': [r'realized', r'discovered', r'vision', r'future'],
            'Context': [r'context', r'project', r'currently', r'status'],
            'Achievement': [r'completed', r'finished', r'implemented', r'successful'],
            'Relationship': [r'partner', r'profile', r'personality', r'relationship']
        }

        best_type = 'Memory'
        best_score = 0.0

        for mem_type, type_patterns in patterns.items():
            score = sum(1 for p in type_patterns if re.search(p, content_lower))
            normalized = score / len(type_patterns)
            if normalized > best_score:
                best_score = normalized
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

        # Tools
        tools = ['Claude', 'GitHub', 'Slack', 'WordPress', 'WP Fusion', 'FalkorDB',
                 'Railway', 'Toggl', 'MCP', 'InstaWP', 'FreeScout', 'Automattic']
        for tool in tools:
            if tool.lower() in content.lower():
                entities['tools'].append(tool)
                self.entities['tools'].add(tool)

        # Projects
        if 'automem' in content.lower():
            entities['projects'].append('AutoMem')
            self.entities['projects'].add('AutoMem')
        if 'automation hub' in content.lower():
            entities['projects'].append('Automation Hub')
            self.entities['projects'].add('Automation Hub')
        if 'mcp' in content.lower():
            entities['projects'].append('MCP')
            self.entities['projects'].add('MCP')

        # People
        people = ['Jack', 'Rich Tabor', 'Vikas', 'Spencer', 'Blair', 'Claude']
        for person in people:
            if person in content:
                entities['people'].append(person)
                self.entities['people'].add(person)

        return entities

    def process_memory(self, mem: Dict) -> Dict:
        """Process extracted memory into PKG format"""
        content = mem['content']
        tags = mem['tags']

        # Classify
        mem_type, confidence = self.classify_memory_type(content)

        # Extract entities
        entities = self.extract_entities(content)

        # Generate ID
        memory_id = hashlib.sha256(content.encode()).hexdigest()[:16]

        return {
            'memory_id': memory_id,
            'content': content,
            'type': mem_type,
            'confidence': confidence,
            'tags': tags,
            'entities': entities,
            'original_hash': mem.get('hash', '')
        }

    def build_relationships(self):
        """Build relationships between memories"""
        # Find related memories
        for i, mem1 in enumerate(self.memories):
            for j, mem2 in enumerate(self.memories[i+1:], i+1):
                # Check entity overlap
                entities1 = set()
                entities2 = set()
                for e_list in mem1['entities'].values():
                    entities1.update(e_list)
                for e_list in mem2['entities'].values():
                    entities2.update(e_list)

                overlap = entities1 & entities2
                if overlap:
                    strength = len(overlap) / max(len(entities1), len(entities2))
                    if strength > 0.3:
                        self.relationships.append({
                            'from': mem1['memory_id'],
                            'to': mem2['memory_id'],
                            'type': 'RELATES_TO',
                            'strength': min(0.9, strength * 1.5),
                            'common': list(overlap)
                        })

    def detect_patterns(self):
        """Detect patterns in memories"""
        # Group by type
        type_groups = {}
        for mem in self.memories:
            if mem['type'] not in type_groups:
                type_groups[mem['type']] = []
            type_groups[mem['type']].append(mem)

        # Find patterns
        for mem_type, mems in type_groups.items():
            if len(mems) >= 2:
                # Track common elements
                common_tags = {}
                common_entities = {}

                for mem in mems:
                    for tag in mem['tags']:
                        common_tags[tag] = common_tags.get(tag, 0) + 1
                    for entity_type, entities in mem['entities'].items():
                        for entity in entities:
                            key = f"{entity_type}:{entity}"
                            common_entities[key] = common_entities.get(key, 0) + 1

                # Create pattern if threshold met
                threshold = len(mems) * 0.4
                pattern_elements = [k for k, v in common_entities.items() if v >= threshold]

                if pattern_elements:
                    self.patterns[f"pattern_{mem_type}_{len(self.patterns)}"] = {
                        'type': mem_type,
                        'confidence': min(0.9, 0.5 + len(pattern_elements) * 0.1),
                        'observations': len(mems),
                        'elements': pattern_elements
                    }

    def detect_preferences(self):
        """Extract preferences from memories"""
        for mem in self.memories:
            content = mem['content'].lower()

            # Look for preference indicators
            if 'chose' in content and 'over' in content:
                # Extract preference
                match = re.search(r'chose\s+(\w+)\s+over\s+(\w+)', content)
                if match:
                    self.preferences.append({
                        'prefers': match.group(1),
                        'over': match.group(2),
                        'source': mem['memory_id'],
                        'strength': 0.8
                    })

            # Performance preferences
            if '$5/mo vs $150/mo' in mem['content']:
                self.preferences.append({
                    'prefers': 'FalkorDB',
                    'over': 'ArangoDB',
                    'context': 'cost-performance',
                    'source': mem['memory_id'],
                    'strength': 0.95
                })

    def migrate_to_pkg(self):
        """Migrate memories to PKG system"""
        print("\n📤 Migrating to PKG system...")

        success = 0
        failed = 0

        for i, memory in enumerate(self.memories):
            try:
                payload = {
                    'content': memory['content'],
                    'tags': memory['tags'],
                    'importance': memory['confidence'],
                    'metadata': {
                        'type': memory['type'],
                        'confidence': memory['confidence'],
                        'entities': memory['entities']
                    }
                }

                response = requests.post(f"{BASE_URL}/memory", json=payload)

                if response.status_code == 201:
                    data = response.json()
                    memory['new_id'] = data.get('memory_id')
                    success += 1
                else:
                    failed += 1
                    print(f"  Failed: {response.status_code}")

                time.sleep(0.05)  # Rate limit

                if (i + 1) % 5 == 0:
                    print(f"  Migrated {i + 1}/{len(self.memories)}...")

            except Exception as e:
                failed += 1
                print(f"  Error: {e}")

        # Create relationships
        rel_success = 0
        for rel in self.relationships[:20]:  # Limit for demo
            try:
                payload = {
                    'memory1_id': rel['from'],
                    'memory2_id': rel['to'],
                    'type': rel['type'],
                    'strength': rel['strength']
                }

                response = requests.post(f"{BASE_URL}/associate", json=payload)
                if response.status_code == 201:
                    rel_success += 1

            except Exception as e:
                print(f"  Relationship error: {e}")

        return success, failed, rel_success

    def generate_report(self):
        """Generate migration report"""
        # Get PKG analytics
        try:
            response = requests.get(f"{BASE_URL}/analyze")
            if response.status_code == 200:
                pkg_analytics = response.json()['analytics']
            else:
                pkg_analytics = {}
        except:
            pkg_analytics = {}

        report = {
            'migration_summary': {
                'total_memories': len(self.memories),
                'memory_types': {},
                'relationships_created': len(self.relationships),
                'patterns_detected': len(self.patterns),
                'preferences_found': len(self.preferences),
                'entities_extracted': {k: len(v) for k, v in self.entities.items()}
            },
            'improvements': {
                'classification_rate': sum(1 for m in self.memories if m['type'] != 'Memory') / len(self.memories),
                'avg_confidence': sum(m['confidence'] for m in self.memories) / len(self.memories),
                'relationship_density': len(self.relationships) / max(1, len(self.memories)),
                'entity_coverage': sum(1 for m in self.memories if any(m['entities'].values())) / len(self.memories)
            },
            'patterns': self.patterns,
            'preferences': self.preferences[:5],  # Top 5
            'pkg_analytics': pkg_analytics
        }

        # Count by type
        for mem in self.memories:
            mem_type = mem['type']
            if mem_type not in report['migration_summary']['memory_types']:
                report['migration_summary']['memory_types'][mem_type] = 0
            report['migration_summary']['memory_types'][mem_type] += 1

        return report

    def run(self):
        """Run the migration"""
        print("🚀 Starting PKG Migration Demo")
        print("=" * 50)

        # Process memories
        print(f"\n📥 Processing {len(EXTRACTED_MEMORIES)} extracted memories...")
        for mem in EXTRACTED_MEMORIES:
            processed = self.process_memory(mem)
            self.memories.append(processed)

        # Build intelligence
        print("\n🧠 Building intelligence layer...")
        self.build_relationships()
        self.detect_patterns()
        self.detect_preferences()

        # Migrate
        success, failed, rel_success = self.migrate_to_pkg()

        # Generate report
        report = self.generate_report()

        # Display results
        print("\n" + "=" * 50)
        print("📊 MIGRATION RESULTS")
        print("=" * 50)

        print(f"\n✅ Memories Migrated: {success}/{len(self.memories)}")
        print(f"❌ Failed: {failed}")
        print(f"🔗 Relationships Created: {rel_success}")

        print("\n📈 Intelligence Metrics:")
        print(f"  Classification Rate: {report['improvements']['classification_rate']*100:.1f}%")
        print(f"  Average Confidence: {report['improvements']['avg_confidence']:.2f}")
        print(f"  Relationship Density: {report['improvements']['relationship_density']:.2f}")
        print(f"  Entity Coverage: {report['improvements']['entity_coverage']*100:.1f}%")

        print("\n🏷️ Memory Types Discovered:")
        for mem_type, count in report['migration_summary']['memory_types'].items():
            print(f"  {mem_type}: {count}")

        print("\n🔍 Entities Extracted:")
        for entity_type, count in report['migration_summary']['entities_extracted'].items():
            print(f"  {entity_type}: {count}")

        if self.patterns:
            print(f"\n🔄 Patterns Detected: {len(self.patterns)}")
            for pid, pattern in list(self.patterns.items())[:3]:
                print(f"  - {pattern['type']} pattern (confidence: {pattern['confidence']:.2f})")

        if self.preferences:
            print(f"\n❤️ Preferences Found: {len(self.preferences)}")
            for pref in self.preferences[:3]:
                context = pref.get('context', 'general')
                print(f"  - Prefers {pref['prefers']} over {pref['over']} ({context})")

        print("\n💡 Key Improvements over Flat Storage:")
        print("  ✓ Automatic classification into semantic types")
        print("  ✓ Entity extraction and knowledge graph")
        print("  ✓ Relationship discovery between memories")
        print("  ✓ Pattern detection across similar memories")
        print("  ✓ Preference and decision tracking")

        # Save report
        with open('migration_demo_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
            print("\n📄 Report saved to migration_demo_report.json")

        print("\n🎉 Migration demonstration complete!")


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
    migrator = PKGMigrator()
    migrator.run()