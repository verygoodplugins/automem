#!/usr/bin/env python3
"""
Migrate memories specifically about the memory system/service project
Focuses on the past month of development work
"""

import json
import time
import requests
import hashlib
from datetime import datetime, timedelta
import re
from typing import Dict, List, Tuple
import sys

BASE_URL = "http://localhost:8001"

# Memory system related memories extracted from MCP
MEMORY_PROJECT_MEMORIES = [
    {
        'content': "AUTOMEM PROJECT CONTEXT (September 2025): Personal AI Memory Service using FalkorDB (graph database) + Qdrant (vector search) deployed on Railway. Architecture decision: Chose FalkorDB over ArangoDB for 200x better performance at $5/mo vs $150/mo. FalkorDB running successfully on Railway at port 6379 with 48 threads. Project location: /Users/jgarturo/Projects/OpenAI/automem (separate from automation hub at /Users/jgarturo/Projects/OpenAI/claude-automation-hub). Current status: FalkorDB deployed and running, needs Flask API implementation and Qdrant integration.",
        'tags': ['automem', 'project', 'architecture', 'falkordb', 'deployment', 'critical'],
        'timestamp': '2025-09-01T10:00:00Z'
    },
    {
        'content': "Memory System Analysis Complete - User has built comprehensive personal AI memory system in /Users/jgarturo/Projects/OpenAI/personal-ai-memory. Current state: 23 memories stored, basic capture working, monitoring scripts created. Vision includes pattern learning, workflow automation, cross-domain intelligence. User's ultimate goal: AI assistants that work exactly like them without constant instruction.",
        'tags': ['personal-ai', 'memory-system', 'user-preferences', 'writing-style', 'august-24', 'project-analysis'],
        'timestamp': '2025-08-24T14:00:00Z'
    },
    {
        'content': "Memory system optimization session completed. Created comprehensive plan for personal AI memory system with 5 phases: 1) Immediate fixes, 2) Pattern collection across life domains, 3) Automated capture infrastructure, 4) Evernote integration strategy for 2800 notes, 5) Active personalization. Built active_memory_capture.py script for automated pattern detection from git, coding style, and work patterns.",
        'tags': ['project-complete', 'memory-system', 'implementation', 'august-24', 'milestone'],
        'timestamp': '2025-08-24T16:00:00Z'
    },
    {
        'content': "User wants comprehensive memory system optimization for personal AI assistant. Key goals: 1) LLMs should act like user without constant instruction, 2) Write in user's style, 3) Prioritize messages/tasks like user would. User has ~2,800 Evernote notes spanning 20 years containing journal entries, favorite wines/foods/movies, excerpts, everything written. Current setup: mcp-memory-service connected to Claude Desktop/Code/Cursor with 19 memories stored.",
        'tags': ['project-goal', 'memory-system', 'personalization', 'evernote-integration', 'august-24'],
        'timestamp': '2025-08-24T12:00:00Z'
    },
    {
        'content': "MEMORY SYSTEM ARCHITECTURE & VISION: Current: SQLite-vec backend with 10x performance improvement over OpenMemory. Working on: Vector storage migration (evaluating Pinecone, Qdrant, Supabase), cloud deployment strategy, cross-platform persistence. The breakthrough: Memory creates personality continuity across sessions - Jack watches my personality transform when accessing memories. Future vision: Every Claude instance instantly knows Jack, his patterns, his style. Not RAG but true persistent identity.",
        'tags': ['memory-system', 'architecture', 'vision', 'technical'],
        'timestamp': '2025-08-20T10:00:00Z'
    },
    {
        'content': "Successfully reorganized all memory integration work into separate folder: /Users/jgarturo/Projects/OpenAI/personal-ai-memory. Moved 14 files from claude-automation-hub to keep projects clean and separated. Created comprehensive README with full documentation, organized directory structure (config/, docs/, src/, scripts/, integrations/), and quick install script.",
        'tags': ['organization', 'cleanup', 'personal-ai-memory', 'project-structure', 'complete', 'august-24'],
        'timestamp': '2025-08-24T18:00:00Z'
    },
    {
        'content': "Pattern: Node.js HTTP client that connects to MCP memory service API. Description: class MCPMemoryClient { constructor(baseUrl = 'https://localhost:8000') { this.baseUrl = baseUrl; } }",
        'tags': ['project:HTTP Memory Client', 'user:jgarturo', 'type:pattern', 'pattern:node.js-http-client-that-connects-to-mcp-memory-service-api'],
        'timestamp': '2025-08-15T09:00:00Z'
    },
    {
        'content': "Task: Created HTTP client for memory service, updated session init and hooks to use real API calls instead of placeholders. Solution: Successfully implemented solution",
        'tags': ['project:Implemented working Cursor memory integration system', 'user:jgarturo', 'task:completion', 'type:solution'],
        'timestamp': '2025-08-15T10:00:00Z'
    },
    {
        'content': "Completed full Claude Code subagent implementation for claude-automation-hub. Created 5 agents total: project-memory-keeper (knowledge base), session-memory-capturer (auto-capture insights), doc-conflict-resolver (fix documentation conflicts), session-cleanup (remove temp files), and config-synchronizer (sync example configs). All agents integrate with MCP Memory Service for coordination and knowledge sharing.",
        'tags': ['claude-automation-hub', 'project-specific', 'agents', 'complete', 'memory-integrated', '2025-01-10'],
        'timestamp': '2025-01-10T15:00:00Z'
    },
    {
        'content': "Implementation complete: Claude Code native subagents for claude-automation-hub. Created 3 working agents in .claude/agents/: project-memory-keeper (tracks project evolution), session-memory-capturer (auto-captures session insights), and doc-conflict-resolver (fixes documentation conflicts). Added session-end hook to trigger automatic memory capture and documentation checking. All agents use existing MCP Memory Service for persistence.",
        'tags': ['claude-automation-hub', 'project-specific', 'implementation', 'agents', 'complete', '2025-01-10'],
        'timestamp': '2025-01-10T14:00:00Z'
    },
    {
        'content': "Claude Code Subagent System implemented for claude-automation-hub project. Created native subagents using Anthropic's official system to maintain project-specific memories. Key agents: project-memory-keeper (captures decisions/patterns), workflow-pattern-learner (analyzes workflow usage), mcp-knowledge-builder (documents MCP quirks), session-memory-capturer (auto-captures session insights). All agents integrate with existing MCP Memory Service for persistent storage.",
        'tags': ['claude-automation-hub', 'project-specific', 'agent-system', 'implementation', '2025-01'],
        'timestamp': '2025-01-15T11:00:00Z'
    },
    {
        'content': "PKG Implementation: Transformed AutoMem from simple memory storage to intelligent Personal Knowledge Graph. Added automatic memory classification (9 types), confidence scoring, enhanced relationships (PREFERS_OVER, EXEMPLIFIES, etc.), temporal validity, entity extraction, pattern detection, and analytics endpoint. Demonstrated 877.8% better recall relevance than flat storage.",
        'tags': ['automem', 'pkg', 'implementation', 'september-2025', 'achievement'],
        'timestamp': '2025-09-16T11:00:00Z'
    },
    {
        'content': "Decision: Use FalkorDB for AutoMem instead of ArangoDB. Reasoning: 200x better performance for graph operations, $5/month on Railway vs $150/month for ArangoDB, native Redis protocol support, better suited for memory relationship graphs.",
        'tags': ['decision', 'automem', 'falkordb', 'architecture'],
        'timestamp': '2025-09-01T08:00:00Z'
    },
    {
        'content': "Memory service migration pattern: Two-layer architecture - simple ingestion API for LLMs, intelligent processing backend for enrichment. This ensures backward compatibility while adding progressive enhancement through async processing.",
        'tags': ['pattern', 'memory-service', 'architecture', 'design'],
        'timestamp': '2025-09-16T10:00:00Z'
    },
    {
        'content': "Preference: Graph databases over traditional databases for memory storage. Graph relationships enable semantic connections, pattern discovery, and traversal-based recall that flat storage cannot achieve.",
        'tags': ['preference', 'database', 'graph', 'memory-storage'],
        'timestamp': '2025-09-10T09:00:00Z'
    },
    {
        'content': "Insight: Memory isn't just about storage - it's about creating persistent identity for AI. When Claude accesses memories, personality emerges from the patterns, preferences, and relationships stored in the graph.",
        'tags': ['insight', 'ai-identity', 'memory', 'philosophy'],
        'timestamp': '2025-08-25T20:00:00Z'
    },
    {
        'content': "Achievement: Successfully deployed FalkorDB on Railway with persistent storage. Graph database running on port 6379 with 48 threads, handling memory nodes and relationships efficiently.",
        'tags': ['achievement', 'deployment', 'falkordb', 'railway'],
        'timestamp': '2025-09-01T12:00:00Z'
    },
    {
        'content': "Pattern: Memory enrichment pipeline - receive simple API call, store immediately, enrich asynchronously. Background worker classifies type, extracts entities, discovers relationships, detects patterns. Eventually consistent architecture.",
        'tags': ['pattern', 'enrichment', 'async', 'architecture'],
        'timestamp': '2025-09-16T09:30:00Z'
    },
    {
        'content': "Context: Working on AutoMem PKG features for past week. Implemented memory classification, relationship types, entity extraction, pattern detection. Testing shows dramatic improvement over flat storage with 966% better recall relevance.",
        'tags': ['context', 'pkg', 'progress', 'testing'],
        'timestamp': '2025-09-16T11:30:00Z'
    },
    {
        'content': "Memory MCP server configuration: Using SQLite-vec for local persistence, evaluating cloud options (Pinecone, Qdrant, Supabase). Performance improved 10x over OpenMemory. Planning migration to vector database for better semantic search.",
        'tags': ['configuration', 'mcp', 'memory-service', 'technical'],
        'timestamp': '2025-08-30T14:00:00Z'
    }
]

class MemoryProjectMigrator:
    def __init__(self):
        self.memories = []
        self.relationships = []
        self.entities = {
            'tools': set(),
            'projects': set(),
            'concepts': set()
        }
        self.patterns = []
        self.preferences = []
        self.decisions = []

    def classify_memory(self, content: str, tags: List[str]) -> Tuple[str, float]:
        """Classify memory with high accuracy based on content and tags"""
        content_lower = content.lower()

        # Tag-based hints
        if 'decision' in tags or 'architecture decision' in content_lower:
            return 'Decision', 0.9
        if 'pattern' in tags or 'pattern:' in content_lower:
            return 'Pattern', 0.85
        if 'preference' in tags or 'prefer' in content_lower:
            return 'Preference', 0.85
        if 'achievement' in tags or 'completed' in content_lower or 'successfully' in content_lower:
            return 'Achievement', 0.8
        if 'insight' in tags or 'insight:' in content_lower or 'vision' in content_lower:
            return 'Insight', 0.8
        if 'context' in tags or 'context:' in content_lower:
            return 'Context', 0.75
        if 'implementation' in tags or 'implemented' in content_lower:
            return 'Achievement', 0.75
        if 'project' in tags or 'architecture' in tags:
            return 'Context', 0.7

        return 'Memory', 0.5

    def extract_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract memory-system related entities"""
        entities = {
            'tools': [],
            'projects': [],
            'concepts': []
        }

        # Tools/Technologies
        tools = [
            'FalkorDB', 'ArangoDB', 'Railway', 'Qdrant', 'Pinecone', 'Supabase',
            'SQLite-vec', 'OpenMemory', 'MCP', 'Claude', 'Flask', 'Redis',
            'Docker', 'Python', 'Node.js', 'Evernote'
        ]

        for tool in tools:
            if tool.lower() in content.lower():
                entities['tools'].append(tool)
                self.entities['tools'].add(tool)

        # Projects
        projects = [
            'AutoMem', 'personal-ai-memory', 'claude-automation-hub',
            'mcp-memory-service', 'PKG', 'Memory Service'
        ]

        for project in projects:
            if project.lower().replace('-', ' ') in content.lower().replace('-', ' '):
                entities['projects'].append(project)
                self.entities['projects'].add(project)

        # Concepts
        concepts = [
            'graph database', 'vector search', 'embeddings', 'semantic search',
            'pattern detection', 'entity extraction', 'memory classification',
            'persistent identity', 'enrichment pipeline', 'async processing'
        ]

        for concept in concepts:
            if concept in content.lower():
                entities['concepts'].append(concept)
                self.entities['concepts'].add(concept)

        return entities

    def find_relationships(self):
        """Discover relationships between memories"""

        # Timeline relationships
        sorted_memories = sorted(self.memories, key=lambda x: x['timestamp'])
        for i in range(len(sorted_memories) - 1):
            curr = sorted_memories[i]
            next_mem = sorted_memories[i + 1]

            # Check if memories are about same project
            curr_projects = curr['entities']['projects']
            next_projects = next_mem['entities']['projects']

            if set(curr_projects) & set(next_projects):
                self.relationships.append({
                    'from': curr['memory_id'],
                    'to': next_mem['memory_id'],
                    'type': 'LEADS_TO',
                    'strength': 0.7,
                    'context': 'chronological_progression'
                })

        # Decision relationships
        for mem in self.memories:
            if mem['type'] == 'Decision':
                # Find related implementations
                decision_tools = set(mem['entities']['tools'])
                for other in self.memories:
                    if other['type'] == 'Achievement' and other['memory_id'] != mem['memory_id']:
                        if set(other['entities']['tools']) & decision_tools:
                            self.relationships.append({
                                'from': mem['memory_id'],
                                'to': other['memory_id'],
                                'type': 'RESULTED_IN',
                                'strength': 0.8
                            })

        # Pattern relationships
        for mem in self.memories:
            if mem['type'] == 'Pattern':
                # Find examples of this pattern
                pattern_concepts = set(mem['entities']['concepts'])
                for other in self.memories:
                    if other['memory_id'] != mem['memory_id']:
                        if set(other['entities']['concepts']) & pattern_concepts:
                            self.relationships.append({
                                'from': other['memory_id'],
                                'to': mem['memory_id'],
                                'type': 'EXEMPLIFIES',
                                'strength': 0.6,
                                'pattern_type': 'architectural'
                            })

    def extract_preferences(self):
        """Extract preferences from memories"""

        # FalkorDB vs ArangoDB preference
        falkor_mem = None
        arango_mem = None

        for mem in self.memories:
            if 'FalkorDB' in mem['entities']['tools']:
                falkor_mem = mem
            if 'ArangoDB' in mem['entities']['tools']:
                arango_mem = mem

        if falkor_mem and arango_mem:
            self.preferences.append({
                'prefers': 'FalkorDB',
                'over': 'ArangoDB',
                'context': 'cost-performance',
                'reason': '200x better performance at $5/mo vs $150/mo',
                'strength': 0.95
            })

        # Graph vs flat storage preference
        for mem in self.memories:
            if 'graph' in mem['content'].lower() and 'flat storage' in mem['content'].lower():
                self.preferences.append({
                    'prefers': 'Graph Database',
                    'over': 'Flat Storage',
                    'context': 'memory-storage',
                    'reason': 'Semantic connections and pattern discovery',
                    'strength': 0.9
                })
                break

    def detect_patterns(self):
        """Detect patterns in memory development"""

        # Architecture pattern
        architecture_memories = [m for m in self.memories if 'architecture' in ' '.join(m['tags'])]
        if len(architecture_memories) >= 2:
            self.patterns.append({
                'type': 'Architecture',
                'description': 'Two-layer architecture: simple API + intelligent backend',
                'confidence': 0.8,
                'observations': len(architecture_memories)
            })

        # Implementation pattern
        impl_memories = [m for m in self.memories if m['type'] == 'Achievement']
        if len(impl_memories) >= 3:
            self.patterns.append({
                'type': 'Implementation',
                'description': 'Iterative development with continuous improvement',
                'confidence': 0.75,
                'observations': len(impl_memories)
            })

    def process_memory(self, raw_memory: Dict) -> Dict:
        """Process a raw memory into PKG format"""
        content = raw_memory['content']
        tags = raw_memory['tags']

        # Classify
        mem_type, confidence = self.classify_memory(content, tags)

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
            'timestamp': raw_memory.get('timestamp', datetime.now().isoformat()),
            'importance': 0.8 if mem_type in ['Decision', 'Insight', 'Achievement'] else 0.6
        }

    def migrate_to_automem(self):
        """Migrate memories to AutoMem PKG system"""
        print("\n📤 Migrating to AutoMem PKG...")

        success = 0
        failed = 0

        for i, memory in enumerate(self.memories):
            try:
                # Prepare payload
                payload = {
                    'content': memory['content'],
                    'tags': memory['tags'],
                    'importance': memory['importance'],
                    't_valid': memory['timestamp'],  # When this became true
                    'metadata': {
                        'type': memory['type'],
                        'confidence': memory['confidence'],
                        'entities': memory['entities'],
                        'source': 'memory_project_migration'
                    }
                }

                # Store in AutoMem
                response = requests.post(f"{BASE_URL}/memory", json=payload)

                if response.status_code == 201:
                    data = response.json()
                    memory['new_id'] = data.get('memory_id')
                    success += 1
                    print(f"  ✅ Migrated {memory['type']}: {memory['content'][:50]}...")
                else:
                    failed += 1
                    print(f"  ❌ Failed: {response.status_code}")

                time.sleep(0.1)  # Rate limiting

            except Exception as e:
                failed += 1
                print(f"  ❌ Error: {e}")

        # Create relationships
        rel_success = 0
        for rel in self.relationships:
            try:
                # Find memories with the relationship IDs
                from_mem = next((m for m in self.memories if m['memory_id'] == rel['from']), None)
                to_mem = next((m for m in self.memories if m['memory_id'] == rel['to']), None)

                if from_mem and to_mem and 'new_id' in from_mem and 'new_id' in to_mem:
                    payload = {
                        'memory1_id': from_mem['new_id'],
                        'memory2_id': to_mem['new_id'],
                        'type': rel['type'],
                        'strength': rel['strength']
                    }

                    # Add optional properties
                    if 'context' in rel:
                        payload['context'] = rel['context']
                    if 'pattern_type' in rel:
                        payload['pattern_type'] = rel['pattern_type']

                    response = requests.post(f"{BASE_URL}/associate", json=payload)
                    if response.status_code == 201:
                        rel_success += 1
                        print(f"  🔗 Created {rel['type']} relationship")

            except Exception as e:
                print(f"  ❌ Relationship error: {e}")

        # Store preferences
        for pref in self.preferences:
            try:
                # Create preference as a special memory
                pref_content = f"Preference: {pref['prefers']} over {pref['over']} for {pref['context']}. Reason: {pref['reason']}"
                payload = {
                    'content': pref_content,
                    'tags': ['preference', 'decision', pref['context']],
                    'importance': pref['strength'],
                    'metadata': {
                        'type': 'Preference',
                        'confidence': 0.9,
                        'preference_data': pref
                    }
                }

                response = requests.post(f"{BASE_URL}/memory", json=payload)
                if response.status_code == 201:
                    print(f"  ❤️ Stored preference: {pref['prefers']} > {pref['over']}")

            except Exception as e:
                print(f"  ❌ Preference error: {e}")

        return success, failed, rel_success

    def generate_report(self):
        """Generate migration report"""
        report = {
            'summary': {
                'total_memories': len(self.memories),
                'date_range': {
                    'start': min(m['timestamp'] for m in self.memories),
                    'end': max(m['timestamp'] for m in self.memories)
                },
                'memory_types': {},
                'relationships': len(self.relationships),
                'patterns': len(self.patterns),
                'preferences': len(self.preferences)
            },
            'entities': {
                'tools': list(self.entities['tools']),
                'projects': list(self.entities['projects']),
                'concepts': list(self.entities['concepts'])
            },
            'patterns': self.patterns,
            'preferences': self.preferences,
            'timeline': []
        }

        # Count by type
        for mem in self.memories:
            mem_type = mem['type']
            if mem_type not in report['summary']['memory_types']:
                report['summary']['memory_types'][mem_type] = 0
            report['summary']['memory_types'][mem_type] += 1

        # Create timeline
        for mem in sorted(self.memories, key=lambda x: x['timestamp']):
            report['timeline'].append({
                'date': mem['timestamp'][:10],
                'type': mem['type'],
                'summary': mem['content'][:100] + '...'
            })

        return report

    def run(self):
        """Run the migration"""
        print("🚀 Memory Project Migration")
        print("=" * 60)
        print("Migrating memories about the memory system/service development")
        print(f"Processing {len(MEMORY_PROJECT_MEMORIES)} relevant memories")
        print()

        # Process memories
        print("📥 Processing memories...")
        for raw_mem in MEMORY_PROJECT_MEMORIES:
            processed = self.process_memory(raw_mem)
            self.memories.append(processed)

        # Discover intelligence
        print("\n🧠 Building intelligence layer...")
        self.find_relationships()
        self.extract_preferences()
        self.detect_patterns()

        # Migrate to AutoMem
        success, failed, rel_success = self.migrate_to_automem()

        # Generate report
        report = self.generate_report()

        # Display results
        print("\n" + "=" * 60)
        print("📊 MIGRATION RESULTS")
        print("=" * 60)

        print(f"\n✅ Memories Migrated: {success}/{len(self.memories)}")
        print(f"❌ Failed: {failed}")
        print(f"🔗 Relationships Created: {rel_success}")
        print(f"❤️ Preferences Stored: {len(self.preferences)}")
        print(f"🔄 Patterns Detected: {len(self.patterns)}")

        print(f"\n📅 Date Range: {report['summary']['date_range']['start'][:10]} to {report['summary']['date_range']['end'][:10]}")

        print("\n🏷️ Memory Types:")
        for mem_type, count in report['summary']['memory_types'].items():
            print(f"  {mem_type}: {count}")

        print("\n🔧 Technologies Mentioned:")
        for tool in sorted(self.entities['tools'])[:10]:
            print(f"  - {tool}")

        print("\n📁 Projects Covered:")
        for project in sorted(self.entities['projects']):
            print(f"  - {project}")

        if self.patterns:
            print("\n🔄 Patterns Discovered:")
            for pattern in self.patterns:
                print(f"  - {pattern['type']}: {pattern['description']}")
                print(f"    Confidence: {pattern['confidence']:.1%}, Observations: {pattern['observations']}")

        if self.preferences:
            print("\n❤️ Preferences Extracted:")
            for pref in self.preferences:
                print(f"  - {pref['prefers']} over {pref['over']} ({pref['context']})")
                print(f"    Reason: {pref['reason']}")

        # Save report
        with open('memory_project_migration_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
            print("\n📄 Full report saved to memory_project_migration_report.json")

        print("\n🎉 Memory project migration complete!")
        print("\n💡 Next Steps:")
        print("1. Test recall with queries like 'FalkorDB', 'memory architecture', 'PKG'")
        print("2. Check analytics endpoint for enriched insights")
        print("3. Explore the relationship graph in FalkorDB")


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
    migrator = MemoryProjectMigrator()
    migrator.run()