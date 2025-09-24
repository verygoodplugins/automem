#!/usr/bin/env python3
"""
Startup Memory Recall System
Automatically surfaces critical lessons at the beginning of conversations
"""

import requests
import json
from typing import List, Dict, Any

BASE_URL = "http://localhost:8001"

def recall_critical_lessons() -> List[Dict[str, Any]]:
    """
    Recall critical lessons for AI assistants at startup.
    This should be called at the beginning of each development session.
    """

    print("\n🧠 STARTUP MEMORY RECALL")
    print("=" * 60)
    print("Recalling critical lessons from previous sessions...\n")

    # Search for critical lessons
    search_queries = [
        "critical lesson ai-assistant",
        "testing docker development",
        "best-practice mistake"
    ]

    all_lessons = []
    seen_ids = set()

    for query in search_queries:
        response = requests.get(
            f"{BASE_URL}/recall",
            params={"query": query, "limit": 5}
        )

        if response.status_code == 200:
            results = response.json().get('results', [])
            for result in results:
                memory = result.get('memory', result)  # Handle both formats
                memory_id = memory.get('memory_id') or memory.get('id', '')

                # Check if this is a critical lesson
                tags = memory.get('tags', [])
                if 'critical' in tags or 'lesson' in tags:
                    if memory_id not in seen_ids:
                        seen_ids.add(memory_id)
                        all_lessons.append(memory)

    return all_lessons

def display_lessons(lessons: List[Dict[str, Any]]) -> None:
    """Display recalled lessons in a formatted way."""

    if not lessons:
        print("📌 No critical lessons found in memory.")
        return

    print(f"📚 Found {len(lessons)} critical lesson(s) to remember:\n")

    for i, lesson in enumerate(lessons, 1):
        print(f"{i}. LESSON (Importance: {lesson.get('importance', 0.5):.1f})")
        print("-" * 50)

        # Display content
        content = lesson.get('content', '')
        if len(content) > 200:
            print(f"   {content[:200]}...")
        else:
            print(f"   {content}")

        # Display tags
        tags = lesson.get('tags', [])
        if tags:
            print(f"\n   Tags: {', '.join(tags[:5])}")

        # Display metadata
        metadata = lesson.get('metadata', {})
        if metadata.get('severity'):
            print(f"   Severity: {metadata['severity']}")
        if metadata.get('applies_to'):
            print(f"   Applies to: {', '.join(metadata['applies_to'])}")

        print()

    print("=" * 60)
    print("💡 Keep these lessons in mind during this session!")
    print()

def create_session_context() -> Dict[str, Any]:
    """
    Create a session context with recalled lessons.
    This can be used to initialize AI assistants with prior knowledge.
    """

    lessons = recall_critical_lessons()
    display_lessons(lessons)

    context = {
        "session_type": "development",
        "recalled_lessons": lessons,
        "lesson_count": len(lessons),
        "has_critical_lessons": any(
            l.get('metadata', {}).get('severity') == 'critical'
            for l in lessons
        )
    }

    # Also check for system rules
    response = requests.get(
        f"{BASE_URL}/recall",
        params={"query": "system memory-recall automation", "limit": 3}
    )

    if response.status_code == 200:
        system_rules = [
            r['memory'] for r in response.json().get('results', [])
            if 'system' in r['memory'].get('tags', [])
        ]
        context['system_rules'] = system_rules

    return context

def main():
    """Run startup recall system."""

    # Create session context with recalled lessons
    context = create_session_context()

    # Summary
    print("\n📋 SESSION CONTEXT INITIALIZED")
    print("=" * 60)
    print(f"• Recalled {context['lesson_count']} lesson(s)")
    print(f"• Has critical lessons: {context['has_critical_lessons']}")
    print(f"• System rules loaded: {len(context.get('system_rules', []))}")

    # Save context for the session
    with open('/tmp/session_context.json', 'w') as f:
        # Convert to serializable format
        serializable_context = {
            "lesson_count": context['lesson_count'],
            "has_critical_lessons": context['has_critical_lessons'],
            "recalled_lessons": [
                {
                    "id": l.get('id'),
                    "content": l.get('content', '')[:500],  # Truncate for storage
                    "tags": l.get('tags', []),
                    "importance": l.get('importance', 0.5)
                }
                for l in context.get('recalled_lessons', [])
            ]
        }
        json.dump(serializable_context, f, indent=2)
        print(f"\n💾 Context saved to /tmp/session_context.json")

    return context

if __name__ == "__main__":
    main()