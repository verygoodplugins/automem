#!/usr/bin/env python3
"""Store critical development lesson in memory system"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8001"

lesson = {
    "content": "CRITICAL LESSON: Always test incrementally in the actual runtime environment (Docker), not just write code blindly. When implementing new features like the consolidation system, I failed to test as I went, leading to missing dependencies (scipy/scikit-learn), Docker build issues, FalkorDB query handling problems, and a recursion bug. The Docker container had different requirements than local. Test each step, verify imports work, check logs for errors, and run actual tests before claiming completion.",
    "tags": ["lesson", "testing", "best-practice", "critical", "docker", "development", "ai-assistant"],
    "importance": 1.0,
    "metadata": {
        "type": "Lesson",
        "learned_from": "consolidation_implementation",
        "date": "2025-09-16",
        "applies_to": ["all_development", "docker_projects", "new_features"],
        "severity": "critical",
        "assistant": "Claude",
        "auto_surface": True
    }
}

# Store the lesson
response = requests.post(f"{BASE_URL}/memory", json=lesson)

if response.status_code == 201:
    result = response.json()
    print(f"✅ Lesson stored with ID: {result['memory_id']}")
    print(f"   Type: {result.get('type', 'Unknown')}")
    print(f"   Confidence: {result.get('confidence', 0)}")

    # Store a companion memory about auto-surfacing
    companion = {
        "content": "SYSTEM RULE: Automatically recall lessons tagged with 'critical' and 'ai-assistant' at the start of new development conversations to prevent repeating mistakes. This ensures Claude remembers important lessons learned from previous sessions.",
        "tags": ["system", "memory-recall", "ai-assistant", "automation"],
        "importance": 0.9,
        "metadata": {
            "type": "System",
            "purpose": "auto_recall_lessons",
            "trigger": "session_start"
        }
    }

    response2 = requests.post(f"{BASE_URL}/memory", json=companion)
    if response2.status_code == 201:
        print(f"✅ Companion rule stored with ID: {response2.json()['memory_id']}")
else:
    print(f"❌ Failed to store lesson: {response.text}")