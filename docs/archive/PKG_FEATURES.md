# AutoMem Personal Knowledge Graph (PKG) Features

## 🚀 What's New

AutoMem has been enhanced with Personal Knowledge Graph capabilities, transforming it from a simple memory store into an intelligent system that learns patterns, preferences, and insights from stored memories.

## ✨ Core Features Implemented

### 1. **Memory Type Classification** 🏷️
Memories are automatically classified into types based on content patterns:
- **Decision** - Choices and selections made
- **Pattern** - Recurring behaviors and habits
- **Preference** - Likes, favorites, and preferences
- **Style** - Communication and work styles
- **Habit** - Regular routines and practices
- **Insight** - Realizations and discoveries
- **Context** - Situational information
- **Memory** - Default base type

Each classification includes a confidence score (0.0-1.0).

### 2. **Enhanced Relationship Types** 🔗
Beyond basic relationships, new PKG relationships with properties:
- **PREFERS_OVER** - Preference relationships (context, reason)
- **EXEMPLIFIES** - Pattern examples (pattern_type, confidence)
- **CONTRADICTS** - Conflicting information (resolution)
- **REINFORCES** - Strengthens patterns (observations)
- **INVALIDATED_BY** - Superseded information (reason, timestamp)
- **EVOLVED_INTO** - Knowledge evolution (confidence)
- **DERIVED_FROM** - Derived knowledge (transformation)
- **PART_OF** - Hierarchical relationships (role)

### 3. **Temporal Validity** ⏰
Memories now support bi-temporal tracking:
- `t_valid` - When something became true
- `t_invalid` - When it stopped being true
- `timestamp` - When we learned about it (ingestion time)

### 4. **Enrichment Pipeline** 🔄
Asynchronous background processing that:
- Classifies memory types
- Extracts entities (tools, projects, people)
- Discovers temporal relationships
- Detects and reinforces patterns
- Builds preference graphs

### 5. **Pattern Detection** 🔍
Automatically discovers recurring patterns:
- Identifies similar memories
- Creates Pattern nodes with increasing confidence
- Links observations via EXEMPLIFIES relationships
- Confidence grows with more observations

### 6. **Analytics Endpoint** 📊
New `/analyze` endpoint provides insights:
- Memory type distribution
- Pattern discoveries
- Preference mappings
- Temporal insights (activity by time)
- Entity frequency analysis
- Confidence distribution

## 📝 API Changes

### Store Memory - Enhanced Response
```json
POST /memory
{
  "content": "I decided to use FalkorDB",
  "tags": ["decision"],
  "t_valid": "2024-01-01T00:00:00Z",  // Optional
  "t_invalid": null                    // Optional
}

Response:
{
  "status": "success",
  "memory_id": "uuid",
  "type": "Decision",        // Auto-classified
  "confidence": 0.85,        // Classification confidence
  "enrichment": "queued"     // Background processing status
}
```

### Create Association - With Properties
```json
POST /associate
{
  "memory1_id": "id1",
  "memory2_id": "id2",
  "type": "PREFERS_OVER",
  "strength": 0.9,
  "context": "cost-effectiveness",  // Type-specific property
  "reason": "30x cost difference"   // Type-specific property
}
```

### Analytics
```json
GET /analyze

Response:
{
  "analytics": {
    "memory_types": {
      "Decision": {"count": 10, "average_confidence": 0.82},
      "Pattern": {"count": 5, "average_confidence": 0.71}
    },
    "patterns": [
      {
        "type": "Habit",
        "description": "Pattern of Habit",
        "confidence": 0.85,
        "observations": 5
      }
    ],
    "preferences": [
      {
        "prefers": "FalkorDB",
        "over": "ArangoDB",
        "context": "cost",
        "strength": 0.95
      }
    ],
    "entity_frequency": {
      "tools": [["Railway", 5], ["FalkorDB", 3]],
      "projects": [["AutoMem", 8]]
    }
  }
}
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/test_app.py -v
```

Run the demo script:
```bash
python scripts/demo_pkg_features.py
```

## 🎯 Benefits

1. **Better Memory Organization** - Automatic classification makes recall more precise
2. **Pattern Discovery** - Automatically finds recurring behaviors and habits
3. **Preference Learning** - Builds knowledge of choices and preferences over time
4. **Temporal Awareness** - Tracks how knowledge and preferences evolve
5. **Richer Relationships** - More nuanced connections between memories
6. **Progressive Intelligence** - System gets smarter with more data

## 🔄 Backward Compatibility

All changes are backward compatible:
- Simple API remains unchanged (just send content, tags, importance)
- New fields are optional
- Enrichment happens asynchronously
- Original endpoints still work exactly as before

## 🚀 Future Enhancements

Potential next steps (not implemented):
- Cross-domain correlation analysis
- Predictive suggestions based on patterns
- Anomaly detection for unusual behaviors
- Complex entity relationships
- Advanced temporal reasoning
- Multi-user pattern learning

## 📚 Implementation Details

The implementation follows a two-layer architecture:
1. **Simple Ingestion Layer** - LLMs continue to use the simple API
2. **Intelligent Processing Layer** - Backend enrichment adds intelligence

This design ensures:
- Zero configuration for LLM clients
- Progressive enhancement over time
- Eventually consistent enrichment
- Source-agnostic processing
