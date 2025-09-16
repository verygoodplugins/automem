# Memory Consolidation Engine

Successfully implemented a dream-inspired memory consolidation system that mimics biological memory processing.

## Features Implemented

### 1. Exponential Decay Scoring
- Memories decay over time using exponential decay function
- Factors affecting relevance:
  - **Age decay**: Older memories naturally fade
  - **Access reinforcement**: Recently accessed memories are strengthened
  - **Relationship preservation**: Connected memories resist decay
  - **Importance weighting**: User-defined priority affects retention
  - **Confidence factor**: Well-classified memories are preserved

### 2. Creative Association Discovery (REM-like)
- Randomly samples disparate memories
- Discovers non-obvious connections:
  - **CONTRASTS_WITH**: Opposite decisions/preferences
  - **EXPLAINS**: Insights that explain patterns
  - **SHARES_THEME**: Similar contexts across different types
  - **PARALLEL_CONTEXT**: Different topics from same time period
- Creates new relationship edges with confidence scores

### 3. Semantic Clustering
- Uses DBSCAN clustering on embeddings
- Groups highly similar memories
- Creates meta-memories for large clusters (5+ memories)
- Tracks temporal span and dominant types
- Enables knowledge compression

### 4. Controlled Forgetting
- Two-tier system:
  - **Archive threshold** (0.2): Low-relevance memories are archived
  - **Delete threshold** (0.05): Very old, unused memories are deleted
- Preserves graph structure
- Dry-run mode for safety

### 5. Consolidation Scheduler
- Different frequencies for different operations:
  - **Decay**: Daily (quick score updates)
  - **Creative**: Weekly (discover associations)
  - **Clustering**: Monthly (reorganize knowledge)
  - **Forgetting**: Quarterly (permanent changes)
- Tracks history and next run times

## API Endpoints

```python
# Run consolidation
POST /consolidate
{
    "mode": "full|decay|creative|cluster|forget",
    "dry_run": true  # Test without making changes
}

# Get scheduler status
GET /consolidate/status
{
    "next_runs": {
        "decay": "Due now",
        "creative": "In 6 days",
        "cluster": "In 29 days",
        "forget": "In 89 days"
    },
    "history": [...]
}
```

## Biological Inspiration

The system mimics how human memory works:

1. **Sleep Consolidation**: Like REM sleep strengthens important memories
2. **Creative Dreams**: Discovers unexpected connections between memories
3. **Memory Compression**: Similar experiences merge into general patterns
4. **Adaptive Forgetting**: Irrelevant details fade while core insights remain

## Configuration

Tunable parameters in `consolidation.py`:

```python
# Decay parameters
base_decay_rate = 0.1          # Daily decay rate
reinforcement_bonus = 0.2       # Strength added when accessed
relationship_preservation = 0.3 # Extra weight for connected memories

# Clustering parameters
min_cluster_size = 3
similarity_threshold = 0.75

# Forgetting thresholds
archive_threshold = 0.2    # Archive below this relevance
delete_threshold = 0.05    # Delete below this
```

## Testing

Run consolidation tests:

```bash
# Test with Docker stack running
make dev

# In another terminal
python test_consolidation_auto.py

# Or interactive test
python test_consolidation.py
```

## Benefits

1. **Self-organizing**: System automatically maintains itself
2. **Relevance-based**: Important memories persist, noise fades
3. **Pattern emergence**: Discovers hidden connections
4. **Space efficient**: Clusters reduce redundancy
5. **Biologically inspired**: Mimics proven memory systems

## Next Steps

Future enhancements could include:
- Importance prediction based on access patterns
- Consolidation triggers based on memory load
- User-specific decay rates
- Export of discovered patterns
- Integration with recall scoring

The consolidation engine makes AutoMem a living, breathing knowledge system that learns, forgets, and discovers - just like biological memory.