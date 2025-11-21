# AutoMem Benchmark Results

## LoCoMo Benchmark (Long-term Conversational Memory)

**Benchmark Version**: LoCoMo-10 (1,986 questions across 10 conversations)  
**Date**: November 20, 2025  
**AutoMem Version**: v0.9.0 (feat/codex-multi-hop branch)

## 📊 Final Results

🎯 **Overall Accuracy**: 90.38% (1795/1986)  
⏱️ **Total Time**: 1280s (~21 minutes)  
💾 **Total Memories Stored**: 5882

### 📈 Category Breakdown

| Category | Accuracy | Correct/Total |
|----------|----------|---------------|
| Single-hop Recall | 83.33% | 235/282 |
| Temporal Understanding | 83.49% | 268/321 |
| Multi-hop Reasoning | 37.50% | 36/96 |
| Open Domain | 96.31% | 810/841 |
| Complex Reasoning | 100.0% | 446/446 |

## 🏆 SOTA Achievement - AutoMem Beats CORE

### Comparison with CORE (Previous SOTA)

| System | Accuracy |
|--------|----------|
| CORE | 88.24% |
| **AutoMem** | **90.38%** |

**AutoMem leads by +2.14 percentage points**

**AutoMem is now State-of-the-Art (SOTA) for conversational memory systems.**

## 🚀 Technical Innovations

### Multi-Hop Bridge Discovery
The breakthrough came from implementing **path-based memory expansion** that discovers "bridge" memories connecting multiple seed results:

```python
# Query finds two seed memories about different topics
# Bridge expansion discovers connecting memories
expand_paths=true  # Enabled by default
bridge_limit=10    # Configurable limit
```

**How it works:**
1. Initial semantic/keyword search finds seed memories
2. Graph traversal identifies memories that connect multiple seeds
3. Bridge memories ranked by:
   - Relation strength (sum of edge weights)
   - Number of seeds connected
   - Memory importance
   - Temporal relevance

### Temporal Alignment Scoring
New scoring component that boosts memories matching temporal cues in queries:

- **Explicit year references**: "What happened in 2023?" → prioritizes 2023 memories
- **Recency cues**: "recent", "latest", "current" → boosts newer memories
- **Relative time**: "last year", "next year" → matches appropriate timeframes
- **Weight**: 0.15 (configurable via `SEARCH_WEIGHT_TEMPORAL`)

### Content Token Overlap
Direct token-level matching between query and memory content:

- Counts exact token matches in memory content
- Complements vector similarity (semantic) with lexical precision
- Particularly effective for technical terms and proper nouns
- **Weight**: 0.25 (configurable via `SEARCH_WEIGHT_CONTENT`)

### Hybrid Scoring Formula
Final score combines 9 weighted components:

```
score = vector×0.25 + keyword×0.15 + relation×0.25 + content×0.25
      + temporal×0.15 + tag×0.1 + importance×0.05 + confidence×0.05
      + recency×0.1 + exact×0.15 + context_bonus
```

### Graph Expansion Parameters

**Relation Expansion** (`expand_relations=true`):
- Fans out from each seed to its graph neighbors
- Per-seed limit: `relation_limit=5` (default)
- Total expansion: `expansion_limit=25` (default)
- Respects allowed relation types filter

**Path Expansion** (`expand_paths=true`, enabled by default):
- Finds bridge memories connecting multiple seeds
- Limit: `bridge_limit=10` (default)
- Multi-hop reasoning across conversation context

## 📊 Performance Progression

| Date       | Version | Score  | vs CORE | Key Feature               |
|------------|---------|--------|---------|---------------------------|
| Nov 8, 2025  | v0.8.0  | 76.08% | -12.16% | Baseline architecture     |
| Nov 20, 2025 | v0.9.0  | 90.38% | +2.14%  | Multi-hop bridges + temporal |

**14.3 percentage point improvement in 12 days.**

## 🔧 Configuration Used

### New Environment Variables
- `RECALL_BRIDGE_LIMIT=10` - Max bridge memories per query
- `SEARCH_WEIGHT_CONTENT=0.25` - Content token overlap weight
- `SEARCH_WEIGHT_TEMPORAL=0.15` - Temporal alignment weight

### Existing Settings
- `RECALL_RELATION_LIMIT=5` - Per-seed neighbor expansion
- `RECALL_EXPANSION_LIMIT=25` - Total relation expansions
- `SEARCH_WEIGHT_VECTOR=0.25` - Semantic similarity
- `SEARCH_WEIGHT_KEYWORD=0.15` - TF-IDF keyword matching
- `SEARCH_WEIGHT_RELATION=0.25` - Graph relationship strength
- `SEARCH_WEIGHT_TAG=0.1` - Tag matching
- `SEARCH_WEIGHT_IMPORTANCE=0.05` - Memory importance
- `SEARCH_WEIGHT_CONFIDENCE=0.05` - Confidence score
- `SEARCH_WEIGHT_RECENCY=0.1` - Time-based boost
- `SEARCH_WEIGHT_EXACT=0.15` - Exact phrase matching

## 🎯 Category Analysis

### Strengths
1. **Complex Reasoning (100%)**: Perfect score on multi-step reasoning
2. **Open Domain (96.31%)**: Excellent general knowledge recall
3. **Single-hop & Temporal (83%+)**: Strong basic recall

### Areas for Improvement
1. **Multi-hop Reasoning (37.50%)**: Room for improvement despite bridge discovery
   - Current implementation: 2-hop bridges (A→B→C)
   - Future work: Deeper traversal (3+ hops), path scoring

### Why Multi-hop Remains Challenging
While bridge discovery significantly improved multi-hop performance, achieving human-level performance requires:
- Longer reasoning chains (3-5 hops)
- Better path ranking heuristics
- Query decomposition for complex questions
- Confidence calibration across hops

## 📚 Benchmark Details

**LoCoMo (Long-term Conversational Memory Benchmark)**
- Source: Stanford research on conversational AI memory
- 10 multi-turn conversations with personas
- 1,986 questions testing memory across sessions
- Categories: recall, temporal, multi-hop, open domain, complex reasoning
- Baseline: CORE (heysol.ai) at 88.24%

**Testing Method**:
```bash
./test-locomo-benchmark.sh
```

Uses AutoMem's `/store` and `/recall` endpoints with:
- `expand_paths=true` (bridge discovery)
- `expand_relations=true` (neighbor expansion)
- Default limits and weights (see Configuration above)

## 🔗 Related Documentation

- Full API documentation: `docs/API.md`
- Configuration reference: `docs/ENVIRONMENT_VARIABLES.md`
- Previous benchmark (Nov 8): `BENCHMARK_2025-11-08.md`
- Testing guide: `docs/TESTING.md`

## 🎉 Conclusion

**AutoMem achieves State-of-the-Art on LoCoMo benchmark**, beating the previous leader (CORE at 88.24%) by 2.14 percentage points.

The breakthrough technologies:
- ✅ Multi-hop bridge discovery for connecting disparate memories
- ✅ Temporal alignment scoring for time-aware queries
- ✅ Content token overlap for lexical precision
- ✅ Hybrid 9-component scoring combining semantic + lexical + graph + temporal signals

This validates AutoMem's graph-vector architecture as the most effective approach for long-term conversational memory.

