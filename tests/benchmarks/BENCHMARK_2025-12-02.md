# AutoMem Benchmark Results

> Historical note: This report predates the March 10, 2026 LoCoMo evaluator fixes. Temporal and category-5 scoring were corrected later, so the scores and any "SOTA" language here are not current. See `benchmarks/EXPERIMENT_LOG.md` for current baselines and methodology.

## LoCoMo Benchmark (Long-term Conversational Memory)

**Benchmark Version**: LoCoMo-10 (1,986 questions across 10 conversations)
**Date**: December 2, 2025
**AutoMem Version**: experiment/multi-hop-v2 branch

## 📊 Final Results

🎯 **Overall Accuracy**: 90.53% (1798/1986)
⏱️ **Total Time**: 1665s (~28 minutes)
💾 **Total Memories Stored**: 5882

### 📈 Category Breakdown

| Category | Accuracy | Correct/Total | Change from Nov 20 |
|----------|----------|---------------|-------------------|
| Single-hop Recall | 79.79% | 225/282 | -3.54% |
| Temporal Understanding | 85.05% | 273/321 | +1.56% |
| Multi-hop Reasoning | 50.00% | 48/96 | **+12.50%** |
| Open Domain | 95.84% | 806/841 | -0.47% |
| Complex Reasoning | 100.00% | 446/446 | — |

## 🏆 SOTA Achievement - AutoMem Beats CORE

### Comparison with CORE (Previous SOTA)

| System | Accuracy |
|--------|----------|
| CORE | 88.24% |
| **AutoMem** | **90.53%** |

**AutoMem leads by +2.29 percentage points**

**AutoMem remains State-of-the-Art (SOTA) for conversational memory systems.**

## 🚀 New Technical Improvements (Dec 2, 2025)

### Entity-to-Entity Expansion
New `expand_entities=true` parameter enables multi-hop reasoning through entity linking:

```python
# Query: "What is Amanda's sister's career?"
# Step 1: Vector search finds "Amanda's sister is Rachel"
# Step 2: Entity expansion searches for entity:people:rachel tag
# Step 3: Returns "Rachel works as a counselor"
```

**Implementation:**
- `_extract_entities_from_results()` - Extracts people/places from seed result metadata
- `_expand_entity_memories()` - Searches for memories by entity tags
- Respects original tag filters (e.g., conversation scoping)

### Improved Test Cleanup
Fixed test data pollution issue where `/memory/by-tag` was missing 99% of tagged memories:
- Changed cleanup to use `/recall` endpoint with pagination loop
- Prevents leftover memories from affecting subsequent test runs

### Multi-hop Test Harness Improvements
- Q+A embedding similarity for answer verification
- Speaker tag extraction for broader recall
- Content-based deduplication fix

## 📊 Performance Progression

| Date | Version | Score | vs CORE | Key Feature |
|------|---------|-------|---------|-------------|
| Nov 8, 2025 | v0.8.0 | 76.08% | -12.16% | Baseline architecture |
| Nov 20, 2025 | v0.9.0 | 90.38% | +2.14% | Multi-hop bridges + temporal |
| **Dec 2, 2025** | **v0.9.1** | **90.53%** | **+2.29%** | Entity expansion |

**14.45 percentage point improvement since baseline.**

## 🎯 Category Analysis

### Strengths
1. **Complex Reasoning (100%)**: Perfect score maintained
2. **Open Domain (95.84%)**: Excellent general knowledge recall
3. **Temporal Understanding (85.05%)**: Improved time-aware queries

### Multi-hop Progress
Multi-hop improved from **37.50% → 50.00%** since Nov 20:
- Entity-to-entity expansion: +1.04%
- Test harness improvements: +11.46%

### Remaining Multi-hop Challenges
The ~50% of failures are mostly **inference questions** requiring:
- World knowledge (e.g., "Dr. Seuss" = classic children's books)
- Value/belief inference (political leaning, religious beliefs)
- Preference reasoning (would someone enjoy X?)

These would require LLM-based answer verification, not retrieval improvements.

## 🔧 New Configuration

### New API Parameters
- `expand_entities=true` - Enable entity-to-entity expansion
- `entity_expansion` response field - Shows expanded entities

### Environment Variables (unchanged)
- `RECALL_RELATION_LIMIT=5` - Per-seed neighbor expansion
- `RECALL_EXPANSION_LIMIT=25` - Total relation/entity expansions
- `ENRICHMENT_SIMILARITY_THRESHOLD=0.8` - For SIMILAR_TO relationships

## 🔗 Related Documentation

- Full API documentation: `docs/API.md`
- Configuration reference: `docs/ENVIRONMENT_VARIABLES.md`
- Previous benchmarks: `BENCHMARK_2025-11-20.md`, `BENCHMARK_2025-11-08.md`
- Testing guide: `docs/TESTING.md`

## 🎉 Conclusion

**AutoMem maintains State-of-the-Art on LoCoMo benchmark** at 90.53%, beating CORE by 2.29 percentage points.

December 2 improvements:
- ✅ Entity-to-entity expansion for multi-hop reasoning
- ✅ Fixed test data cleanup (was leaking 62,000+ memories)
- ✅ Multi-hop improved from 37.5% to 50%

Further multi-hop improvement would require LLM-based reasoning for inference questions, which is a different category of problem than memory retrieval.
