# LoCoMo Benchmark - Improvement Plan

## Current Results (Baseline)

**Overall Accuracy**: 70.69% (1404/1986 questions)
**Target (CORE SOTA)**: 88.24%
**Gap**: 17.55%

### Category Performance

| Category | AutoMem | Notes |
|----------|---------|-------|
| **Complex Reasoning** | 99.78% (445/446) | ✅ Nearly perfect |
| **Open Domain** | 83.12% (699/841) | ✅ Strong |
| **Single-hop Recall** | 54.96% (155/282) | ⚠️ Moderate |
| **Temporal Understanding** | 26.17% (84/321) | ❌ Weak |
| **Multi-hop Reasoning** | 21.88% (21/96) | ❌ Weak |

---

## Root Cause Analysis

### 1. Temporal Understanding (26.17%)

**Problem**: Questions about dates, times, sequences, and "when" events occurred.

**Example Questions**:
- "When did Caroline go to the LGBTQ support group?" → Expected: "7 May 2023"
- "When did Melanie run a charity race?" → Expected: "The sunday before 25 May 2023"

**Why We're Failing**:
1. **Date format mismatch**: Memory contains "I went to a LGBTQ support group yesterday" but doesn't store the absolute date "7 May 2023"
2. **Relative time**: "yesterday", "last week", "next month" aren't converted to absolute dates
3. **Timestamp metadata**: Dialog metadata contains `session_datetime` but we're not using it for temporal queries
4. **No temporal enrichment**: AutoMem's enrichment pipeline doesn't extract/normalize dates from content

**Solutions**:

#### Short-term (Quick Wins)
1. **Use session metadata for temporal queries**
   - When question contains "when", include `session_datetime` in matching
   - Parse dates from session metadata (currently "2:01 pm on 21 October, 2022")
   
2. **Temporal keywords in recall**
   - Boost memories with date keywords when question has temporal indicators
   - Add `time_query` parameter to `/recall` API for date range filtering

#### Medium-term (Requires Changes)
3. **Date normalization in enrichment**
   - Extract dates from content ("yesterday" → actual date based on session_datetime)
   - Store normalized dates in metadata
   - Add temporal tags: `date:2023-05-07`, `month:2023-05`, `year:2023`

4. **Temporal reasoning in answer matching**
   - Parse expected date formats (various formats in dataset)
   - Match against session_datetime, not just content
   - Handle relative dates ("the sunday before 25 May 2023")

#### Long-term (Architecture)
5. **Temporal knowledge graph**
   - Build temporal relationships: `OCCURRED_BEFORE`, `OCCURRED_AFTER`
   - Query by time range: "What happened between March and May?"
   - Timeline reconstruction

---

### 2. Multi-hop Reasoning (21.88%)

**Problem**: Questions requiring multiple pieces of information from different dialogs.

**Example**:
- Question: "What fields would Caroline be likely to pursue in her education?"
- Answer: "Psychology, counseling certification"
- Evidence: Requires connecting D1:9 (psychology interest) + D1:11 (counseling goal)

**Why We're Failing**:
1. **Single-recall approach**: We query once, get top 50 memories, but may miss one of the hops
2. **No graph traversal**: Not using FalkorDB relationships to "follow" connections
3. **Evidence matching**: We check if evidence dialog is in top 50, but don't verify we got ALL evidence dialogs

**Solutions**:

#### Short-term
1. **Increase recall limit for multi-hop questions**
   - Detect multi-hop questions (multiple evidence dialogs)
   - Increase limit from 50 → 100 for these questions

2. **Multiple recall passes**
   - First pass: Get initial memories
   - Extract entities/topics from recalled memories
   - Second pass: Query for related memories using extracted entities

#### Medium-term
3. **Use graph relationships**
   - After initial recall, traverse `RELATES_TO` edges in FalkorDB
   - Pull in connected memories that might contain other evidence

4. **Evidence completeness check**
   - For questions with N evidence dialogs, verify we recalled N dialogs
   - If missing, do targeted recall for missing dialog IDs

#### Long-term
5. **Multi-hop query planning**
   - Decompose question into sub-questions
   - Execute sub-queries in sequence
   - Combine results for final answer

---

### 3. Single-hop Recall (54.96%)

**Problem**: Even simple "recall one fact" questions only get 55% accuracy.

**Why We're Failing**:
1. **Semantic search limitations**: Question phrasing differs from memory content
2. **Answer format mismatch**: Answer might be paraphrased in memory
3. **Confidence threshold**: 0.5 threshold might be too strict OR too lenient

**Solutions**:

#### Short-term
1. **Use evidence dialog IDs more effectively**
   - We have the ground truth dialog IDs in `evidence` field
   - Current approach: check if any recalled memory matches evidence ID
   - Improved: Directly fetch evidence dialog IDs, guarantee they're in context

2. **Query expansion**
   - Extract key entities from question
   - Add entity synonyms to query
   - Example: "Caroline" → "Caroline", "she", "her"

#### Medium-term
3. **Hybrid ranking optimization**
   - Tune weights: semantic similarity vs keyword match vs tag match
   - Currently using default Qdrant scoring
   - Experiment with re-ranking recalled memories

4. **Answer extraction improvement**
   - Use LLM to extract answer from recalled memories
   - Current: Simple word overlap matching
   - Better: GPT-4o-mini to read memories and answer question

---

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 days)
**Target**: 70% → 75%

- [ ] Increase recall limit for multi-hop questions (50 → 100)
- [ ] Use session_datetime metadata for temporal question matching
- [ ] Implement query expansion for entity extraction
- [ ] Add temporal keywords boost in scoring

### Phase 2: Core Improvements (1 week)
**Target**: 75% → 82%

- [ ] Date normalization in enrichment pipeline
- [ ] Multiple recall passes for multi-hop
- [ ] Graph relationship traversal for evidence finding
- [ ] LLM-based answer extraction (replace word overlap)

### Phase 3: Advanced Features (2-3 weeks)
**Target**: 82% → 88%+

- [ ] Temporal knowledge graph with time-based relationships
- [ ] Multi-hop query planning and decomposition
- [ ] Evidence completeness verification
- [ ] Hybrid ranking optimization with learned weights

---

## Testing Strategy

### Continuous Testing
- Run benchmark after each improvement
- Track per-category scores
- Use `--test-one` for fast iteration

### A/B Testing
- Keep baseline version
- Test improvements in isolation
- Measure delta for each change

### Regression Prevention
- Save successful runs as fixtures
- Add category-specific test cases
- Don't break Complex Reasoning (99.78%)!

---

## Next Steps

1. **Analyze failure cases**
   ```bash
   python tests/benchmarks/test_locomo.py --debug --save-failures failures.json
   ```

2. **Profile temporal questions**
   - Extract all category=2 questions
   - Manual review of top 10 failures
   - Identify common patterns

3. **Profile multi-hop questions**
   - Extract all questions with len(evidence) > 1
   - Check if we're recalling ANY evidence vs ALL evidence
   - Measure hop coverage

4. **Implement Phase 1 improvements**
   - Start with temporal metadata matching (easiest)
   - Then multi-hop recall limit increase
   - Measure impact

---

## Resources

- LoCoMo paper: https://arxiv.org/abs/2407.03350
- CORE results: 88.24% (SOTA as of 2024)
- AutoMem API: http://localhost:8001/docs
- Benchmark code: `tests/benchmarks/test_locomo.py`

---

**Updated**: 2025-10-15
**Status**: ✅ Baseline established, improvement plan ready

