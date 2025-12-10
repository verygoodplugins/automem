# AutoMem Benchmark Results

## LoCoMo Benchmark (Long-term Conversational Memory)

**Benchmark Version**: LoCoMo-10 (1,986 questions across 10 conversations)
**Date**: October 15, 2025
**AutoMem Version**: Latest (as of benchmark)

### Overall Performance

| Metric | AutoMem | CORE (SOTA) | Gap |
|--------|---------|-------------|-----|
| **Overall Accuracy** | **70.69%** | 88.24% | -17.55% |
| Total Correct | 1,404 / 1,986 | - | - |
| Avg. Response Time | 0.5s | - | - |
| Total Memories Stored | 5,882 | - | - |

### Category Breakdown

| Category | Questions | Correct | Accuracy | Analysis |
|----------|-----------|---------|----------|----------|
| **Complex Reasoning** | 446 | 445 | **99.78%** | ✅ Exceptional - Nearly perfect on complex multi-step reasoning |
| **Open Domain** | 841 | 699 | **83.12%** | ✅ Strong - Handles broad knowledge synthesis well |
| **Single-hop Recall** | 282 | 155 | **54.96%** | ⚠️ Moderate - Room for improvement in basic fact retrieval |
| **Temporal Understanding** | 321 | 84 | **26.17%** | ⚠️ Weak - Date/time queries need better metadata extraction |
| **Multi-hop Reasoning** | 96 | 21 | **21.88%** | ⚠️ Weak - Needs graph traversal for connecting facts |

### Per-Conversation Results

| Conversation | Memories | Questions | Accuracy |
|--------------|----------|-----------|----------|
| conv-50 | 568 | 204 | 78.92% |
| conv-43 | 680 | 242 | 76.86% |
| conv-49 | 509 | 196 | 75.00% |
| conv-48 | 681 | 239 | 74.90% |
| conv-44 | 675 | 158 | 74.68% |
| conv-41 | 663 | 193 | 74.61% |
| conv-47 | 689 | 190 | 67.37% |
| conv-42 | 629 | 260 | 61.54% |
| conv-26 | 419 | 199 | 60.30% |
| conv-30 | 369 | 105 | 58.10% |

**Average**: 70.69% (fairly consistent across conversations)

---

## Strengths

### 1. Complex Reasoning (99.78%)
AutoMem excels at questions requiring sophisticated reasoning across multiple pieces of information. The hybrid graph-vector architecture enables rich semantic understanding.

**Example questions handled well**:
- "What are the key factors influencing Maria's career decisions?"
- "How do John's basketball goals relate to his personal values?"

### 2. Open Domain (83.12%)
Strong performance on broad knowledge synthesis and open-ended questions. The vector search effectively captures semantic similarity.

**Example questions handled well**:
- "What fields would Caroline be likely to pursue in her education?"
- "What are John's suspected health problems?"

---

## Weaknesses & Improvement Plan

### 1. Temporal Understanding (26.17%) ⚠️

**Problem**: Questions about dates, times, and temporal sequences fail due to:
- Relative time references ("yesterday", "last week") not converted to absolute dates
- Session datetime metadata not used in matching
- Date format mismatches between questions and stored content

**Improvements Planned**:
1. **Phase 1**: Use session_datetime metadata for temporal matching (Target: +15%)
2. **Phase 2**: Date normalization in enrichment pipeline (Target: +10%)
3. **Phase 3**: Temporal knowledge graph with time-based relationships (Target: +10%)

**Target**: 26% → 60%

### 2. Multi-hop Reasoning (21.88%) ⚠️

**Problem**: Questions requiring multiple facts from different dialogs fail due to:
- Single-pass recall misses some evidence dialogs
- Graph relationships not traversed to find connected memories
- No verification that all evidence is present

**Improvements Planned**:
1. **Phase 1**: Increase recall limit for multi-hop questions (Target: +10%)
2. **Phase 2**: Graph relationship traversal for evidence finding (Target: +15%)
3. **Phase 3**: Multi-hop query planning and decomposition (Target: +15%)

**Target**: 22% → 65%

### 3. Single-hop Recall (54.96%) ⚠️

**Problem**: Even simple fact retrieval only achieves 55% due to:
- Query phrasing differs from memory content
- Simple word-overlap matching misses paraphrased answers
- Not fully utilizing evidence dialog IDs

**Improvements Planned**:
1. **Phase 1**: Query expansion with entity extraction (Target: +5%)
2. **Phase 2**: LLM-based answer extraction replacing word overlap (Target: +15%)
3. **Phase 3**: Hybrid ranking optimization (Target: +5%)

**Target**: 55% → 80%

---

## Projected Improvements

With the planned improvements across 3 phases:

| Phase | Timeline | Target Accuracy | Key Changes |
|-------|----------|-----------------|-------------|
| **Baseline** | Current | 70.69% | Initial implementation |
| **Phase 1** | 1-2 days | 75% (+4.31%) | Quick wins: temporal metadata, recall tuning |
| **Phase 2** | 1 week | 82% (+7%) | Core improvements: LLM extraction, graph traversal |
| **Phase 3** | 2-3 weeks | 88%+ (+6%+) | Advanced: temporal graphs, query planning |

---

## Technical Details

### Test Configuration
- **Base URL**: http://localhost:8001 (Docker)
- **Recall Limit**: 50 memories per question
- **Match Threshold**: 0.5 (word overlap confidence)
- **Enrichment Wait**: 10 seconds
- **API Token**: test-token

### Infrastructure
- **Vector DB**: Qdrant (cloud-hosted)
- **Graph DB**: FalkorDB (Railway)
- **Embeddings**: OpenAI text-embedding-3-small (768d)
- **Test Duration**: ~16 minutes (993s)

### Memory Storage
- Conversations stored with rich metadata:
  - `conversation_id`, `dialog_id`, `session_id`, `speaker`
  - `session_datetime` for temporal context
  - Tags: `conversation:conv-XX`, `session:XX`, `speaker:name`

---

## How to Reproduce

```bash
# Run the full benchmark
make test-locomo

# Test with one conversation (fast iteration)
python tests/benchmarks/test_locomo.py --test-one

# Save results to JSON
python tests/benchmarks/test_locomo.py --output results.json

# Test against production
make test-locomo-live
```

---

## References

- **LoCoMo Paper**: https://arxiv.org/abs/2407.03350
- **CORE SOTA**: 88.24% (best published result)
- **Benchmark Dataset**: 10 conversations, 1,986 questions
- **Improvement Plan**: [docs/LOCOMO_IMPROVEMENTS.md](LOCOMO_IMPROVEMENTS.md)

---

**Last Updated**: 2025-10-15
**Status**: ✅ Baseline established, improvement roadmap defined
