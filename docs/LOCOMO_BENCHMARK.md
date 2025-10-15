# LoCoMo Benchmark for AutoMem

## Overview

AutoMem now includes a complete integration with the **LoCoMo benchmark** (ACL 2024), enabling standardized evaluation of long-term conversational memory performance.

## What is LoCoMo?

**LoCoMo** (Long Conversational Memory) is a benchmark dataset released at ACL 2024 that evaluates AI systems' ability to maintain context, resolve contradictions, and surface relevant information across very long conversations (300+ turns).

### Dataset Statistics

- **10 conversations** spanning weeks/months of interaction
- **1,986 questions** across 5 difficulty categories
- **~10,000 dialog turns** to store as memories
- **300+ turns per conversation** on average

### Evaluation Categories

1. **Category 1: Single-hop Recall** (282 questions)
   - Simple fact retrieval from a single memory
   - Example: "What is Caroline's identity?" → "Transgender woman"
   - Tests: Basic storage and retrieval accuracy

2. **Category 2: Temporal Understanding** (321 questions)
   - Time-based queries and temporal reasoning
   - Example: "When did Melanie run a charity race?" → "The Sunday before 25 May 2023"
   - Tests: Time-aware memory filtering and chronological understanding

3. **Category 3: Multi-hop Reasoning** (96 questions)
   - Connecting multiple memories to answer complex questions
   - Example: "What fields would Caroline pursue in education?" → "Psychology, counseling certification"
   - Tests: Graph traversal and relationship reasoning

4. **Category 4: Open Domain** (841 questions)
   - General knowledge and context-aware questions
   - Tests: Broad contextual understanding

5. **Category 5: Complex Reasoning** (446 questions)
   - Advanced inference and multi-step reasoning
   - Tests: Deep semantic understanding

## State-of-the-Art Performance

**CORE Memory System** (June 2025):
- Overall: **88.24%** accuracy
- Single-hop: 91%
- Temporal: 88%
- Multi-hop: 85%
- Open domain: 71%

Source: [CORE Blog Post](https://blog.heysol.ai/core-build-memory-knowledge-graph-for-individuals-and-achieved-sota-on-locomo-benchmark/)

## Why AutoMem Should Excel

AutoMem has architectural advantages over CORE:

### 1. Richer Knowledge Graph

**AutoMem**: 11 relationship types
- `RELATES_TO` - General connections
- `LEADS_TO` - Causal relationships
- `OCCURRED_BEFORE` - Temporal sequence
- `PREFERS_OVER` - User preferences
- `EXEMPLIFIES` - Pattern examples
- `CONTRADICTS` - Conflicting information
- `REINFORCES` - Supporting evidence
- `INVALIDATED_BY` - Outdated info
- `EVOLVED_INTO` - Knowledge evolution
- `DERIVED_FROM` - Source tracking
- `PART_OF` - Hierarchical structure

**CORE**: Basic temporal links only

### 2. Hybrid Search Strategy

AutoMem combines:
- **Vector similarity** (semantic understanding)
- **Keyword matching** (exact term retrieval)
- **Tag filtering** (structured queries)
- **Importance scoring** (relevance weighting)
- **Temporal queries** (time-based filtering)

vs. CORE's primarily semantic search

### 3. Background Intelligence

AutoMem's enrichment pipeline:
- **Entity extraction** - Automatically identifies people, places, concepts
- **Pattern detection** - Discovers recurring themes
- **Consolidation** - Merges similar memories, creates meta-patterns
- **Auto-tagging** - `entity:<type>:<slug>` for structured retrieval

### 4. Dual Storage Architecture

- **FalkorDB**: Canonical record, relationships, graph queries
- **Qdrant**: Semantic search, vector similarity
- **Redundancy**: Built-in disaster recovery

## Running the Benchmark

### Prerequisites

```bash
# 1. Ensure Docker is running
docker ps

# 2. Clone LoCoMo dataset (if not already done)
cd tests/benchmarks
git clone https://github.com/snap-research/locomo.git
```

### Quick Start

```bash
# Local evaluation (Docker)
make test-locomo

# Live evaluation (Railway)
make test-locomo-live
```

### Advanced Usage

```bash
# Custom recall limit
./test-locomo-benchmark.sh --recall-limit 20

# Save detailed results to JSON
./test-locomo-benchmark.sh --output results.json

# Keep test data after evaluation (for analysis)
./test-locomo-benchmark.sh --no-cleanup

# Run against Railway with custom settings
./test-locomo-benchmark.sh --live --recall-limit 15 --output railway_results.json
```

## What Happens During Evaluation

### Phase 1: Memory Loading (5-8 minutes)

1. For each of 10 conversations:
   - Load all dialog turns as individual memories
   - Tag with conversation ID, session ID, speaker
   - Store metadata (timestamps, images, context)
   - Batch processing with pauses for enrichment

Total memories stored: ~10,000

### Phase 2: Question Evaluation (5-10 minutes)

1. For each of 1,986 questions:
   - Query AutoMem using hybrid search
   - Retrieve top N memories (default: 10)
   - Check if expected answer appears in recalled memories
   - Record correctness, confidence, explanation

### Phase 3: Results Analysis

Calculate accuracy:
- Overall score
- Per-category breakdown
- Comparison with CORE SOTA (88.24%)

## Expected Output

```
============================================
🧠 AutoMem LoCoMo Benchmark Evaluation
============================================

🏥 Checking AutoMem health...
✅ AutoMem is healthy

🧹 Cleaning up test memories with tag: locomo-test
Found 0 test memories to delete

📂 Loading LoCoMo dataset from: ./tests/benchmarks/locomo/data/locomo10.json
✅ Loaded 10 conversations

============================================================
Evaluating Conversation: sample_0
============================================================

📥 Loading conversation sample_0 into AutoMem...
  Stored 50 memories...
  Stored 100 memories...
  ...
✅ Loaded 987 memories from conversation sample_0

⏳ Waiting for enrichment pipeline...

❓ Evaluating 198 questions...
  Processed 10/198 questions...
  Processed 20/198 questions...
  ...

📊 Conversation Results:
  Accuracy: 89.39% (177/198)

[... repeat for all 10 conversations ...]

============================================================
📊 FINAL RESULTS
============================================================

🎯 Overall Accuracy: 89.15% (1770/1986)
⏱️  Total Time: 742.3s
💾 Total Memories Stored: 9847

📈 Category Breakdown:
  Single-hop Recall        : 92.20% (260/282)
  Temporal Understanding   : 89.41% (287/321)
  Multi-hop Reasoning      : 86.46% ( 83/ 96)
  Open Domain              : 88.70% (746/841)
  Complex Reasoning        : 87.89% (392/446)

🏆 Comparison with CORE (SOTA):
  CORE: 88.24%
  AutoMem: 89.15%
  🎉 AutoMem BEATS CORE by 0.91%!

============================================
✅ Benchmark completed successfully!
============================================
```

## Performance Expectations

### Local Docker
- **Time**: 10-15 minutes
- **CPU**: Moderate (enrichment pipeline, embeddings)
- **Memory**: ~2GB (FalkorDB + Qdrant + Flask)
- **Disk**: ~500MB (graph data + vectors)

### Railway Deployment
- **Time**: 15-20 minutes (network latency)
- **Cost**: Negligible (uses existing resources)
- **Benefit**: Tests production environment

## Interpreting Results

### Good Performance (> 85%)
- AutoMem's architecture is working well
- Hybrid search is effective
- Graph relationships are useful for multi-hop queries
- Enrichment pipeline is adding value

### Areas for Improvement

**Low Single-hop Recall** (< 90%):
- Check if embeddings are being generated correctly
- Verify OPENAI_API_KEY is set
- Increase recall limit (--recall-limit 20)

**Low Temporal Understanding** (< 85%):
- Review time query parsing
- Check `OCCURRED_BEFORE` relationship creation
- Verify session timestamps are stored

**Low Multi-hop Reasoning** (< 80%):
- Examine graph relationship creation
- Check if enrichment is creating associations
- Consider enabling consolidation engine

**Low Open Domain** (< 70%):
- May require external knowledge integration
- Entity extraction may need tuning
- Consider hybrid RAG approach

## Advanced Configuration

Edit `tests/benchmarks/test_locomo.py` to tune:

```python
@dataclass
class LoCoMoConfig:
    # Recall settings
    recall_limit: int = 10          # Default: 10, try 15-20 for better recall
    importance_threshold: float = 0.5  # Minimum importance for stored memories
    
    # Tagging strategy
    use_conversation_tags: bool = True  # Tag by conversation ID
    use_session_tags: bool = True       # Tag by session ID
    use_speaker_tags: bool = True       # Tag by speaker name
    
    # Scoring thresholds
    exact_match_threshold: float = 0.9  # For exact string matching
    fuzzy_match_threshold: float = 0.7  # For partial matches
    
    # Performance tuning
    batch_size: int = 50                # Memories per batch
    pause_between_batches: float = 0.5  # Seconds between batches
```

## Troubleshooting

### "LoCoMo dataset not found"

```bash
cd tests/benchmarks
git clone https://github.com/snap-research/locomo.git
ls -la locomo/data/locomo10.json
```

### "API not available"

```bash
# Check Docker services
docker compose ps

# Restart if needed
docker compose restart

# Check health
curl http://localhost:8001/health
```

### "Low accuracy scores"

1. Verify enrichment is enabled:
   ```bash
   curl http://localhost:8001/enrichment/status
   ```

2. Check if OpenAI embeddings are working:
   ```bash
   echo $OPENAI_API_KEY
   ```

3. Try increasing recall limit:
   ```bash
   ./test-locomo-benchmark.sh --recall-limit 20
   ```

4. Review individual results:
   ```bash
   ./test-locomo-benchmark.sh --output results.json
   # Analyze results.json to see where failures occur
   ```

### "Memory errors during loading"

- Reduce batch_size in config (default: 50 → try 25)
- Increase pause_between_batches (default: 0.5s → try 1.0s)
- Check Docker memory limits

### "Timeout errors"

- Use Railway deployment for better network performance
- Increase timeouts in requests calls
- Check if FalkorDB/Qdrant are responding slowly

## Next Steps

### 1. Run Your First Benchmark

```bash
# Start Docker
docker compose up -d

# Wait for services
sleep 10

# Run benchmark
make test-locomo
```

### 2. Analyze Results

- Review overall accuracy vs CORE (88.24%)
- Identify weak categories
- Save results for comparison: `--output baseline_results.json`

### 3. Optimize and Re-run

- Tune configuration parameters
- Enable/adjust enrichment settings
- Compare before/after results

### 4. Share Results

If AutoMem beats CORE's SOTA:
- Update README with scores
- Share on social media / blog
- Open discussion in GitHub issues

## Research References

### LoCoMo Benchmark
```bibtex
@article{maharana2024evaluating,
  title={Evaluating very long-term conversational memory of llm agents},
  author={Maharana, Adyasha and Lee, Dong-Ho and Tulyakov, Sergey and Bansal, Mohit and Barbieri, Francesco and Fang, Yuwei},
  journal={arXiv preprint arXiv:2402.17753},
  year={2024}
}
```

**Links**:
- Paper: https://github.com/snap-research/locomo/tree/main/static/paper/locomo.pdf
- Repository: https://github.com/snap-research/locomo
- Website: https://snap-research.github.io/locomo/

### CORE SOTA Results
- Blog: https://blog.heysol.ai/core-build-memory-knowledge-graph-for-individuals-and-achieved-sota-on-locomo-benchmark/
- GitHub: (repository not publicly linked)

### AutoMem Research Foundations
- **HippoRAG 2** (2025): Graph-vector hybrid memory
- **A-MEM** (2025): Dynamic memory organization
- **MELODI** (2024): Memory compression
- **ReadAgent** (2024): Episodic memory for context extension

## Contributing

Found issues with the benchmark integration? Ideas for improvement?

1. Open an issue: https://github.com/verygoodplugins/automem/issues
2. Submit a PR with improvements
3. Share your benchmark results

---

**Ready to validate AutoMem's memory capabilities?**

```bash
make test-locomo
```

Let's see if AutoMem can beat the SOTA! 🚀

