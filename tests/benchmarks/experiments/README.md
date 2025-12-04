# AutoMem Experiment Framework

Automated experimentation system for finding optimal memory retrieval configurations.

## Overview

This framework enables:
- **Grid Search**: Test predefined configurations
- **AI-Guided Exploration**: LLM agent decides what to try next
- **Ablation Studies**: Vary one parameter while holding others constant
- **Parallel Testing**: Deploy multiple Railway instances simultaneously

## Quick Start

### 1. Basic Grid Search (Local)

Run all predefined configurations against your local AutoMem instance:

```bash
# Quick test (subset of questions)
python experiment_runner.py --mode grid --quick

# Full benchmark
python experiment_runner.py --mode grid
```

### 2. AI-Guided Optimization

Let the analysis agent find optimal configuration:

```bash
# Quick exploration
python analysis_agent.py --quick --iterations 5

# Full optimization
python analysis_agent.py --iterations 20
```

### 3. Ablation Study

Test impact of a single parameter:

```bash
# Test different recall limits
python experiment_runner.py --mode ablation --param recall_limit --values 5,10,15,20,30

# Test different model combinations
python experiment_runner.py --mode ablation --param enrichment_model --values gpt-4o-mini,gpt-4.1,gpt-5.1
```

### 4. Railway Deployment (Parallel Testing)

Deploy multiple instances on Railway for parallel testing:

```bash
# Deploy 3 instances in parallel
python experiment_runner.py --mode grid --railway --parallel 3
```

## Configuration Options

### ExperimentConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_model` | text-embedding-3-small | Embedding model |
| `embedding_dim` | 1536 | Embedding dimensions |
| `enrichment_model` | gpt-4o-mini | LLM for enrichment |
| `extract_facts` | false | Enable statement extraction |
| `recall_limit` | 10 | Max memories to retrieve |
| `recall_strategy` | hybrid | vector_only, graph_only, hybrid |
| `vector_weight` | 0.6 | Weight for vector similarity |
| `graph_weight` | 0.4 | Weight for graph relationships |
| `graph_expansion_depth` | 2 | Hops for graph traversal |

### Scoring Weights

The optimization score balances:
- **Accuracy**: 40%
- **Multi-hop reasoning**: 20% (our weakness)
- **Response time**: 15%
- **Temporal understanding**: 15%
- **Cost efficiency**: 10%

## Predefined Configurations

| Name | Focus | Key Changes |
|------|-------|-------------|
| `baseline` | Default settings | Standard config |
| `large_embeddings` | Better semantic understanding | 3072-dim embeddings |
| `better_enrichment` | Better fact extraction | GPT-4.1 for enrichment |
| `with_facts` | Statement-level extraction | `extract_facts=true` |
| `more_recall` | More context | `recall_limit=20` |
| `graph_heavy` | Emphasize relationships | Higher graph weight |
| `vector_only` | Comparison baseline | No graph features |
| `premium` | Best everything | Expensive but optimal |

## Output Structure

```
experiment_results/
├── 20251204_103000_baseline.json       # Individual results
├── 20251204_103000_large_embeddings.json
├── 20251204_103000_report.txt          # Summary report
└── optimization_report_20251204.json   # Agent optimization log
```

## Analysis Agent

The analysis agent uses GPT-5.1 to:
1. Analyze experiment results
2. Identify patterns and bottlenecks
3. Suggest next configurations to try
4. Decide when to stop (convergence)

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    OPTIMIZATION LOOP                        │
├─────────────────────────────────────────────────────────────┤
│  1. Run experiments with current configs                    │
│  2. Collect results (accuracy, response time, etc.)         │
│  3. Agent analyzes: "What's working? What's not?"          │
│  4. Agent proposes: "Try these changes next"               │
│  5. If improvement < threshold for 3 iterations → STOP     │
│  6. Else → Go to step 1                                    │
└─────────────────────────────────────────────────────────────┘
```

### Example Agent Decision

```json
{
  "continue_exploring": true,
  "reasoning": "Multi-hop accuracy (35%) is still below target. The graph_heavy 
               config showed 10% improvement. I hypothesize increasing 
               graph_expansion_depth further while adding extract_facts could 
               help connect more related memories.",
  "next_experiments": [
    {
      "name": "deep_graph_facts",
      "changes": {
        "graph_expansion_depth": 4,
        "extract_facts": true,
        "enrichment_model": "gpt-4.1"
      },
      "hypothesis": "Deeper graph traversal + statement extraction will improve 
                    multi-hop by enabling better fact chaining"
    }
  ],
  "confidence": 0.75
}
```

## Tips for Best Results

1. **Start with quick tests** (`--quick`) to eliminate obviously bad configs
2. **Focus on weaknesses** - our multi-hop is 35% vs CORE's 85%
3. **Monitor response times** - stay under 100ms for production
4. **Watch token costs** - track `total_cost_usd` in results
5. **Use Railway** for final validation with production-like setup

## Railway Integration

### Setup

1. Install Railway CLI: `npm install -g @railway/cli`
2. Login: `railway login`
3. Set template ID in environment or code

### How It Works

```
┌────────────────────────────────────────────────────────────┐
│  For each config:                                          │
│  1. railway init --template automem-template               │
│  2. Set environment variables from ExperimentConfig        │
│  3. railway up --detach                                    │
│  4. Wait for healthy                                       │
│  5. Run benchmark against instance URL                     │
│  6. Collect results                                        │
│  7. railway delete                                         │
└────────────────────────────────────────────────────────────┘
```

### Environment Variables

The framework converts `ExperimentConfig` to env vars:

```bash
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIM=3072
ENRICHMENT_MODEL=gpt-4.1
EXTRACT_FACTS=true
RECALL_LIMIT=20
...
```

## Extending

### Add New Configuration Parameters

1. Add field to `ExperimentConfig` in `experiment_config.py`
2. Add to `to_env_vars()` method
3. Update `generate_experiment_grid()` with test values

### Custom Scoring Function

Modify `ExperimentResult.score()` to change optimization target:

```python
def score(self, weights: Dict[str, float] = None) -> float:
    if weights is None:
        weights = {
            "accuracy": 0.4,
            "multi_hop": 0.3,  # Increase for your priorities
            "response_time": 0.2,
            "cost": 0.1,
        }
    # ...
```

### Add Analysis Metrics

Update `ANALYSIS_PROMPT` in `analysis_agent.py` to include new metrics.

