# AutoMem LoCoMo Benchmark Reproducibility Guide

This document provides everything needed to reproduce our benchmark results.

## Quick Start

```bash
# 1. Start AutoMem
make dev

# 2. Run benchmark with all modes
./tests/benchmarks/run_all_benchmarks.sh
```

## Detailed Steps

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/your-org/automem.git
cd automem

# Install dependencies
make install
source venv/bin/activate

# Install benchmark dependencies
pip install nltk==3.9.1
python -c "import nltk; nltk.download('punkt')"
```

### 2. Start AutoMem Stack

```bash
# Start FalkorDB + Qdrant + AutoMem
make dev

# Verify health
curl http://localhost:8001/health
```

### 3. Run Benchmarks

#### Option A: Python Benchmark (Recommended)

```bash
# Default mode (retrieval + official F1)
python tests/benchmarks/test_locomo.py --output results_default.json

# Strict mode (no evidence hints)
python tests/benchmarks/test_locomo.py --no-evidence-hints --output results_strict.json

# E2E mode (LLM generates answers)
python tests/benchmarks/test_locomo.py --eval-mode e2e --output results_e2e.json
```

#### Option B: CORE-Compatible Benchmark (For Comparison)

```bash
cd tests/benchmarks/core_adapter
npm install
export OPENAI_API_KEY="your-key"
node evaluate.js
```

### 4. Compare Results

| Mode | Description | Expected Accuracy |
|------|-------------|-------------------|
| Default | Retrieval + F1 + evidence hints | ~90% |
| Strict | No evidence hints | ~85% |
| E2E | LLM generates answers | ~87% |
| CORE-compat | CORE methodology | ~85% |

## Evaluation Modes Explained

### Retrieval Mode (Default)
- Stores conversations as memories
- Retrieves relevant memories for each question
- Checks if expected answer appears in retrieved memories
- Uses official F1 scoring with Porter stemmer

### E2E Mode
- Same retrieval as above
- Generates answer using LLM (GPT-4o-mini by default)
- Scores generated answer against expected
- Matches CORE's evaluation methodology

### Strict Mode
- Same as retrieval mode
- Disables evidence ID hints (no "cheating")
- More realistic evaluation of retrieval quality

## Configuration Options

```bash
# All available options
python tests/benchmarks/test_locomo.py --help

# Key options:
--eval-mode {retrieval,e2e}  # Evaluation mode
--no-official-f1             # Use original word overlap (not recommended)
--no-evidence-hints          # Disable evidence ID hints
--e2e-model MODEL            # Model for E2E generation (default: gpt-4o-mini)
--f1-threshold FLOAT         # F1 threshold for "correct" (default: 0.5)
```

## Verifying Results

### 1. Check Output Files

```bash
# View results summary
cat results_default.json | jq '.overall'

# View category breakdown
cat results_default.json | jq '.categories'

# View official F1 metrics
cat results_default.json | jq '.official_f1'
```

### 2. Compare with Baseline

Our baseline scores (as of Dec 2025):

| Category | Retrieval | E2E | CORE (claimed) |
|----------|-----------|-----|----------------|
| Single-hop | 79.8% | 82% | 91% |
| Temporal | 85.1% | 84% | 88% |
| Multi-hop | 50.0% | 55% | 85% |
| Open Domain | 95.8% | 92% | 71% |
| Complex Reasoning | 100% | 95% | N/A |
| **Overall** | **90.5%** | **87%** | **~85%** |

### 3. Run CORE's Benchmark

To directly compare with CORE:

```bash
# Their benchmark
git clone https://github.com/RedPlanetHQ/core-benchmark
cd core-benchmark

# Replace search service with AutoMem adapter
cp ../automem/tests/benchmarks/core_adapter/automem-search.js services/

# Run their evaluation
npm run evaluate
```

## Known Differences from CORE

1. **Multi-hop scoring**: CORE claims 85%, we get ~50%. Their definition of "multi-hop" may differ.
2. **Open Domain**: We get 95.8%, CORE claims 71%. Category interpretation may differ.
3. **Overall formula**: CORE's 88.24% doesn't match their category breakdown math.

## Troubleshooting

### "AutoMem API is not accessible"
- Verify `make dev` is running
- Check `docker ps` shows all containers
- Try `curl http://localhost:8001/health`

### Low scores on first run
- Wait for enrichment pipeline (2-3 min after data load)
- Re-run benchmark after enrichment completes

### Different results each run
- Check for leftover test data: `curl http://localhost:8001/memory/stats`
- Clean up: The benchmark cleans up automatically, but you can force it

## Dependencies

### Python
```
nltk==3.9.1
openai>=1.0.0
requests>=2.28.0
python-dateutil>=2.8.0
```

### Node.js (CORE adapter)
```
axios>=1.6.0
openai>=4.20.0
dotenv>=16.3.0
```

## Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...          # For E2E mode and CORE adapter

# Optional
AUTOMEM_TEST_BASE_URL=http://localhost:8001
AUTOMEM_TEST_API_TOKEN=test-token
EVAL_MODEL=gpt-4o-mini         # Model for E2E evaluation
```

## Citing Results

When citing AutoMem benchmark results, please specify:
- Evaluation mode (retrieval/e2e)
- Whether evidence hints were used
- F1 threshold used
- AutoMem version/commit hash
- Date of evaluation

Example:
> AutoMem achieved 90.53% accuracy on LoCoMo-10 benchmark (retrieval mode,
> official F1 scoring with Porter stemmer, evidence hints enabled, F1 threshold 0.5)
> using commit abc123 on December 3, 2025.
