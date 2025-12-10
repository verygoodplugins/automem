# AutoMem CORE-Compatible Benchmark Adapter

This adapter allows running CORE's exact evaluation methodology against AutoMem,
enabling direct apples-to-apples comparison.

## Background

CORE (https://github.com/RedPlanetHQ/core-benchmark) claims:
- Single Hop: 91%
- Multi Hop: 85%
- Temporal: 88%
- Open Domain: 71%
- Overall: ~85%

However, their published overall number (88.24%) doesn't match their category breakdown.
This adapter allows us to verify our results using their exact methodology.

## How CORE Evaluates

1. **Search** - Retrieve relevant context from memory system
2. **Generate** - Use LLM to generate answer from context
3. **Evaluate** - Use LLM to compare generated answer vs expected

Key differences from our original evaluation:
- We originally checked if answer appears in retrieved memories (retrieval-only)
- CORE generates an actual answer and scores it (E2E)

## Usage

### Prerequisites

```bash
cd tests/benchmarks/core_adapter
npm install
```

### Environment Variables

```bash
export OPENAI_API_KEY="your-key"
export AUTOMEM_API_TOKEN="test-token"
export AUTOMEM_BASE_URL="http://localhost:8001"
```

### Run Evaluation

```bash
# Make sure AutoMem is running with LoCoMo data loaded
npm run evaluate
```

### Results

Results are saved to `evaluation_results.json` with:
- Overall accuracy
- Per-category breakdown
- Individual question details

## Methodology Comparison

| Aspect | AutoMem Original | CORE | This Adapter |
|--------|------------------|------|--------------|
| Retrieval | AutoMem /recall | Custom search | AutoMem /recall |
| Answer Generation | None (retrieval-only) | GPT-4o | GPT-4o-mini |
| Scoring | Word overlap / F1 | LLM-based | LLM-based |
| Category 5 | Adversarial word check | Phrase match | Phrase match |

## Files

- `automem-search.js` - AutoMem adapter for CORE's search interface
- `evaluate.js` - Main evaluation script (CORE methodology)
- `package.json` - Node.js dependencies

## Verifying CORE's Results

To verify CORE's claimed results:

1. Clone their benchmark repo:
   ```bash
   git clone https://github.com/RedPlanetHQ/core-benchmark
   ```

2. Replace their search service with our adapter:
   ```javascript
   // In services/search.server.js
   import { AutoMemSearchService } from './automem-search.js';
   export default new AutoMemSearchService();
   ```

3. Run their evaluation:
   ```bash
   npm run evaluate
   ```

This gives us a direct comparison using their exact code.

