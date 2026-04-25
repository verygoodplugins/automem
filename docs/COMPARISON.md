# How AutoMem compares

This doc covers how AutoMem differs from the three things people most often compare it against: traditional RAG, pure vector databases, and rolling your own memory layer. It also covers the recall scoring formula in enough detail to evaluate whether AutoMem will solve your specific retrieval problem.

For the short version, skip to the README's [Should you use AutoMem?](../README.md#should-you-use-automem) section. For research foundations, see [RESEARCH.md](RESEARCH.md).

---

## vs. Traditional RAG

Retrieval-Augmented Generation retrieves *similar documents* and stuffs them into the prompt. AutoMem retrieves *connected memories* with reasoning attached.

Concrete example. You've stored: "Migrated to PostgreSQL for operational simplicity. Considered MongoDB but the team has stronger SQL operational experience. This continued our preference for boring technology."

Ask "what database should we use for the new service?":

- **RAG** returns the PostgreSQL document. Maybe a related doc about MongoDB. The model has to infer the reasoning from the document text.
- **AutoMem** returns the PostgreSQL memory, the `PREFERS_OVER` edge to the MongoDB evaluation, the `DERIVED_FROM` edge to the "boring technology" principle, and the `EXEMPLIFIES` edge linking it to your Redis decision from six months earlier. The model gets the reasoning *as structured graph data*, not as prose to be re-interpreted.

What this buys you:

- **Typed relationships.** Not just "similar", but "causes", "contradicts", "evolved from", "invalidates".
- **Temporal awareness.** Recall knows what preceded what, what was invalidated, and what emerged in response.
- **Pattern detection.** AutoMem reinforces emerging themes across memories so the system learns your style without you tagging anything.
- **Consolidation.** Memories strengthen with use and connection, fade without — so wrong rabbit holes don't pollute future retrieval.

What it costs you:

- More moving parts (a graph database alongside the vector store).
- Higher write latency on enrichment-heavy workloads (mitigated by background processing — graph writes never block on enrichment).
- A fuller mental model. Devs used to "stuff documents in, ask questions out" need to think about edges and types.

---

## vs. Pure vector databases (Pinecone, Weaviate, Qdrant alone)

Vector databases match embeddings. AutoMem builds knowledge graphs *on top of* a vector database. (Qdrant is one of AutoMem's two storage layers — they're complementary, not competing.)

What AutoMem adds over a bare vector store:

- **Multi-hop reasoning.** Bridge discovery follows graph relationships from seed memories to find the connecting nodes — see the README's [Multi-hop bridge discovery](../README.md#multi-hop-bridge-discovery) example.
- **Eleven relationship types.** Structured semantics rather than cosine similarity alone.
- **Background intelligence.** Auto-enrichment, pattern detection, scheduled decay cycles — none of which exist in a raw vector store.
- **Hybrid scoring.** A 9-component recall score (see below) that combines vector similarity with graph strength, temporal alignment, importance, confidence, and tag overlap. Pure vector search collapses all of this into a single distance metric.

When a vector database alone is the right answer:

- Your queries are document-similarity ("find docs that look like this one"), not relational ("find what led to this decision").
- You don't need temporal awareness or pattern reinforcement.
- You're optimizing for raw QPS at scale and don't want the latency overhead of graph queries.

---

## vs. Building your own

You can absolutely build a custom memory layer on Postgres + pgvector + a queue + your own enrichment service. That's how a lot of teams start. What you're signing up for:

- **The recall scoring problem.** Tuning a hybrid score across vector, keyword, graph, temporal, and importance signals takes weeks of iteration. AutoMem ships with a [validated baseline](#9-component-recall-scoring) and a [recall-quality lab](https://github.com/verygoodplugins/automem-evals) for tuning.
- **The consolidation problem.** Decay rates, importance floors, archival thresholds — get any of these wrong and recall quality silently degrades over months. AutoMem encodes the [neuroscience-derived defaults](../README.md#memory-consolidation-neuroscience-inspired) and exposes them as env vars.
- **The integration problem.** MCP, REST, SSE, embedding provider abstraction (Voyage / OpenAI / Ollama / FastEmbed), client-specific configs for Cursor / Claude Desktop / ChatGPT — this is months of plumbing.
- **The benchmark problem.** Without a benchmark, "it feels like it works" is the only quality signal you have. AutoMem publishes transparent [LoCoMo baselines](../benchmarks/EXPERIMENT_LOG.md) and a reproducible harness.

When building your own is the right answer:

- You have requirements AutoMem explicitly doesn't meet — row-level ACLs, per-agent memory isolation, SOC2 compliance, an enterprise audit log.
- The integration has to live deep inside an existing service that can't tolerate an external dependency.
- Memory isn't a meaningful component of your product and a flat embedding store is genuinely sufficient.

---

## 9-component recall scoring

`GET /recall` ranks results by a weighted sum of nine signals. Every weight is tunable via the `SEARCH_WEIGHT_*` environment variables — see [`docs/ENVIRONMENT_VARIABLES.md`](ENVIRONMENT_VARIABLES.md) for the canonical list.

| Component | Default weight | What it measures |
|---|---|---|
| **Vector** | 0.35 | Cosine similarity between query embedding and memory embedding |
| **Keyword** | 0.35 | TF-IDF style overlap between query terms and memory content |
| **Relation** | 0.25 | Strength of graph edges into the memory (boosted by traversal depth) |
| **Tag** | 0.20 | Tag overlap between query and memory |
| **Exact** | 0.20 | Exact phrase match in memory metadata |
| **Importance** | 0.10 | The memory's `importance` score (0–1) |
| **Recency** | 0.10 | Linear decay over a 180-day window |
| **Confidence** | 0.05 | The memory's `confidence` score (0–1) |
| **Relevance** | 0.00 | Consolidation decay relevance — disabled by default |

These defaults reflect the current LoCoMo baseline (89.27% judge-off, 87.56% judge-on). For a query like `GET /recall?query=database+migration&tags=decision&time_query=last+month`, the temporal-alignment and tag components dominate; for `GET /recall?query=why+postgres&expand_relations=true`, the relation component does.

The Recall Quality Lab (`scripts/lab/`) lets you sweep any weight and A/B-compare configs against snapshots of production data without touching the service.

---

## Where to go next

- **Want to deploy?** [INSTALLATION.md](../INSTALLATION.md)
- **Want to tune recall?** [`scripts/lab/`](../scripts/lab/) and [`docs/ENVIRONMENT_VARIABLES.md`](ENVIRONMENT_VARIABLES.md)
- **Want to understand the science?** [RESEARCH.md](RESEARCH.md)
- **Want to talk to humans?** [Discord](https://automem.ai/discord)
