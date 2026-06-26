# Recall Quality Lab

The Recall Quality Lab is a data-driven harness for tuning AutoMem's recall
scoring against a **clone of real data** instead of guesswork. You clone
production into an isolated local stack, generate a test set of natural
questions with known-correct answers, then measure how scoring changes move
information-retrieval metrics (Recall@K, MRR, NDCG) — with a statistical
comparison so you can tell a real improvement from noise.

This is contributor/maintainer tooling. It is **not** required to run AutoMem.
The scripts live in [`scripts/lab/`](../scripts/lab/) and are fronted by
`make lab-*` targets.

## Why it exists

Recall quality is a balance of many weighted signals (vector similarity,
keyword match, tags, recency, importance — see
[ENVIRONMENT_VARIABLES.md](ENVIRONMENT_VARIABLES.md)). Changing a weight to fix
one query can quietly regress ten others. The lab makes that tradeoff
measurable: same corpus, same test set, change one thing, read the metric delta.

Guiding methodology: prefer **one legible primary metric (NDCG@10) plus a
precision guardrail** over an opaque blended score, and only trust a change when
the paired comparison says it's outside the noise band.

## Prerequisites

- Docker (the lab runs an isolated AutoMem stack locally).
- `.env` (repo root or `~/.config/automem/.env`) with production credentials for
  the initial clone — a FalkorDB TCP proxy and Qdrant access. Once you have a
  saved snapshot you can re-run without touching production (see below).
- An LLM key for test-query generation (`create_test_queries.py` uses
  GPT-4o-mini by default).

## Workflow

### 1. Clone production into an isolated stack

```bash
make lab-clone
```

[`clone_production.sh`](../scripts/lab/clone_production.sh) takes a direct DB
backup from Railway and restores it into a local Docker stack. For repeated
experiments, **avoid hitting production every time** by saving the API tarball
once and restoring from it:

```bash
# restore from a saved snapshot instead of re-pulling production
./scripts/lab/clone_production.sh --restore-only lab/snapshots/prod-api-YYYYMMDD-HHMMSS/snapshot.tar.gz

# bring up a second, isolated stack on custom ports for a parallel sweep
./scripts/lab/clone_production.sh --restore-only prod-api-YYYYMMDD-HHMMSS \
  --compose-project automem-sweep --api-port 8011 --qdrant-port 6343 --falkordb-port 6389
```

### 2. Generate a test set

```bash
make lab-queries          # or: python scripts/lab/create_test_queries.py --count 100
```

[`create_test_queries.py`](../scripts/lab/create_test_queries.py) samples diverse
memories from the local clone and uses an LLM to write the natural questions a
user would ask to retrieve each one. The output is a JSON test set — each entry
pairs a query with its expected memory id(s) — consumed by the test harness.

### 3. Run a recall test

```bash
make lab-test CONFIG=baseline
# underlying: python scripts/lab/run_recall_test.py --config baseline
```

[`run_recall_test.py`](../scripts/lab/run_recall_test.py) runs every query in the
test set against the local stack under the named config and reports the IR
metrics across the set.

### 4. A/B compare two configs

```bash
make lab-compare CONFIG=fix_v1 BASELINE=baseline
# underlying: python scripts/lab/run_recall_test.py --config fix_v1 --compare baseline
```

Runs both configs over the same test set and prints the per-metric delta with a
paired comparison, so you can separate a genuine win from run-to-run variance.

### 5. Sweep a parameter

```bash
make lab-sweep PARAM=SEARCH_WEIGHT_VECTOR VALUES=0.20,0.30,0.40,0.50
# underlying: python scripts/lab/run_recall_test.py --sweep SEARCH_WEIGHT_VECTOR 0.20,0.30,0.40,0.50
```

Runs the test set once per value and tabulates how the metric moves across the
range — the fastest way to find a weight's sweet spot.

## Configs

Scoring configs live in [`scripts/lab/configs/`](../scripts/lab/configs/) as JSON
files that override the `SEARCH_WEIGHT_*` (and related) values for a run:

- `baseline.json` — the reference config to compare against.
- `issue78_*.json` — experimental variants attached to a specific investigation.

Add a new config by copying `baseline.json`, changing the weights you want to
test, and passing its name (without `.json`) to `--config`. The keys mirror the
environment variables documented in
[ENVIRONMENT_VARIABLES.md](ENVIRONMENT_VARIABLES.md).

## Metrics and internals

Two library modules (imported by the harness, not run directly) keep the scoring
honest and unit-testable:

- [`lab_metrics.py`](../scripts/lab/lab_metrics.py) — pure, deterministic IR
  scoring functions: `recall_at_k`, MRR, NDCG@K, and distractor rate. No I/O, so
  the metric math is covered by unit tests.
- [`lab_corpus.py`](../scripts/lab/lab_corpus.py) — recall/corpus HTTP helpers
  behind injectable clients, so the retrieval logic can be tested without a live
  server.

**Reading the output:** lead with NDCG@10 as the primary signal, watch a
precision/distractor guardrail so a recall gain isn't bought with junk results,
and trust a config change only when the paired comparison (step 4) clears the
noise band.

## Safety

The lab runs an **isolated** Docker stack and reads production only for the
initial clone. Use `--restore-only` for all subsequent runs so experiments never
touch live data. Custom `--compose-project` / `--*-port` values let multiple lab
stacks coexist for parallel sweeps without colliding.

## Related documentation

- [Environment Variables](ENVIRONMENT_VARIABLES.md) — the `SEARCH_WEIGHT_*` knobs configs override
- [Testing Guide](TESTING.md) — unit, integration, and benchmark testing
- [scripts/README.md](../scripts/README.md) — the full script catalog
- [Benchmark history](../benchmarks/EXPERIMENT_LOG.md) — LoCoMo / LongMemEval / BEAM runs
