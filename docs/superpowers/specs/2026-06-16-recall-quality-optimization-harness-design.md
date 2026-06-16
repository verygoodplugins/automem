# AutoMem Recall-Quality Optimization Harness — Design Spec

- **Date:** 2026-06-16
- **Status:** Draft for review
- **Owner:** Jack Arturo (with Claude Code)
- **Branch:** `eval/recall-quality-harness`

---

## 1. Goal & non-goals

**Goal.** Improve AutoMem's recall quality measured against Jack's *real production
corpus*, then confirm the gains on public benchmarks (AMB / BEAM / PersonaMem) and
publish. The optimization target is intrinsic service quality, not benchmark score.

**North star (operator's words):** *elegant, efficient, easy to understand, effective.*
This governs both the service changes we accept and the evaluation harness we build.

**Non-goals.**
- Optimizing directly for a benchmark (overfitting to the eval).
- Adding machinery for marginal wins. A change that improves quality but adds
  complexity loses to one that matches quality with less.
- Net-new tools/endpoints unless an existing one can't be extended.

## 2. Guiding principle: tune on the corpus, confirm on the benchmark

We optimize configs against the cloned production corpus (the *validation* surface),
then run only the internal winners against the public benchmarks (the *holdout/test*
surface). Because the corpus and the benchmark share no data, every published
benchmark number is genuinely out-of-sample — this is the validation→test discipline,
with a more realistic validation set than a benchmark slice.

## 3. The quality metric — a legible scorecard, not a composite

A weighted blend of five sub-metrics is precise but not *understandable*. We use one
primary number, one guardrail, and two tiebreakers. Anyone can read why a config won.

| Role | Metric | Why |
|---|---|---|
| **Primary (effective)** | **NDCG@10** on known-item queries | One standard IR metric that rewards *finding* the right memory **and** *ranking it high*. Burying or missing the answer drops it — no separate recall/precision math. |
| **Guardrail (effective)** | **Distractor-precision** — share of top-10 results that are known-irrelevant (lower is better), measured against a salted distractor query set | Junk must not crowd out relevant results. Also the only metric that can *see* the forget/dreaming arm working (known-item recall is blind to suppression). |
| **Tiebreaker (efficient)** | **Recall latency / result-economy** | Faster, tighter result sets win. |
| **Tiebreaker (elegant)** | **Enabled-knob count** | Among configs within noise, the *simpler* one wins. Encodes "simplify if possible." |

**Decision rule (one sentence):** *pick the highest NDCG@10 that does not regress
distractor-precision; break ties toward fewer knobs and lower latency.*

Why NDCG@10 over the current Recall@20: Recall@20 has no precision term — a config
that returns the target at rank 20 under 19 junk results scores 100%. Optimizing it
rewards casting wider nets (more complexity, the opposite of the goal) and is blind to
forgetting. NDCG@10 fixes all three and is already computed at
`scripts/lab/run_recall_test.py:47`.

## 4. The three-tier funnel

Each tier is more expensive and runs on fewer configs. The slow Gemini-3-Pro
benchmark only ever sees configs that already won on Jack's data.

| Tier | Surface | Configs | Cost | Purpose |
|---|---|---|---|---|
| **1 — Sweep** | Scorecard (§3) on cloned corpus | all ~10–14 | cheap, deterministic, parallel | rank every config + simplification arm |
| **2 — Usefulness gate** | LLM-judged "is this recall actually useful?" on the corpus | top ~3 | moderate | catch what synthetic known-item retrieval misses (relevant-but-not-source) |
| **3 — Publish** | AMB (ByteRover-matched) + BEAM + PersonaMem/32k | 1–2 finalists | expensive | out-of-sample confirmation → leaderboard PR |

## 5. Experiment arms (the ~10–14 configs)

OFAT: each arm perturbs **one factor** off a fixed, pinned baseline (current
production defaults). The baseline config and the win condition are pre-registered
(§6) before any numbers are seen.

- **Additive** (PR #194 knobs, currently off by default): relevance-gate,
  tag-score-token cap, recency-bias on/auto, min_score, recall-k.
- **Subtractive / simplify**: enrichment off, JIT-enrichment off, zeroed/collapsed
  search weights (do we need all 11?), recency-bias off. Goal: find what we can
  remove without quality loss.
- **Dreaming (labeled experimental)**: backdate timestamps → `/consolidate
  {mode, dry_run:false}` → measure scorecard delta, especially distractor-precision.
  See §8.

## 6. Statistical methodology

- **Validation = corpus, test = benchmark.** Sweep on the corpus; open each benchmark
  surface once, for the frozen winner only.
- **Paired comparisons, fixed item set.** Same queries across all configs; store
  per-query outcome per config. The lab already does paired t-test + Cohen's d on
  per-query IR metrics (`run_recall_test.py:81`). For binary correct/incorrect
  (benchmark tier) use **McNemar (mid-p)**.
- **Confidence intervals:** BCa bootstrap on items (≥10k resamples for any published
  number; 1k during the sweep). Accuracies sit near the ceiling, where percentile CIs
  mis-cover.
- **Seeds:** 3 per config on the sweep/usefulness tiers, 5 on the published test runs;
  report mean ± 95% CI, never a single run.
- **Multiple comparisons:** Benjamini-Hochberg (q=0.10) to *screen* on validation;
  Bonferroni across the published surfaces for the headline (α≈0.0167 for 3 surfaces).
- **Power reality:** at corpus/benchmark item counts a +2–3pt single-surface gap is
  likely underpowered. Mitigations: pool across surfaces for the primary decision;
  exploit pairing (McNemar power depends on discordant count). If still underpowered,
  **report "within noise" — do not declare a winner.**
- **Pre-registration:** a one-pager fixed before numbers: baseline config, primary
  surface+metric, win condition, split/seed, repeats, judge model+temperature, the
  declared family of comparisons.

## 7. Orchestration & isolation

Worktrees isolate *code*, not *runtime* — each config needs its own FalkorDB+Qdrant or
results bleed across cells.

- **One stack per cell** via `matrix_stack.py`: `COMPOSE_PROJECT_NAME=lab-<config>-<commit8>-<seed>`,
  dynamic host ports derived deterministically from the cell id (so a resumed cell
  re-lands on identical ports).
- **No fixed `container_name`, no literal host ports, no shared DB volumes.** Add a
  lint that fails a matrix compose file containing either. Add a CI/precheck.
- **Shared model cache, read-only:** pre-download FastEmbed `bge-base-en-v1.5` once,
  mount `:ro` with `HF_HUB_OFFLINE=1` → byte-identical weights, no download races.
- **Concurrency = RAM, not CPU.** Measure one cell's peak RSS; `max ≈ floor(0.8·80GB /
  peak)`. Pin `mem_limit`+`cpus` per service so one OOM kills only its cell. Admit via
  semaphore with a 5–15s stagger (cold-start I/O is the real contention).
- **Provenance manifest:** one row per cell keyed on a content hash of
  `{config, git_commit, seed, corpus_snapshot_id, embedding_model+version,
  image_digest, answer/judge model+version, temperature}`. Pin the corpus snapshot;
  do not re-clone live prod mid-sweep.
- **Idempotent resume:** a cell is done iff `results/<hash>.json` exists (temp +
  atomic rename). A crashed run re-runs the identical command and converges on only
  the missing cells.
- **Shared API keys → global token bucket** across the matrix (not per-cell backoff),
  retry only transient 429/5xx with jittered backoff honoring Retry-After.
- **Teardown:** `down -v --remove-orphans` in a finally block; a labeled reaper for
  stacks whose process died.

## 8. Consolidation ("dreaming") arm

Experimental, kept identical to the no-dream baseline except one inserted step, so any
delta is attributable to consolidation alone.

- **Force a real pass:** `dry_run` defaults to **true** (`automem/api/consolidation.py:26`).
  Send `{"mode": <mode>, "dry_run": false}` or creative/cluster/forget are no-ops.
- **Time compression by backdating timestamps, not looping cycles.** Decay relevance
  is a pure function of age (`consolidation.py:243`), recomputed wholesale each call —
  N cycles ≠ N intervals. To represent "6 months old," ingest with a backdated
  `timestamp` and run decay once. (Single `POST /memory` accepts backdated
  `timestamp`/`last_accessed`/`t_valid`; batch accepts only `timestamp`.)
- **Order:** ingest (backdated) → drain enrichment → decay → creative → cluster →
  forget. One real pass per mode (idempotent on age).
- **Gotchas:** creative & cluster only sample `relevance_score > 0.3`, which store does
  *not* set — run decay first or they see zero candidates. Enrichment SIMILAR_TO edges
  suppress creative on high-similarity pairs — for a clean ablation compare
  enrich-then-consolidate vs consolidate-only. The HTTP forget endpoint uses class
  defaults that forget **more aggressively than prod** — state this or call the
  scheduler builder in-process to match prod. Don't recall before the measured pass
  (recall bumps `last_accessed`, resetting decay).
- **Observability:** only **forget** changes default recall; decay is invisible
  (`SEARCH_WEIGHT_RELEVANCE=0.0`); creative/cluster surface only with
  `expand_relations=true`. So measure the dreaming arm via distractor-precision
  (forget) and multi-hop recall with `expand_relations` (creative/cluster).

## 9. AMB submission integrity (publish tier)

- **Verbatim board judge/answer prompts.** Zero custom judge prompt, equivalence
  rules, or dataset-specific answer instructions. All tuning lives in the recall path.
  (The Mem0 −19.6pt reproduction scandal was custom judge rules.)
- **Comparability anchor: ByteRover 2.0** — Gemini 3 Flash (curate/retrieve/judge) +
  Gemini 3 Pro (justify). Set `OMB_ANSWER_LLM`/`OMB_ANSWER_MODEL` **explicitly** (the
  default is buggy — `OMB_ANSWER_LLM` defaults to `groq`, issue #15).
- **Fix two harness bugs before publishing:** (a) the groq answer-model default;
  (b) `gemini.py:74-76` returns `correct=<last_text>` (truthy) on judge parse failure
  — a silently fabricated "correct." Make parse-failure abstain/error. Land these as a
  small upstream PR (helps the whole board, strengthens our credibility).
- **Baselines through the identical pipeline:** bm25, full-context, closed-book
  (no-memory), oracle (gold-docs). The AutoMem delta must sit *above* the closed-book
  parametric-knowledge floor.
- **Contamination caveat** on LoCoMo/LongMemEval (predate the answer model). Lead with
  PersonaMem MCQ (deterministic, judge-free) and BEAM long-context where "dump
  everything in context" can't win.
- **Reproducibility:** response-cache artifact (cached answer/judge responses so the
  maintainer re-judges identically) + mean±std. Hosted APIs aren't byte-reproducible;
  this is the pragmatic, elegant choice (no self-hosting).
- **Run full splits** (drop `--query-limit`); normalize **k** across providers; report
  all four AMB axes (accuracy, retrieval latency, ingestion time, token cost). Current
  committed outputs are smoke tests (locomo 152/1540, longmemeval 30/500, personamem
  absent) and are **not** publishable.

## 10. Build vs reuse

**Reuse:** corpus clone (`scripts/lab/clone_production.sh`), known-item query gen
(`create_test_queries.py`), IR metrics + paired stats (`run_recall_test.py`), isolated
stacks (`automem-evals/scripts/matrix_stack.py`), AMB self-spinning provider
(`agent-memory-benchmark/.../memory/automem.py`).

**Build (small):** distractor metric + distractor query set; timestamp-backdate
utility; consolidation step inside the recall loop; simplicity (knob-count) score;
matrix-compose lint.

**Build (the one real lift):** wire the lab recall test to run as a *matrix cell* —
accept `endpoint + config` and drop the docker-restart coupling
(`run_recall_test.py:158`) so the sweep runs in parallel across isolated stacks.

## 11. Repo placement (per `CLAUDE.md` benchmark-ownership)

- **Exploratory sweep orchestration + scorecard runner → `automem-evals`** (ruleset
  sweeps, seeded corpora are explicitly its domain).
- **Lab metric changes (NDCG objective, distractor metric, backdate util,
  consolidation-in-loop, lab-as-matrix-cell) → `automem/scripts/lab`.**
- **AMB provider, results, submission PR → `agent-memory-benchmark`.**
- **Any official release-gating benchmark flow → `automem`.**

## 12. Risks & open items

- **Synthetic known-item ground truth.** Queries are GPT-generated, one source memory
  = the only "relevant" doc; blind to relevant-but-not-source. Mitigated by the Tier-2
  usefulness gate and the benchmark holdout — a config must win on all three to be
  declared a winner.
- **Concurrency constraints (carried, not re-litigated):** automem `main` has unrelated
  uncommitted WIP — do not stash/checkout over it. Don't run two judged runs against
  the same stack. Keep external LLM calls out of the AutoMem service path; no secrets
  in images.
- **`:8001` dev-instance data loss** (separate from this work) — recover via
  `scripts/recover_from_qdrant.py` or dismiss; tracked outside this spec.

## 13. Success criteria

One of:
- A config that beats baseline on **NDCG@10** with **no distractor-precision
  regression**, is **simpler and/or faster**, and confirms on ≥1 benchmark surface
  with significance after correction; **or**
- A defensible "current config is near-optimal; here is exactly what we can safely
  *remove* to make it more elegant/efficient without quality loss."

Either outcome ships: a tuned-and-simplified service, plus an honest, reproducible AMB
leaderboard submission with baselines and caveats.
