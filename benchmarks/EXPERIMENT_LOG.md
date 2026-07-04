# AutoMem Experiment Log

Tracks recall quality experiments with before/after benchmark results.

**Internal-harness baselines** in this log are created with Voyage 4 embeddings
(`EMBEDDING_PROVIDER=voyage`, `VECTOR_SIZE=1024`) on the snapshot-based bench
infrastructure (PR #97, merged 2026-03-02). These are AutoMem's *own* LoCoMo /
LongMemEval harness — engineering baselines, **not** the neutral Agent Memory
Benchmark. For the current outbound/release numbers (the neutral AMB run on
FastEmbed `bge-base-en-v1.5` 768d with a Gemini answerer + judge), see the
**Neutral Agent Memory Benchmark (AMB)** section immediately below.

## Neutral Agent Memory Benchmark (AMB) — 0.16.0

Everything else in this log is AutoMem's **internal** LoCoMo / LongMemEval
harness. For **outbound / release claims**, the current numbers come from the
neutral [Agent Memory Benchmark](https://automem.ai/benchmarks) (AMB, by
vectorize-io) — a third-party harness AutoMem does not control. These supersede
the internal-harness numbers for all outbound use.

**Regime stamp (always cite with the numbers):** AMB neutral harness, Gemini
answerer (`gemini-3.1-pro-preview`) + Gemini judge (`gemini-2.5-flash-lite`),
single-query / RAG mode. AutoMem provider: self-spinning Docker (FalkorDB +
Qdrant), **FastEmbed-local `bge-base-en-v1.5` (768d)**, no embedding API keys,
`ENRICHMENT_ENABLED=false`. Run name `automem-sub`.

**BEAM — the apples-to-apples axis** (same benchmark, same harness vs published
competitors). Rubric-mean (0 / 0.5 / 1 per item, averaged — a different scale
than the pass/fail Core-3 benchmarks):

| Split | AutoMem (95% CI) | Honcho | Δ vs Honcho | Mean ctx tokens |
|-------|------------------|--------|-------------|-----------------|
| beam/100k | 67.5% (×3, spread 1.8pp) | 63.0% | **+4.5pp** | ~3.8k |
| beam/500k | 65.6% ±2.8 (n=700) | 64.9% | +0.7pp | ~3.9k |
| beam/1m | 63.8% ±2.7 (n=700) | 63.1% | +0.7pp | ~3.9k |
| beam/10m | **57.4% ±5.5 (n=200)** | **40.6%** | **+16.8pp** | ~3.9k |

AutoMem degrades gracefully (67.5% → 57.4%, −10pp across a 100× haystack
increase) where Honcho collapses (63.0% → 40.6% — flat through 1M, then a cliff
at 10M). Clear **#2 on BEAM** behind vectorize's own Hindsight (~73→64% across
the curve). The **10M result is the centerpiece:** at 10M tokens context-stuffing
is physically impossible, so the score reflects retrieval architecture, not
context window. Efficiency is architectural — AutoMem feeds **~2.6–4.8k** mean
context tokens to the answerer at every scale vs the board leader's 17–27k.

**Core-3 (conversational) — AutoMem trails the leaders (the honest other half):**

| Split | AutoMem (95% CI) | Hindsight (yardstick) |
|-------|------------------|------------------------|
| locomo/locomo10 | 85.1% ±1.8 (n=1540) | 92% |
| longmemeval/s | 74.4% ±3.8 (n=500) | 94.6% |
| personamem/32k | 76.1% ±3.4 (n=589) | 86.6% |

AutoMem's strength is large-context BEAM scaling + token efficiency, **not**
verbatim conversational recall. Honcho's Core-3 numbers are self-reported on its
own harness (directional, not head-to-head) — only BEAM is the clean comparison;
Hindsight is the apples-to-apples Core-3 yardstick (same AMB harness).

**Status — submitted, not official.** AMB results are submitted to the neutral
board; the vectorize-io provider PR ([#24](https://github.com/vectorize-io/agent-memory-benchmark/pull/24))
is **under review**. Frame as "submitted, PR under review" and "run it yourself"
— never "live/official on the leaderboard" until the PR merges. Outputs are committed and public; `AUTOMEM_REPRODUCE.md`
gives one command per split, and the public GHCR image
(`ghcr.io/verygoodplugins/automem:amb-v1`) self-spins the full stack with no API
keys. **No cross-system latency/speed claims** — AMB timings are AutoMem's own
hardware only.

**Canonical homes:** the full per-tier head-to-head and reproducibility recipe
live on [automem.ai/benchmarks](https://automem.ai/benchmarks); the full
per-split run summary lives in `automem-evals`
(`data/results/SUMMARY-amb-submission-2026-06.md`).

## Tiered Benchmarking

| Tier | Benchmark | Runtime | Cost | When to use |
|------|-----------|---------|------|-------------|
| 0 | `make test` (unit) | 30s | free | Every change |
| 0 | `make bench-current-state` (state-aware recall smoke) | <30s | free | Recall state contract changes |
| 1 | `locomo-mini` (2 convos, 304 Qs) | 2-3 min | free / ~$0.20 with judge | Rapid iteration |
| 2 | `locomo` (10 convos, 1986 Qs) | 5-10 min | free / ~$1-3 with judge | Before merge |
| 3 | `longmemeval-mini` (stratified 30 Qs) | 15 min | ~$1 | Scoring/entity changes |
| 4 | `longmemeval` (500 Qs) | 1-2 hr | ~$10 | Milestones only |

## Results

Internal-harness headline results (engineering baselines — superseded for
outbound/release claims by the **Neutral Agent Memory Benchmark (AMB)** section
above):

| Benchmark | Scope | Score | Retrieval | Notes |
|-----------|-------|-------|-----------|-------|
| LongMemEval full | 500 questions | **87.00% (435/500)** | recall@5 **97.00% (485/500)** | Fresh publication verification run with `gpt-5-mini` answerer + `gpt-5.4-mini-2026-03-17` judge; `judge_errors=0`, `memory_ingest_failures=0`, harness `publishable=true`. |
| LongMemEval mini | 30 questions, stratified 5 per type | **70.00% (21/30)** | recall@5 **96.67% (29/30)** | Representative canary from the May 2026 publication verification run; do not compare to legacy prefix slices. |
| LoCoMo full | 10 conversations, 1986 questions | **84.74% (1683/1986)** | -- | Fresh publication verification run with pinned `gpt-5.4-mini-2026-03-17` judge; 444 judge calls, 0 skips/errors. |

Detailed experiment history:

| Date | Issue/PR | Branch | LoCoMo-mini | LoCoMo-full | LME-mini | LME-full | Notes |
|------|----------|--------|-------------|-------------|----------|----------|-------|
| 2026-03-02 | baseline | main | 76.97% (234/304) | 80.06% (1590/1986) | -- | -- | Voyage 4, 1024d. Health: DEGRADED (low score variance) |
| 2026-03-02 | #73 | exp/73-min-score-threshold | 76.97% (+0.0) | -- | -- | -- | min_score + adaptive floor. No regression. Needs #78 for impact |
| 2026-03-02 | PR #80 | jescalan/feat/enhanced-recall | BLOCKED | -- | -- | -- | Merge conflicts with main (recall.py), needs rebase before eval |
| 2026-03-02 | PR #87 | jescalan/feat/write-time-dedup | 76.97% (+0.0) | -- | -- | -- | Write-time dedup gate. Neutral on recall (expected) |
| 2026-03-02 | #78 | exp/78-decay-fix | 76.97% (+0.0) | 79.51% (-0.55) | -- | -- | Decay rate 0.1→0.01, importance floor, archive filter. Within variance. Impact is on production (rehabilitated via rescore) |
| 2026-03-10 | pre-refactor | main (@ 795368a) | 76.97% (+0.0) | -- | -- | -- | Baseline re-confirmed after #73, #78, #115, #116 merged. Stable. Pre-relation-tier-refactor checkpoint. |
| 2026-03-10 | eval-fix | docs/benchmark-agent-guidelines | **89.27% (208/233)** | -- | -- | -- | Fix temporal matching (answer vs memory dates) + skip cat5 (no ground truth). Honest corrected baseline after evaluator fixes. |
| 2026-03-10 | cat5-judge | feat/bench-cat5-judge | **89.80% (273/304)** | **87.56% (1739/1986)** | -- | -- | Opt-in GPT-4o judge for cat5. Full run scored cat5 at 95.74% (427/446) with 0 judge skips/errors; added 90s OpenAI request timeout to prevent stuck full runs. |
| 2026-03-10 | main-refresh (no judge) | main | **89.36% (210/235)** | -- | -- | -- | Fresh current-main rerun before PR #80 experiment. Comparison anchor for judge-off. |
| 2026-03-10 | main-refresh (judge) | main | **90.13% (274/304)** | -- | -- | -- | Fresh current-main rerun with `BENCH_JUDGE_MODEL=gpt-4o`. Comparison anchor for judge-on. |
| 2026-03-11 | PR #80 port (no judge) | exp/pr80-enhanced-recall-v2 | 85.53% (201/235) | -- | -- | -- | BM25 + query expansion + rerank port. Regressed **-3.83pp** vs fresh main. Open-domain -11.4pp. Runtime 7.4x. → [postmortem](postmortems/2026-03-11_pr80_enhanced_recall.md) |
| 2026-03-11 | PR #80 port (judge) | exp/pr80-enhanced-recall-v2 | 88.16% (268/304) | -- | -- | -- | Same branch with GPT-4o cat5 judge. Regressed **-1.97pp** vs fresh main. Runtime 10.2x. → [postmortem](postmortems/2026-03-11_pr80_enhanced_recall.md) |
| 2026-03-11 | PR #80 BM25-only f10 | exp/pr80-bm25-only-f10 | 88.09% (-1.28) | -- | -- | -- | Best config variant, still regressed. → [postmortem](postmortems/2026-03-11_pr80_enhanced_recall.md) |
| 2026-03-11 | PR #80 BM25-only f20 | exp/pr80-bm25-only-f20 | 87.66% (-1.70) | -- | -- | -- | More BM25 results = more dilution. → [postmortem](postmortems/2026-03-11_pr80_enhanced_recall.md) |
| 2026-03-11 | PR #80 BM25+rerank top5 | exp/pr80-bm25-rerank-top5 | 87.23% (-2.13) | -- | -- | -- | Reranking didn't recover regression. → [postmortem](postmortems/2026-03-11_pr80_enhanced_recall.md) |
| 2026-03-11 | PR #80 BM25+rerank top10 | exp/pr80-bm25-rerank-top10 | 86.81% (-2.55) | -- | -- | -- | Wider rerank window = worse. → [postmortem](postmortems/2026-03-11_pr80_enhanced_recall.md) |
| 2026-03-11 | #74 entity expansion | exp/74-entity-expansion-precision-v1 | 89.36% (+0.0) | -- | -- | -- | Hub-node detection. Zero delta — benchmark doesn't exercise graph expansion. → [postmortem](postmortems/2026-03-11_issue74_entity_expansion_precision.md) |
| 2026-03-12 | #79 (PR #125) | exp/79-priority-ids-fetch-v1 | 89.36% (+0.0) | -- | -- | -- | Bug fix: priority_ids now fetches by ID. Merged. → [postmortem](postmortems/2026-03-12_issue79_priority_ids_fetch.md) |
| 2026-04-23 | #128 | fix/128-recall-keyword-scoring-dead-for-vector-results-adaptive-floor-too-aggressive | **85.53% (201/235)** | -- | -- | -- | Content keyword fallback + gentler adaptive floor. Improved **+3.40pp** vs the same-day baseline (`82.13%`, `193/235`) with no sampled question-level regressions across conv-26/conv-30. |
| 2026-04-23 | #128 full judge | fix/128-recall-keyword-scoring-dead-for-vector-results-adaptive-floor-too-aggressive | -- | **83.99% (1668/1986)** | -- | -- | Full judge-on rerun after harness fixes; category-5 judge model was `gpt-5.1`. Judge preflight passed and cat-5 scored **92.83% (414/446)** with **0 skips**. Improves **+3.93pp** vs full baseline (`80.06%`, `1590/1986`), so #128 is strong enough to move forward to broader validation. |
| 2026-04-23 | #142 | fix/142-expansion-tag-filter | -- | 77.30% (-0.07) | -- | -- | Expansion tag-filter bypass. Effectively flat vs pre-fix `77.37%` — canonical configs don't exercise `expand_relations`. Validated via scoped repro + helper/API tests. |
| 2026-04-26 | LongMemEval harness | fix/longmemeval-harness-resume-and-stratified-mini | -- | -- | **60.0% (18/30)** | **86.20% (431/500)** | Historical canonical milestone. Full recall@5 **97.20% (486/500)**; `judge_errors=0`, `memory_ingest_failures=0`. Superseded for publication claims by the 2026-05-17 verification run. |
| 2026-05-17 | Publication verification | feat/automem-arxiv-publication | **85.20% (259/304)** | **84.74% (1683/1986)** | **70.00% (21/30)** | **87.00% (435/500)** | Fresh local publication reruns. LoCoMo full used pinned `gpt-5.4-mini-2026-03-17` judge, 444 judge calls, 0 skips/errors, estimated judge cost `$0.7909`, artifact `benchmarks/results/locomo_baseline_20260517_193934.json`, sha256 `a75816e9a6d3302c22b34852b75ac19a9d9f5cb27d1a109e0af7e49359330716`. LongMemEval full used `gpt-5-mini` answerer + `gpt-5.4-mini-2026-03-17` judge, recall@5 **97.00% (485/500)**, `memory_ingest_failures=0`, `judge_errors=0`, `publishable=true`, artifact `benchmarks/results/longmemeval-full-publication-20260518.json`, sha256 `ed6f7cf69b7be6fa0050536ec2b0f947f5510afd8c2a374b3fafb9cde009da75`. |
| 2026-06-06 | main-refresh (no judge) | main @ b1df86c | **83.40% (196/235)** | -- | -- | -- | Same local `.env`, snapshot eval after #173. Comparison anchor for PR #124/#72 hardening; cat-5 judge disabled, so 69 complex questions are skipped and the result is directional. |
| 2026-06-06 | PR #124 + #72 hardening | feat/entity-identity-hardening | **83.40% (+0.0)** | **81.71% (1260/1542)** | -- | -- | Entity quality gates, safe Entity-node migration/dedup, current-state identity synthesis, and disabled scheduled synthesis. Flat vs same-env main on LoCoMo-mini; full run is judge-off with 444 cat-5 questions skipped, so focused graph/entity regressions carry the entity-pollution risk. |
| 2026-06-11 | Ranking release (develop) | develop @ 0b522a9 | -- | -- | **43.3% (13/30)**, recall@5 30/30 | **86.00% (430/500)**, recall@5 **96.60% (483/500)** | Full judged run, ship config (`RECALL_RECENCY_BIAS=auto`, harness `temporal-answer`); `gpt-5-mini` answerer + pinned `gpt-5.4-mini-2026-03-17` judge; `judge_errors=0`, `memory_ingest_failures=0`. Artifact `lme_full_ship_20260611.json` (sha256 `1e53f71…`). **Recall is the deterministic signal**; the accuracy delta vs the 2026-05-17 run (87.00%) is within answerer noise (±1pp replicate). Churn attribution shows the April 97.2% recall floor is stale — current main at defaults measures ~97.0% (485/500 est). Mini-floor row (43.3%) is the judge-off develop canary, not comparable to judged rows. → full forensic breakdown: [2026-06-11 ranking release: recall churn attribution](#2026-06-11-ranking-release-recall-churn-attribution). |
| 2026-07-04 | PR #205 infra-gated validation | fix/recall-overfetch-and-artifact-exclusion @ b9d89c2 | -- | **83.98% (1295/1542)** | -- | -- | Candidate-pool sizing + artifact exclusion validation on an isolated production-clone worktree. Directionally above prior main judge-off baseline `81.58% (1258/1542)`, but not a same-run A/B. Recall Quality Lab was non-regressing, not improving: `fix_v1` vs `baseline_pr205_legacy` both NDCG@10 **0.029**, Recall@10 **0.045**, distractor@10 **0.000**; latency 189ms vs 204ms. Sweep `RECALL_VECTOR_OVERFETCH=1,2,4,8`: 1/2/4 tied on quality, 8 slightly lower NDCG **0.028**. Local smoke excluded `MetaPattern` rows, but failed the unscoped positive-id check (`ee830782` absent up to limit 100; rank 2 when scoped to `tags=lennard`). Live PR validation skipped because Railway production was `f207a40` from PR #213 and did not contain `b9d89c2`. |

### 2026-06-11 ranking release: recall churn attribution

Deep dive for the `2026-06-11` ship-config row above (develop @ `0b522a9`), kept
verbatim out of the table cell for readability. Nothing here is dropped.

- **Server-change vs harness-prompt split.** Recall deltas are server-side only
  (the harness prompt can't move recall). The accuracy delta vs the 2026-05-17
  verification run (87.00%) is within answerer noise: the two identical-config
  reference runs (2026-04-26 vs 2026-05-17) flip 28 answers (12 newly wrong / 16
  newly right) while recall flips just 1 question — so recall is the
  deterministic signal and accuracy ±1pp is replicate noise.
- **Recall churn attribution (17 questions vs the 2026-04-26 canonical).**
  Targeted re-runs of exactly the 17 churned questions on current main at
  defaults and develop at defaults show 8 of the 10 new misses also miss on
  current main (and the 7 newly-fixed questions already hit on current main) —
  i.e. 15 of the 17 churned questions (8 misses + 7 fixes) moved with that week's
  main merges (#191 keyword normalization), making the April canonical 97.2%
  floor stale; current main at defaults measures ~97.0% (485/500 est).
- **Develop vs current main: 1 question.** Develop at defaults differs from
  current main by one question (`9ea5eabc`, a near-tie rank-5/6 flip consistent
  with #187's deterministic timestamp tiebreak). The remaining churned question
  (`00ca467f`) hits at defaults on both codebases and has no recency-trigger
  keyword (likely residual run noise, possibly ship-env).
- **Artifacts.** `lme_churn17_main_defaults.*`, `lme_churn17_dev_defaults.*`,
  `analyze_churn17.py`; primary result
  `benchmarks/results/lme_full_ship_20260611.json` (sha256
  `1e53f715220d2ab4e2666106d56fd954fa4b3e4818a1bce7d060738b1bdd2d4b`).
- **Failure-mode distribution** vs canonical is stable
  (`failure_modes_ship_llm_20260612.json`: answer-construction 39 vs 41,
  retrieval-gap 7 vs 6). Watch item: `missing-date-use` 2→6 in temporal-reasoning
  despite `temporal_answer_hint` (weak evidence given answerer noise).
- **Per-category accuracy vs canonical:** preference 63% vs 60%,
  single-session-user 94% vs 91%, assistant 100% vs 98%, multi-session 81% flat,
  knowledge-update 86% vs 88%, temporal 86% vs 88%.
- **Mini-floor row (43.3% accuracy)** is the judge-off develop canary, not
  comparable to judged rows.

### Category Breakdown (LoCoMo-mini)

Categories 1-4 are scored by word-overlap/date matching. Category 5 uses an opt-in LLM judge when `BENCH_JUDGE_MODEL` or `--judge` is enabled; otherwise it remains `N/A`.

| Date | Issue/PR | Single-hop | Temporal | Multi-hop | Open Domain | Complex | Overall |
|------|----------|------------|----------|-----------|-------------|---------|---------|
| 2026-03-02 | baseline | 76.7% (33/43) | 22.2%\* (14/63) | 46.2% (6/13) | 96.5% (110/114) | 100%\*\* (71/71) | 76.97% (234/304) |
| 2026-03-10 | pre-refactor | 76.7% (33/43) | 22.2%\* (14/63) | 46.2% (6/13) | 96.5% (110/114) | 100%\*\* (71/71) | 76.97% |
| 2026-03-10 | eval-fix | **79.1% (34/43)** | **92.1% (58/63)** | 46.2% (6/13) | 96.5% (110/114) | N/A (71 skipped) | **89.27% (208/233)** |
| 2026-03-10 | cat5-judge | **79.1% (34/43)** | **92.1% (58/63)** | 46.2% (6/13) | 96.5% (110/114) | **91.5% (65/71)** | **89.80% (273/304)** |
| 2026-03-10 | main-refresh (no judge) | **79.1% (34/43)** | **92.1% (58/63)** | 46.2% (6/13) | **96.5% (110/114)** | 100.0% (2/2, 69 skipped) | **89.36% (210/235)** |
| 2026-03-10 | main-refresh (judge) | **79.1% (34/43)** | **92.1% (58/63)** | 46.2% (6/13) | **96.5% (110/114)** | **93.0% (66/71)** | **90.13% (274/304)** |
| 2026-03-11 | PR #80 port (no judge) | **86.0% (37/43)** | **93.7% (59/63)** | 46.2% (6/13) | 85.1% (97/114) | 100.0% (2/2, 69 skipped) | 85.53% (201/235) |
| 2026-03-11 | PR #80 port (judge) | **88.4% (38/43)** | **92.1% (58/63)** | 46.2% (6/13) | 86.0% (98/114) | **95.8% (68/71)** | 88.16% (268/304) |
| 2026-03-11 | PR #80 BM25-only f10 | 81.4% (35/43) | 92.1% (58/63) | 46.2% (6/13) | 93.0% (106/114) | N/A | 88.09% |
| 2026-03-11 | #74 entity expansion | 79.1% (34/43) | 92.1% (58/63) | 46.2% (6/13) | 96.5% (110/114) | N/A | 89.36% |
| 2026-03-12 | #79 (PR #125) | 79.1% (34/43) | 92.1% (58/63) | 46.2% (6/13) | 96.5% (110/114) | N/A | 89.36% |
| 2026-04-23 | #128 | **65.1% (28/43)** | 92.1% (58/63) | **38.5% (5/13)** | **94.7% (108/114)** | 100.0% (2/2, 69 skipped) | **85.53% (201/235)** |
| 2026-06-06 | main @ b1df86c (no judge) | 51.2% (22/43) | 92.1% (58/63) | 53.8% (7/13) | 93.9% (107/114) | 100.0% (2/2, 69 skipped) | **83.40% (196/235)** |
| 2026-06-06 | PR #124 + #72 hardening (no judge) | 51.2% (22/43) | 92.1% (58/63) | 53.8% (7/13) | 93.9% (107/114) | 100.0% (2/2, 69 skipped) | **83.40% (196/235)** |

\* Temporal was artificially low: evaluator compared question dates (empty) vs memory dates instead of answer dates.
\*\* Complex was artificially 100%: dataset has no `answer` field for cat5 → empty string → `"" in content` always True.

### Category Breakdown (LongMemEval full)

Canonical run: `benchmarks/results/longmemeval-full-publication-20260518.json`.
Answerer `gpt-5-mini`; judge `gpt-5.4-mini-2026-03-17`.

| Question type | Accuracy | Recall@5 |
|---------------|----------|----------|
| knowledge-update | 88.46% (69/78) | 100.00% (78/78) |
| multi-session | 84.21% (112/133) | 97.74% (130/133) |
| single-session-assistant | 98.21% (55/56) | 100.00% (56/56) |
| single-session-preference | 56.67% (17/30) | 90.00% (27/30) |
| single-session-user | 92.86% (65/70) | 92.86% (65/70) |
| temporal-reasoning | 87.97% (117/133) | 96.99% (129/133) |

Failure split from the result-analysis helper: 54 wrong answers had the answer
session retrieved at recall@5; 11 wrong answers were retrieval misses. This is the basis for
follow-up issues #158 and #159.

## Exploratory and Historical Benchmarks

These results are useful validation signals, but they are not current headline
claims. BEAM entries came from `automem-evals` exploratory runners. LongMemEval
prefix entries are legacy/provisional because dataset prefixes are biased.
Memora/FAMA and WRIT lifecycle diagnostics are diagnostic signals unless their
scenario, harness, and artifacts are promoted through this repo's official
benchmark policy.

| Date | Benchmark | Scope | Score | Retrieval | Notes |
|------|-----------|-------|-------|-----------|-------|
| 2026-04-22 | BEAM 100K V1 raw-dialogue shim | 20 conversations, 400 questions | **76.25% (305/400)**, avg 0.677 | top-k 200 | `gpt-5-mini` answerer/judge, zero errors. Diagnostic result only: BEAM 100K is easier than mem0's published 1M/10M settings, and the V1 shim stores raw dialogue rather than mem0-style extracted facts. |
| 2026-04-22 | BEAM 100K V2 fact-extraction shim | 20 conversations, 400 questions | **73.75% (295/400)**, avg 0.653 | top-k 200 | -2.50pp vs V1 overall, within the estimated noise floor. Category signal was useful: abstention +15pp and knowledge_update +7.5pp; event_ordering -20pp and information_extraction -12.5pp. |
| 2026-04-24 | LongMemEval partial legacy prefix (50q) | 50 questions, single-session-user type | **82.0% (41/50)** | recall@5 **92.0% (46/50)** | Provisional prefix run with legacy `gpt-4o` answerer; recall_limit=10, no entity/relation expansion. Not reproduced by the current stratified `bench-mini-longmemeval` target and not directly comparable to the older 35.6% / 500-question setup or to a full LongMemEval claim. |

## How to add an entry

1. Run the benchmark: `make bench-eval BENCH=locomo-mini CONFIG=baseline`
2. Record the overall accuracy from the output JSON
3. Add a row to the detailed experiment history with the date, issue/PR, branch, and scores
4. Add a row to the Category Breakdown table with per-category scores
5. For deltas, show as `XX.X% (+Y.Y)` relative to the baseline row

## Snapshot metadata

| Snapshot | Created | Git SHA | Embedding | Memories |
|----------|---------|---------|-----------|----------|
| locomo-mini | 2026-03-02 | main @ 80a6f93 | voyage:voyage-4 1024d | 788 (2 convos) |
| locomo | 2026-03-02 | main @ 80a6f93 | voyage:voyage-4 1024d | 5828 (10 convos) |
