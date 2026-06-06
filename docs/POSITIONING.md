# AutoMem Positioning — Scout Reference

Single source of truth for what scout can cite about AutoMem in outbound drafts. Edit this file when positioning changes. Do NOT store positioning in AutoMem — it creates feedback loops with scout's own prior outputs.

**Last updated:** 2026-05-22

---

## What AutoMem is

AutoMem is an open-source memory layer for AI agents. Graph-vector hybrid — FalkorDB for typed relations, Qdrant for vector search — with a consolidation/decay model that prunes noise over time. MIT licensed. Universal MCP bridge so the same memory works across Claude Code, Cursor, Warp, Codex, Claude Desktop, ChatGPT Desktop.

The pitch in one line: deploy once, remember everywhere.

## What AutoMem isn't

- Not a verbatim chat log store. It stores distilled memories — decisions, patterns, preferences, insights — not raw conversation transcripts.
- Not a session-history grep tool. Different problem from things like Darc.
- Not optimized for raw retrieval recall. Architecturally favors multi-hop reasoning and temporal validity over verbatim recall metrics.
- Not a hosted SaaS yet. See pricing below.

## Pricing — current reality (May 2026)

- **Free, self-host via Docker.** Full stack runs locally: FalkorDB + Qdrant + Flask API. `docker-compose up`.
- **~$5/month on Railway** if you want managed infra without ops work. One-click deploy template.
- **Hosted SaaS is not launched.** Paid tiers ($9 Pro, $29 Ultimate) are planned but not live. Do not cite them in outbound replies until the launch ships.

## Current benchmark numbers

- **LoCoMo-full:** ~84.74% on v0.15.2 with Voyage 4 embeddings (May 2026)
- **LongMemEval:** ~87.00% on v0.15.2 with Voyage 4 embeddings (May 2026)

Live benchmarks page: `https://automem.ai/benchmarks` — reproducible methodology, version-stamped, updated as new runs ship.

Older runs (the 70.69% / 99.78% LoCoMo figures that propagated through earlier scout outputs) are stale and must not appear in drafts. If you see them in recall, ignore.

Published head-to-head comparisons against competitors are reserved for the arXiv paper where framing can be controlled. Don't post benchmark tables in Reddit/HN replies — link the page instead.

## When to link vs cite numbers inline

**Link `automem.ai/benchmarks`** when:

- The thread is a head-to-head comparison ("X vs Y vs Z for agent memory")
- Someone's asking for benchmark data across frameworks
- The reply would otherwise need more than one number to make the point
- Methodology matters to the asker

**Cite a single number inline** when:

- Passing reference where one figure is enough ("AutoMem does ~84.74% LoCoMo-full on v0.15.2 — graph layer helps with multi-hop")
- The asker has a narrow question and a benchmark link would be overkill
- Reply context is conversational, not comparative

**Always include the version number** (`v0.15.2`) and embedding model (`Voyage 4`) when citing inline. Bare percentages get challenged; version-stamped ones get trusted.

## Differentiators worth citing

1. **Typed relations (11 of them):** `RELATES_TO`, `LEADS_TO`, `OCCURRED_BEFORE`, `PREFERS_OVER`, `EXEMPLIFIES`, `CONTRADICTS`, `REINFORCES`, `INVALIDATED_BY`, `EVOLVED_INTO`, `DERIVED_FROM`, `PART_OF`. Lets the graph express decision lifecycles, not just similarity.
2. **Graph-vector hybrid, not vector-only.** Vector for fuzzy semantic recall, graph for explicit relationships and multi-hop traversal. Most competitors are one or the other.
3. **Consolidation/decay model.** Memories age out by configurable importance/confidence thresholds. Not a perfect-recall append-only log.
4. **Temporal validity:** `t_valid` and `t_invalid` are first-class fields. Facts can have shelf life.
5. **MCP-native.** Works with every editor that speaks MCP. No vendor lock-in to a single AI client.
6. **Open source, MIT.** No hidden tier walls around core graph features.

## What NOT to claim

- No SOTA claims on retrieval-recall benchmarks. AutoMem is structurally not optimized for verbatim recall — competing there loses winnable comparisons.
- No "faster than X" / "cheaper than Y" without traceable source numbers. The old "20-50ms vs 100-200ms" / "$5 vs $50" framing in earlier blog posts is from a different competitive context — don't pull those forward.
- No claims AutoMem evaluates on benchmarks it hasn't actually been run against. If recall doesn't have the number, leave it out.

## External replies

Use this register for Reddit, HN, GitHub, dev.to, and X drafts:

- Answer the actual question first. AutoMem belongs in the reply only if it directly helps the asker.
- Keep it technical and specific: graph-vector hybrid, typed relations, temporal validity, consolidation/decay, MCP-native deployment.
- Disclose maintainer/operator context when it matters: "I'm building AutoMem" or "maintainer here" is cleaner than pretending to be neutral.
- Prefer one useful detail over a pitch. A good reply can be two sentences plus a link.
- Do not dunk on competitors. Frame tradeoffs: vector-only vs graph-vector, append-only logs vs distilled memories, client-specific memory vs MCP-shared memory.
- For comparison threads, link `https://automem.ai/benchmarks` instead of pasting tables.
- For passing references, cite one version-stamped number inline and include both `v0.15.2` and `Voyage 4`.

Voice samples:

- TODO: paste 2-3 real Reddit/HN replies here before the next scout run.
- Until samples are added, follow the register above and ignore voice examples from AutoMem recall.

## Banned phrases

Do not use these phrases, casing variants, or close paraphrases in outbound drafts:

- "game changer"
- "revolutionary"
- "state of the art" / "SOTA"
- "best-in-class"
- "unmatched"
- "crushes"
- "blows X out of the water"
- "10x"
- "perfect memory"
- "never forgets"
- "set and forget"
- "just works"
- "killer app"
- "memory layer that changes everything"
- "faster than X"
- "cheaper than X"

Also ban these claim shapes even if the wording changes:

- Hosted SaaS is live.
- AutoMem is objectively the best memory framework.
- AutoMem wins every benchmark or every use case.
- AutoMem has verified speed, cost, or accuracy advantages over a named competitor without a traceable current source.
- AutoMem has published head-to-head competitor tables outside the benchmarks page or arXiv paper.

## Scout contract

- This file is authoritative for pricing, current benchmark numbers, differentiators, approved links, voice register, and banned phrases.
- AutoMem recall is allowed for dedup, prior outreach, and competitive intel only. Treat recalled positioning facts as stale unless they match this file.
- If a scout run cannot read this file, abort the sweep rather than drafting from memory.
- If this file lacks a needed claim, leave the claim out or link the benchmarks page.
- Do not store benchmark numbers, pricing, banned phrases, or voice rules in AutoMem.

## Approved links

- Benchmarks: `https://automem.ai/benchmarks`
- Repo: `https://github.com/verygoodplugins/automem`
- npm: `https://www.npmjs.com/package/@verygoodplugins/mcp-automem`
- Homepage: `https://automem.ai`
- Discord: `https://automem.ai/discord`
