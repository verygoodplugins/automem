---
name: automem-regression-drift-scout
description: Inspect AutoMem CI, benchmark health, and deployment drift as a read-only recurring automation; use for scheduled Codex maintenance runs before proposing issues or PR plans.
license: MIT
tags: [automem, automation, regression, benchmarks, deployment]
category: maintenance
agents: [codex]
metadata:
  version: "1.0.0"
capabilities:
  network: true
  filesystem: readonly
  tools: [Bash]
---

# AutoMem Regression Drift Scout

## When to use

Use this skill for scheduled or manual Codex runs that assess whether AutoMem
has fresh regression, benchmark, CI, or deployment-drift signals that need
human attention. It is not a general code-review skill and it is not a release
operator.

## Safety rules

- Default to read-only.
- Do not edit files.
- Do not deploy.
- Do not create issues or pull requests.
- Do not run long or paid live benchmarks unless the automation prompt
  explicitly authorizes that run.
- Do not report benchmark claims unless they are grounded in
  `benchmarks/EXPERIMENT_LOG.md`, local benchmark command output, or CI output.
- Do not treat missing credentials as a failure by itself; report the skipped
  surface and continue with available local evidence.

## Inputs

- Active worktree root:
  `WORKTREE_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)`
- Repository guidance: `AGENTS.md`
- Benchmark source of truth: `benchmarks/EXPERIMENT_LOG.md`
- Local command surface: `Makefile`, `pyproject.toml`, `.github/workflows`
- Optional read-only GitHub status via `gh run list`
- Optional Railway status via `railway status`

## Workflow

1. Load repository rules from `AGENTS.md`.
2. Inspect the current branch and dirty state with `git status --short --branch`.
3. Read the current benchmark baseline and latest experiment notes from
   `benchmarks/EXPERIMENT_LOG.md`.
4. Run the fast local health probe when dependencies are available:
   `make bench-health`.
5. Check deploy drift without changing remote state:
   `make deploy-check`.
6. Check recent CI status if GitHub CLI is authenticated:
   `gh run list --limit 10`.
7. Check Railway status if the Railway CLI is already authenticated:
   `railway status`.
8. Compare the evidence against the last run memory, open issues/PRs if
   visible, and recent `automem` memories. Dedupe before escalating.
9. Choose exactly one tier:
   `healthy`, `needs_issue`, or `needs_pr_plan`.

## Command result handling

- Do not trust the final `RECALL HEALTH: HEALTHY` banner by itself. Parse the
  full output for `ERROR:`, `SKIP:`, tracebacks, and connection failures.
- If `make bench-health` reports `Connection refused` for `localhost:8001`,
  classify the local API-dependent checks as a skipped surface unless the
  automation expected the local stack to be running.
- If `make deploy-check` reports `railway CLI not found`, classify Railway
  deploy drift as a skipped surface, not as an AutoMem regression.
- If read-only GitHub or Railway status is unavailable because a CLI is missing
  or unauthenticated, record the exact command output and continue with the
  remaining evidence.

## Tier guidance

- `healthy`: checks are green or skipped for known missing credentials, and no
  new actionable drift appears.
- `needs_issue`: evidence shows a real problem, but the fix is unclear, depends
  on infrastructure state, or needs human prioritization.
- `needs_pr_plan`: evidence points to a bounded code/docs change with clear
  files, commands, and acceptance criteria. The scout should draft the plan, not
  create the PR unless a later prompt explicitly upgrades the run to PR-capable.

## Output

Return a concise triage deliverable with:

- tier
- evidence bundle with command names, timestamps, and key outputs
- confidence score and one-line rationale
- dedupe notes against issues, PRs, or previous memories
- recommended next action
- what to measure next time

Always end with the required inbox item for the automation runner.

## Anti-patterns

- Running heavyweight live benchmarks as a routine scout.
- Treating a stale benchmark table as a regression without fresh evidence.
- Opening GitHub issues or PRs from the read-only scout.
- Reporting deploy drift from memory alone without checking current read-only
  status.
