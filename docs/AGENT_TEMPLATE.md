# Long-Haul Agent Template

Reusable playbook for long-running coding sessions with GitHub PRs and CodeRabit reviews. Copy into repo-level `AGENTS.md` and fill the placeholders.

## Scope & Guardrails
- Repos in scope: <names/paths>. Allowed commands: <lint/test/build>. Secrets: env vars only; never log or commit secrets.
- Stop conditions: <time budget>, <issue count>, CI green on open PRs. Checkpoint cadence: <interval>.
- Approvals: destructive ops and production actions require explicit approval.

## Intake & Queue
- Source backlog: GitHub issues/PRs (labels include <labels>, exclude <blocked labels>).
- Maintain an active set of 1–3 items. Reprioritize after each completion or CI result.
- Note dependencies; prioritize unblockers and small wins before larger items.

## Workflow Loop
1) Read context (recent commits, open PRs, failing checks) and make a short plan.
2) Implement minimal scoped changes; add/adjust tests.
3) Run targeted tests first (`<targeted test cmd>`), then format/lint (`<fmt/lint cmd>`).
4) Commit with Conventional Commits; keep diffs tight. Push and open a draft PR early for CI.
5) While CI runs, pick up the next queued item or address feedback.

## PR & CodeRabit Reviews
- PR body must include: Goal; Scope; Tests (commands + results); Risks/Rollback; Follow-ups; Screenshots if UI.
- Rebase/merge main regularly (e.g., every 60–90 minutes or after each issue).
- Request CodeRabit review when ready; re-request after changes; summarize deltas in PR comments to speed review.

## Safety & Hygiene
- No destructive commands without approval. Keep secrets out of logs/commits.
- Log checkpoints/decisions so the next session can resume quickly.
- Throttle scope creep; open follow-up issues for larger refactors.

## Quick Commands (fill per repo)
- Format/Lint: `<fmt command>`, `<lint command>`
- Unit/targeted tests: `<cmd>`
- Full test suite: `<cmd>`
- Health check: `<cmd>`

## Sync & Drift
- `git fetch` regularly; rebase/merge main after each completed issue or when CI conflicts appear.
- Resolve conflicts promptly; keep PRs small to ease merges.

## Template Change Log
- v1.0: Initial long-haul agent template (PR discipline, CodeRabit flow, safety, sync guidance).
