#!/usr/bin/env bash
# Compare the latest Railway deployment commit against origin/main HEAD
# to detect when Railway's GitHub integration silently disconnects.
#
# Usage:
#   ./scripts/deploy_check.sh                    # check memory-service (default)
#   ./scripts/deploy_check.sh automem-mcp-sse    # check a specific service
#   DEPLOY_CHECK_QUIET=1 ./scripts/deploy_check.sh  # exit code only (for CI)

set -euo pipefail

SERVICE="${1:-memory-service}"
QUIET="${DEPLOY_CHECK_QUIET:-0}"
MAX_AGE_HOURS="${DEPLOY_CHECK_MAX_AGE:-24}"

log() { [[ "$QUIET" == "1" ]] || echo "$@"; }
err() { echo "❌ $*" >&2; }

if ! command -v railway &>/dev/null; then
    err "railway CLI not found. Install: https://docs.railway.app/guides/cli"
    exit 2
fi

if ! command -v gh &>/dev/null; then
    err "gh CLI not found. Install: https://cli.github.com"
    exit 2
fi

log "🔍 Checking deploy freshness for service: $SERVICE"

deploy_json=$(railway deployment list --json --service "$SERVICE" 2>/dev/null) || {
    err "Failed to list deployments for $SERVICE. Is the Railway CLI linked?"
    exit 2
}

deploy_commit=$(echo "$deploy_json" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for d in data:
    if d.get('status') == 'SUCCESS':
        print(d.get('meta', {}).get('commitHash', ''))
        break
" 2>/dev/null)

deploy_ts=$(echo "$deploy_json" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for d in data:
    if d.get('status') == 'SUCCESS':
        print(d.get('createdAt', ''))
        break
" 2>/dev/null)

if [[ -z "$deploy_commit" ]]; then
    err "No successful deployment found for $SERVICE"
    exit 1
fi

repo_slug=$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null) || {
    err "Failed to detect GitHub repo. Run from within the repo directory."
    exit 2
}

main_sha=$(gh api "repos/$repo_slug/commits/main" --jq .sha 2>/dev/null) || {
    err "Failed to fetch HEAD of main from GitHub."
    exit 2
}

deploy_short="${deploy_commit:0:7}"
main_short="${main_sha:0:7}"

if [[ "$main_sha" == "$deploy_commit"* ]] || [[ "$deploy_commit" == "$main_sha"* ]]; then
    log "✅ $SERVICE is up to date (deployed: $deploy_short, main: $main_short)"
    exit 0
fi

git fetch origin main --quiet 2>/dev/null || true

is_ancestor=0
if git merge-base --is-ancestor "$deploy_commit" origin/main 2>/dev/null; then
    is_ancestor=1
fi

behind_count=""
if [[ "$is_ancestor" == "1" ]]; then
    behind_count=$(git rev-list --count "$deploy_commit..origin/main" 2>/dev/null || echo "?")
fi

age_hours=""
if [[ -n "$deploy_ts" ]]; then
    deploy_epoch=$(python3 -c "
from datetime import datetime, timezone
ts = '$deploy_ts'.replace('Z', '+00:00')
print(int(datetime.fromisoformat(ts).timestamp()))
" 2>/dev/null || echo "")
    if [[ -n "$deploy_epoch" ]]; then
        now_epoch=$(date +%s)
        age_secs=$((now_epoch - deploy_epoch))
        age_hours=$((age_secs / 3600))
    fi
fi

log ""
log "⚠️  $SERVICE is BEHIND origin/main"
log "   Deployed commit:  $deploy_short ($deploy_ts)"
log "   main HEAD:        $main_short"
[[ -n "$behind_count" ]] && log "   Commits behind:   $behind_count"
[[ -n "$age_hours" ]]    && log "   Deploy age:       ${age_hours}h"
log ""

if [[ -n "$age_hours" ]] && (( age_hours > MAX_AGE_HOURS )); then
    log "🚨 Deploy is ${age_hours}h old (threshold: ${MAX_AGE_HOURS}h)"
    log "   Railway's GitHub integration may have disconnected."
    log "   Fix: Railway dashboard → $SERVICE → Settings → Source → Reconnect repo"
    log ""
fi

exit 1
