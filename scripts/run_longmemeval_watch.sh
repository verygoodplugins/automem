#!/usr/bin/env bash
#
# Run LongMemEval with persistent logging and local crash/completion notifications.
#
# Usage:
#   scripts/run_longmemeval_watch.sh
#   scripts/run_longmemeval_watch.sh --max-questions 50
#   LONGMEMEVAL_CONFIG=per-turn scripts/run_longmemeval_watch.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f "venv/bin/activate" ]]; then
  # Use project virtualenv so benchmark deps (requests/openai/etc.) are available.
  # shellcheck disable=SC1091
  source "venv/bin/activate"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
CONFIG="${LONGMEMEVAL_CONFIG:-baseline}"
OUT_BASE="${LONGMEMEVAL_OUT_BASE:-tests/benchmarks/results/longmemeval_${CONFIG}_${TIMESTAMP}}"
LOG_FILE="${LONGMEMEVAL_LOG_FILE:-${OUT_BASE}.run.log}"
STATUS_FILE="${LONGMEMEVAL_STATUS_FILE:-${OUT_BASE}.status.json}"

mkdir -p "$(dirname "$OUT_BASE")"

notify() {
  local title="$1"
  local message="$2"

  # macOS desktop notification (best effort)
  if command -v osascript >/dev/null 2>&1; then
    osascript -e "display notification \"${message//\"/\\\"}\" with title \"${title//\"/\\\"}\"" >/dev/null 2>&1 || true
  fi

  # Terminal bell fallback (interactive terminals only)
  if [[ -t 1 ]]; then
    printf '\a'
  fi
}

echo "Starting LongMemEval watch run"
echo "  config:      $CONFIG"
echo "  out base:    $OUT_BASE"
echo "  log file:    $LOG_FILE"
echo "  status file: $STATUS_FILE"
echo

set +e
./test-longmemeval-benchmark.sh --config "$CONFIG" --output "$OUT_BASE" "$@" 2>&1 | tee "$LOG_FILE"
exit_code=${PIPESTATUS[0]}
set -e

status_word="success"
if [[ $exit_code -ne 0 ]]; then
  status_word="failed"
fi

cat >"$STATUS_FILE" <<EOF
{
  "status": "$status_word",
  "exit_code": $exit_code,
  "config": "$CONFIG",
  "output_base": "$OUT_BASE",
  "log_file": "$LOG_FILE",
  "finished_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

if [[ $exit_code -eq 0 ]]; then
  notify "LongMemEval Complete" "Run finished successfully. Output: $OUT_BASE.json"
  echo
  echo "LongMemEval completed successfully."
  echo "Results: $OUT_BASE.json"
  echo "Hypotheses: $OUT_BASE.jsonl"
  echo "Status: $STATUS_FILE"
else
  notify "LongMemEval Failed" "Run crashed (exit $exit_code). See: $LOG_FILE"
  echo
  echo "LongMemEval failed with exit code $exit_code."
  echo "See log: $LOG_FILE"
  echo "Status: $STATUS_FILE"
fi

exit "$exit_code"
