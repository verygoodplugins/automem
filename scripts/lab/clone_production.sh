#!/usr/bin/env bash
# Clone AutoMem production data into an isolated local Docker stack.
#
# Usage:
#   ./scripts/lab/clone_production.sh
#   ./scripts/lab/clone_production.sh my-snapshot
#   ./scripts/lab/clone_production.sh --restore-only lab/snapshots/prod-api-.../snapshot.tar.gz
#   ./scripts/lab/clone_production.sh --restore-only prod-api-20260501-021907 \
#     --compose-project automem-sweep --api-port 8011 --qdrant-port 6343 --falkordb-port 6389
#
# Default behavior still takes a direct DB backup from Railway, which requires
# FALKORDB_TCP_PROXY and Qdrant access. Use --restore-only with a saved API
# tarball to avoid touching production on repeated local experiment runs.
#
# Prerequisites:
#   - Railway CLI installed and linked to AutoMem project for non-restore-only mode
#   - Docker running
#   - Python venv with requirements installed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LAB_DIR="$PROJECT_ROOT/lab"

RESTORE_ONLY=false
SNAPSHOT_NAME=""
SNAPSHOT_INPUT=""
COMPOSE_PROJECT="${COMPOSE_PROJECT_NAME:-automem}"
AUTOMEM_API_HOST_PORT="${AUTOMEM_API_HOST_PORT:-8001}"
QDRANT_HOST_PORT="${QDRANT_HOST_PORT:-6333}"
FALKORDB_HOST_PORT="${FALKORDB_HOST_PORT:-6379}"
FALKORDB_BROWSER_HOST_PORT="${FALKORDB_BROWSER_HOST_PORT:-3000}"
LOCAL_FALKORDB_PASSWORD="${LOCAL_FALKORDB_PASSWORD:-}"
LOCAL_QDRANT_API_KEY="${LOCAL_QDRANT_API_KEY:-}"
LOCAL_AUTOMEM_API_TOKEN="${LOCAL_AUTOMEM_API_TOKEN:-test-token}"
LOCAL_ADMIN_API_TOKEN="${LOCAL_ADMIN_API_TOKEN:-test-admin-token}"
PYTHON_BIN="${PYTHON_BIN:-}"

usage() {
    cat <<EOF
Usage:
  $0 [snapshot-name]
  $0 --restore-only NAME_OR_PATH [options]

Clone Railway AutoMem data to local Docker for testing.

Options:
  --restore-only NAME_OR_PATH       Restore an existing snapshot and skip production backup.
                                    Accepts:
                                      - lab/snapshots/<name>/snapshot.tar.gz
                                      - lab/snapshots/<name>/
                                      - <name> under lab/snapshots/
  --compose-project NAME            Docker Compose project name for isolated containers/volumes.
                                    Default: $COMPOSE_PROJECT
  --api-port PORT                   Host port for Flask API. Default: $AUTOMEM_API_HOST_PORT
  --qdrant-port PORT                Host port for Qdrant. Default: $QDRANT_HOST_PORT
  --falkordb-port PORT              Host port for FalkorDB. Default: $FALKORDB_HOST_PORT
  --falkordb-browser-port PORT      Host port for FalkorDB browser. Default: $FALKORDB_BROWSER_HOST_PORT
  --python PATH                     Python executable. Default: .venv/bin/python if present, else python3
  --help                            Show this help

Examples:
  # Restore the saved production API tarball into an isolated local stack.
  $0 --restore-only lab/snapshots/prod-api-20260501-021907/snapshot.tar.gz \\
    --compose-project automem-sweep --api-port 8011 --qdrant-port 6343 --falkordb-port 6389

  # Re-run a destructive experiment from the same local tarball without hitting production.
  $0 --restore-only prod-api-20260501-021907 --compose-project automem-sweep
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --restore-only)
            RESTORE_ONLY=true
            SNAPSHOT_INPUT="${2:-}"
            if [ -z "$SNAPSHOT_INPUT" ]; then
                echo "ERROR: --restore-only requires a snapshot name or path."
                exit 1
            fi
            shift 2
            ;;
        --compose-project)
            COMPOSE_PROJECT="${2:-}"
            if [ -z "$COMPOSE_PROJECT" ]; then
                echo "ERROR: --compose-project requires a value."
                exit 1
            fi
            shift 2
            ;;
        --api-port)
            AUTOMEM_API_HOST_PORT="${2:-}"
            if [ -z "$AUTOMEM_API_HOST_PORT" ]; then
                echo "ERROR: --api-port requires a value."
                exit 1
            fi
            shift 2
            ;;
        --qdrant-port)
            QDRANT_HOST_PORT="${2:-}"
            if [ -z "$QDRANT_HOST_PORT" ]; then
                echo "ERROR: --qdrant-port requires a value."
                exit 1
            fi
            shift 2
            ;;
        --falkordb-port)
            FALKORDB_HOST_PORT="${2:-}"
            if [ -z "$FALKORDB_HOST_PORT" ]; then
                echo "ERROR: --falkordb-port requires a value."
                exit 1
            fi
            shift 2
            ;;
        --falkordb-browser-port)
            FALKORDB_BROWSER_HOST_PORT="${2:-}"
            if [ -z "$FALKORDB_BROWSER_HOST_PORT" ]; then
                echo "ERROR: --falkordb-browser-port requires a value."
                exit 1
            fi
            shift 2
            ;;
        --python)
            PYTHON_BIN="${2:-}"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            if [ -n "$SNAPSHOT_NAME" ]; then
                echo "ERROR: Unexpected argument: $1"
                usage
                exit 1
            fi
            SNAPSHOT_NAME="$1"
            shift
            ;;
    esac
done

if [ -z "$PYTHON_BIN" ]; then
    if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
        PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

if [ "$RESTORE_ONLY" = true ] && [ -n "$SNAPSHOT_NAME" ]; then
    echo "ERROR: pass either --restore-only NAME_OR_PATH or a new snapshot name, not both."
    exit 1
fi

SNAPSHOT_NAME="${SNAPSHOT_NAME:-production-$(date +%Y-%m-%d-%H%M)}"
SNAPSHOT_DIR="$LAB_DIR/snapshots/$SNAPSHOT_NAME"
RESTORE_SOURCE=""

compose() {
    FALKORDB_HOST_PORT="$FALKORDB_HOST_PORT" \
    FALKORDB_BROWSER_HOST_PORT="$FALKORDB_BROWSER_HOST_PORT" \
    QDRANT_HOST_PORT="$QDRANT_HOST_PORT" \
    AUTOMEM_API_HOST_PORT="$AUTOMEM_API_HOST_PORT" \
    FALKORDB_PASSWORD="$LOCAL_FALKORDB_PASSWORD" \
    QDRANT_API_KEY="$LOCAL_QDRANT_API_KEY" \
    AUTOMEM_API_TOKEN="$LOCAL_AUTOMEM_API_TOKEN" \
    ADMIN_API_TOKEN="$LOCAL_ADMIN_API_TOKEN" \
    docker compose -p "$COMPOSE_PROJECT" "$@"
}

resolve_restore_source() {
    local input="$1"
    local candidate=""

    if [ -e "$input" ]; then
        candidate="$input"
    elif [ -e "$PROJECT_ROOT/$input" ]; then
        candidate="$PROJECT_ROOT/$input"
    elif [ -e "$LAB_DIR/snapshots/$input" ]; then
        candidate="$LAB_DIR/snapshots/$input"
    else
        echo "ERROR: Snapshot not found: $input"
        echo "Available snapshots:"
        ls "$LAB_DIR/snapshots/" 2>/dev/null || echo "  (none)"
        exit 1
    fi

    if [ -d "$candidate" ] && [ -f "$candidate/snapshot.tar.gz" ]; then
        candidate="$candidate/snapshot.tar.gz"
    fi

    echo "$candidate"
}

wait_for_falkordb() {
    for i in $(seq 1 30); do
        if [ -n "$LOCAL_FALKORDB_PASSWORD" ]; then
            if compose exec -T falkordb env REDISCLI_AUTH="$LOCAL_FALKORDB_PASSWORD" redis-cli ping 2>/dev/null | grep -q PONG; then
                return 0
            fi
        elif compose exec -T falkordb redis-cli ping 2>/dev/null | grep -q PONG; then
            return 0
        fi
        sleep 1
    done
    return 1
}

wait_for_qdrant() {
    for i in $(seq 1 30); do
        if curl -fsS "http://localhost:${QDRANT_HOST_PORT}/collections" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

wait_for_api() {
    for i in $(seq 1 45); do
        if curl -fsS "http://localhost:${AUTOMEM_API_HOST_PORT}/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

echo "=== AutoMem Recall Quality Lab ==="
echo "Compose project: $COMPOSE_PROJECT"
echo "Ports: API=$AUTOMEM_API_HOST_PORT Qdrant=$QDRANT_HOST_PORT FalkorDB=$FALKORDB_HOST_PORT Browser=$FALKORDB_BROWSER_HOST_PORT"
echo "Python: $PYTHON_BIN"
echo ""

# ---------- Phase 1: Backup or resolve existing snapshot ----------
if [ "$RESTORE_ONLY" = false ]; then
    echo "[1/4] Backing up from Railway..."

    if ! command -v railway &>/dev/null; then
        echo "ERROR: Railway CLI not installed. Install with: brew install railway"
        exit 1
    fi

    RAILWAY_VARS=$(railway variables --json 2>/dev/null || true)
    if [ -z "$RAILWAY_VARS" ] || [ "$RAILWAY_VARS" = "{}" ]; then
        echo "ERROR: Could not fetch Railway variables. Run 'railway link' first."
        exit 1
    fi

    RAILWAY_FALKORDB_PASSWORD=$(echo "$RAILWAY_VARS" | "$PYTHON_BIN" -c "import sys,json; print(json.load(sys.stdin).get('FALKORDB_PASSWORD',''))" 2>/dev/null)
    RAILWAY_QDRANT_URL=$(echo "$RAILWAY_VARS" | "$PYTHON_BIN" -c "import sys,json; print(json.load(sys.stdin).get('QDRANT_URL',''))" 2>/dev/null)
    RAILWAY_QDRANT_API_KEY=$(echo "$RAILWAY_VARS" | "$PYTHON_BIN" -c "import sys,json; print(json.load(sys.stdin).get('QDRANT_API_KEY',''))" 2>/dev/null)

    FALKORDB_PROXY="${FALKORDB_TCP_PROXY:-}"
    if [ -z "$FALKORDB_PROXY" ]; then
        echo "ERROR: FALKORDB_TCP_PROXY not set."
        echo "Enable TCP Proxy on FalkorDB in Railway dashboard, then run:"
        echo "  FALKORDB_TCP_PROXY=host:port $0"
        exit 1
    fi

    RAILWAY_FALKORDB_HOST=$(echo "$FALKORDB_PROXY" | cut -d: -f1)
    RAILWAY_FALKORDB_PORT=$(echo "$FALKORDB_PROXY" | cut -d: -f2)

    echo "  Snapshot: $SNAPSHOT_NAME"
    echo "  FalkorDB: $RAILWAY_FALKORDB_HOST:$RAILWAY_FALKORDB_PORT"
    echo "  Qdrant:   ${RAILWAY_QDRANT_URL:0:40}..."

    mkdir -p "$SNAPSHOT_DIR"

    FALKORDB_HOST="$RAILWAY_FALKORDB_HOST" \
    FALKORDB_PORT="$RAILWAY_FALKORDB_PORT" \
    FALKORDB_PASSWORD="$RAILWAY_FALKORDB_PASSWORD" \
    QDRANT_URL="$RAILWAY_QDRANT_URL" \
    QDRANT_API_KEY="$RAILWAY_QDRANT_API_KEY" \
    AUTOMEM_BACKUP_DIR="$SNAPSHOT_DIR" \
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/backup_automem.py"

    RESTORE_SOURCE="$SNAPSHOT_DIR"
    echo "  Backup saved to: $SNAPSHOT_DIR"
else
    echo "[1/4] Using existing snapshot..."
    RESTORE_SOURCE="$(resolve_restore_source "$SNAPSHOT_INPUT")"
    echo "  Restore source: $RESTORE_SOURCE"
fi
echo ""

# ---------- Phase 2: Prepare isolated local Docker ----------
echo "[2/4] Preparing local Docker environment..."

cd "$PROJECT_ROOT"

compose down -v 2>/dev/null || true
compose up -d falkordb qdrant

echo "  Waiting for databases to be ready..."
if ! wait_for_falkordb; then
    echo "ERROR: FalkorDB did not become ready in time"
    exit 1
fi
echo "  FalkorDB: ready"

if ! wait_for_qdrant; then
    echo "ERROR: Qdrant did not become ready in time"
    exit 1
fi
echo "  Qdrant: ready"
echo ""

# ---------- Phase 3: Restore data ----------
echo "[3/4] Restoring data from snapshot..."

FALKORDB_HOST="localhost" \
FALKORDB_PORT="$FALKORDB_HOST_PORT" \
FALKORDB_PASSWORD="$LOCAL_FALKORDB_PASSWORD" \
QDRANT_URL="http://localhost:${QDRANT_HOST_PORT}" \
QDRANT_API_KEY="$LOCAL_QDRANT_API_KEY" \
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/restore_from_backup.py" \
    --backup-dir "$RESTORE_SOURCE" \
    --force

echo ""

# ---------- Phase 4: Start API (consolidation disabled) ----------
echo "[4/4] Starting test API server (consolidation disabled)..."

CONSOLIDATION_DECAY_INTERVAL_SECONDS=0 \
CONSOLIDATION_CREATIVE_INTERVAL_SECONDS=0 \
CONSOLIDATION_CLUSTER_INTERVAL_SECONDS=0 \
CONSOLIDATION_FORGET_INTERVAL_SECONDS=0 \
compose up -d flask-api

if ! wait_for_api; then
    echo "WARNING: API health check did not respond, but the container may still be starting."
fi

LOCAL_COUNT=$(curl -fsS "http://localhost:${AUTOMEM_API_HOST_PORT}/health" 2>/dev/null \
    | "$PYTHON_BIN" -c "import sys,json; d=json.load(sys.stdin); print(d.get('memory_count','?'))" 2>/dev/null \
    || echo "?")

echo ""
echo "=== Clone Complete ==="
echo "  Compose:   $COMPOSE_PROJECT"
echo "  Snapshot:  $RESTORE_SOURCE"
echo "  API:       http://localhost:${AUTOMEM_API_HOST_PORT}"
echo "  Qdrant:    http://localhost:${QDRANT_HOST_PORT}"
echo "  FalkorDB:  localhost:${FALKORDB_HOST_PORT}"
echo "  Memories:  $LOCAL_COUNT"
echo "  Console:   consolidation disabled"
echo ""
echo "Next steps:"
echo "  python3 runners/sweep_corpus.py --scenario corpus_sweep_v1 --endpoint http://localhost:${AUTOMEM_API_HOST_PORT} --token $LOCAL_AUTOMEM_API_TOKEN"
