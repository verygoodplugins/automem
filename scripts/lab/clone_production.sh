#!/usr/bin/env bash
# Clone AutoMem production data from Railway to local Docker.
#
# Usage:
#   ./scripts/lab/clone_production.sh                     # Auto-named snapshot
#   ./scripts/lab/clone_production.sh my-snapshot          # Custom name
#   ./scripts/lab/clone_production.sh --restore-only NAME  # Restore existing snapshot
#
# Prerequisites:
#   - Railway CLI installed and linked to AutoMem project
#   - Docker running
#   - Python venv with requirements installed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LAB_DIR="$PROJECT_ROOT/lab"

# Parse args
RESTORE_ONLY=false
SNAPSHOT_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --restore-only)
            RESTORE_ONLY=true
            SNAPSHOT_NAME="${2:-}"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [snapshot-name] [--restore-only NAME]"
            echo ""
            echo "Clone Railway AutoMem data to local Docker for testing."
            echo ""
            echo "Options:"
            echo "  --restore-only NAME  Restore an existing snapshot (skip backup)"
            echo "  --help               Show this help"
            exit 0
            ;;
        *)
            SNAPSHOT_NAME="$1"
            shift
            ;;
    esac
done

SNAPSHOT_NAME="${SNAPSHOT_NAME:-production-$(date +%Y-%m-%d-%H%M)}"
SNAPSHOT_DIR="$LAB_DIR/snapshots/$SNAPSHOT_NAME"

echo "=== AutoMem Recall Quality Lab ==="
echo "Snapshot: $SNAPSHOT_NAME"
echo ""

# ---------- Phase 1: Backup from Railway ----------
if [ "$RESTORE_ONLY" = false ]; then
    echo "[1/4] Backing up from Railway..."

    # Fetch Railway env vars
    if ! command -v railway &>/dev/null; then
        echo "ERROR: Railway CLI not installed. Install with: brew install railway"
        exit 1
    fi

    # Get Railway variables
    RAILWAY_VARS=$(railway variables --json 2>/dev/null || true)
    if [ -z "$RAILWAY_VARS" ] || [ "$RAILWAY_VARS" = "{}" ]; then
        echo "ERROR: Could not fetch Railway variables. Run 'railway link' first."
        exit 1
    fi

    # Extract connection info
    RAILWAY_FALKORDB_HOST=$(echo "$RAILWAY_VARS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('FALKORDB_HOST',''))" 2>/dev/null)
    RAILWAY_FALKORDB_PORT=$(echo "$RAILWAY_VARS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('FALKORDB_PORT','6379'))" 2>/dev/null)
    RAILWAY_FALKORDB_PASSWORD=$(echo "$RAILWAY_VARS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('FALKORDB_PASSWORD',''))" 2>/dev/null)
    RAILWAY_QDRANT_URL=$(echo "$RAILWAY_VARS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('QDRANT_URL',''))" 2>/dev/null)
    RAILWAY_QDRANT_API_KEY=$(echo "$RAILWAY_VARS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('QDRANT_API_KEY',''))" 2>/dev/null)

    if [ -z "$RAILWAY_FALKORDB_HOST" ]; then
        echo "ERROR: FALKORDB_HOST not found in Railway variables."
        echo "Ensure TCP Proxy is enabled on your FalkorDB service."
        exit 1
    fi

    echo "  FalkorDB: $RAILWAY_FALKORDB_HOST:$RAILWAY_FALKORDB_PORT"
    echo "  Qdrant:   ${RAILWAY_QDRANT_URL:0:40}..."

    mkdir -p "$SNAPSHOT_DIR"

    FALKORDB_HOST="$RAILWAY_FALKORDB_HOST" \
    FALKORDB_PORT="$RAILWAY_FALKORDB_PORT" \
    FALKORDB_PASSWORD="$RAILWAY_FALKORDB_PASSWORD" \
    QDRANT_URL="$RAILWAY_QDRANT_URL" \
    QDRANT_API_KEY="$RAILWAY_QDRANT_API_KEY" \
    AUTOMEM_BACKUP_DIR="$SNAPSHOT_DIR" \
    python3 "$PROJECT_ROOT/scripts/backup_automem.py"

    echo "  Backup saved to: $SNAPSHOT_DIR"
    echo ""
else
    echo "[1/4] Skipped (using existing snapshot)"
    if [ ! -d "$SNAPSHOT_DIR" ]; then
        echo "ERROR: Snapshot not found: $SNAPSHOT_DIR"
        echo "Available snapshots:"
        ls "$LAB_DIR/snapshots/" 2>/dev/null || echo "  (none)"
        exit 1
    fi
    echo ""
fi

# ---------- Phase 2: Prepare local Docker ----------
echo "[2/4] Preparing local Docker environment..."

cd "$PROJECT_ROOT"

# Stop existing containers and remove volumes for clean state
docker compose down -v 2>/dev/null || true

# Start only the databases (not the API yet)
docker compose up -d falkordb qdrant

echo "  Waiting for databases to be ready..."
sleep 5

# Wait for FalkorDB health check
for i in $(seq 1 30); do
    if docker compose exec -T falkordb redis-cli ping 2>/dev/null | grep -q PONG; then
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: FalkorDB did not become ready in time"
        exit 1
    fi
    sleep 1
done
echo "  FalkorDB: ready"

# Wait for Qdrant
for i in $(seq 1 30); do
    if curl -s http://localhost:6333/healthz 2>/dev/null | grep -q ok; then
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: Qdrant did not become ready in time"
        exit 1
    fi
    sleep 1
done
echo "  Qdrant: ready"
echo ""

# ---------- Phase 3: Restore data ----------
echo "[3/4] Restoring data from snapshot..."

FALKORDB_HOST="localhost" \
FALKORDB_PORT="6379" \
FALKORDB_PASSWORD="" \
QDRANT_URL="http://localhost:6333" \
QDRANT_API_KEY="" \
AUTOMEM_BACKUP_DIR="$SNAPSHOT_DIR" \
python3 "$PROJECT_ROOT/scripts/restore_from_backup.py" --force

echo ""

# ---------- Phase 4: Start API (consolidation disabled) ----------
echo "[4/4] Starting test API server (consolidation disabled)..."

# Build and start the API with consolidation disabled
CONSOLIDATION_DECAY_INTERVAL_SECONDS=0 \
CONSOLIDATION_CREATIVE_INTERVAL_SECONDS=0 \
CONSOLIDATION_CLUSTER_INTERVAL_SECONDS=0 \
CONSOLIDATION_FORGET_INTERVAL_SECONDS=0 \
docker compose up -d flask-api

# Wait for API
for i in $(seq 1 30); do
    if curl -s http://localhost:8001/health 2>/dev/null | grep -q "ok\|healthy"; then
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "WARNING: API health check didn't respond, but may still be starting..."
        break
    fi
    sleep 1
done

# Validate memory count
LOCAL_COUNT=$(curl -s -H "Authorization: Bearer test-token" http://localhost:8001/analyze 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('total_memories', d.get('node_count', '?')))" 2>/dev/null || echo "?")

echo ""
echo "=== Clone Complete ==="
echo "  Snapshot:  $SNAPSHOT_DIR"
echo "  API:       http://localhost:8001"
echo "  Memories:  $LOCAL_COUNT"
echo "  Console:   consolidation DISABLED (read-only mode)"
echo ""
echo "Next steps:"
echo "  python scripts/lab/create_test_queries.py"
echo "  python scripts/lab/run_recall_test.py --config baseline"
