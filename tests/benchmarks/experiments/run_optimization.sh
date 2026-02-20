#!/bin/bash
#
# AutoMem Optimization Runner
# Finds optimal configuration through automated experimentation
#

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}  AutoMem Configuration Optimizer          ${NC}"
echo -e "${BLUE}=============================================${NC}"
echo ""

# Check if AutoMem is running
check_automem() {
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ AutoMem is running${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  AutoMem not running${NC}"
        return 1
    fi
}

# Activate virtualenv
activate_venv() {
    if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
        source "$PROJECT_ROOT/.venv/bin/activate"
    elif [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
        source "$PROJECT_ROOT/venv/bin/activate"
    else
        echo -e "${RED}❌ No virtualenv found${NC}"
        exit 1
    fi
}

# Run mode selection
select_mode() {
    echo "Select optimization mode:"
    echo "  1) Quick Grid Search (fast, limited questions)"
    echo "  2) Full Grid Search (comprehensive, slow)"
    echo "  3) AI-Guided Optimization (autonomous agent)"
    echo "  4) Ablation Study (single parameter)"
    echo "  5) Railway Parallel Testing"
    echo ""
    read -p "Enter choice [1-5]: " choice

    case $choice in
        1) MODE="grid_quick" ;;
        2) MODE="grid_full" ;;
        3) MODE="explore" ;;
        4) MODE="ablation" ;;
        5) MODE="railway" ;;
        *) MODE="grid_quick" ;;
    esac
}

# Main execution
main() {
    cd "$SCRIPT_DIR"
    activate_venv

    if ! check_automem; then
        echo -e "${YELLOW}Starting AutoMem...${NC}"
        cd "$PROJECT_ROOT"
        make dev &
        sleep 15
        cd "$SCRIPT_DIR"
    fi

    select_mode

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="$SCRIPT_DIR/results_${TIMESTAMP}"
    mkdir -p "$OUTPUT_DIR"

    echo ""
    echo -e "${GREEN}▶️  Starting optimization...${NC}"
    echo "   Mode: $MODE"
    echo "   Output: $OUTPUT_DIR"
    echo ""

    case $MODE in
        grid_quick)
            python experiment_runner.py \
                --mode grid \
                --quick \
                --output-dir "$OUTPUT_DIR" \
                2>&1 | tee "$OUTPUT_DIR/optimization.log"
            ;;
        grid_full)
            python experiment_runner.py \
                --mode grid \
                --output-dir "$OUTPUT_DIR" \
                2>&1 | tee "$OUTPUT_DIR/optimization.log"
            ;;
        explore)
            read -p "Max iterations [10]: " iterations
            iterations=${iterations:-10}
            python analysis_agent.py \
                --iterations "$iterations" \
                --output-dir "$OUTPUT_DIR" \
                2>&1 | tee "$OUTPUT_DIR/optimization.log"
            ;;
        ablation)
            read -p "Parameter to test: " param
            read -p "Values (comma-separated): " values
            python experiment_runner.py \
                --mode ablation \
                --param "$param" \
                --values "$values" \
                --output-dir "$OUTPUT_DIR" \
                2>&1 | tee "$OUTPUT_DIR/optimization.log"
            ;;
        railway)
            read -p "Number of parallel instances [3]: " parallel
            parallel=${parallel:-3}
            python experiment_runner.py \
                --mode grid \
                --railway \
                --parallel "$parallel" \
                --output-dir "$OUTPUT_DIR" \
                2>&1 | tee "$OUTPUT_DIR/optimization.log"
            ;;
    esac

    echo ""
    echo -e "${GREEN}✅ Optimization complete!${NC}"
    echo "   Results: $OUTPUT_DIR"

    # Show summary
    if [ -f "$OUTPUT_DIR/"*"_report.txt" ]; then
        echo ""
        echo -e "${BLUE}=== SUMMARY ===${NC}"
        cat "$OUTPUT_DIR/"*"_report.txt" | head -30
    fi
}

# Handle interrupts
trap 'echo -e "\n${RED}⚠️ Interrupted${NC}"; exit 1' INT TERM

main "$@"
