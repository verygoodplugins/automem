#!/bin/bash
#
# LoCoMo Benchmark Runner for AutoMem
#
# Evaluates AutoMem against the LoCoMo benchmark (ACL 2024)
# to measure long-term conversational memory performance.
#
# Usage:
#   ./test-locomo-benchmark.sh                    # Run against local Docker
#   ./test-locomo-benchmark.sh --live             # Run against Railway
#   ./test-locomo-benchmark.sh --help             # Show help
#

set -euo pipefail

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Shared utilities (colors + wait_for_api)
source "${SCRIPT_DIR}/scripts/lib/common.sh"

if [ -x "${SCRIPT_DIR}/venv/bin/python" ]; then
    PYTHON_BIN="${SCRIPT_DIR}/venv/bin/python"
else
    PYTHON_BIN="python3"
fi

# Default configuration
RUN_LIVE=false
CONVERSATIONS=""
RECALL_LIMIT=10
NO_CLEANUP=false
OUTPUT_FILE=""
JUDGE=false
JUDGE_MODEL=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --live)
            RUN_LIVE=true
            shift
            ;;
        --recall-limit)
            RECALL_LIMIT="$2"
            shift 2
            ;;
        --no-cleanup)
            NO_CLEANUP=true
            shift
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --judge)
            JUDGE=true
            shift
            ;;
        --judge-model)
            JUDGE=true
            JUDGE_MODEL="$2"
            shift 2
            ;;
        --conversations)
            CONVERSATIONS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --live              Run against Railway deployment (default: local Docker)"
            echo "  --recall-limit N    Number of memories to recall per question (default: 10)"
            echo "  --conversations I,J Comma-separated conversation indices (e.g. 0,1 for mini mode)"
            echo "  --judge             Enable category-5 LLM judge (defaults to gpt-5.1)"
            echo "  --judge-model MODEL Set the category-5 judge model (also enables judge)"
            echo "  --no-cleanup        Don't cleanup test data after evaluation"
            echo "  --output FILE       Save results to JSON file"
            echo "  --help, -h          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run locally"
            echo "  $0 --live                             # Run against Railway"
            echo "  $0 --conversations 0,1                # Mini mode (2 conversations)"
            echo "  $0 --conversations 0,1 --judge        # Mini mode with cat-5 judge"
            echo "  $0 --recall-limit 20 --output results.json"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}🧠 AutoMem LoCoMo Benchmark Runner${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Check if locomo dataset exists
LOCOMO_DATA="$SCRIPT_DIR/tests/benchmarks/locomo/data/locomo10.json"
if [ ! -f "$LOCOMO_DATA" ]; then
    echo -e "${RED}❌ LoCoMo dataset not found at: $LOCOMO_DATA${NC}"
    echo -e "${YELLOW}Please ensure the benchmark repository is cloned correctly.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Found LoCoMo dataset${NC}"

# Configure based on target environment
if [ "$RUN_LIVE" = true ]; then
    echo -e "${YELLOW}⚠️  Running against LIVE Railway deployment${NC}"
    echo ""
    echo -e "${YELLOW}This will:${NC}"
    echo -e "${YELLOW}  - Store ~10,000 test memories on Railway${NC}"
    echo -e "${YELLOW}  - Evaluate 1,986 questions${NC}"
    echo -e "${YELLOW}  - Take approximately 10-15 minutes${NC}"
    echo ""
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Cancelled.${NC}"
        exit 0
    fi

    # Check Railway CLI
    if ! command -v railway &> /dev/null; then
        echo -e "${RED}❌ Railway CLI not found${NC}"
        echo -e "${YELLOW}Install with: npm i -g @railway/cli${NC}"
        exit 1
    fi

    # Get Railway credentials
    echo -e "${BLUE}📡 Fetching Railway credentials...${NC}"

    export AUTOMEM_TEST_BASE_URL=$(railway variables get PUBLIC_URL 2>/dev/null || echo "")
    if [ -z "$AUTOMEM_TEST_BASE_URL" ]; then
        echo -e "${RED}❌ Could not fetch PUBLIC_URL from Railway${NC}"
        echo -e "${YELLOW}Make sure you're linked to the project: railway link${NC}"
        exit 1
    fi

    export AUTOMEM_TEST_API_TOKEN=$(railway variables get AUTOMEM_API_TOKEN 2>/dev/null || echo "")
    if [ -z "$AUTOMEM_TEST_API_TOKEN" ]; then
        echo -e "${RED}❌ Could not fetch AUTOMEM_API_TOKEN from Railway${NC}"
        exit 1
    fi

    echo -e "${GREEN}✅ Connected to Railway: $AUTOMEM_TEST_BASE_URL${NC}"

    # Enable live testing
    export AUTOMEM_ALLOW_LIVE=1

else
    echo -e "${BLUE}🐳 Running against local Docker${NC}"

    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        echo -e "${RED}❌ Docker is not running${NC}"
        echo -e "${YELLOW}Please start Docker and try again${NC}"
        exit 1
    fi

    # Check if services are running and healthy
    if ! curl -fsS "http://localhost:8001/health" > /dev/null 2>&1; then
        echo -e "${YELLOW}AutoMem services not running or unhealthy${NC}"
        echo -e "${BLUE}Starting services...${NC}"
        docker compose up -d
        wait_for_api "http://localhost:8001" 60 || exit 1
    fi

    export AUTOMEM_TEST_BASE_URL="http://localhost:8001"
    export AUTOMEM_TEST_API_TOKEN="test-token"

    echo -e "${GREEN}Docker services ready${NC}"
fi

# Build python command
PYTHON_CMD="$PYTHON_BIN $SCRIPT_DIR/tests/benchmarks/test_locomo.py"
PYTHON_CMD="$PYTHON_CMD --base-url $AUTOMEM_TEST_BASE_URL"
PYTHON_CMD="$PYTHON_CMD --api-token $AUTOMEM_TEST_API_TOKEN"
PYTHON_CMD="$PYTHON_CMD --recall-limit $RECALL_LIMIT"

if [ "$NO_CLEANUP" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --no-cleanup"
fi

if [ -n "$CONVERSATIONS" ]; then
    PYTHON_CMD="$PYTHON_CMD --conversations $CONVERSATIONS"
fi

if [ -n "$OUTPUT_FILE" ]; then
    PYTHON_CMD="$PYTHON_CMD --output $OUTPUT_FILE"
fi

if [ "$JUDGE" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --judge"
fi

if [ -n "$JUDGE_MODEL" ]; then
    PYTHON_CMD="$PYTHON_CMD --judge-model $JUDGE_MODEL"
fi

echo ""
echo -e "${BLUE}🚀 Starting benchmark evaluation...${NC}"
echo ""

# Run the benchmark
if $PYTHON_CMD; then
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}✅ Benchmark completed successfully!${NC}"
    echo -e "${GREEN}============================================${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}============================================${NC}"
    echo -e "${RED}❌ Benchmark failed${NC}"
    echo -e "${RED}============================================${NC}"
    exit 1
fi
