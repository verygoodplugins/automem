#!/bin/bash
#
# LongMemEval Benchmark Runner for AutoMem
#
# Evaluates AutoMem against the LongMemEval benchmark (ICLR 2025)
# to measure long-term conversational memory performance across
# 500 questions testing 5 core memory abilities.
#
# Usage:
#   ./test-longmemeval-benchmark.sh                        # Run against local Docker
#   ./test-longmemeval-benchmark.sh --live                 # Run against Railway
#   ./test-longmemeval-benchmark.sh --config full-graph    # Use full-graph config
#   ./test-longmemeval-benchmark.sh --max-questions 10     # Quick test with 10 questions
#   ./test-longmemeval-benchmark.sh --help                 # Show help
#

set -euo pipefail

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Shared utilities (colors + wait_for_api)
source "${SCRIPT_DIR}/scripts/lib/common.sh"

# Default configuration
RUN_LIVE=false
CONFIG="baseline"
RECALL_LIMIT=""
NO_CLEANUP=false
OUTPUT_FILE=""
MAX_QUESTIONS=""
PER_TYPE=""
LLM_MODEL=""
EVAL_LLM_MODEL=""
LLM_EVAL=false
RESUME=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --live)
            RUN_LIVE=true
            shift
            ;;
        --config)
            CONFIG="$2"
            shift 2
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
        --max-questions)
            MAX_QUESTIONS="$2"
            shift 2
            ;;
        --per-type)
            PER_TYPE="$2"
            shift 2
            ;;
        --llm-model)
            LLM_MODEL="$2"
            shift 2
            ;;
        --eval-llm-model)
            EVAL_LLM_MODEL="$2"
            shift 2
            ;;
        --llm-eval)
            LLM_EVAL=true
            shift
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --live              Run against Railway deployment (default: local Docker)"
            echo "  --config NAME       Benchmark config: baseline, per-turn, expand-entities,"
            echo "                      expand-relations, high-k, temporal, full-graph"
            echo "                      (default: baseline)"
            echo "  --recall-limit N    Override recall limit from config"
            echo "  --llm-model MODEL   LLM for answer generation (default: gpt-5-mini)"
            echo "  --eval-llm-model MODEL"
            echo "                      LLM judge override (default: canonical pinned judge)"
            echo "  --llm-eval          Use canonical OpenAI judge for evaluation"
            echo "  --max-questions N   Prefix-limit questions for debug smoke only"
            echo "  --per-type N        Select up to N questions per question_type"
            echo "  --resume            Resume from <output>.partial.jsonl (requires --output)"
            echo "  --no-cleanup        Don't cleanup test data after evaluation"
            echo "  --output FILE       Save results to specified path (without extension)"
            echo "  --help, -h          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                        # Baseline, local, all 500"
            echo "  $0 --per-type 5 --llm-eval                # Representative stratified mini"
            echo "  $0 --max-questions 10                     # Prefix debug smoke"
            echo "  $0 --config full-graph --llm-eval         # Full config with LLM eval"
            echo "  $0 --live --config temporal               # Temporal config on Railway"
            echo "  $0 --config high-k --recall-limit 30      # Custom recall limit"
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
echo -e "${BLUE}AutoMem LongMemEval Benchmark Runner${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Check if dataset exists
DATA_FILE="$SCRIPT_DIR/tests/benchmarks/longmemeval/data/longmemeval_s_cleaned.json"
if [ ! -f "$DATA_FILE" ]; then
    echo -e "${RED}Dataset not found at: $DATA_FILE${NC}"
    echo -e "${YELLOW}Download with:${NC}"
    echo "  mkdir -p tests/benchmarks/longmemeval/data"
    echo "  cd tests/benchmarks/longmemeval/data"
    echo "  wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
    exit 1
fi

echo -e "${GREEN}Found LongMemEval dataset${NC}"

# Configure based on target environment
if [ "$RUN_LIVE" = true ]; then
    echo -e "${YELLOW}Running against LIVE Railway deployment${NC}"
    echo ""

    QUESTION_COUNT=${MAX_QUESTIONS:-500}
    echo -e "${YELLOW}This will:${NC}"
    echo -e "${YELLOW}  - Ingest ~40 sessions per question into Railway${NC}"
    echo -e "${YELLOW}  - Evaluate $QUESTION_COUNT questions with LLM calls${NC}"
    echo ""
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Cancelled.${NC}"
        exit 0
    fi

    # Check Railway CLI
    if ! command -v railway &> /dev/null; then
        echo -e "${RED}Railway CLI not found${NC}"
        echo -e "${YELLOW}Install with: npm i -g @railway/cli${NC}"
        exit 1
    fi

    # Get Railway credentials
    echo -e "${BLUE}Fetching Railway credentials...${NC}"

    base_url="$(railway variables get PUBLIC_URL 2>/dev/null)" || base_url=""
    if [ -z "$base_url" ]; then
        echo -e "${RED}Could not fetch PUBLIC_URL from Railway${NC}"
        echo -e "${YELLOW}Make sure you're linked to the project: railway link${NC}"
        exit 1
    fi
    export AUTOMEM_TEST_BASE_URL="$base_url"

    api_token="$(railway variables get AUTOMEM_API_TOKEN 2>/dev/null)" || api_token=""
    if [ -z "$api_token" ]; then
        echo -e "${RED}Could not fetch AUTOMEM_API_TOKEN from Railway${NC}"
        exit 1
    fi
    export AUTOMEM_TEST_API_TOKEN="$api_token"

    echo -e "${GREEN}Connected to Railway: $AUTOMEM_TEST_BASE_URL${NC}"
    export AUTOMEM_ALLOW_LIVE=1

else
    echo -e "${BLUE}Running against local Docker${NC}"

    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        echo -e "${RED}Docker is not running${NC}"
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
    if [ -z "${AUTOMEM_TEST_API_TOKEN:-}" ]; then
        env_api_token=""
        if [ -f "$SCRIPT_DIR/.env" ]; then
            env_api_token="$(awk -F= '/^AUTOMEM_API_TOKEN=/ {print substr($0, index($0, "=") + 1); exit}' "$SCRIPT_DIR/.env")"
            env_api_token="${env_api_token%\"}"
            env_api_token="${env_api_token#\"}"
            env_api_token="${env_api_token%\'}"
            env_api_token="${env_api_token#\'}"
        fi
        export AUTOMEM_TEST_API_TOKEN="${env_api_token:-test-token}"
    else
        export AUTOMEM_TEST_API_TOKEN
    fi

    echo -e "${GREEN}Docker services ready${NC}"
fi

# Build python command
PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi
PYTHON_CMD=("$PYTHON_BIN" "$SCRIPT_DIR/tests/benchmarks/longmemeval/test_longmemeval.py")
PYTHON_CMD+=(--base-url "$AUTOMEM_TEST_BASE_URL")
PYTHON_CMD+=(--api-token "$AUTOMEM_TEST_API_TOKEN")
PYTHON_CMD+=(--config "$CONFIG")

if [ -n "$RECALL_LIMIT" ]; then
    PYTHON_CMD+=(--recall-limit "$RECALL_LIMIT")
fi

if [ "$NO_CLEANUP" = true ]; then
    PYTHON_CMD+=(--no-cleanup)
fi

if [ -n "$OUTPUT_FILE" ]; then
    PYTHON_CMD+=(--output "$OUTPUT_FILE")
fi

if [ -n "$MAX_QUESTIONS" ]; then
    PYTHON_CMD+=(--max-questions "$MAX_QUESTIONS")
fi

if [ -n "$PER_TYPE" ]; then
    PYTHON_CMD+=(--per-type "$PER_TYPE")
fi

if [ -n "$LLM_MODEL" ]; then
    PYTHON_CMD+=(--llm-model "$LLM_MODEL")
fi

if [ -n "$EVAL_LLM_MODEL" ]; then
    PYTHON_CMD+=(--eval-llm-model "$EVAL_LLM_MODEL")
fi

if [ "$LLM_EVAL" = true ]; then
    PYTHON_CMD+=(--llm-eval)
fi

if [ "$RESUME" = true ]; then
    PYTHON_CMD+=(--resume)
fi

echo ""
echo -e "${BLUE}Config: $CONFIG${NC}"
echo -e "${BLUE}Starting benchmark evaluation...${NC}"
echo ""

# Run the benchmark
if "${PYTHON_CMD[@]}"; then
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}Benchmark completed successfully!${NC}"
    echo -e "${GREEN}============================================${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}============================================${NC}"
    echo -e "${RED}Benchmark failed${NC}"
    echo -e "${RED}============================================${NC}"
    exit 1
fi
