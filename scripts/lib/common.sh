#!/usr/bin/env bash
# Shared utilities for benchmark scripts

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

wait_for_api() {
    local url="$1" max="${2:-60}" attempt=0
    echo -e "${BLUE}Waiting for API at ${url}...${NC}"
    while [ $attempt -lt $max ]; do
        if curl -fsS "${url}/health" > /dev/null 2>&1; then
            echo -e "${GREEN}API ready after ${attempt}s${NC}"
            return 0
        fi
        sleep 1; attempt=$((attempt + 1))
        [ $((attempt % 10)) -eq 0 ] && echo -e "${YELLOW}  Still waiting... (${attempt}s)${NC}"
    done
    echo -e "${RED}ERROR: API not ready after ${max}s${NC}"
    return 1
}
