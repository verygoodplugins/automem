#!/bin/bash
# Backwards-compatible wrapper for non-interactive live test run
set -e
cd "$(dirname "$0")"
./test-live-server.sh --non-interactive "$@"
