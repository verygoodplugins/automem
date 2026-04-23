#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

resolve_python() {
    if [[ -n "${AUTOMEM_PYTHON:-}" ]]; then
        if command -v "$AUTOMEM_PYTHON" >/dev/null 2>&1; then
            echo "$AUTOMEM_PYTHON"
            return 0
        fi
        echo "❌ AUTOMEM_PYTHON is set to '$AUTOMEM_PYTHON' but was not found or is not executable." >&2
        return 1
    fi

    if command -v python3.12 >/dev/null 2>&1; then
        echo "python3.12"
        return 0
    fi

    if [[ -x /opt/homebrew/bin/python3.12 ]]; then
        echo "/opt/homebrew/bin/python3.12"
        return 0
    fi

    if command -v python3 >/dev/null 2>&1; then
        local version
        version="$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')"
        if [[ "$version" == "3.12" ]]; then
            echo "python3"
            return 0
        fi
        echo "❌ Found python3=$version, but AutoMem local dev currently expects Python 3.12." >&2
        echo "   Install python3.12 (for example with Homebrew) or set AUTOMEM_PYTHON explicitly." >&2
        return 1
    fi

    echo "❌ Could not find a usable Python interpreter. Install Python 3.12 first." >&2
    return 1
}

PYTHON_BIN="$(resolve_python)"
PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"

echo "🔧 Bootstrapping AutoMem with Python $PYTHON_VERSION ($PYTHON_BIN)"

if [[ -x .venv/bin/python ]]; then
    EXISTING_VERSION="$(.venv/bin/python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')"
    if [[ "$EXISTING_VERSION" != "3.12" ]]; then
        echo "♻️  Replacing incompatible .venv (found Python $EXISTING_VERSION)"
        rm -rf .venv
    fi
fi

if [[ -e venv && ! -L venv ]]; then
    echo "♻️  Removing legacy venv directory so it can point at .venv"
    rm -rf venv
fi

if [[ ! -x .venv/bin/python ]]; then
    "$PYTHON_BIN" -m venv .venv
fi

ln -sfn .venv venv

.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements-dev.txt

if [[ -x .venv/bin/pre-commit ]]; then
    .venv/bin/pre-commit install
fi

echo "✅ Virtual environment ready at .venv"
echo "💡 Run 'source .venv/bin/activate' to activate"
