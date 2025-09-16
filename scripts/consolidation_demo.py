#!/usr/bin/env python3
"""Convenience script to exercise the consolidation endpoints manually."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

import requests


def request_consolidation(base_url: str, mode: str, dry_run: bool) -> Dict[str, Any]:
    response = requests.post(
        f"{base_url.rstrip('/')}/consolidate",
        json={"mode": mode, "dry_run": dry_run},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["consolidation"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["full", "decay", "creative", "cluster", "forget"], help="Consolidation mode to run")
    parser.add_argument("--base-url", default="http://localhost:8001", help="Memory service base URL")
    parser.add_argument("--apply", action="store_true", help="Mutable run instead of dry-run")
    args = parser.parse_args()

    try:
        result = request_consolidation(args.base_url, args.mode, not args.apply)
    except requests.RequestException as exc:  # pragma: no cover - manual helper
        sys.exit(f"Request failed: {exc}")

    steps = result.get("steps", {})
    summary = {
        "mode": result.get("mode"),
        "dry_run": result.get("dry_run"),
        "success": result.get("success"),
        "steps": {step: list(info.keys()) for step, info in steps.items()},
    }

    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
