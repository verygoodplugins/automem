#!/usr/bin/env python3
"""AutoMem real-time monitor - Terminal UI for observing memory service activity.

Usage:
    python scripts/automem_watch.py --url https://automem.railway.app --token $AUTOMEM_API_TOKEN
    python scripts/automem_watch.py --url http://localhost:8001 --token dev

Features:
    - Real-time SSE event streaming
    - Garbage pattern detection (short content, test data, duplicates, bursts)
    - Consolidation monitoring with timing
    - Auto-reconnect on connection loss
"""

import argparse
import hashlib
import json
import sys
import threading
import time
from collections import Counter, deque
from datetime import datetime
from typing import Deque, Dict, List, Optional, Set

try:
    import httpx
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print("Required dependencies missing. Install with:")
    print("  pip install rich httpx")
    sys.exit(1)


class GarbageDetector:
    """Detect suspicious patterns in memory stores."""

    SUSPICIOUS_KEYWORDS = {
        "test",
        "asdf",
        "xxx",
        "lorem",
        "foo",
        "bar",
        "baz",
        "debug",
        "tmp",
        "placeholder",
        "example",
        "sample",
    }

    def __init__(self, min_content_length: int = 10, burst_threshold: int = 5):
        self.min_content_length = min_content_length
        self.burst_threshold = burst_threshold
        self.recent_hashes: Deque[str] = deque(maxlen=100)
        self.warnings: Deque[str] = deque(maxlen=50)
        self.counts: Counter = Counter()
        self.burst_times: Deque[float] = deque(maxlen=20)

    def analyze(self, event: Dict) -> Optional[str]:
        """Analyze a memory.store event for garbage patterns."""
        if event.get("type") != "memory.store":
            return None

        data = event.get("data", {})
        content = data.get("content_preview", "")
        now = time.time()

        # Check: Burst detection (>N stores in 10 seconds)
        self.burst_times.append(now)
        recent_count = sum(1 for t in self.burst_times if now - t < 10)
        if recent_count >= self.burst_threshold:
            warning = f"Burst detected: {recent_count} stores in 10s"
            self._record(warning, "burst")
            # Don't return - continue checking other patterns

        # Check: Very short content
        if len(content) < self.min_content_length:
            warning = f"Short ({len(content)} chars): '{content[:30]}'"
            self._record(warning, "short_content")
            return warning

        # Check: Test/debug keywords
        content_lower = content.lower()
        for kw in self.SUSPICIOUS_KEYWORDS:
            if kw in content_lower and len(content) < 50:
                warning = f"Test keyword '{kw}': '{content[:40]}...'"
                self._record(warning, "test_keyword")
                return warning

        # Check: No tags
        tags = data.get("tags", [])
        if not tags:
            warning = f"No tags: '{content[:40]}...'"
            self._record(warning, "no_tags")
            return warning

        # Check: Very low importance with generic content
        importance = data.get("importance", 0.5)
        if importance < 0.3 and data.get("type") == "Memory":
            warning = f"Low importance ({importance}): '{content[:40]}...'"
            self._record(warning, "low_importance")
            return warning

        # Check: Duplicate content (hash-based)
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        if content_hash in self.recent_hashes:
            warning = f"Duplicate hash: '{content[:40]}...'"
            self._record(warning, "duplicate")
            return warning
        self.recent_hashes.append(content_hash)

        return None

    def _record(self, warning: str, category: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.warnings.appendleft(f"[{ts}] {warning}")
        self.counts[category] += 1


class AutoMemMonitor:
    """Real-time monitor for AutoMem service."""

    def __init__(self, url: str, token: str, min_content_length: int = 10):
        self.url = url.rstrip("/")
        self.token = token
        self.events: Deque[Dict] = deque(maxlen=100)
        self.errors: Deque[str] = deque(maxlen=20)
        self.stats = {
            "stores": 0,
            "recalls": 0,
            "enriched": 0,
            "consolidated": 0,
            "errors": 0,
        }
        self.garbage = GarbageDetector(min_content_length)
        self.start_time = datetime.now()
        self.connected = False
        self.last_event_time: Optional[datetime] = None
        self.consolidation_history: Deque[Dict] = deque(maxlen=10)

    def connect(self):
        """Connect to SSE stream and process events."""
        headers = {"Authorization": f"Bearer {self.token}"}

        with httpx.Client(timeout=None) as client:
            with client.stream("GET", f"{self.url}/stream", headers=headers) as response:
                if response.status_code != 200:
                    raise Exception(f"HTTP {response.status_code}: {response.text[:100]}")
                self.connected = True
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        try:
                            event = json.loads(line[6:])
                            self._process_event(event)
                        except json.JSONDecodeError:
                            pass
                    elif line.startswith(":"):
                        # Keepalive comment - ignore
                        pass

    def _process_event(self, event: Dict) -> None:
        """Process an incoming event."""
        self.last_event_time = datetime.now()
        self.events.appendleft(event)

        event_type = event.get("type", "")

        # Update stats
        if event_type == "memory.store":
            self.stats["stores"] += 1
            # Check for garbage
            warning = self.garbage.analyze(event)
            if warning:
                ts = datetime.now().strftime("%H:%M:%S")
                self.errors.appendleft(f"[{ts}] [GARBAGE] {warning}")
        elif event_type == "memory.recall":
            self.stats["recalls"] += 1
        elif event_type == "enrichment.complete":
            self.stats["enriched"] += 1
        elif event_type == "enrichment.failed":
            self.stats["errors"] += 1
            ts = datetime.now().strftime("%H:%M:%S")
            err = event.get("data", {}).get("error", "unknown")[:50]
            self.errors.appendleft(f"[{ts}] [ENRICH] {err}")
        elif event_type == "consolidation.run":
            self.stats["consolidated"] += 1
            self.consolidation_history.appendleft(event.get("data", {}))
        elif event_type == "error":
            self.stats["errors"] += 1
            ts = datetime.now().strftime("%H:%M:%S")
            err = event.get("data", {}).get("error", "unknown")[:50]
            self.errors.appendleft(f"[{ts}] [ERROR] {err}")

    def render(self) -> Layout:
        """Render the TUI layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=2),
            Layout(name="bottom", size=14),
        )

        layout["main"].split_row(
            Layout(name="events", ratio=2),
            Layout(name="consolidation", ratio=1),
        )

        layout["bottom"].split_row(
            Layout(name="stats", ratio=1),
            Layout(name="garbage", ratio=1),
            Layout(name="errors", ratio=1),
        )

        # Header
        uptime = datetime.now() - self.start_time
        uptime_str = (
            f"{int(uptime.total_seconds() // 3600)}h {int((uptime.total_seconds() % 3600) // 60)}m"
        )
        status = "[green]Connected[/]" if self.connected else "[red]Disconnected[/]"
        last_event = ""
        if self.last_event_time:
            ago = (datetime.now() - self.last_event_time).total_seconds()
            if ago < 60:
                last_event = f"  |  Last event: {int(ago)}s ago"
            else:
                last_event = f"  |  Last event: {int(ago // 60)}m ago"

        layout["header"].update(
            Panel(
                f"[bold]AutoMem Watch[/]  |  {self.url}  |  {status}  |  Uptime: {uptime_str}{last_event}",
                style="bold white on blue",
            )
        )

        # Events table
        events_table = Table(show_header=True, header_style="bold", expand=True, show_lines=False)
        events_table.add_column("Time", width=8)
        events_table.add_column("Type", width=18)
        events_table.add_column("Details", overflow="ellipsis")

        for event in list(self.events)[:15]:
            ts = event.get("timestamp", "")
            if "T" in ts:
                ts = ts.split("T")[1][:8]
            event_type = event.get("type", "unknown")
            data = event.get("data", {})

            # Format based on event type
            if event_type == "memory.store":
                preview = data.get("content_preview", "")[:40]
                details = f"{preview}... ({data.get('type', '?')})"
                type_style = "green"
            elif event_type == "memory.recall":
                query = data.get("query", "")[:30]
                details = (
                    f"'{query}' -> {data.get('result_count', 0)} ({data.get('elapsed_ms', 0)}ms)"
                )
                type_style = "cyan"
            elif event_type == "enrichment.start":
                details = f"{data.get('memory_id', '')[:8]}... attempt {data.get('attempt', 1)}"
                type_style = "yellow"
            elif event_type == "enrichment.complete":
                status = "skipped" if data.get("skipped") else "done"
                details = (
                    f"{data.get('memory_id', '')[:8]}... {status} ({data.get('elapsed_ms', 0)}ms)"
                )
                type_style = "green"
            elif event_type == "enrichment.failed":
                details = f"{data.get('memory_id', '')[:8]}... {data.get('error', '')[:30]}"
                type_style = "red"
            elif event_type == "consolidation.run":
                details = f"{data.get('task_type', '?')} - {data.get('affected_count', 0)} affected ({data.get('elapsed_ms', 0)}ms)"
                type_style = "magenta"
            else:
                details = str(data)[:50]
                type_style = "white"

            events_table.add_row(ts, f"[{type_style}]{event_type}[/]", details)

        layout["events"].update(Panel(events_table, title="Events (latest 15)"))

        # Consolidation panel
        consol_text = Text()
        if self.consolidation_history:
            for run in list(self.consolidation_history)[:5]:
                task = run.get("task_type", "?")
                affected = run.get("affected_count", 0)
                elapsed = run.get("elapsed_ms", 0)
                success = "[green]OK[/]" if run.get("success") else "[red]FAIL[/]"
                next_run = run.get("next_scheduled", "?")
                consol_text.append(f"{task}: ", style="bold")
                consol_text.append(f"{affected} affected, {elapsed}ms ")
                consol_text.append_markup(success)
                consol_text.append(f"\n  Next: {next_run}\n", style="dim")
        else:
            consol_text.append("No consolidation runs yet", style="dim")
        layout["consolidation"].update(Panel(consol_text, title="Consolidation"))

        # Stats panel
        stats_text = Text()
        stats_text.append(f"Stores:       {self.stats['stores']}\n", style="green")
        stats_text.append(f"Recalls:      {self.stats['recalls']}\n", style="cyan")
        stats_text.append(f"Enriched:     {self.stats['enriched']}\n", style="yellow")
        stats_text.append(f"Consolidated: {self.stats['consolidated']}\n", style="magenta")
        stats_text.append(f"Errors:       {self.stats['errors']}\n", style="red")
        layout["stats"].update(Panel(stats_text, title="Stats"))

        # Garbage detector panel
        garbage_text = Text()
        if self.garbage.counts:
            for category, count in self.garbage.counts.most_common(5):
                garbage_text.append(f"{category}: {count}\n", style="yellow")
            garbage_text.append("\n")
            for warning in list(self.garbage.warnings)[:3]:
                garbage_text.append(f"{warning[:45]}...\n", style="dim yellow")
        else:
            garbage_text.append("No suspicious patterns", style="green")
        layout["garbage"].update(Panel(garbage_text, title="Garbage Detector"))

        # Errors panel
        errors_text = Text()
        if self.errors:
            for err in list(self.errors)[:5]:
                errors_text.append(f"{err[:50]}\n", style="red")
        else:
            errors_text.append("No errors", style="green")
        layout["errors"].update(Panel(errors_text, title="Errors"))

        return layout


def main():
    parser = argparse.ArgumentParser(
        description="AutoMem real-time monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local development
  python scripts/automem_watch.py --url http://localhost:8001 --token dev

  # Against Railway
  python scripts/automem_watch.py --url https://automem.railway.app --token $AUTOMEM_API_TOKEN
        """,
    )
    parser.add_argument("--url", required=True, help="AutoMem API URL")
    parser.add_argument("--token", required=True, help="API token")
    parser.add_argument(
        "--min-content-length",
        type=int,
        default=10,
        help="Minimum content length before flagging as garbage (default: 10)",
    )
    args = parser.parse_args()

    console = Console()
    monitor = AutoMemMonitor(args.url, args.token, args.min_content_length)

    console.print(f"[bold]Connecting to {args.url}/stream...[/]")

    def connect_loop():
        while True:
            try:
                monitor.connect()
            except KeyboardInterrupt:
                break
            except Exception as e:
                monitor.connected = False
                ts = datetime.now().strftime("%H:%M:%S")
                monitor.errors.appendleft(f"[{ts}] [CONN] {str(e)[:50]}")
                time.sleep(5)  # Reconnect delay

    thread = threading.Thread(target=connect_loop, daemon=True)
    thread.start()

    # Give connection time to establish
    time.sleep(1)

    with Live(monitor.render(), console=console, refresh_per_second=2) as live:
        try:
            while True:
                live.update(monitor.render())
                time.sleep(0.5)
        except KeyboardInterrupt:
            console.print("\n[bold]Disconnected.[/]")


if __name__ == "__main__":
    main()
