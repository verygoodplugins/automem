#!/usr/bin/env python3
"""AutoMem tail - Simple streaming log of memory operations.

Usage:
    python scripts/automem_watch.py --url https://automem.up.railway.app --token $AUTOMEM_API_TOKEN
    python scripts/automem_watch.py --url http://localhost:8001 --token dev
"""

import argparse
import json
import sys
import textwrap
import time

try:
    import httpx
    from rich.console import Console
    from rich.syntax import Syntax
except ImportError:
    print("Required dependencies missing. Install with:")
    print("  pip install rich httpx")
    sys.exit(1)

console = Console()


def format_timestamp(ts: str) -> str:
    """Extract just the time from ISO timestamp."""
    if "T" in ts:
        return ts.split("T")[1][:12]
    return ts[:12]


def print_store_event(event: dict) -> None:
    """Print a memory.store event with full content."""
    data = event.get("data", {})
    ts = format_timestamp(event.get("timestamp", ""))

    console.print()
    console.print(f"[bold green]━━━ STORE[/] [dim]{ts}[/]")

    # Memory info line
    mem_id = data.get("memory_id", "?")[:8]
    mem_type = data.get("type", "Memory")
    importance = data.get("importance", 0.5)
    elapsed = data.get("elapsed_ms", 0)

    console.print(
        f"  [cyan]id:[/] {mem_id}  "
        f"[cyan]type:[/] {mem_type}  "
        f"[cyan]importance:[/] {importance}  "
        f"[dim]{elapsed}ms[/]"
    )

    # Tags
    tags = data.get("tags", [])
    if tags:
        console.print(f"  [cyan]tags:[/] {', '.join(tags)}")

    # Full content
    content = data.get("content_preview", "")
    if content:
        console.print("  [cyan]content:[/]")
        # Wrap long content
        wrapped = textwrap.fill(content, width=100, initial_indent="    ", subsequent_indent="    ")
        console.print(f"[white]{wrapped}[/]")


def print_recall_event(event: dict) -> None:
    """Print a memory.recall event with query details."""
    data = event.get("data", {})
    ts = format_timestamp(event.get("timestamp", ""))

    console.print()
    console.print(f"[bold cyan]━━━ RECALL[/] [dim]{ts}[/]")

    # Query info
    query = data.get("query", "")
    result_count = data.get("result_count", 0)
    elapsed = data.get("elapsed_ms", 0)

    console.print(f'  [yellow]query:[/] "{query}"')
    console.print(f"  [yellow]results:[/] {result_count}  [dim]{elapsed}ms[/]")

    # Filters
    filters = []
    if data.get("has_time_filter"):
        filters.append("time")
    if data.get("has_tag_filter"):
        filters.append("tags")
    if data.get("vector_search"):
        filters.append("vector")
    if filters:
        console.print(f"  [yellow]filters:[/] {', '.join(filters)}")


def print_enrichment_event(event: dict) -> None:
    """Print enrichment events."""
    data = event.get("data", {})
    ts = format_timestamp(event.get("timestamp", ""))
    event_type = event.get("type", "")

    mem_id = data.get("memory_id", "?")[:8]

    if event_type == "enrichment.start":
        attempt = data.get("attempt", 1)
        console.print(f"[dim]{ts}[/] [yellow]ENRICH[/] {mem_id} attempt {attempt}")
    elif event_type == "enrichment.complete":
        elapsed = data.get("elapsed_ms", 0)
        if data.get("skipped"):
            console.print(f"[dim]{ts}[/] [yellow]ENRICH[/] {mem_id} skipped")
        else:
            console.print(f"[dim]{ts}[/] [green]ENRICH[/] {mem_id} done ({elapsed}ms)")
    elif event_type == "enrichment.failed":
        error = data.get("error", "unknown")[:50]
        console.print(f"[dim]{ts}[/] [red]ENRICH FAIL[/] {mem_id}: {error}")


def print_consolidation_event(event: dict) -> None:
    """Print consolidation events."""
    data = event.get("data", {})
    ts = format_timestamp(event.get("timestamp", ""))

    task_type = data.get("task_type", "?")
    affected = data.get("affected_count", 0)
    elapsed = data.get("elapsed_ms", 0)
    success = data.get("success", True)

    status = "[green]OK[/]" if success else "[red]FAIL[/]"
    console.print(
        f"[dim]{ts}[/] [magenta]CONSOLIDATE[/] "
        f"{task_type}: {affected} affected ({elapsed}ms) {status}"
    )


def print_error_event(event: dict) -> None:
    """Print error events."""
    data = event.get("data", {})
    ts = format_timestamp(event.get("timestamp", ""))

    error = data.get("error", "unknown")
    console.print(f"[dim]{ts}[/] [bold red]ERROR[/] {error}")


def print_raw_event(event: dict) -> None:
    """Print any other event as JSON."""
    ts = format_timestamp(event.get("timestamp", ""))
    event_type = event.get("type", "unknown")

    console.print(f"[dim]{ts}[/] [blue]{event_type}[/]")
    # Print data as formatted JSON
    data = event.get("data", {})
    if data:
        json_str = json.dumps(data, indent=2, default=str)
        syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
        console.print(syntax)


def process_event(event: dict) -> None:
    """Route event to appropriate printer."""
    event_type = event.get("type", "")

    if event_type == "memory.store":
        print_store_event(event)
    elif event_type == "memory.recall":
        print_recall_event(event)
    elif event_type.startswith("enrichment."):
        print_enrichment_event(event)
    elif event_type == "consolidation.run":
        print_consolidation_event(event)
    elif event_type == "error":
        print_error_event(event)
    else:
        print_raw_event(event)


def stream_events(url: str, token: str) -> None:
    """Connect to SSE stream and print events."""
    headers = {"Authorization": f"Bearer {token}"}

    console.print(f"[bold]Connecting to {url}/stream...[/]")
    console.print("[dim]Press Ctrl+C to stop[/]")
    console.print()

    while True:
        try:
            with httpx.Client(timeout=None) as client:
                with client.stream("GET", f"{url}/stream", headers=headers) as resp:
                    if resp.status_code != 200:
                        console.print(f"[red]HTTP {resp.status_code}[/]")
                        break

                    console.print("[green]Connected[/]")

                    for line in resp.iter_lines():
                        if line.startswith("data: "):
                            try:
                                event = json.loads(line[6:])
                                process_event(event)
                            except json.JSONDecodeError:
                                pass
                        elif line.startswith(":"):
                            # Keepalive - ignore silently
                            pass

        except KeyboardInterrupt:
            console.print("\n[bold]Disconnected.[/]")
            break
        except Exception as e:
            console.print(f"[red]Connection error:[/] {e}")
            console.print("[dim]Reconnecting in 5s...[/]")
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser(
        description="AutoMem tail - stream memory operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/automem_watch.py --url http://localhost:8001 --token dev
  python scripts/automem_watch.py --url https://automem.up.railway.app --token $TOKEN
        """,
    )
    parser.add_argument("--url", required=True, help="AutoMem API URL")
    parser.add_argument("--token", required=True, help="API token")
    args = parser.parse_args()

    stream_events(args.url.rstrip("/"), args.token)


if __name__ == "__main__":
    main()
