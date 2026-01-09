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
    type_conf = data.get("type_confidence", 0)
    importance = data.get("importance", 0.5)
    elapsed = data.get("elapsed_ms", 0)

    console.print(
        f"  [cyan]id:[/] {mem_id}  "
        f"[cyan]type:[/] {mem_type} ({type_conf:.2f})  "
        f"[cyan]importance:[/] {importance}  "
        f"[dim]{elapsed}ms[/]"
    )

    # Tags
    tags = data.get("tags", [])
    if tags:
        console.print(f"  [cyan]tags:[/] {', '.join(tags)}")

    # Metadata
    metadata = data.get("metadata", {})
    if metadata:
        meta_preview = ", ".join(f"{k}={v}" for k, v in list(metadata.items())[:5])
        if len(metadata) > 5:
            meta_preview += f", ... (+{len(metadata) - 5} more)"
        console.print(f"  [cyan]metadata:[/] {{{meta_preview}}}")

    # Embedding status
    emb_status = data.get("embedding_status", "")
    qdrant_status = data.get("qdrant_status", "")
    if emb_status or qdrant_status:
        console.print(f"  [cyan]embedding:[/] {emb_status}  [cyan]qdrant:[/] {qdrant_status}")

    # Full content
    content = data.get("content", "")
    if content:
        console.print("  [cyan]content:[/]")
        # Wrap long content
        wrapped = textwrap.fill(content, width=100, initial_indent="    ", subsequent_indent="    ")
        console.print(f"[white]{wrapped}[/]")


def print_recall_event(event: dict) -> None:
    """Print a memory.recall event with query details and result summaries."""
    data = event.get("data", {})
    ts = format_timestamp(event.get("timestamp", ""))

    console.print()
    console.print(f"[bold cyan]━━━ RECALL[/] [dim]{ts}[/]")

    # Query info
    query = data.get("query", "")
    result_count = data.get("result_count", 0)
    dedup = data.get("dedup_removed", 0)
    elapsed = data.get("elapsed_ms", 0)

    dedup_str = f" (dedup: {dedup})" if dedup else ""
    console.print(f'  [yellow]query:[/] "{query}"')
    console.print(f"  [yellow]results:[/] {result_count}{dedup_str}  [dim]{elapsed}ms[/]")

    # Filters
    filters = []
    if data.get("has_time_filter"):
        filters.append("time")
    if data.get("tags_filter"):
        filters.append(f"tags({len(data['tags_filter'])})")
    if data.get("vector_search"):
        filters.append("vector")
    if filters:
        console.print(f"  [yellow]filters:[/] {', '.join(filters)}")

    # Stats
    stats = data.get("stats", {})
    if stats:
        avg_len = stats.get("avg_length", 0)
        avg_tags = stats.get("avg_tags", 0)
        score_range = stats.get("score_range", [0, 0])
        console.print(
            f"  [yellow]stats:[/] avg_len={avg_len} avg_tags={avg_tags} "
            f"score_range={score_range[0]:.2f}-{score_range[1]:.2f}"
        )

    # Top results
    summaries = data.get("result_summaries", [])
    if summaries:
        console.print("  [yellow]top results:[/]")
        for i, r in enumerate(summaries[:3], 1):
            console.print(
                f"    #{i} [{r.get('type', '?'):8s}] "
                f"score={r.get('score', 0):.2f} "
                f"len={r.get('content_length', 0)} "
                f"tags={r.get('tags_count', 0)}"
            )


def print_enrichment_event(event: dict) -> None:
    """Print enrichment events with details."""
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
            reason = data.get("skip_reason", "")
            console.print(f"[dim]{ts}[/] [yellow]ENRICH[/] {mem_id} skipped ({reason})")
        else:
            console.print(f"[dim]{ts}[/] [green]ENRICH[/] {mem_id} done ({elapsed}ms)")

            # Entity counts
            entities = data.get("entities", {})
            if entities:
                entity_parts = [f"{k}={v}" for k, v in entities.items()]
                console.print(f"  [dim]entities:[/] {' '.join(entity_parts)}")

            # Links
            temporal = data.get("temporal_links", 0)
            semantic = data.get("semantic_neighbors", 0)
            patterns = data.get("patterns_detected", 0)
            entity_tags = data.get("entity_tags_added", 0)

            if temporal or semantic or patterns:
                console.print(
                    f"  [dim]links:[/] temporal={temporal} semantic={semantic} patterns={patterns}"
                )

            if entity_tags:
                console.print(f"  [dim]entity_tags_added:[/] {entity_tags}")

            # Summary preview
            summary = data.get("summary_preview", "")
            if summary:
                console.print(f'  [dim]summary:[/] "{summary}"')

    elif event_type == "enrichment.failed":
        error = data.get("error", "unknown")[:80]
        attempt = data.get("attempt", 1)
        will_retry = data.get("will_retry", False)
        retry_str = " (will retry)" if will_retry else ""
        console.print(
            f"[dim]{ts}[/] [red]ENRICH FAIL[/] {mem_id} attempt {attempt}: {error}{retry_str}"
        )


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
    reconnect_count = 0

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

                    if reconnect_count > 0:
                        console.print(f"[green]Connected[/] [dim](reconnect #{reconnect_count})[/]")
                    else:
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
            reconnect_count += 1
            console.print(f"[red]Connection error:[/] {e}")
            console.print(f"[dim]Reconnecting in 5s... (attempt #{reconnect_count})[/]")
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
