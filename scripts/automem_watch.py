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
from typing import Any, Mapping

try:
    import httpx
    from rich.console import Console
    from rich.syntax import Syntax
except ImportError:
    print("Required dependencies missing. Install with:")
    print("  pip install rich httpx")
    sys.exit(1)

console = Console()

# Type alias for SSE event payloads
Event = Mapping[str, Any]


def format_timestamp(ts: Any) -> str:
    """Extract just the time from ISO timestamp."""
    if ts is None:
        return ""
    ts = str(ts)
    if "T" in ts:
        return ts.split("T")[1][:12]
    return ts[:12]


def _safe_float(val: object, default: float = 0.0) -> float:
    """Safely convert value to float."""
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def _safe_int(val: object, default: int = 0) -> int:
    """Safely convert value to int."""
    try:
        return int(float(val)) if val is not None else default
    except (TypeError, ValueError):
        return default


def print_store_event(event: Event) -> None:
    """Print a memory.store event with full content."""
    data = event.get("data", {})
    ts = format_timestamp(event.get("timestamp", ""))

    console.print()
    console.print(f"[bold green]━━━ STORE[/] [dim]{ts}[/]")

    # Memory info line - defensive conversions
    mem_id = str(data.get("memory_id", "?"))[:8]
    mem_type = data.get("type", "Memory")
    type_conf = _safe_float(data.get("type_confidence"), 0.0)
    importance = _safe_float(data.get("importance"), 0.5)
    elapsed = _safe_int(data.get("elapsed_ms"), 0)

    console.print(
        f"  [cyan]id:[/] {mem_id}  "
        f"[cyan]type:[/] {mem_type} ({type_conf:.2f})  "
        f"[cyan]importance:[/] {importance}  "
        f"[dim]{elapsed}ms[/]"
    )

    # Tags - coerce to strings for safety
    tags = data.get("tags", [])
    if tags:
        safe_tags = [str(t) for t in tags]
        console.print(f"  [cyan]tags:[/] {', '.join(safe_tags)}")

    # Metadata - truncate large values
    metadata = data.get("metadata", {})
    if metadata:

        def safe_val(v: object, max_len: int = 50) -> str:
            s = str(v) if isinstance(v, (str, int, float, bool)) else repr(v)
            return s[:max_len] + "..." if len(s) > max_len else s

        meta_preview = ", ".join(f"{k}={safe_val(v)}" for k, v in list(metadata.items())[:5])
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


def print_recall_event(event: Event) -> None:
    """Print a memory.recall event with query details and result summaries."""
    data = event.get("data", {})
    ts = format_timestamp(event.get("timestamp", ""))

    console.print()
    console.print(f"[bold cyan]━━━ RECALL[/] [dim]{ts}[/]")

    # Query info - defensive conversions
    query = data.get("query", "")
    result_count = _safe_int(data.get("result_count"), 0)
    dedup = _safe_int(data.get("dedup_removed"), 0)
    elapsed = _safe_int(data.get("elapsed_ms"), 0)

    dedup_str = f" (dedup: {dedup})" if dedup else ""
    console.print(f'  [yellow]query:[/] "{query}"')
    console.print(f"  [yellow]results:[/] {result_count}{dedup_str}  [dim]{elapsed}ms[/]")

    # Filters - defensive check for tags_filter
    filters = []
    if data.get("has_time_filter"):
        filters.append("time")
    tags_filter = data.get("tags_filter")
    if isinstance(tags_filter, (list, tuple, set)):
        filters.append(f"tags({len(tags_filter)})")
    elif tags_filter:
        filters.append("tags(?)")
    if data.get("vector_search"):
        filters.append("vector")
    if filters:
        console.print(f"  [yellow]filters:[/] {', '.join(filters)}")

    # Stats - defensive score_range handling
    stats = data.get("stats", {})
    if stats:
        avg_len = stats.get("avg_length", 0)
        avg_tags = stats.get("avg_tags", 0)
        score_range = stats.get("score_range", [])

        # Safely format score range
        try:
            if len(score_range) >= 2:
                score_str = f"{float(score_range[0]):.2f}-{float(score_range[1]):.2f}"
            else:
                score_str = "n/a"
        except (TypeError, ValueError):
            score_str = "n/a"

        console.print(
            f"  [yellow]stats:[/] avg_len={avg_len} avg_tags={avg_tags} " f"score_range={score_str}"
        )

    # Top results - defensive field conversions
    summaries = data.get("result_summaries", [])
    if summaries:
        console.print("  [yellow]top results:[/]")
        for i, r in enumerate(summaries[:3], 1):
            r_type = str(r.get("type", "?"))[:8]
            r_score = _safe_float(r.get("score"), 0.0)
            r_len = _safe_int(r.get("content_length"), 0)
            r_tags = _safe_int(r.get("tags_count"), 0)
            console.print(
                f"    #{i} [{r_type:8s}] " f"score={r_score:.2f} " f"len={r_len} " f"tags={r_tags}"
            )


def print_enrichment_event(event: Event) -> None:
    """Print enrichment events with details."""
    data = event.get("data", {})
    ts = format_timestamp(event.get("timestamp", ""))
    event_type = event.get("type", "")

    mem_id = str(data.get("memory_id", "?"))[:8]

    if event_type == "enrichment.start":
        attempt = data.get("attempt", 1)
        console.print(f"[dim]{ts}[/] [yellow]ENRICH[/] {mem_id} attempt {attempt}")

    elif event_type == "enrichment.complete":
        elapsed = data.get("elapsed_ms", 0)

        if data.get("skipped"):
            reason = data.get("skip_reason", "")
            console.print(f"[dim]{ts}[/] [yellow]ENRICH[/] {mem_id} skipped ({reason})")
        else:
            console.print(f"\n[bold cyan]━━━ ENRICH[/] [dim]{ts}[/] [cyan]({elapsed}ms)[/]")
            console.print(f"  [dim]memory:[/] {mem_id}")

            # Content
            content = data.get("content", "")
            if content:
                # Indent content lines
                content_lines = content[:500].split("\n")
                console.print("  [dim]content:[/]")
                for line in content_lines[:8]:
                    console.print(f"    {line}")
                if len(content) > 500 or len(content_lines) > 8:
                    console.print("    [dim]...[/]")

            # Tags before/after
            tags_before = data.get("tags_before", [])
            tags_added = data.get("tags_added", [])
            if tags_before or tags_added:
                console.print("")
                console.print(f"  [dim]tags before:[/] {tags_before}")
                if tags_added:
                    console.print(f"  [green]tags added:[/]  {tags_added}")

            # Entities by category
            entities = data.get("entities", {})
            if entities and any(entities.values()):
                console.print("")
                console.print("  [dim]entities:[/]")
                for category, values in entities.items():
                    if values:
                        console.print(f"    {category}: {', '.join(values)}")

            # Links created
            temporal_links = data.get("temporal_links", [])
            semantic_neighbors = data.get("semantic_neighbors", [])
            patterns = data.get("patterns_detected", [])

            if temporal_links or semantic_neighbors or patterns:
                console.print("")
                console.print("  [dim]links created:[/]")
                if temporal_links:
                    ids = [str(tid)[:8] for tid in temporal_links]
                    console.print(f"    temporal: {', '.join(ids)} ({len(ids)} memories)")
                if semantic_neighbors:
                    neighbor_strs = [f"{nid} ({score})" for nid, score in semantic_neighbors]
                    console.print(f"    semantic: {', '.join(neighbor_strs)}")
                if patterns:
                    for p in patterns:
                        ptype = p.get("type", "?")
                        similar = p.get("similar_memories", 0)
                        console.print(f"    patterns: {ptype} ({similar} similar memories)")

            # Summary
            summary = data.get("summary", "")
            if summary:
                console.print("")
                console.print(f'  [dim]summary:[/] "{summary[:100]}"')

            console.print("")  # Blank line after

    elif event_type == "enrichment.failed":
        error = data.get("error", "unknown")[:80]
        attempt = data.get("attempt", 1)
        will_retry = data.get("will_retry", False)
        retry_str = " (will retry)" if will_retry else ""
        console.print(
            f"[dim]{ts}[/] [red]ENRICH FAIL[/] {mem_id} attempt {attempt}: {error}{retry_str}"
        )


def print_consolidation_event(event: Event) -> None:
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


def print_error_event(event: Event) -> None:
    """Print error events."""
    data = event.get("data", {})
    ts = format_timestamp(event.get("timestamp", ""))

    error = data.get("error", "unknown")
    console.print(f"[dim]{ts}[/] [bold red]ERROR[/] {error}")


def print_associate_event(event: Event) -> None:
    """Print a memory.associate event."""
    data = event.get("data", {})
    ts = format_timestamp(event.get("timestamp", ""))

    mem1 = str(data.get("memory1_id", "?"))[:8]
    mem2 = str(data.get("memory2_id", "?"))[:8]
    rel_type = data.get("relation_type", "?")
    strength = data.get("strength", 0.5)

    console.print(
        f"[dim]{ts}[/] [bold blue]ASSOCIATE[/] "
        f"{mem1} [cyan]--{rel_type}-->[/] {mem2}  "
        f"[dim]strength={strength}[/]"
    )


def print_raw_event(event: Event) -> None:
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


def process_event(event: Event) -> None:
    """Route event to appropriate printer."""
    event_type = event.get("type", "")

    if event_type == "memory.store":
        print_store_event(event)
    elif event_type == "memory.recall":
        print_recall_event(event)
    elif event_type == "memory.associate":
        print_associate_event(event)
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
            # Explicit timeout: finite connect/write, infinite read for SSE
            timeout = httpx.Timeout(connect=10.0, read=None, write=10.0)
            with httpx.Client(timeout=timeout) as client:
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


def main() -> None:
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
