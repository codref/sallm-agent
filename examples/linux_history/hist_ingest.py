"""hist_ingest — search history, queue blocks for Agent.remember (meaning-first).

This CLI does **not** talk to SQLite/Lance. It writes pending blocks to
``.sallm/pending_remember.jsonl``; the example agent drains that file and calls
``agent.remember(...)`` so an LLM can turn command lines into English facts.

```run
hist_ingest --query docker --limit 40 --block-lines 30
hist_ingest --latest 300
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hist_common import (
    HERE,
    days_to_since,
    format_row,
    parse_histfile,
    parse_time_bound,
    resolve_histfile,
    search_history,
)

PENDING = HERE / ".sallm" / "pending_remember.jsonl"

# Soft caps: search stays modest; --latest can queue a heavier remember phase.
_MAX_SEARCH_LIMIT = 200
_MAX_LATEST = 2000


def _blocks_from_entries(
    entries_with_match: list[tuple],
    block_lines: int,
) -> list[tuple[str, str]]:
    """Split (entry, is_match) pairs into text blocks with source labels."""
    entries = [(e, is_m) for e, is_m in entries_with_match if e.idx > 0]
    if not entries:
        return []
    block_lines = max(5, min(int(block_lines), 120))
    out: list[tuple[str, str]] = []
    for i in range(0, len(entries), block_lines):
        chunk = entries[i : i + block_lines]
        lo = chunk[0][0].idx
        hi = chunk[-1][0].idx
        lines = [format_row(e, match=is_m) for e, is_m in chunk]
        text = "Shell history block:\n" + "\n".join(lines)
        out.append((f"zsh_history:{lo}-{hi}", text))
    return out


def _blocks_from_rows(rows, block_lines: int) -> list[tuple[str, str]]:
    """Split search rows into text blocks with source labels."""
    return _blocks_from_entries(
        [(e, is_m) for e, is_m, _s in rows],
        block_lines,
    )


def queue_latest_blocks(
    *,
    n: int,
    histfile: str | None = None,
    block_lines: int = 30,
) -> tuple[Path, list[tuple[str, str]]]:
    """Take the newest ``n`` history commands (oldest→newest within that window)."""
    path = resolve_histfile(histfile)
    all_entries = parse_histfile(path)
    n = max(1, min(int(n), _MAX_LATEST))
    # Newest last in file; take the tail, keep chronological order for blocks.
    selected = all_entries[-n:] if len(all_entries) > n else list(all_entries)
    blocks = _blocks_from_entries([(e, True) for e in selected], block_lines)
    return path, blocks


def write_pending(blocks: list[tuple[str, str]]) -> Path:
    PENDING.parent.mkdir(parents=True, exist_ok=True)
    with PENDING.open("a", encoding="utf-8") as fh:
        for source, text in blocks:
            fh.write(json.dumps({"source": source, "text": text}) + "\n")
    return PENDING


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="hist_ingest",
        description=(
            "Search shell history and queue blocks for meaning-first Agent.remember. "
            "Use when the user wants history indexed into durable English memory "
            "(hosts, IPs, usage patterns) — not for live bash execution."
        ),
    )
    parser.add_argument("--query", "-q", default="", help="Search query")
    parser.add_argument(
        "--latest",
        type=int,
        default=None,
        metavar="N",
        help=(
            f"Queue the newest N history commands (no search; max {_MAX_LATEST}). "
            "Useful to simulate a large remember phase."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("token", "substr", "semantic"),
        default="token",
        help="hist_search match mode (default: token)",
    )
    parser.add_argument("--limit", "-n", type=int, default=40)
    parser.add_argument("--file", "-f", default=None)
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--since", default=None)
    parser.add_argument("--until", default=None)
    parser.add_argument("--context", "-C", type=int, default=4)
    parser.add_argument(
        "--block-lines",
        type=int,
        default=30,
        help="History lines per remember() block (default: 30)",
    )
    parser.add_argument(
        "--include-undated",
        action="store_true",
    )
    args = parser.parse_args(argv)

    if args.latest is not None:
        path, blocks = queue_latest_blocks(
            n=int(args.latest),
            histfile=args.file,
            block_lines=args.block_lines,
        )
        note = f"latest={int(args.latest)}"
    else:
        since = parse_time_bound(args.since)
        until = parse_time_bound(args.until)
        if args.days is not None:
            days_since = days_to_since(int(args.days))
            since = days_since if since is None else max(since, days_since)

        path, rows = search_history(
            histfile=args.file,
            query=args.query or "",
            limit=min(int(args.limit), _MAX_SEARCH_LIMIT),
            since=since,
            until=until,
            include_undated=bool(args.include_undated),
            context=max(0, int(args.context or 0)),
            mode=args.mode,
        )
        blocks = _blocks_from_rows(rows, args.block_lines)
        note = f"query={args.query!r} mode={args.mode}"

    if not blocks:
        sys.stdout.write("(no history rows to ingest)\n")
        return 0

    write_pending(blocks)
    cmds = sum(text.count("\n") for _, text in blocks)  # rough line count
    sys.stdout.write(
        f"queued={len(blocks)} file={path} {note} pending={PENDING}\n"
    )
    for source, text in blocks:
        preview = text.replace("\n", " ")[:100]
        sys.stdout.write(f"  - {source}: {preview}…\n")
    sys.stdout.write(
        f"Agent will run remember() on these blocks "
        f"(~{cmds} history lines → LLM English facts).\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
