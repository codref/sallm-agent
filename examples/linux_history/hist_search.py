"""hist_search — search bash/zsh history; compact rows with optional time + context.

Default matching is **token** mode (whole shell tokens), so ``git`` does not
match a path like ``.../git-codref/...``.

Keyword search:

```run
hist_search --query git --limit 10
```

Semantic search (Ollama qwen3-embedding — natural language / related commands):

```run
hist_search --query "docker authentication remote host" --mode semantic --limit 10
```

Usage story with neighbors:

```run
hist_search --query docker --days 14 --context 8 --limit 20
```

Rows: `idx | time | command` (newest matches first). Context neighbors are
prefixed with `|`. Never invent hosts/IPs — only report what appears here.
"""

from __future__ import annotations

import argparse
import os
import sys

from hist_common import (
    days_to_since,
    format_row,
    parse_time_bound,
    search_history,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="hist_search",
        description=(
            "Search the user's bash/zsh history file. "
            "Default --mode token: match whole shell tokens "
            "(avoids path false positives like git inside git-codref). "
            "--mode semantic: embed query + recent history via Ollama "
            "(qwen3-embedding) and rank by similarity. "
            "--mode substr: legacy substring match. "
            "For usage stories: --query SEED --days N --context K. "
            "Do not invent hostnames or IPs — only report tool output."
        ),
    )
    parser.add_argument(
        "--query",
        "-q",
        default="",
        help="Search text (token/substr) or natural-language cue (semantic)",
    )
    parser.add_argument(
        "--mode",
        choices=("token", "substr", "semantic"),
        default="token",
        help="Match mode (default: token)",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=20,
        help="Max matching commands (default: 20; context rows are extra)",
    )
    parser.add_argument(
        "--file",
        "-f",
        default=None,
        help="History file path (default: HISTFILE / ~/.zsh_history / ~/.bash_history)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Only entries from the last N days (needs timestamps; zsh extended)",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="Lower bound: Unix epoch or ISO date YYYY-MM-DD",
    )
    parser.add_argument(
        "--until",
        default=None,
        help="Upper bound: Unix epoch or ISO date YYYY-MM-DD",
    )
    parser.add_argument(
        "--context",
        "-C",
        type=int,
        default=0,
        help="Include K history lines before/after each match (for story inference)",
    )
    parser.add_argument(
        "--include-undated",
        action="store_true",
        help="When a time filter is set, also keep lines without timestamps",
    )
    parser.add_argument(
        "--semantic-pool",
        type=int,
        default=3000,
        help="For --mode semantic: newest N commands to embed (default: 3000)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.25,
        help="For --mode semantic: minimum cosine similarity (default: 0.25)",
    )
    args = parser.parse_args(argv)

    since = parse_time_bound(args.since)
    until = parse_time_bound(args.until)
    if args.days is not None:
        days_since = days_to_since(int(args.days))
        since = days_since if since is None else max(since, days_since)

    embed_model = (
        (os.environ.get("SALLM_EMBED_MODEL") or "").strip()
        or "ollama/qwen3-embedding:0.6b"
    )
    api_base = (
        (os.environ.get("SALLM_API_BASE") or "").strip()
        or "http://localhost:11434"
    )

    try:
        path, rows = search_history(
            histfile=args.file,
            query=args.query or "",
            limit=int(args.limit),
            since=since,
            until=until,
            include_undated=bool(args.include_undated),
            context=max(0, int(args.context or 0)),
            mode=args.mode,
            semantic_pool=int(args.semantic_pool),
            min_score=float(args.min_score),
            embed_model=embed_model,
            api_base=api_base,
        )
    except Exception as exc:
        sys.stderr.write(f"hist_search failed: {exc}\n")
        return 1

    match_count = sum(1 for e, is_m, _s in rows if is_m and e.idx > 0)
    sys.stdout.write(
        f"file={path} query={args.query!r} mode={args.mode} matches={match_count} "
        f"since={since or '-'} until={until or '-'} context={args.context}\n"
    )
    if not rows:
        sys.stdout.write("(no matching history entries)\n")
        return 0

    for entry, is_match, score in rows:
        if entry.idx < 0:
            sys.stdout.write("\n")
            continue
        sys.stdout.write(
            format_row(
                entry,
                match=is_match,
                score=score if args.mode == "semantic" else None,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
