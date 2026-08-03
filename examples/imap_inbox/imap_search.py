"""imap_search — search a mailbox; print compact UID + header lines.

Prefer dedicated flags (avoids broken IMAP quoting from the model):

```run
imap_search --folder INBOX --from noreply@tricol.it --limit 5
```

Most recent N messages:

```run
imap_search --folder INBOX --query ALL --limit 8
```

Raw criteria still work (double quotes only — never single quotes):

```run
imap_search --folder INBOX --query 'FROM "billing" SINCE 1-Jan-2026' --limit 5
```

Each result line is: UID | date | from | subject
Never dump full bodies here — use imap_fetch for one message.
"""

from __future__ import annotations

import argparse
import sys

from imap_common import (
    build_search_criteria,
    connect,
    fetch_headers,
    format_search_row,
    select_folder,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="imap_search",
        description=(
            "Search an IMAP folder and print compact header lines (newest first). "
            "Prefer --from ADDRESS or --subject TEXT for sender/subject searches. "
            "For the N most recent messages: --query ALL --limit N. "
            "Presets ALL/LATEST/NEWEST/RECENT/NEW all mean ALL. "
            "Unread only: --query UNSEEN. "
            "Raw --query must use IMAP double quotes, never single quotes. "
            "Do not invent message contents — only report what this tool prints."
        ),
    )
    parser.add_argument(
        "--folder",
        "-f",
        default=None,
        help="Mailbox name (default: IMAP_FOLDER from .env, usually INBOX)",
    )
    parser.add_argument(
        "--from",
        dest="from_addr",
        default=None,
        help="Match From header (preferred over raw FROM in --query)",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help="Match Subject header substring",
    )
    parser.add_argument(
        "--query",
        "-q",
        default="ALL",
        help=(
            "ALL/LATEST/NEWEST/RECENT/NEW, UNSEEN, SEEN, "
            "or raw IMAP criteria (default: ALL)"
        ),
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=10,
        help="Max messages to return (newest UIDs first; default: 10)",
    )
    args = parser.parse_args(argv)

    limit = max(1, min(int(args.limit), 50))
    raw_query = (args.query or "ALL").strip()
    criteria = build_search_criteria(
        query=raw_query,
        from_addr=args.from_addr,
        subject=args.subject,
    )

    client, cfg = connect()
    folder = (args.folder or cfg["folder"]).strip() or "INBOX"
    try:
        select_folder(client, folder)
        # Parenthesize so multi-token criteria stay one SEARCH expression.
        search_arg = criteria if criteria.startswith("(") else f"({criteria})"
        status, data = client.uid("search", None, search_arg)
        if status != "OK":
            sys.stderr.write(f"SEARCH failed: status={status} criteria={criteria!r}\n")
            return 1
        raw = data[0] if data else b""
        if isinstance(raw, bytes):
            uids = [u for u in raw.decode("ascii", errors="ignore").split() if u]
        else:
            uids = []
        # Newest last in many servers' UID order — take the tail, then show newest first.
        uids = uids[-limit:]
        uids = list(reversed(uids))

        if not uids:
            sys.stdout.write(f"(no messages in {folder!r} matching {criteria!r})\n")
            return 0

        note = ""
        if (
            not args.from_addr
            and not args.subject
            and raw_query.upper() in ("RECENT", "NEW", "LATEST", "NEWEST")
            and criteria == "ALL"
        ):
            note = f" (preset {raw_query!r} → ALL; newest {limit} by UID)"
        sys.stdout.write(
            f"folder={folder} query={criteria!r} count={len(uids)}{note}\n"
        )
        by_uid = {uid: msg for uid, msg in fetch_headers(client, uids)}
        for uid in uids:
            msg = by_uid.get(uid)
            if msg is None:
                sys.stdout.write(f"{uid} | ? | ? | (headers unavailable)\n")
            else:
                sys.stdout.write(format_search_row(uid, msg) + "\n")
    finally:
        try:
            client.logout()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
