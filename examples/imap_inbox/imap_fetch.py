"""imap_fetch — fetch one message by UID; structured headers + truncated body.

Example model invocation:

```run
imap_fetch --folder INBOX --uid 12345 --max-chars 1500
```

Keep --max-chars modest. Huge bodies fall out of the recent-history window and
hurt later recall; durable facts should be short (sender, subject, order id).
"""

from __future__ import annotations

import argparse
import sys

from imap_common import (
    DEFAULT_BODY_CHARS,
    connect,
    fetch_full,
    format_fetch,
    select_folder,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="imap_fetch",
        description=(
            "Fetch a single IMAP message by UID. "
            "Prints From/To/Subject/Date/Message-ID and a truncated plaintext body. "
            "Use after imap_search when the user needs details of one message. "
            "Do not invent body text — only report what this tool returns."
        ),
    )
    parser.add_argument(
        "--folder",
        "-f",
        default=None,
        help="Mailbox name (default: IMAP_FOLDER from .env)",
    )
    parser.add_argument(
        "--uid",
        "-u",
        required=True,
        help="IMAP UID from imap_search output",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_BODY_CHARS,
        help=f"Body snippet cap (default: {DEFAULT_BODY_CHARS})",
    )
    args = parser.parse_args(argv)

    uid = str(args.uid).strip()
    if not uid.isdigit():
        sys.stderr.write("--uid must be a numeric IMAP UID\n")
        return 2
    max_chars = max(200, min(int(args.max_chars), 8000))

    client, cfg = connect()
    folder = (args.folder or cfg["folder"]).strip() or "INBOX"
    try:
        select_folder(client, folder)
        msg = fetch_full(client, uid)
        if msg is None:
            sys.stdout.write(f"(message UID {uid} not found in {folder!r})\n")
            return 0
        out = format_fetch(uid, msg, max_chars=max_chars)
        sys.stdout.write(out)
        if not out.endswith("\n"):
            sys.stdout.write("\n")
    finally:
        try:
            client.logout()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
