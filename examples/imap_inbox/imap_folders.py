"""imap_folders — list mailboxes on the configured IMAP server.

SALLM invokes this as a CLI tool via a ```run block, for example:

```run
imap_folders
```

Stdout is a plain list of folder names (one per line). Keep it short so the
observation fits the agent's token budget.
"""

from __future__ import annotations

import argparse
import sys

from imap_common import connect, list_mailbox_names


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="imap_folders",
        description=(
            "List IMAP mailbox / folder names. "
            "Use when the user asks which folders exist or where mail lives. "
            "No flags required — credentials come from .env."
        ),
    )
    parser.parse_args(argv)

    client, _cfg = connect()
    try:
        names = list_mailbox_names(client)
    finally:
        try:
            client.logout()
        except Exception:
            pass

    if not names:
        sys.stdout.write("(no folders returned)\n")
        return 0

    for name in names:
        sys.stdout.write(name + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
