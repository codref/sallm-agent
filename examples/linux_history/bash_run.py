"""bash_run — execute a bash command as the current user.

WARNING: This runs real commands with your privileges. Prefer read-only
hist_search for past activity. Use bash_run only when the user wants live
execution.

```run
bash_run --command "pwd && date"
```

Stdout/stderr are truncated so one noisy command cannot blow the prompt budget.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Allow importing load_env from the same directory when run as a script.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from hist_common import load_env  # noqa: E402

DEFAULT_TIMEOUT = 30
DEFAULT_MAX_CHARS = 4000


def _clip(text: str, max_chars: int) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + "\n… [truncated]\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="bash_run",
        description=(
            "Run a bash command (-c) as the current user and print exit code plus "
            "truncated stdout/stderr. "
            "DANGER: full shell privileges. "
            "Do not use for reading shell history — use hist_search instead. "
            "Only report what the command actually printed."
        ),
    )
    parser.add_argument(
        "--command",
        "-c",
        required=True,
        help="Bash command string to execute",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help=f"Seconds before kill (default: {DEFAULT_TIMEOUT} or BASH_RUN_TIMEOUT)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help=f"Stdout+stderr soft cap (default: {DEFAULT_MAX_CHARS})",
    )
    args = parser.parse_args(argv)

    load_env()
    timeout = args.timeout
    if timeout is None:
        env_t = (os.environ.get("BASH_RUN_TIMEOUT") or "").strip()
        timeout = int(env_t) if env_t else DEFAULT_TIMEOUT
    timeout = max(1, min(int(timeout), 600))

    max_chars = args.max_chars
    if max_chars is None:
        env_m = (os.environ.get("BASH_RUN_MAX_CHARS") or "").strip()
        max_chars = int(env_m) if env_m else DEFAULT_MAX_CHARS
    max_chars = max(200, min(int(max_chars), 100_000))

    cmd = args.command
    try:
        proc = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        sys.stdout.write(f"exit=timeout after {timeout}s\n")
        sys.stdout.write(f"command={cmd!r}\n")
        return 0
    except OSError as exc:
        sys.stderr.write(f"Failed to run bash: {exc}\n")
        return 1

    out = proc.stdout or ""
    err = proc.stderr or ""
    # Split the char budget roughly between streams.
    half = max_chars // 2
    sys.stdout.write(f"exit={proc.returncode}\n")
    sys.stdout.write(f"command={cmd!r}\n")
    if out:
        sys.stdout.write("--- stdout ---\n")
        sys.stdout.write(_clip(out, half if err else max_chars))
        if not out.endswith("\n"):
            sys.stdout.write("\n")
    if err:
        sys.stdout.write("--- stderr ---\n")
        sys.stdout.write(_clip(err, half if out else max_chars))
        if not err.endswith("\n"):
            sys.stdout.write("\n")
    if not out and not err:
        sys.stdout.write("(no output)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
