"""echo — print text unchanged."""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="echo",
        description="Echo text back unchanged. Use only when the user asks to echo/repeat text.",
    )
    parser.add_argument(
        "text",
        nargs="?",
        default="",
        help="Text to echo (or use --text)",
    )
    parser.add_argument(
        "--text",
        dest="text_flag",
        default=None,
        help="Text to echo (overrides positional)",
    )
    args = parser.parse_args(argv)
    value = args.text_flag if args.text_flag is not None else args.text
    sys.stdout.write(str(value))
    if value and not str(value).endswith("\n"):
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
