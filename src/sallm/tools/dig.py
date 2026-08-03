"""dig — multi-step treasure dig (file-backed progress across subprocesses)."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from sallm.tools.runner import intermediate

STATE_PATH = Path(tempfile.gettempdir()) / "sallm-dig-state.json"

DIG_SUMMARY = (
    "Toy treasure game only. Flags: --site NAME (short label like beach|cave). "
    "Use ONLY if the user asks to dig/play treasure. "
    "NEVER for documents, transcripts, search, memory, Q&A, or metaphors like 'dig into'."
)


def _load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state))


def reset_dig_state() -> None:
    """Clear dig progress (e.g. on /clear)."""
    try:
        STATE_PATH.unlink(missing_ok=True)
    except OSError:
        pass


def dig(site: str = "default") -> str:
    site = (site or "default").strip() or "default"
    state = _load_state()
    n = int(state.get(site, 0)) + 1
    state[site] = n
    if n == 1:
        _save_state(state)
        return intermediate(
            f"At '{site}' you found loose soil. Dig again at the same site."
        )
    if n == 2:
        _save_state(state)
        return intermediate(
            f"At '{site}' you uncovered a locked chest. Dig once more at the same site."
        )
    state[site] = 0
    _save_state(state)
    return f"At '{site}' you found gold coins worth 42."


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="dig",
        description=(
            "Toy treasure game. Dig at a short site name several times until finished. "
            "Early calls return [intermediate] — dig again at the same site. "
            "Do not invent treasure; only report what dig returns. "
            "Never use for documents, transcripts, search, or Q&A."
        ),
    )
    parser.add_argument(
        "--site",
        "-s",
        default="default",
        help="Short dig site label (default: default)",
    )
    args = parser.parse_args(argv)
    out = dig(args.site)
    sys.stdout.write(out + ("\n" if not out.endswith("\n") else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
