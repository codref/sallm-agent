"""memory — store and search text (CliTool subprocess entry)."""

from __future__ import annotations

import argparse
import sys

from .backend import (
    chunk_text,
    default_memory_path,
    default_session_id,
    open_store,
    _chunk_id,
)


def _join_words(value) -> str:
    """Normalize argparse nargs='+' / str into one string."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(x) for x in value).strip()
    return str(value).strip()


def _cmd_add(store, session_id: str, text: str, chunk_tokens: int, overlap: int) -> str:
    text = text or ""
    if not text.strip():
        return "Error: empty text"
    n = 0
    for chunk in chunk_text(text, max_tokens=chunk_tokens, overlap_tokens=overlap):
        store.add(chunk, id=_chunk_id(session_id, chunk), session_id=session_id)
        n += 1
    return f"stored {n} chunk(s)"


def _cmd_search(store, session_id: str, query: str, k: int) -> str:
    hits = store.query(query or "", k=k, session_id=session_id)
    if not hits:
        return "(no matches)"
    return "\n---\n".join(hits)


def _cmd_clear(store, session_id: str | None) -> str:
    store.clear(session_id=session_id)
    if session_id is None:
        return "cleared all sessions"
    return f"cleared session {session_id}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="memory",
        description=(
            "Store and search text for later recall. "
            "Add documents/transcripts first, then search. "
            "Multi-word --text / --query values do not need quotes."
        ),
    )
    parser.add_argument(
        "--path",
        default=None,
        help="Store directory (default: $SALLM_MEMORY_PATH or temp)",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Session id (default: $SALLM_MEMORY_SESSION or 'default')",
    )
    parser.add_argument(
        "--backend",
        choices=("file", "lance"),
        default=None,
        help="file (default) or lance (needs uv sync --extra memory)",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=512,
        help="Max estimated tokens per chunk on add",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=64,
        help="Overlap tokens between chunks on add",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    add_p = sub.add_parser("add", help="Chunk and store text")
    add_p.add_argument(
        "--text",
        "-t",
        nargs="+",
        required=True,
        help="Text to store (one or more words; quotes optional)",
    )

    search_p = sub.add_parser("search", help="Search stored chunks")
    search_p.add_argument(
        "--query",
        "-q",
        nargs="+",
        required=True,
        help="Search query (one or more words; quotes optional)",
    )
    search_p.add_argument("-k", type=int, default=4, help="Top-k hits (default 4)")

    clear_p = sub.add_parser("clear", help="Clear stored chunks")
    clear_p.add_argument(
        "--all",
        action="store_true",
        help="Clear every session under --path (ignore --session)",
    )

    args = parser.parse_args(argv)
    path = args.path or str(default_memory_path())
    session = args.session or default_session_id()
    backend = args.backend or "file"

    try:
        store = open_store(path=path, backend=backend)
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1

    if args.command == "add":
        out = _cmd_add(
            store,
            session,
            _join_words(args.text),
            args.chunk_tokens,
            args.chunk_overlap,
        )
    elif args.command == "search":
        out = _cmd_search(store, session, _join_words(args.query), args.k)
    elif args.command == "clear":
        out = _cmd_clear(store, None if args.all else session)
    else:
        out = f"Error: unknown command {args.command}"

    err = out.startswith("Error:")
    sys.stdout.write(out + ("\n" if not out.endswith("\n") else ""))
    return 1 if err else 0


if __name__ == "__main__":
    raise SystemExit(main())
