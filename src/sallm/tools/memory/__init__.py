"""memory CliTool — optional long-term text store/search."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from sallm.tools.runner import CliTool

from .backend import default_memory_path

MEMORY_SUMMARY = (
    "Store and search text. "
    "First: memory add --text <document or transcript>. "
    "Then: memory search --query <keywords> [-k N]. "
    "clear wipes the store. "
    "Use for documents/transcripts only — not games. "
    "Multi-word --query/--text need no quotes."
)

__all__ = [
    "MEMORY_SUMMARY",
    "memory_tool",
    "reset_memory_store",
]


def memory_tool(
    *,
    path: str | Path | None = None,
    session: str | None = None,
    backend: str = "file",
) -> CliTool:
    """Build a CliTool that invokes ``python -m sallm.tools.memory``."""
    argv = [sys.executable, "-m", "sallm.tools.memory"]
    root = str(path) if path is not None else str(default_memory_path())
    argv.extend(["--path", root])
    if session:
        argv.extend(["--session", session])
    if backend and backend != "file":
        argv.extend(["--backend", backend])
    return CliTool(name="memory", argv=argv, summary=MEMORY_SUMMARY)


def reset_memory_store(path: str | Path | None = None) -> None:
    """Remove the memory directory (e.g. on /clear)."""
    root = Path(path) if path is not None else default_memory_path()
    env = os.environ.get("SALLM_MEMORY_PATH")
    if env and path is None:
        root = Path(env)
    try:
        if root.is_dir():
            shutil.rmtree(root, ignore_errors=True)
        elif root.exists():
            root.unlink(missing_ok=True)
    except OSError:
        pass
