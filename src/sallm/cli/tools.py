"""Chat-app tool registry — CLI tools registered by the REPL, not the library core."""

from __future__ import annotations

import sys

from sallm.tools import CliTool
from sallm.toolapps.dig import reset_dig_state

__all__ = ["CHAT_TOOLS", "reset_dig_state"]


def _py_module(module: str) -> list[str]:
    return [sys.executable, "-m", module]


CHAT_TOOLS = {
    "echo": CliTool(
        name="echo",
        argv=_py_module("sallm.toolapps.echo"),
        summary="Echo text unchanged. Flags: --text TEXT (or positional).",
    ),
    "calc": CliTool(
        name="calc",
        argv=_py_module("sallm.toolapps.calc"),
        summary="Evaluate a math expression. Flags: --expression EXPR (-e).",
    ),
    "dig": CliTool(
        name="dig",
        argv=_py_module("sallm.toolapps.dig"),
        summary=(
            "Dig for treasure at a site (multi-step). Flags: --site NAME. "
            "Repeat until result is not [intermediate]."
        ),
    ),
}
