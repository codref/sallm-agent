"""CLI tools: runner + shipped deployable tool apps."""

from __future__ import annotations

import sys

from .runner import (
    DEFAULT_TIMEOUT,
    INTERMEDIATE_PREFIX,
    CliTool,
    ToolResult,
    format_observations,
    help_text,
    intermediate,
    is_intermediate,
    normalize_registry,
    parse_run_blocks,
    run_many,
    run_tool,
    split_run_line,
    tool_descriptions,
)
from .dig import DIG_SUMMARY, reset_dig_state

__all__ = [
    "DEFAULT_TIMEOUT",
    "INTERMEDIATE_PREFIX",
    "CliTool",
    "ToolResult",
    "BUILTIN_TOOL_NAMES",
    "DEFAULT_TOOLS",
    "builtin_tools",
    "format_observations",
    "help_text",
    "intermediate",
    "is_intermediate",
    "normalize_registry",
    "parse_run_blocks",
    "reset_dig_state",
    "run_many",
    "run_tool",
    "split_run_line",
    "tool_descriptions",
]

BUILTIN_TOOL_NAMES = ("echo", "calc", "dig", "memory")
DEFAULT_TOOLS = ("echo", "calc", "dig")


def _py_module(module: str) -> list[str]:
    return [sys.executable, "-m", module]


def _echo_tool() -> CliTool:
    return CliTool(
        name="echo",
        argv=_py_module("sallm.tools.echo"),
        summary="Echo text unchanged. Flags: --text TEXT (or positional).",
    )


def _calc_tool() -> CliTool:
    return CliTool(
        name="calc",
        argv=_py_module("sallm.tools.calc"),
        summary="Evaluate a math expression. Flags: --expression EXPR (-e).",
    )


def _dig_tool() -> CliTool:
    return CliTool(
        name="dig",
        argv=_py_module("sallm.tools.dig"),
        summary=DIG_SUMMARY,
    )


def builtin_tools(
    names=None,
    *,
    memory_path=None,
    memory_session=None,
    memory_backend: str = "file",
) -> dict[str, CliTool]:
    """Build a registry of shipped tools.

    ``names``: iterable of tool names, or ``None`` / ``\"all\"`` for every
    built-in, or ``\"none\"`` / empty for ``{}``. Default chat set is
    :data:`DEFAULT_TOOLS` (echo, calc, dig) — pass those explicitly or use
    ``None`` only when you want all including memory.
    """
    if names is None or names == "all":
        wanted = list(BUILTIN_TOOL_NAMES)
    elif names == "none" or names == ():
        return {}
    elif isinstance(names, str):
        wanted = [n.strip() for n in names.split(",") if n.strip()]
    else:
        wanted = [str(n).strip() for n in names if str(n).strip()]

    unknown = [n for n in wanted if n not in BUILTIN_TOOL_NAMES]
    if unknown:
        raise ValueError(
            f"unknown tool(s): {', '.join(unknown)}; "
            f"choose from {', '.join(BUILTIN_TOOL_NAMES)}"
        )

    out: dict[str, CliTool] = {}
    for name in wanted:
        if name == "echo":
            out[name] = _echo_tool()
        elif name == "calc":
            out[name] = _calc_tool()
        elif name == "dig":
            out[name] = _dig_tool()
        elif name == "memory":
            from .memory import memory_tool

            out[name] = memory_tool(
                path=memory_path,
                session=memory_session,
                backend=memory_backend,
            )
    return out
