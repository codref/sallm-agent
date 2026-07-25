"""CLI tool runner: tools are subprocesses, invoked via ```run blocks."""

from __future__ import annotations

import re
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

# Tools return this prefix when more rounds are required before a final answer.
INTERMEDIATE_PREFIX = "[intermediate]"

_RUN_BLOCK_RE = re.compile(r"```run\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

DEFAULT_TIMEOUT = 60


@dataclass
class CliTool:
    """A registered CLI tool: name maps to an argv prefix (executable + fixed args)."""

    name: str
    argv: list[str]
    summary: str = ""


@dataclass
class ToolResult:
    """Outcome of one subprocess tool run."""

    name: str
    # What the model asked for: [name, ...user args]
    command: list[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    error: str | None = None  # set when the command never ran (unknown tool, etc.)

    @property
    def observation(self) -> str:
        if self.error:
            return self.error
        out = (self.stdout or "").strip()
        err = (self.stderr or "").strip()
        if self.returncode != 0:
            parts = [f"Error: exit {self.returncode}"]
            if out:
                parts.append(out)
            if err:
                parts.append(err)
            return "\n".join(parts)
        return out if out else (err or "")

    @property
    def intermediate(self) -> bool:
        return is_intermediate(self.observation)


def is_intermediate(observation):
    """True if a tool result is an intermediate step (not a finished result)."""
    text = str(observation or "").lstrip()
    return text.lower().startswith(INTERMEDIATE_PREFIX)


def intermediate(message):
    """Build an intermediate tool observation string."""
    message = str(message or "").strip()
    if is_intermediate(message):
        return message
    return f"{INTERMEDIATE_PREFIX} {message}".rstrip()


def normalize_registry(tools) -> dict[str, CliTool]:
    """Accept a dict or list of CliTool and return {name: CliTool}."""
    if not tools:
        return {}
    if isinstance(tools, dict):
        out = {}
        for key, value in tools.items():
            if isinstance(value, CliTool):
                out[value.name] = value
            else:
                raise TypeError(f"tools[{key!r}] must be a CliTool, got {type(value)}")
        return out
    out = {}
    for item in tools:
        if not isinstance(item, CliTool):
            raise TypeError(f"expected CliTool, got {type(item)}")
        out[item.name] = item
    return out


def tool_descriptions(tools: dict[str, CliTool]) -> str:
    """One-line summaries for the system prompt."""
    if not tools:
        return "(none)"
    lines = []
    for name, tool in tools.items():
        summary = (tool.summary or "").strip() or "no description"
        lines.append(f"- {name}: {summary}")
    return "\n".join(lines)


def parse_run_blocks(text) -> list[list[str]]:
    """Extract argv lists from ```run fenced blocks (one command per line)."""
    text = text or ""
    commands: list[list[str]] = []
    for match in _RUN_BLOCK_RE.finditer(text):
        body = match.group(1)
        for line in body.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                argv = shlex.split(line)
            except ValueError:
                argv = [line]
            if argv:
                commands.append(argv)
    return commands


def run_tool(
    tool: CliTool,
    extra_args=None,
    timeout=DEFAULT_TIMEOUT,
    command=None,
) -> ToolResult:
    """Run one CliTool as a subprocess with optional extra argv."""
    extra_args = list(extra_args or [])
    full_argv = list(tool.argv) + extra_args
    display = list(command) if command is not None else [tool.name, *extra_args]
    try:
        proc = subprocess.run(
            full_argv,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return ToolResult(
            name=tool.name,
            command=display,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            returncode=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            name=tool.name,
            command=display,
            error=f"Error: tool '{tool.name}' timed out after {timeout}s",
            returncode=-1,
        )
    except OSError as exc:
        return ToolResult(
            name=tool.name,
            command=display,
            error=f"Error running tool '{tool.name}': {exc}",
            returncode=-1,
        )


def help_text(tool: CliTool, timeout=DEFAULT_TIMEOUT) -> str:
    """Fetch --help stdout for a tool."""
    result = run_tool(tool, ["--help"], timeout=timeout)
    return result.observation


def _run_one_command(
    registry: dict[str, CliTool], argv: list[str], timeout
) -> ToolResult:
    if not argv:
        return ToolResult(name="", error="Error: empty command", returncode=-1)
    name = argv[0]
    tool = registry.get(name)
    if tool is None:
        available = ", ".join(registry) or "(none)"
        return ToolResult(
            name=name,
            command=argv,
            error=f"Error: unknown tool '{name}'. Available: {available}",
            returncode=-1,
        )
    return run_tool(tool, argv[1:], timeout=timeout, command=argv)


def run_many(
    registry: dict[str, CliTool],
    commands: list[list[str]],
    timeout=DEFAULT_TIMEOUT,
) -> list[ToolResult]:
    """Run parsed commands; concurrent when there is more than one."""
    if not commands:
        return []
    if len(commands) == 1:
        return [_run_one_command(registry, commands[0], timeout)]

    results: list[ToolResult | None] = [None] * len(commands)
    with ThreadPoolExecutor(max_workers=len(commands)) as pool:
        futures = {
            pool.submit(_run_one_command, registry, cmd, timeout): i
            for i, cmd in enumerate(commands)
        }
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    return list(results)  # type: ignore[arg-type]


def format_observations(results: list[ToolResult]) -> str:
    """Format subprocess results for the conversation (fed back to the model)."""
    blocks = []
    for r in results:
        display = shlex.join(r.command) if r.command else (r.name or "?")
        blocks.append(f"$ {display}\n{r.observation}")
    return "\n\n".join(blocks)
