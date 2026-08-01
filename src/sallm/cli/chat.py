"""Interactive chat REPL — agent + optional CLI tools."""

from __future__ import annotations

import secrets
import tempfile
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from sallm import Agent
from sallm.context import MaxMessages, SummarizeOverflow
from sallm.llm import complete
from sallm.messages import DEFAULT_API_BASE, DEFAULT_MODEL
from sallm.prom import SessionMetrics
from sallm.tools import DEFAULT_TOOLS, builtin_tools, reset_dig_state
from sallm.tools.memory import reset_memory_store
from sallm.trace import DEFAULT_TRUNCATE, Tracer, jsonl_sink, multi_sink, otlp_http_sink

app = typer.Typer(
    name="sallm",
    help="Minimal CLI-tool agent over LiteLLM / Ollama.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()

CONTEXT_NONE = "none"
CONTEXT_MAX_MESSAGES = "max-messages"
CONTEXT_SUMMARIZE = "summarize"
CONTEXT_CHOICES = (CONTEXT_NONE, CONTEXT_MAX_MESSAGES, CONTEXT_SUMMARIZE)

_PROMPT_PREVIEW_CHARS = 400


@app.callback()
def _root():
    """Minimal CLI-tool agent over LiteLLM / Ollama."""
    pass


def _print_help():
    console.print(
        Panel(
            "[bold]/help[/]  this help\n"
            "[bold]/clear[/]  reset conversation (+ dig/memory tool state)\n"
            "[bold]/history[/]  show message roles + lengths\n"
            "[bold]/prompt[/]  show system prompt (+ last LLM view if any)\n"
            "[bold]/prompt system[/]  system / templates only\n"
            "[bold]/prompt last[/]  last messages sent to the model\n"
            "[bold]/quit[/]  exit",
            title="commands",
            border_style="dim",
        )
    )


def _print_history(agent):
    table = Table(title="history", show_header=True, header_style="bold")
    table.add_column("#", style="dim", width=4)
    table.add_column("role", width=10)
    table.add_column("chars", justify="right", width=8)
    table.add_column("preview")
    for i, msg in enumerate(agent.messages):
        content = msg.get("content") or ""
        preview = content.replace("\n", " ")
        if len(preview) > 60:
            preview = preview[:57] + "..."
        table.add_row(str(i), msg.get("role", "?"), str(len(content)), preview)
    console.print(table)


def _print_last_prompt(agent, *, truncate=_PROMPT_PREVIEW_CHARS):
    """Show the exact message list last sent to complete()."""
    msgs = agent.last_prompt
    if not msgs:
        console.print("[yellow]no last LLM prompt yet — ask something first[/]")
        return
    table = Table(title="last LLM prompt", show_header=True, header_style="bold")
    table.add_column("#", style="dim", width=4)
    table.add_column("role", width=10)
    table.add_column("chars", justify="right", width=8)
    table.add_column("content")
    for i, msg in enumerate(msgs):
        content = msg.get("content") or ""
        shown = content
        if truncate > 0 and len(shown) > truncate:
            shown = shown[: truncate - 3] + "..."
        table.add_row(
            str(i),
            msg.get("role", "?"),
            str(len(content)),
            escape(shown),
        )
    console.print(table)


def _print_prompt_cmd(agent, arg: str | None = None):
    """Handle /prompt [system|last]."""
    kind = (arg or "").strip().lower()
    if kind in ("", "all"):
        console.print(
            Panel(
                escape(agent.prompt.preview()),
                title="prompt system",
                border_style="cyan",
            )
        )
        if agent.last_prompt:
            _print_last_prompt(agent)
        return
    if kind == "system":
        console.print(
            Panel(
                escape(agent.prompt.preview()),
                title="prompt system",
                border_style="cyan",
            )
        )
        return
    if kind == "last":
        _print_last_prompt(agent)
        return
    console.print(
        f"[red]unknown /prompt arg:[/] {kind!r}  (try system|last)"
    )


def _print_steps(steps):
    for i, step in enumerate(steps, 1):
        if step.get("kind") == "action":
            calls = step.get("tool_calls") or []
            for tc in calls:
                tag = " [intermediate]" if tc.get("intermediate") else ""
                obs = escape(str(tc.get("observation") or ""))
                args = escape(str(tc.get("action_input") or ""))
                name = escape(str(tc.get("action") or ""))
                body = (
                    f"[bold]cmd[/]    {name} {args}{escape(tag)}\n"
                    f"[bold]result[/] {obs}"
                )
                console.print(
                    Panel(
                        body,
                        title="tool",
                        border_style="yellow" if tc.get("intermediate") else "magenta",
                    )
                )
            continue

        if step.get("kind") == "nudge":
            console.print(
                Panel(
                    escape(step.get("raw") or "continue"),
                    title="nudge",
                    border_style="dim",
                )
            )
            continue

        if step.get("kind") == "rejected":
            console.print(
                Panel(
                    escape(step.get("raw") or ""),
                    title="rejected early answer",
                    border_style="red",
                )
            )
            if step.get("nudge"):
                console.print(
                    Panel(
                        escape(step["nudge"]),
                        title="nudge",
                        border_style="dim",
                    )
                )
            continue

        if step.get("reasoning"):
            console.print(
                Panel(
                    str(step["reasoning"]),
                    title=f"reasoning ({i})",
                    border_style="dim",
                )
            )


def _print_metrics(metrics):
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    table.add_row("in tokens", str(metrics.get("prompt_tokens", 0)))
    table.add_row("out tokens", str(metrics.get("completion_tokens", 0)))
    table.add_row("total tokens", str(metrics.get("total_tokens", 0)))
    table.add_row("elapsed", f"{metrics.get('elapsed_ms', 0):.1f} ms")
    table.add_row("context msgs", str(metrics.get("context_messages", 0)))
    if "prompt_messages" in metrics:
        table.add_row("prompt msgs", str(metrics.get("prompt_messages", 0)))
    console.print(Panel(table, title="metrics", border_style="cyan"))


def _build_trace(trace_path, otlp_url, debug=False, truncate=512, metrics_port=0):
    sinks = []
    if trace_path:
        sinks.append(jsonl_sink(trace_path))
    if otlp_url:
        sinks.append(otlp_http_sink(otlp_url))
    if not sinks and not metrics_port:
        return None
    emit = (lambda _event: None)
    if sinks:
        emit = sinks[0] if len(sinks) == 1 else multi_sink(*sinks)
    tracer = Tracer(emit, debug=debug, truncate=truncate)
    if metrics_port:
        metrics = SessionMetrics(tracer.session_id)
        metrics.start_server(port=int(metrics_port))
        tracer.metrics = metrics
    return tracer


def _make_summarize_fn(model, api_base):
    def summarize(text: str) -> str:
        result = complete(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Summarize the conversation briefly. "
                        "Keep all key facts, names, numbers, and codes.\n\n" + text
                    ),
                }
            ],
            api_base=api_base,
        )
        return result.get("content") or ""

    return summarize


def _session_id(tracer) -> str:
    if tracer is not None:
        return tracer.session_id
    return secrets.token_hex(8)


def _build_context(
    kind,
    *,
    model,
    api_base,
    max_context_messages,
    context_threshold,
    context_keep_last,
):
    """Build a context optimizer for the chat CLI, or None."""
    if kind is None or kind == CONTEXT_NONE:
        return None, ""
    if kind == CONTEXT_MAX_MESSAGES:
        n = 40 if max_context_messages is None else max_context_messages
        return MaxMessages(n), f"MaxMessages({n})"
    if kind == CONTEXT_SUMMARIZE:
        threshold = 2000 if context_threshold is None else context_threshold
        keep_last = 10 if context_keep_last is None else context_keep_last
        opt = SummarizeOverflow(
            threshold=threshold,
            keep_last=keep_last,
            summarize_fn=_make_summarize_fn(model, api_base),
        )
        return opt, f"SummarizeOverflow(threshold={threshold}, keep_last={keep_last})"
    raise typer.BadParameter(
        f"unknown context optimizer {kind!r}; choose from {', '.join(CONTEXT_CHOICES)}"
    )


def _parse_tools_flag(value: str | None) -> str | tuple | None:
    """Return names for builtin_tools: 'all', 'none', comma-list, or DEFAULT_TOOLS."""
    if value is None:
        return DEFAULT_TOOLS
    raw = value.strip().lower()
    if raw in ("", "default"):
        return DEFAULT_TOOLS
    if raw in ("none", "off"):
        return "none"
    if raw == "all":
        return "all"
    return raw


def iter_prompts(path):
    """Yield non-empty stripped lines from a script file (one prompt per line)."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def iter_repl():
    """Yield user lines from the interactive prompt until EOF / Ctrl-C."""
    while True:
        try:
            yield console.input("[bold green]you>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            return


@app.command()
def chat(
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="LiteLLM model id"),
    api_base: str = typer.Option(
        DEFAULT_API_BASE, "--api-base", help="Provider base URL"
    ),
    system: str = typer.Option(None, "--system", "-s", help="Extra system prompt"),
    max_steps: int = typer.Option(
        5, "--max-steps", help="Max tool rounds per turn"
    ),
    multi_step: bool = typer.Option(
        True,
        "--multi-step/--no-multi-step",
        help="Allow chained tool rounds within one turn",
    ),
    tools: str = typer.Option(
        "echo,calc,dig",
        "--tools",
        help=(
            "Comma-separated tools, or all|none|default "
            "(echo,calc,dig,memory — memory uses file store by default)"
        ),
    ),
    memory_path: str = typer.Option(
        None,
        "--memory-path",
        help="Directory for the memory tool store (default: temp per session)",
    ),
    memory_backend: str = typer.Option(
        "file",
        "--memory-backend",
        help="memory tool backend: file | lance (lance needs --extra memory)",
    ),
    script: str = typer.Option(
        None,
        "--script",
        help="Run prompts from a file (one non-empty line = one turn), then exit",
    ),
    show_prompt: bool = typer.Option(
        False,
        "--show-prompt/--no-show-prompt",
        help="After each turn, print the last LLM prompt view",
    ),
    trace: str = typer.Option(
        None,
        "--trace",
        help="Append OpenTelemetry-shaped JSONL events to this file",
    ),
    otlp: str = typer.Option(
        None,
        "--otlp",
        help="POST spans to OTLP/HTTP endpoint (e.g. http://localhost:4318)",
    ),
    trace_debug: bool = typer.Option(
        False,
        "--trace-debug/--no-trace-debug",
        help="Store prompt/context/completion text on spans (truncated)",
    ),
    trace_truncate: int = typer.Option(
        DEFAULT_TRUNCATE,
        "--trace-truncate",
        help="Max chars per content field (each message/completion); 0 = unlimited",
    ),
    metrics_port: int = typer.Option(
        0,
        "--metrics-port",
        help="Expose Prometheus /metrics on this port (0 = off); scrape from Docker via host.docker.internal",
    ),
    context: str = typer.Option(
        CONTEXT_NONE,
        "--context",
        help="Context optimizer: none | max-messages | summarize",
    ),
    max_context_messages: int | None = typer.Option(
        None,
        "--max-context-messages",
        help="For max-messages: keep system + last N messages (default 40)",
    ),
    context_threshold: int | None = typer.Option(
        None,
        "--context-threshold",
        help="For summarize: overflow token budget before summarizing (default 2000)",
    ),
    context_keep_last: int | None = typer.Option(
        None,
        "--context-keep-last",
        help="For summarize: recent messages kept verbatim (default 10)",
    ),
):
    """Interactive chat REPL for testing the agent."""
    kind = (context or CONTEXT_NONE).strip().lower()
    if kind not in CONTEXT_CHOICES:
        raise typer.BadParameter(
            f"unknown --context {context!r}; choose from {', '.join(CONTEXT_CHOICES)}"
        )
    if kind == CONTEXT_NONE and max_context_messages is not None:
        kind = CONTEXT_MAX_MESSAGES

    backend = (memory_backend or "file").strip().lower()
    if backend not in ("file", "lance"):
        raise typer.BadParameter("--memory-backend must be file or lance")

    tracer = _build_trace(
        trace,
        otlp,
        debug=trace_debug,
        truncate=trace_truncate,
        metrics_port=metrics_port,
    )
    sid = _session_id(tracer)
    mem_path = memory_path or str(
        Path(tempfile.gettempdir()) / f"sallm-memory-{sid}"
    )

    try:
        tool_names = _parse_tools_flag(tools)
        registry = builtin_tools(
            tool_names,
            memory_path=mem_path,
            memory_session=sid,
            memory_backend=backend,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    ctx, context_label = _build_context(
        kind,
        model=model,
        api_base=api_base,
        max_context_messages=max_context_messages,
        context_threshold=context_threshold,
        context_keep_last=context_keep_last,
    )

    agent = Agent(
        model=model,
        api_base=api_base,
        system=system,
        max_steps=max_steps,
        multi_step=multi_step,
        tools=registry,
        trace=tracer,
        context=ctx,
    )

    tool_label = ", ".join(registry) or "(none)"
    trace_bits = []
    if trace:
        trace_bits.append(f"jsonl={trace}")
    if otlp:
        trace_bits.append(f"otlp={otlp}")
    if tracer is not None:
        trace_bits.append(f"session={tracer.session_id}")
        if trace_debug:
            trace_bits.append(f"debug truncate={trace_truncate}")
        if metrics_port:
            trace_bits.append(f"metrics=:{metrics_port}/metrics")
    else:
        trace_bits.append(f"session={sid}")
    trace_line = (
        f"trace: [cyan]{', '.join(trace_bits)}[/]\n" if trace_bits else ""
    )
    script_line = f"script: [cyan]{script}[/]\n" if script else ""
    context_line = (
        f"context: [cyan]{context_label}[/]\n" if context_label else ""
    )
    show_prompt_line = (
        "show_prompt: [cyan]on[/]\n" if show_prompt else ""
    )
    mem_line = ""
    if "memory" in registry:
        mem_line = (
            f"memory: [cyan]{backend}[/] path=[cyan]{mem_path}[/]\n"
        )
    console.print(
        Panel(
            f"[bold]sallm[/] chat\nmodel: [cyan]{model}[/]\napi_base: [cyan]{api_base}[/]\n"
            f"tools: [cyan]{tool_label}[/] (CLI subprocesses)\n"
            f"multi_step: [cyan]{multi_step}[/]  max_steps: [cyan]{max_steps}[/]\n"
            f"{script_line}{context_line}{show_prompt_line}{mem_line}{trace_line}"
            "type /help for commands",
            border_style="green",
        )
    )

    prompts = iter_prompts(script) if script else iter_repl()
    for line in prompts:
        if script:
            preview = line if len(line) <= 200 else line[:197] + "..."
            console.print(f"[bold green]you>[/] {preview}")

        if not line:
            continue

        if line.startswith("/"):
            parts = line.split(None, 1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else None
            if cmd in ("/quit", "/exit", "/q"):
                console.print("bye")
                return
            if cmd == "/help":
                _print_help()
                continue
            if cmd == "/clear":
                agent.clear()
                reset_dig_state()
                if "memory" in registry:
                    reset_memory_store(mem_path)
                console.print("[dim]conversation cleared[/]")
                continue
            if cmd == "/history":
                _print_history(agent)
                continue
            if cmd == "/prompt":
                _print_prompt_cmd(agent, arg)
                continue
            console.print(f"[red]unknown command:[/] {cmd}  (try /help)")
            continue

        with console.status("[dim]thinking…[/]", spinner="dots"):
            try:
                result = agent.ask(line)
            except Exception as exc:
                console.print(f"[red]error:[/] {exc}")
                continue

        if show_prompt:
            _print_last_prompt(agent)
        _print_steps(result.get("steps") or [])
        console.print(
            Panel(
                Markdown(result.get("answer") or ""),
                title="assistant",
                border_style="blue",
            )
        )
        _print_metrics(result.get("metrics") or {})

    console.print("bye")


if __name__ == "__main__":
    app()
