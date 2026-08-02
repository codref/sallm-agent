"""Interactive chat REPL — agent + optional CLI tools + durable sessions."""

from __future__ import annotations

import secrets
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from sallm import Agent
from sallm.cli import display
from sallm.cli.optimize_cmd import register as register_optimize
from sallm.context import MaxMessages, SummarizeOverflow
from sallm.llm import complete
from sallm.messages import DEFAULT_API_BASE, DEFAULT_MODEL
from sallm.models import resolve_embedding_profile
from sallm.prom import SessionMetrics
from sallm.prompt import CompiledProfile
from sallm.tools import DEFAULT_TOOLS, builtin_tools, reset_dig_state
from sallm.trace import DEFAULT_TRUNCATE, Tracer, jsonl_sink, multi_sink, otlp_http_sink

app = typer.Typer(
    name="sallm",
    help="Minimal CLI-tool agent over LiteLLM / Ollama.",
    add_completion=False,
    no_args_is_help=True,
)
register_optimize(app)
console = Console()

CONTEXT_NONE = "none"
CONTEXT_MAX_MESSAGES = "max-messages"
CONTEXT_SUMMARIZE = "summarize"
CONTEXT_CHOICES = (CONTEXT_NONE, CONTEXT_MAX_MESSAGES, CONTEXT_SUMMARIZE)
_DEFAULT_PROFILE = (
    Path(__file__).resolve().parents[1] / "profiles" / "gemma4-e4b-v1.json"
)


@app.callback()
def _root():
    """Minimal CLI-tool agent over LiteLLM / Ollama."""
    pass


def _build_trace(
    trace_path,
    otlp_url,
    debug=False,
    truncate=512,
    metrics_port=0,
    session_id=None,
):
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
    # Use the same id as Agent/--session so Prometheus + Tempo match the CLI name.
    tracer = Tracer(emit, debug=debug, truncate=truncate, session_id=session_id)
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


def _build_context(
    kind,
    *,
    model,
    api_base,
    max_context_messages,
    context_threshold,
    context_keep_last,
):
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
        return (
            opt,
            f"SummarizeOverflow(threshold={threshold}, keep_last={keep_last})",
        )
    raise typer.BadParameter(
        f"unknown context optimizer {kind!r}; choose from {', '.join(CONTEXT_CHOICES)}"
    )


def _parse_tools_flag(value: str | None) -> str | tuple | None:
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
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def iter_repl():
    while True:
        try:
            yield console.input("[bold green]you>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            return


@app.command()
def chat(
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m"),
    api_base: str = typer.Option(DEFAULT_API_BASE, "--api-base"),
    system: str = typer.Option(None, "--system", "-s"),
    max_steps: int = typer.Option(5, "--max-steps"),
    multi_step: bool = typer.Option(True, "--multi-step/--no-multi-step"),
    tools: str = typer.Option("echo,calc,dig", "--tools"),
    script: str = typer.Option(None, "--script"),
    show_prompt: bool = typer.Option(False, "--show-prompt/--no-show-prompt"),
    trace: str = typer.Option(None, "--trace"),
    otlp: str = typer.Option(None, "--otlp"),
    trace_debug: bool = typer.Option(False, "--trace-debug/--no-trace-debug"),
    trace_truncate: int = typer.Option(DEFAULT_TRUNCATE, "--trace-truncate"),
    metrics_port: int = typer.Option(0, "--metrics-port"),
    context: str = typer.Option(CONTEXT_NONE, "--context"),
    max_context_messages: int | None = typer.Option(None, "--max-context-messages"),
    context_threshold: int | None = typer.Option(None, "--context-threshold"),
    context_keep_last: int | None = typer.Option(None, "--context-keep-last"),
    state_path: str | None = typer.Option(None, "--state-path"),
    vector_path: str | None = typer.Option(None, "--vector-path"),
    session: str | None = typer.Option(None, "--session"),
    profile_path: str | None = typer.Option(None, "--profile"),
    embedding_model: str = typer.Option(
        "ollama/qwen3-embedding:0.6b", "--embedding-model"
    ),
    embedding_dimensions: int = typer.Option(1024, "--embedding-dimensions"),
    retrieval_query: str = typer.Option("instruct", "--retrieval-query"),
    search: str = typer.Option("dense", "--search"),
    memory_gate: bool = typer.Option(True, "--memory-gate/--no-memory-gate"),
    extract: str = typer.Option(
        "waterfall",
        "--extract",
        help="waterfall (sync extract) | queue (defer extract; flush on retrieval miss)",
    ),
    top_k: int = typer.Option(4, "--top-k"),
):
    """Interactive chat REPL for testing the agent."""
    kind = (context or CONTEXT_NONE).strip().lower()
    if kind not in CONTEXT_CHOICES:
        raise typer.BadParameter(
            f"unknown --context {context!r}; choose from {', '.join(CONTEXT_CHOICES)}"
        )
    if kind == CONTEXT_NONE and max_context_messages is not None:
        kind = CONTEXT_MAX_MESSAGES

    from sallm.memory import resolve_retrieval_config

    try:
        retrieval = resolve_retrieval_config(
            retrieval_query=retrieval_query or "instruct",
            search_mode=search or "dense",
            memory_gate=memory_gate,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    rq = retrieval.label

    from sallm.extract_queue import normalize_extract_mode

    try:
        extract_mode = normalize_extract_mode(extract)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    sid = (session or "").strip() or secrets.token_hex(8)
    tracer = _build_trace(
        trace,
        otlp,
        debug=trace_debug,
        truncate=trace_truncate,
        metrics_port=metrics_port,
        session_id=sid,
    )

    compiled = None
    ppath = Path(profile_path) if profile_path else _DEFAULT_PROFILE
    if ppath.is_file():
        try:
            compiled = CompiledProfile.load(ppath)
        except Exception as exc:
            console.print(f"[yellow]profile load failed:[/] {exc}")

    try:
        registry = builtin_tools(_parse_tools_flag(tools))
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
    emb = resolve_embedding_profile(
        embedding_model,
        api_base=api_base,
        dimensions=embedding_dimensions,
        top_k=top_k,
    )
    agent_kwargs = dict(
        model=model,
        api_base=api_base,
        system=system,
        max_steps=max_steps,
        multi_step=multi_step,
        tools=registry,
        trace=tracer,
        context=ctx,
        session_id=sid,
        compiled_profile=compiled,
        retrieval=retrieval,
        embedding_profile=emb,
        extract_mode=extract_mode,
    )
    if state_path:
        agent_kwargs["state_path"] = state_path
        agent_kwargs["vector_path"] = vector_path
    agent = Agent(**agent_kwargs)

    bits = [
        f"[bold]sallm[/] chat",
        f"model: [cyan]{model}[/]",
        f"tools: [cyan]{', '.join(registry) or '(none)'}[/]",
        f"session: [cyan]{sid}[/]",
    ]
    if state_path:
        vp = vector_path or str(Path(state_path).parent / "vectors")
        bits.append(f"state: [cyan]{state_path}[/]  vectors: [cyan]{vp}[/]")
        bits.append(
            f"retrieval: [cyan]{rq}[/]  search: [cyan]{retrieval.search_mode}[/]  "
            f"gate: [cyan]{'on' if retrieval.memory_gate else 'off'}[/]  "
            f"extract: [cyan]{extract_mode}[/]  "
            f"embed: [cyan]{embedding_model}[/]"
        )
    if context_label:
        bits.append(f"context: [cyan]{context_label}[/]")
    bits.append("type /help for commands")
    console.print(Panel("\n".join(bits), border_style="green"))

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
                display.print_help()
            elif cmd == "/clear":
                agent.clear()
                reset_dig_state()
                console.print("[dim]conversation cleared[/]")
            elif cmd == "/history":
                display.print_history(agent)
            elif cmd == "/prompt":
                display.print_prompt_cmd(agent, arg)
            elif cmd == "/state":
                display.print_state(agent, state_path, vector_path)
            elif cmd == "/stack":
                display.print_stack(agent)
            elif cmd == "/memory":
                display.print_memory(agent)
            elif cmd == "/context":
                display.print_context_receipt(agent)
            else:
                console.print(f"[red]unknown command:[/] {cmd}  (try /help)")
            continue

        with console.status("[dim]thinking…[/]", spinner="dots"):
            try:
                result = agent.ask(line)
            except Exception as exc:
                console.print(f"[red]error:[/] {exc}")
                continue

        if show_prompt:
            display.print_last_prompt(agent)
        display.print_steps(result.get("steps") or [])
        console.print(
            Panel(
                Markdown(result.get("answer") or ""),
                title="assistant",
                border_style="blue",
            )
        )
        display.print_metrics(result.get("metrics") or {})

    console.print("bye")


if __name__ == "__main__":
    app()
