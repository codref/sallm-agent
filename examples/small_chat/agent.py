#!/usr/bin/env python3
"""Small-model chat example — inject a CompiledProfile at runtime.

What this proves
----------------
You do **not** need to ship a profile under ``src/sallm/profiles/``.
Pass any CompiledProfile JSON; ``target_model`` and ``budgets`` select the
student. Switch models by switching ``--profile``, not a hard-coded tag list.

Prefer LiteLLM's ``ollama_chat/`` prefix in ``target_model`` (Ollama
``/api/chat``). Plain ``ollama/`` hits ``/api/generate``, which is unreliable
for ``think``. ``budgets.think`` is false/true or ``low``/``medium``/``high``
/``max``; optional ``think_hint`` is extra system text when thinking is on.
Embeddings stay on the usual ``qwen3-embedding:0.6b``.

How to run (from the repo root)
-------------------------------
    ollama pull qwen3.5:0.8b          # or whatever target_model you set
    ollama pull qwen3-embedding:0.6b
    # optional CPU-only tag:
    ollama create qwen3.5:0.8b-cpu -f examples/small_chat/Modelfile

    # Observability stack (Tempo + Prometheus + Grafana)
    docker compose up -d

    # Interactive REPL — OTLP + Prometheus metrics are ON by default
    uv run python examples/small_chat/agent.py
    uv run python examples/small_chat/agent.py --profile examples/small_chat/profile-cpu.json
    uv run python examples/small_chat/agent.py --profile examples/small_chat/profile-gemma.json
    uv run python examples/small_chat/agent.py --script examples/small_chat/qa_script.txt

Telemetry defaults (same as docs/tracing-tempo.md):
    --otlp http://localhost:4318
    --metrics-port 9464
Disable with --no-otlp and --metrics-port 0. Optional JSONL: --trace /tmp/small.jsonl

Grafana: http://localhost:3000 → dashboard "sallm session" → session_id=small-chat-demo

    uv run sallm optimize --dataset examples/small_chat/opt_cases.jsonl \
        --profile examples/small_chat/profile.json --task converse --evaluate-only
    uv run sallm optimize --dataset examples/small_chat/opt_cases.jsonl \
        --profile examples/small_chat/profile.json --task converse \
        --budgets-only --out examples/small_chat/profile.budgets.json
    uv run sallm optimize --dataset examples/small_chat/opt_cases.jsonl \
        --profile examples/small_chat/profile.json --task converse \
        --out examples/small_chat/profile.opt.json

Slash commands: /help /clear /context /memory /quit
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel

from sallm import Agent, CompiledProfile, RetrievalConfig
from sallm.llm import ThinkingTruncated
from sallm.models import resolve_embedding_profile, resolve_model_profile
from sallm.prom import SessionMetrics
from sallm.tools import builtin_tools
from sallm.trace import (
    DEFAULT_TRUNCATE,
    Tracer,
    jsonl_sink,
    multi_sink,
    otlp_http_sink,
)

console = Console()

HERE = Path(__file__).resolve().parent
STATE_DIR = HERE / ".sallm"
STATE_DB = STATE_DIR / "state.db"
VECTOR_DIR = STATE_DIR / "vectors"
DEFAULT_SESSION = "small-chat-demo"
DEFAULT_PROFILE = HERE / "profile.json"
DEFAULT_OTLP = "http://localhost:4318"
DEFAULT_METRICS_PORT = 9464


def _resolve_profile_path(raw: str | None) -> Path:
    env = (os.environ.get("SALLM_PROFILE") or "").strip()
    text = (raw or env or str(DEFAULT_PROFILE)).strip()
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    if not path.is_file():
        raise SystemExit(f"profile not found: {path}")
    return path


def build_trace(
    *,
    session_id: str,
    otlp_url: str | None,
    trace_path: Path | None,
    metrics_port: int,
    debug: bool = False,
    truncate: int = DEFAULT_TRUNCATE,
) -> Tracer | None:
    """OTLP and/or JSONL sinks plus optional Prometheus /metrics.

    ``session_id`` is shared across Agent state, Tempo ``session.id``, and
    Prometheus labels so the Grafana "sallm session" dashboard lines up.
    """
    sinks = []
    if trace_path:
        sinks.append(jsonl_sink(str(trace_path)))
    if otlp_url:
        sinks.append(otlp_http_sink(otlp_url))
    if not sinks and not metrics_port:
        return None
    emit = (lambda _event: None)
    if sinks:
        emit = sinks[0] if len(sinks) == 1 else multi_sink(*sinks)
    tracer = Tracer(
        emit,
        debug=debug,
        truncate=truncate,
        session_id=session_id,
    )
    if metrics_port:
        metrics = SessionMetrics(tracer.session_id)
        metrics.start_server(port=int(metrics_port))
        tracer.metrics = metrics
    return tracer


def build_agent(
    session_id: str,
    *,
    profile_path: Path,
    trace: Tracer | None = None,
) -> Agent:
    """Build a durable agent from a CompiledProfile JSON path."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    compiled = CompiledProfile.load(profile_path)
    model = (compiled.target_model or "").strip()
    if not model:
        raise SystemExit(
            f"profile {profile_path} has empty target_model — set it in the JSON"
        )
    api_base = (os.environ.get("SALLM_API_BASE") or "").strip() or None
    version = str(
        (compiled.metadata or {}).get("version") or profile_path.stem
    )

    # Budgets (think, temperature, token caps) overlay via compiled_profile.
    profile = resolve_model_profile(model, api_base=api_base, version=version)
    embedding = resolve_embedding_profile()

    return Agent(
        model=model,
        api_base=api_base,
        profile=profile,
        embedding_profile=embedding,
        compiled_profile=compiled,
        tools=builtin_tools(("echo", "calc")),
        state_path=STATE_DB,
        vector_path=VECTOR_DIR,
        session_id=session_id,
        trace=trace,
        retrieval=RetrievalConfig(
            memory_gate=True,
            search_mode="dense",
            use_instruct=True,
            # Prefer controller retrieval_query over raw user text (mixed tool+recall).
            use_rewrite=True,
            use_hyde=False,
        ),
        max_steps=5,
    )


def print_help() -> None:
    console.print(
        Panel(
            "[bold]/help[/]     this help\n"
            "[bold]/clear[/]    wipe this session (SQLite + vectors)\n"
            "[bold]/context[/]  last ContextReceipt (token budget)\n"
            "[bold]/memory[/]   chunk / derived-fact counts\n"
            "[bold]/quit[/]     exit",
            title="commands",
            border_style="dim",
        )
    )


def print_context(agent: Agent) -> None:
    receipt = agent.last_receipt
    if receipt is None:
        console.print("[yellow]no receipt yet — ask something first[/]")
        return
    d = receipt.as_dict()
    lines = [
        f"budget: {d['budget']}  used≈{d['total_tokens']}  "
        f"omitted_msgs={d['omitted_messages']}",
        f"profile: {d.get('profile_version') or '(none)'}",
    ]
    for s in d.get("sections") or []:
        flag = "✓" if s.get("included") else "·"
        note = f" ({s['note']})" if s.get("note") else ""
        lines.append(f"  {flag} {s['name']}: {s['tokens']} tok{note}")
    console.print(Panel("\n".join(lines), title="context receipt", border_style="cyan"))


def print_memory(agent: Agent) -> None:
    if agent.repo is None:
        console.print("[dim]no durable memory[/]")
        return
    chunks = agent.repo.list_chunks(agent.session_id)
    derived = agent.repo.list_derived(agent.session_id)
    indexed = sum(1 for c in chunks if c.indexed)
    console.print(
        Panel(
            f"chunks: [cyan]{len(chunks)}[/] (indexed={indexed})\n"
            f"derived facts: [cyan]{len(derived)}[/]",
            title="memory",
            border_style="cyan",
        )
    )


def handle_slash(agent: Agent, line: str) -> bool:
    """Return True when the REPL should exit."""
    cmd = line.split(None, 1)[0].lower()
    if cmd in ("/quit", "/exit", "/q"):
        console.print("bye")
        return True
    if cmd == "/help":
        print_help()
    elif cmd == "/clear":
        agent.clear()
        console.print("[dim]session cleared[/]")
    elif cmd == "/context":
        print_context(agent)
    elif cmd == "/memory":
        print_memory(agent)
    else:
        console.print(f"[red]unknown command:[/] {cmd}  (try /help)")
    return False


def print_result(result: dict, *, show_receipt: bool) -> None:
    for step in result.get("steps") or []:
        if step.get("kind") == "action":
            for call in step.get("tool_calls") or []:
                console.print(
                    f"[dim]tool[/] {call.get('action')} → {call.get('observation')}"
                )
    _print_thinking(result)
    answer = (result.get("answer") or "").strip()
    if answer:
        # Unclosed ``` fences vanish inside Rich Markdown — print those raw.
        if "```" in answer:
            console.print(escape(answer))
        else:
            console.print(Markdown(answer))
    if show_receipt and result.get("receipt"):
        console.print(f"[dim]receipt:[/] {result['receipt']}")


def _print_thinking(result: dict) -> None:
    """One line: thinking cost + a short snippet, never the full trace."""
    metrics = result.get("metrics") or {}
    rtok = int(metrics.get("reasoning_tokens") or 0)
    rch = int(metrics.get("reasoning_chars") or 0)
    snippet = ""
    for step in result.get("steps") or []:
        raw = str(step.get("reasoning") or "").strip()
        if raw:
            snippet = raw.replace("\n", " ")
    if not rtok and not rch and not snippet:
        return
    if len(snippet) > 96:
        snippet = snippet[:93] + "..."
    extra = f"  {snippet}" if snippet else ""
    console.print(f"[dim]think[/] {rtok} tok / {rch} chars{extra}")


def iter_repl():
    while True:
        try:
            line = console.input("[bold green]you>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        yield line


def iter_script(path: Path):
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        yield line


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SALLM small_chat — any model via CompiledProfile JSON"
    )
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--script", type=Path, default=None)
    parser.add_argument(
        "--profile",
        default=None,
        help=(
            f"CompiledProfile JSON (target_model + instructions + budgets). "
            f"Default: {DEFAULT_PROFILE} or $SALLM_PROFILE"
        ),
    )
    parser.add_argument(
        "--show-receipt",
        action="store_true",
        help="Print ContextReceipt dict after each answer",
    )
    # Telemetry ON by default (docker compose Tempo + Prometheus + Grafana).
    parser.add_argument(
        "--otlp",
        default=None,
        metavar="URL",
        help=f"OTLP/HTTP endpoint (default: {DEFAULT_OTLP}; env SALLM_OTLP)",
    )
    parser.add_argument(
        "--no-otlp",
        action="store_true",
        help="Disable OTLP export to Tempo",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=None,
        metavar="PORT",
        help=f"Prometheus /metrics port (default: {DEFAULT_METRICS_PORT}; 0=off)",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=None,
        metavar="PATH",
        help="Also write spans to a JSONL file",
    )
    parser.add_argument(
        "--trace-debug",
        action="store_true",
        help="Print truncated span payloads to stderr",
    )
    parser.add_argument(
        "--trace-truncate",
        type=int,
        default=DEFAULT_TRUNCATE,
        help=f"Max chars per traced field (default: {DEFAULT_TRUNCATE})",
    )
    args = parser.parse_args(argv)

    profile_path = _resolve_profile_path(args.profile)

    if args.no_otlp:
        otlp_url = None
    elif args.otlp is not None:
        otlp_url = args.otlp.strip() or None
    else:
        otlp_url = (
            (os.environ.get("SALLM_OTLP") or "").strip()
            or DEFAULT_OTLP
        )

    if args.metrics_port is not None:
        metrics_port = int(args.metrics_port)
    else:
        env_port = (os.environ.get("SALLM_METRICS_PORT") or "").strip()
        metrics_port = int(env_port) if env_port else DEFAULT_METRICS_PORT

    tracer = build_trace(
        session_id=args.session,
        otlp_url=otlp_url,
        trace_path=args.trace,
        metrics_port=metrics_port,
        debug=args.trace_debug,
        truncate=args.trace_truncate,
    )
    agent = build_agent(
        args.session, profile_path=profile_path, trace=tracer
    )

    tel_bits = []
    if otlp_url:
        tel_bits.append(f"otlp:    [cyan]{otlp_url}[/]")
    else:
        tel_bits.append("otlp:    [dim]off[/]")
    if metrics_port:
        tel_bits.append(f"metrics: [cyan]:{metrics_port}/metrics[/]")
    else:
        tel_bits.append("metrics: [dim]off[/]")
    if args.trace:
        tel_bits.append(f"jsonl:   [cyan]{args.trace}[/]")

    console.print(
        Panel(
            "[bold]sallm[/] small_chat — runtime profile injection\n"
            f"model:   [cyan]{agent.model}[/]\n"
            f"think:   [cyan]{agent.profile.think!r}[/]\n"
            f"profile: [cyan]{profile_path}[/] "
            f"(version={agent.profile.version})\n"
            f"embed:   [cyan]{agent.embedding_profile.model}[/]\n"
            f"session: [cyan]{args.session}[/]\n"
            f"state:   [cyan]{STATE_DB}[/]\n"
            + "\n".join(tel_bits)
            + "\n"
            "type /help — edit the profile JSON and restart to try new instructions\n"
            "[dim]Grafana: set session_id to this session name[/]",
            border_style="green",
        )
    )

    prompts = iter_script(args.script) if args.script else iter_repl()
    for line in prompts:
        if args.script:
            preview = line if len(line) <= 200 else line[:197] + "..."
            console.print(f"[bold green]you>[/] {preview}")
        if not line:
            continue
        if line.startswith("/"):
            if handle_slash(agent, line):
                return 0
            continue

        with console.status("[dim]thinking…[/]", spinner="dots"):
            try:
                result = agent.ask(line)
            except ThinkingTruncated as exc:
                console.print(f"[yellow]thinking truncated:[/] {exc}")
                continue
            except Exception as exc:
                console.print(f"[red]error:[/] {exc}")
                continue

        print_result(result, show_receipt=args.show_receipt)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
