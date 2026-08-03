#!/usr/bin/env python3
"""IMAP inbox example — durable SALLM agent over a live mailbox.

What this proves
----------------
A long Q&A session about email can run for many turns (hours of wall-clock
chatter) without the prompt ballooning. Early facts leave the recent-history
window; they come back via SQLite extract + LanceDB retrieval — not by
re-stuffing every past message into the model.

How to run (from the repo root)
-------------------------------
    cp examples/imap_inbox/.env.example examples/imap_inbox/.env
    # edit IMAP_HOST / IMAP_USER / IMAP_PASSWORD (Gmail: use an App Password)

    # Observability stack (Tempo + Prometheus + Grafana)
    docker compose up -d

    # Interactive REPL — OTLP + Prometheus metrics are ON by default
    uv run python examples/imap_inbox/agent.py

    # Scripted multi-phase session (one non-empty line = one ask)
    uv run python examples/imap_inbox/agent.py --script examples/imap_inbox/qa_script.txt

    # Resume the same durable session later
    uv run python examples/imap_inbox/agent.py --session imap-demo

Telemetry defaults (same as docs/tracing-tempo.md):
    --otlp http://localhost:4318
    --metrics-port 9464
Disable with --no-otlp and --metrics-port 0. Optional JSONL: --trace /tmp/imap.jsonl

Grafana: http://localhost:3000 → dashboard "sallm session" → session_id=imap-demo

Slash commands: /help /clear /context /memory /stack /quit

Requires: Ollama with gemma4:e4b-it-qat and qwen3-embedding:0.6b.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# ---------------------------------------------------------------------------
# Make sibling tool modules importable when this file is run as a script.
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from imap_common import load_env  # noqa: E402

from sallm import Agent, CliTool, RetrievalConfig, Skill, SkillRegistry  # noqa: E402
from sallm.prom import SessionMetrics  # noqa: E402
from sallm.trace import (  # noqa: E402
    DEFAULT_TRUNCATE,
    Tracer,
    jsonl_sink,
    multi_sink,
    otlp_http_sink,
)

console = Console()

# Durable state lives next to the example — gitignored via .sallm*
STATE_DIR = HERE / ".sallm"
STATE_DB = STATE_DIR / "state.db"
VECTOR_DIR = STATE_DIR / "vectors"
DEFAULT_SESSION = "imap-demo"
DEFAULT_OTLP = "http://localhost:4318"
DEFAULT_METRICS_PORT = 9464


# ---------------------------------------------------------------------------
# Tools — each is a separate CLI file (SALLM contract: subprocess + --help).
# ---------------------------------------------------------------------------

def build_tools() -> dict[str, CliTool]:
    """Register the three IMAP CLIs as CliTool entries."""
    py = sys.executable
    return {
        "imap_folders": CliTool(
            name="imap_folders",
            argv=[py, str(HERE / "imap_folders.py")],
            summary=(
                "List IMAP mailbox names. No flags. "
                "Use when the user asks which folders exist."
            ),
        ),
        "imap_search": CliTool(
            name="imap_search",
            argv=[py, str(HERE / "imap_search.py")],
            summary=(
                "Search mail; compact header lines (newest UIDs first). "
                "Flags: --folder FOLDER --from ADDRESS --subject TEXT "
                "--query CRITERIA --limit N. "
                "For sender search ALWAYS use --from addr@host (not raw FROM). "
                "For most recent N: --query ALL --limit N "
                "(never rely on IMAP RECENT). "
                "Unread: --query UNSEEN. "
                "NEVER invent messages — only report tool output."
            ),
        ),
        "imap_fetch": CliTool(
            name="imap_fetch",
            argv=[py, str(HERE / "imap_fetch.py")],
            summary=(
                "Fetch one message by UID. "
                "Flags: --folder FOLDER --uid UID --max-chars N. "
                "Returns headers + truncated body. Prefer small max-chars. "
                "Use after imap_search when details of one mail are needed."
            ),
        ),
    }


# ---------------------------------------------------------------------------
# Skills — modes, not tools. Controller routes; prompt shapes behaviour.
# ---------------------------------------------------------------------------

INBOX_SKILL = Skill(
    name="inbox",
    description=(
        "User wants to browse, search, read, or recall email from IMAP. "
        "Use when the question is about mailboxes, senders, subjects, or message bodies."
    ),
    prompt=(
        "Active skill: inbox.\n"
        "Use imap_folders / imap_search / imap_fetch via ```run blocks when live "
        "mailbox data is needed.\n"
        "For the N most recent messages always run: "
        "imap_search --folder INBOX --query ALL --limit N "
        "(do not use RECENT — that IMAP flag is often empty).\n"
        "For sender search always run: "
        "imap_search --from addr@host --limit N "
        "(use --from, never --query with single-quoted FROM).\n"
        "Keep answers short. Prefer durable facts: From, Subject, Date, UID, "
        "and any order/invoice ids — never paste whole MIME payloads.\n"
        "When recalling earlier findings, say what you remember; only re-fetch "
        "if the user asks for fresh server data or memory is insufficient."
    ),
    tools=("imap_folders", "imap_search", "imap_fetch"),
)


def build_skills() -> SkillRegistry:
    # SkillRegistry always ensures `converse` exists even if we omit it.
    return SkillRegistry([INBOX_SKILL])


# ---------------------------------------------------------------------------
# Tracing + Prometheus — same wiring as `sallm chat --otlp … --metrics-port …`
# ---------------------------------------------------------------------------

def build_trace(
    *,
    session_id: str,
    otlp_url: str | None,
    trace_path: Path | None,
    metrics_port: int,
    debug: bool = False,
    truncate: int = DEFAULT_TRUNCATE,
) -> Tracer | None:
    """Build a Tracer with OTLP and/or JSONL sinks, plus optional /metrics.

    session_id is shared across Agent state, Tempo ``session.id``, and Prometheus
    labels so the Grafana "sallm session" dashboard lines up.
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


# ---------------------------------------------------------------------------
# Agent construction — always durable for this example.
# ---------------------------------------------------------------------------

def build_agent(session_id: str, trace: Tracer | None = None) -> Agent:
    """One Agent per process; resume = same state paths + session_id."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tools = build_tools()
    return Agent(
        tools=tools,
        skills=build_skills(),
        state_path=STATE_DB,
        vector_path=VECTOR_DIR,
        session_id=session_id,
        trace=trace,
        retrieval=RetrievalConfig(
            memory_gate=True,
            search_mode="dense",
            use_instruct=True,
            use_rewrite=False,
            use_hyde=False,
        ),
        # Local Ollama defaults from sallm; override via env if needed.
        max_steps=6,
    )


# ---------------------------------------------------------------------------
# REPL helpers — thin, human-readable debugging of the durable pipeline.
# ---------------------------------------------------------------------------

def print_help() -> None:
    console.print(
        Panel(
            "[bold]/help[/]     this help\n"
            "[bold]/clear[/]    wipe this session (SQLite + vectors)\n"
            "[bold]/context[/]  last ContextReceipt (token budget)\n"
            "[bold]/memory[/]   chunk / derived-fact counts\n"
            "[bold]/stack[/]    active skill stack\n"
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
    ]
    for s in d.get("sections") or []:
        flag = "✓" if s.get("included") else "·"
        note = f" ({s['note']})" if s.get("note") else ""
        lines.append(f"  {flag} {s['name']}: {s['tokens']} tok{note}")
    retrieved = d.get("retrieved") or []
    if retrieved:
        lines.append(f"retrieved: {len(retrieved)} hit(s)")
        for h in retrieved[:6]:
            lines.append(
                f"  - score={h.get('score')} src={h.get('source_id')}"
            )
    else:
        lines.append("retrieved: (none)")
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


def print_stack(agent: Agent) -> None:
    frames = agent.stack
    if not frames:
        console.print("[dim]stack empty[/]")
        return
    for f in frames:
        console.print(f"  depth={f.depth} skill=[cyan]{f.skill}[/] note={f.note or ''}")


def handle_slash(agent: Agent, line: str) -> bool:
    """Handle /commands. Return True if the REPL should exit."""
    parts = line.split(None, 1)
    cmd = parts[0].lower()
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
    elif cmd == "/stack":
        print_stack(agent)
    else:
        console.print(f"[red]unknown command:[/] {cmd}  (try /help)")
    return False


def print_result(result: dict, *, show_receipt: bool) -> None:
    # Steps use kind=action with nested tool_calls (same shape as sallm chat).
    for step in result.get("steps") or []:
        if not isinstance(step, dict) or step.get("kind") != "action":
            continue
        for tc in step.get("tool_calls") or []:
            name = tc.get("action") or "?"
            obs = str(tc.get("observation") or "").replace("\n", " ")
            if len(obs) > 140:
                obs = obs[:137] + "..."
            console.print(f"[dim]tool[/] {name} → {obs}")
    console.print(
        Panel(
            Markdown(result.get("answer") or ""),
            title="assistant",
            border_style="blue",
        )
    )
    if show_receipt and result.get("receipt"):
        r = result["receipt"]
        total = r.get("total_tokens") if isinstance(r, dict) else None
        if total is not None:
            console.print(f"[dim]receipt total_tokens≈{total}[/]")


# ---------------------------------------------------------------------------
# Prompt sources: interactive stdin, or a script file (one turn per line).
# ---------------------------------------------------------------------------

def iter_repl():
    while True:
        try:
            line = console.input("[bold green]you>[/] ")
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        yield line


def iter_script(path: Path):
    """One non-empty, non-# line → one agent.ask turn (same as sallm chat --script)."""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        yield line


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Durable SALLM agent that queries an IMAP inbox via CLI tools.",
    )
    parser.add_argument(
        "--session",
        default=DEFAULT_SESSION,
        help=f"Durable session id (default: {DEFAULT_SESSION})",
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=None,
        help="Run prompts from a file (one turn per non-empty line)",
    )
    parser.add_argument(
        "--show-receipt",
        action="store_true",
        help="Print total_tokens after each answer",
    )
    # Telemetry is ON by default for this example (docker compose stack).
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

    # Credentials for tools (subprocess children also load .env themselves).
    load_env(HERE / ".env")
    if not (HERE / ".env").is_file():
        console.print(
            f"[yellow]No .env yet.[/] Copy {HERE / '.env.example'} → "
            f"{HERE / '.env'} and set IMAP_* before searching mail."
        )

    # Resolve telemetry: CLI > .env > demo defaults (enabled).
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
    agent = build_agent(args.session, trace=tracer)

    tel_bits = []
    if otlp_url:
        tel_bits.append(f"otlp: [cyan]{otlp_url}[/]")
    else:
        tel_bits.append("otlp: [dim]off[/]")
    if metrics_port:
        tel_bits.append(f"metrics: [cyan]:{metrics_port}/metrics[/]")
    else:
        tel_bits.append("metrics: [dim]off[/]")
    if args.trace:
        tel_bits.append(f"jsonl: [cyan]{args.trace}[/]")

    console.print(
        Panel(
            f"[bold]sallm[/] IMAP inbox example\n"
            f"session: [cyan]{args.session}[/]\n"
            f"state:   [cyan]{STATE_DB}[/]\n"
            f"vectors: [cyan]{VECTOR_DIR}[/]\n"
            f"tools:   [cyan]imap_folders, imap_search, imap_fetch[/]\n"
            + "\n".join(tel_bits)
            + "\n"
            "type /help — long sessions rely on /context + /memory, not infinite history",
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
            except Exception as exc:
                console.print(f"[red]error:[/] {exc}")
                continue

        print_result(result, show_receipt=args.show_receipt)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
