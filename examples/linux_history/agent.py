#!/usr/bin/env python3
"""Linux shell-history example — durable SALLM agent over bash/zsh history.

What this proves
----------------
Long Q&A about past console work stays coherent: early facts leave the
recent-history window and return via SQLite extract + LanceDB retrieval.
The agent can also reconstruct **usage stories** (e.g. “the day I was
authenticating on docker… what remote host did I use?”) by searching history
with time windows and nearby context — without dumping the whole histfile.

How to run (from the repo root)
-------------------------------
    # optional overrides
    cp examples/linux_history/.env.example examples/linux_history/.env

    docker compose up -d   # Tempo + Prometheus + Grafana

    uv run python examples/linux_history/agent.py

    uv run python examples/linux_history/agent.py \\
      --script examples/linux_history/qa_script.txt --show-receipt

    uv run python examples/linux_history/agent.py --session hist-demo

Telemetry defaults: --otlp http://localhost:4318 --metrics-port 9464
Disable with --no-otlp and --metrics-port 0.

WARNING: bash_run executes real commands as your user.

Slash commands: /help /clear /context /memory /stack /feed [N] /remember /quit

Requires: Ollama with gemma4:e4b-it-qat and qwen3-embedding:0.6b.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from hist_common import load_env  # noqa: E402

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

STATE_DIR = HERE / ".sallm"
STATE_DB = STATE_DIR / "state.db"
VECTOR_DIR = STATE_DIR / "vectors"
DEFAULT_SESSION = "hist-demo"
DEFAULT_OTLP = "http://localhost:4318"
DEFAULT_METRICS_PORT = 9464


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def build_tools() -> dict[str, CliTool]:
    py = sys.executable
    return {
        "hist_search": CliTool(
            name="hist_search",
            argv=[py, str(HERE / "hist_search.py")],
            summary=(
                "Search bash/zsh shell history (read-only). "
                "Flags: --query TEXT --mode token|substr|semantic --limit N "
                "--file PATH --days N --since ISO --until ISO --context K "
                "--include-undated --semantic-pool N --min-score F. "
                "Default mode=token (whole tokens; avoids path false positives). "
                "Use --mode semantic for natural-language / related-command search "
                "(needs Ollama qwen3-embedding). "
                "For usage stories: --query seed --days N --context 8. "
                "NEVER invent hosts/IPs — only report tool output."
            ),
        ),
        "hist_ingest": CliTool(
            name="hist_ingest",
            argv=[py, str(HERE / "hist_ingest.py")],
            summary=(
                "Queue shell-history blocks for meaning-first Agent.remember "
                "(LLM turns commands into English facts in durable memory). "
                "Flags: --query TEXT --mode token|semantic --limit N --days N "
                "--context K --block-lines N, or --latest N (newest N commands, "
                "no search; up to 2000) to load a large remember phase. "
                "Use when the user asks to index/load/remember history into memory. "
                "Not for live command execution."
            ),
        ),
        "bash_run": CliTool(
            name="bash_run",
            argv=[py, str(HERE / "bash_run.py")],
            summary=(
                "Execute a live bash command as the current user. "
                "Flags: --command CMD (-c) --timeout SEC --max-chars N. "
                "DANGER: full privileges. "
                "Do NOT use for reading history — use hist_search. "
                "Only report real exit code and truncated stdout/stderr."
            ),
        ),
    }


# ---------------------------------------------------------------------------
# Skills — shell mode specializes on history stories + console scripting
# ---------------------------------------------------------------------------

SHELL_SKILL = Skill(
    name="shell",
    description=(
        "User asks about past shell commands, bash/zsh history, console scripting, "
        "or reconstructing what they did on a past day (usage patterns, docker auth, "
        "ssh hosts, IPs). Also when they want history indexed into durable memory, "
        "or a command executed live."
    ),
    prompt=(
        "Active skill: shell.\n"
        "You help with bash/zsh history and console scripting.\n"
        "Past activity: use hist_search via ```run blocks. "
        "Keyword lookups: hist_search --query <word> --mode token --limit N. "
        "Vague memories: hist_search --query \"...\" --mode semantic --limit N. "
        "To store history as English-retrievable memory (hosts, IPs, patterns): "
        "hist_ingest --query <seed> --limit 40 --context 4 — the runtime will "
        "call Agent.remember (LLM meaning extract). For a large bulk load of "
        "recent activity: hist_ingest --latest 200 (or larger). Prefer ingest "
        "before long English story questions about past ops.\n"
        "For day-level stories after ingest: answer from memory; re-search only "
        "if needed. Do not invent hosts/IPs.\n"
        "Live execution: bash_run --command \"...\".\n"
        "Keep answers short. Never dump whole history files."
    ),
    tools=("hist_search", "hist_ingest", "bash_run"),
)


def build_skills() -> SkillRegistry:
    return SkillRegistry([SHELL_SKILL])


# ---------------------------------------------------------------------------
# Tracing
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


def build_agent(session_id: str, trace: Tracer | None = None) -> Agent:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return Agent(
        tools=build_tools(),
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
        max_steps=8,
    )


# ---------------------------------------------------------------------------
# REPL helpers
# ---------------------------------------------------------------------------

def print_help() -> None:
    console.print(
        Panel(
            "[bold]/help[/]         this help\n"
            "[bold]/clear[/]        wipe this session (SQLite + vectors)\n"
            "[bold]/context[/]      last ContextReceipt (token budget)\n"
            "[bold]/memory[/]       chunk / derived-fact counts\n"
            "[bold]/stack[/]        active skill stack\n"
            "[bold]/feed[/] [N]     queue newest N history cmds → remember "
            "(default 120)\n"
            "[bold]/remember[/]     drain pending hist_ingest → agent.remember\n"
            "[bold]/quit[/]         exit",
            title="commands",
            border_style="dim",
        )
    )


def drain_pending_remember(agent: Agent) -> list[dict]:
    """Apply blocks queued by hist_ingest via Agent.remember (meaning-first)."""
    pending = HERE / ".sallm" / "pending_remember.jsonl"
    if not pending.is_file():
        return []
    lines = pending.read_text(encoding="utf-8").splitlines()
    pending.unlink(missing_ok=True)
    items = []
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        text = str(obj.get("text") or "").strip()
        if not text:
            continue
        items.append({"text": text, "source": obj.get("source")})
    if not items:
        return []
    console.print(
        f"[dim]remember[/] ingesting {len(items)} history block(s) (LLM meaning)…"
    )
    results = agent.remember_many(items)
    for r in results:
        n = r.get("facts") or 0
        src = r.get("source") or ""
        console.print(f"[dim]remember[/] {src}: facts={n}")
    return results


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
    elif cmd == "/feed":
        # Simulate a larger remember phase: newest N cmds → pending → remember.
        from hist_ingest import queue_latest_blocks, write_pending

        n = 120
        if len(parts) > 1:
            try:
                n = int(parts[1].strip().split()[0])
            except (ValueError, IndexError):
                console.print("[red]/feed needs an integer N[/]  (e.g. /feed 200)")
                return False
        path, blocks = queue_latest_blocks(n=n)
        if not blocks:
            console.print("[dim]no history to feed[/]")
            return False
        write_pending(blocks)
        console.print(
            f"[dim]feed[/] queued {len(blocks)} block(s) from newest {n} "
            f"cmds ({path.name})"
        )
        drain_pending_remember(agent)
    elif cmd == "/remember":
        results = drain_pending_remember(agent)
        if not results:
            console.print(
                "[dim]nothing pending — try /feed 200, or ask the agent to "
                "hist_ingest[/]"
            )
    else:
        console.print(f"[red]unknown command:[/] {cmd}  (try /help)")
    return False


def print_result(result: dict, *, show_receipt: bool) -> None:
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


def iter_repl():
    while True:
        try:
            line = console.input("[bold green]you>[/] ")
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        yield line


def iter_script(path: Path):
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
        description=(
            "Durable SALLM agent over bash/zsh history with live bash_run. "
            "WARNING: bash_run executes real commands as your user."
        ),
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

    load_env(HERE / ".env")

    if args.no_otlp:
        otlp_url = None
    elif args.otlp is not None:
        otlp_url = args.otlp.strip() or None
    else:
        otlp_url = (os.environ.get("SALLM_OTLP") or "").strip() or DEFAULT_OTLP

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
            f"[bold]sallm[/] Linux shell-history example\n"
            f"session: [cyan]{args.session}[/]\n"
            f"state:   [cyan]{STATE_DB}[/]\n"
            f"vectors: [cyan]{VECTOR_DIR}[/]\n"
            f"tools:   [cyan]hist_search, hist_ingest, bash_run[/]\n"
            + "\n".join(tel_bits)
            + "\n"
            "[yellow]bash_run executes real commands as your user[/]\n"
            "hist_ingest /feed → Agent.remember (English facts). type /help",
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
        drain_pending_remember(agent)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
