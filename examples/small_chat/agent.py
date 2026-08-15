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

    uv run python examples/small_chat/agent.py
    uv run python examples/small_chat/agent.py --profile examples/small_chat/profile-cpu.json
    uv run python examples/small_chat/agent.py --profile examples/small_chat/profile-gemma.json
    uv run python examples/small_chat/agent.py --script examples/small_chat/qa_script.txt
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
from sallm.tools import builtin_tools

console = Console()

HERE = Path(__file__).resolve().parent
STATE_DIR = HERE / ".sallm"
STATE_DB = STATE_DIR / "state.db"
VECTOR_DIR = STATE_DIR / "vectors"
DEFAULT_SESSION = "small-chat-demo"
DEFAULT_PROFILE = HERE / "profile.json"


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


def build_agent(session_id: str, *, profile_path: Path) -> Agent:
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
    args = parser.parse_args(argv)

    profile_path = _resolve_profile_path(args.profile)
    agent = build_agent(args.session, profile_path=profile_path)

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
            f"type /help — edit the profile JSON and restart to try new instructions",
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
