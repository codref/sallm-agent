"""Chat REPL display helpers."""

from __future__ import annotations

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

console = Console()
_PROMPT_PREVIEW_CHARS = 400


def print_help():
    console.print(
        Panel(
            "[bold]/help[/]  this help\n"
            "[bold]/clear[/]  reset conversation (+ dig/session state)\n"
            "[bold]/history[/]  show message roles + lengths\n"
            "[bold]/prompt[/]  show system prompt (+ last LLM view if any)\n"
            "[bold]/prompt system[/]  system / templates only\n"
            "[bold]/prompt last[/]  last messages sent to the model\n"
            "[bold]/state[/]  session id, goal, paths\n"
            "[bold]/stack[/]  skill stack\n"
            "[bold]/memory[/]  stored chunk counts\n"
            "[bold]/context[/]  last ContextReceipt\n"
            "[bold]/quit[/]  exit",
            title="commands",
            border_style="dim",
        )
    )


def print_state(agent, state_path, vector_path):
    console.print(
        Panel(
            f"session: [cyan]{agent.session_id}[/]\n"
            f"goal: [cyan]{agent.goal or '(none)'}[/]\n"
            f"state: [cyan]{state_path or '(in-memory)'}[/]\n"
            f"vectors: [cyan]{vector_path or '(none)'}[/]\n"
            f"retrieval: [cyan]{agent.retrieval_mode}[/]  "
            f"search: [cyan]{getattr(agent.retrieval, 'search_mode', 'dense')}[/]  "
            f"gate: [cyan]{'on' if getattr(agent.retrieval, 'memory_gate', True) else 'off'}[/]\n"
            f"extract: [cyan]{getattr(agent, 'extract_mode', 'waterfall')}[/]",
            title="state",
            border_style="cyan",
        )
    )


def print_stack(agent):
    frames = agent.stack
    if not frames:
        console.print("[dim]stack empty (no durable session)[/]")
        return
    table = Table(title="skill stack", show_header=True, header_style="bold")
    table.add_column("depth", width=6)
    table.add_column("skill")
    table.add_column("note")
    for f in frames:
        table.add_row(str(f.depth), f.skill, f.note or "")
    console.print(table)


def print_memory(agent):
    if agent.repo is None:
        console.print("[dim]no durable memory (pass --state-path)[/]")
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


def print_context_receipt(agent):
    receipt = agent.last_receipt
    if receipt is None:
        console.print("[yellow]no context receipt yet — ask something first[/]")
        return
    d = receipt.as_dict()
    lines = [
        f"profile: {d['profile']} ({d['profile_version']})",
        f"budget: {d['budget']}  used≈{d['total_tokens']}  "
        f"omitted_msgs={d['omitted_messages']}",
    ]
    for s in d.get("sections") or []:
        flag = "✓" if s.get("included") else "·"
        note = f" ({s['note']})" if s.get("note") else ""
        lines.append(f"  {flag} {s['name']}: {s['tokens']} tok{note}")
    if d.get("retrieved"):
        lines.append("retrieved:")
        for h in d["retrieved"][:8]:
            lines.append(
                f"  - {str(h.get('id', ''))[:12]}… "
                f"score={h.get('score')} src={h.get('source_id')}"
            )
    if d.get("fallbacks"):
        lines.append("fallbacks: " + ", ".join(d["fallbacks"]))
    console.print(
        Panel("\n".join(lines), title="context receipt", border_style="cyan")
    )


def print_history(agent):
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


def print_last_prompt(agent, *, truncate=_PROMPT_PREVIEW_CHARS):
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
            str(i), msg.get("role", "?"), str(len(content)), escape(shown)
        )
    console.print(table)


def print_prompt_cmd(agent, arg: str | None = None):
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
            print_last_prompt(agent)
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
        print_last_prompt(agent)
        return
    console.print(f"[red]unknown /prompt arg:[/] {kind!r}  (try system|last)")


def print_steps(steps):
    for i, step in enumerate(steps, 1):
        if step.get("kind") == "action":
            for tc in step.get("tool_calls") or []:
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
                        border_style="yellow"
                        if tc.get("intermediate")
                        else "magenta",
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
                        escape(step["nudge"]), title="nudge", border_style="dim"
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


def print_metrics(metrics):
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
