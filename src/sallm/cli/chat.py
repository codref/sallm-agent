import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from sallm import Agent
from sallm.messages import DEFAULT_API_BASE, DEFAULT_MODEL

from .tools import CHAT_TOOLS, reset_dig_state

app = typer.Typer(
    name="sallm",
    help="Minimal CLI-tool agent over LiteLLM / Ollama.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


@app.callback()
def _root():
    """Minimal CLI-tool agent over LiteLLM / Ollama."""
    pass


def _print_help():
    console.print(
        Panel(
            "[bold]/help[/]  this help\n"
            "[bold]/clear[/]  reset conversation\n"
            "[bold]/history[/]  show message roles + lengths\n"
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
    console.print(Panel(table, title="metrics", border_style="cyan"))


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
):
    """Interactive chat REPL for testing the agent."""
    agent = Agent(
        model=model,
        api_base=api_base,
        system=system,
        max_steps=max_steps,
        multi_step=multi_step,
        tools=CHAT_TOOLS,
    )

    tool_names = ", ".join(CHAT_TOOLS) or "(none)"
    console.print(
        Panel(
            f"[bold]sallm[/] chat\nmodel: [cyan]{model}[/]\napi_base: [cyan]{api_base}[/]\n"
            f"tools: [cyan]{tool_names}[/] (CLI subprocesses)\n"
            f"multi_step: [cyan]{multi_step}[/]  max_steps: [cyan]{max_steps}[/]\n"
            "type /help for commands",
            border_style="green",
        )
    )

    while True:
        try:
            line = console.input("[bold green]you>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye")
            break

        if not line:
            continue

        if line.startswith("/"):
            cmd = line.split(None, 1)[0].lower()
            if cmd in ("/quit", "/exit", "/q"):
                console.print("bye")
                break
            if cmd == "/help":
                _print_help()
                continue
            if cmd == "/clear":
                agent.clear()
                reset_dig_state()
                console.print("[dim]conversation cleared[/]")
                continue
            if cmd == "/history":
                _print_history(agent)
                continue
            console.print(f"[red]unknown command:[/] {cmd}  (try /help)")
            continue

        with console.status("[dim]thinking…[/]", spinner="dots"):
            try:
                result = agent.ask(line)
            except Exception as exc:
                console.print(f"[red]error:[/] {exc}")
                continue

        _print_steps(result.get("steps") or [])
        console.print(
            Panel(
                Markdown(result.get("answer") or ""),
                title="assistant",
                border_style="blue",
            )
        )
        _print_metrics(result.get("metrics") or {})


if __name__ == "__main__":
    app()
