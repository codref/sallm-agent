import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from sallm import Agent
from sallm.messages import DEFAULT_API_BASE, DEFAULT_MODEL

app = typer.Typer(
    name="sallm",
    help="Minimal tool-calling agent over LiteLLM / Ollama.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


@app.callback()
def _root():
    """Minimal tool-calling agent over LiteLLM / Ollama."""
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
        if not content and msg.get("tool_calls"):
            names = []
            for tc in msg["tool_calls"]:
                fn = tc.get("function") or {}
                names.append(fn.get("name") or "?")
            content = f"tool_calls: {', '.join(names)}"
        preview = content.replace("\n", " ")
        if len(preview) > 60:
            preview = preview[:57] + "..."
        table.add_row(str(i), msg.get("role", "?"), str(len(content)), preview)
    console.print(table)


def _print_steps(steps):
    for i, step in enumerate(steps, 1):
        if step.get("kind") == "decide":
            use = step.get("use_tools")
            lines = [f"use_tools = [bold]{use}[/]"]
            if step.get("tools") is not None:
                lines.append(f"tools = {step.get('tools')}")
            if step.get("raw"):
                lines.append(f"[dim]{step['raw']}[/]")
            console.print(
                Panel(
                    "\n".join(lines),
                    title="decide",
                    border_style="yellow",
                )
            )
            continue

        if step.get("kind") == "action":
            calls = step.get("tool_calls") or []
            if not calls and step.get("action"):
                calls = [
                    {
                        "action": step.get("action"),
                        "action_input": step.get("action_input"),
                        "observation": step.get("observation"),
                    }
                ]
            for tc in calls:
                body = (
                    f"[bold]name[/]   {tc.get('action')}\n"
                    f"[bold]args[/]   {tc.get('action_input')}\n"
                    f"[bold]result[/] {tc.get('observation')}"
                )
                console.print(
                    Panel(
                        body,
                        title="tool",
                        border_style="magenta",
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
    max_steps: int = typer.Option(5, "--max-steps", help="Max tool-loop steps per turn"),
):
    """Interactive chat REPL for testing the agent."""
    agent = Agent(
        model=model,
        api_base=api_base,
        system=system,
        max_steps=max_steps,
    )

    console.print(
        Panel(
            f"[bold]sallm[/] chat\nmodel: [cyan]{model}[/]\napi_base: [cyan]{api_base}[/]\n"
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
