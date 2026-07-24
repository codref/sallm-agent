---
name: sallm agent scaffold
overview: Scaffold a minimal, human-readable Python library (`sallm`) with a working ReAct loop over LiteLLM/Ollama, one echo tool, and a Rich/Typer chat CLI that surfaces tokens, timing, context, and reasoning.
todos:
  - id: packaging
    content: Add pyproject.toml (uv, litellm/typer/rich), .gitignore, README with Ollama + gemma4 defaults
    status: completed
  - id: library-core
    content: "Implement src/sallm: llm wrapper, echo tool, ReAct agent loop, metrics helpers (plain dicts, no pydantic)"
    status: completed
  - id: cli
    content: Implement Typer+Rich chat REPL with turn metrics and slash commands
    status: completed
  - id: smoke
    content: Install editable with uv and verify import + CLI /help
    status: completed
isProject: false
---

# sallm-agent v1 scaffold

## Goals

- Library first: small, plain Python (no Pydantic, no type annotations for now).
- Working **ReAct** loop: Thought → Action → Observation, with one built-in **echo** tool so the path is real.
- Default backend: **Ollama** via LiteLLM, model `ollama/gemma4:e4b-it-qat`, `api_base=http://localhost:11434`.
- Chat CLI to exercise the library and print turn metrics (tokens, elapsed time, context size, reasoning/tool steps).

## Layout

```
sallm-agent/
  pyproject.toml          # uv, package + console script
  README.md               # short usage for library + CLI
  .gitignore
  src/sallm/
    __init__.py           # export Agent, run helpers
    agent.py              # ReAct loop
    llm.py                # thin litellm wrapper
    tools.py              # Tool registry + echo
    messages.py           # plain dict message helpers
    metrics.py            # usage / timing extraction
  src/sallm/cli/
    __main__.py
    chat.py               # Typer + Rich REPL
```

Package name: **`sallm`**. Console entry: `sallm` → chat command.

## Library design (keep tiny)

### `llm.py`
- One function: `complete(model, messages, api_base=..., **kwargs)` calling `litellm.completion`.
- Return a plain dict: `content`, `reasoning` (if present), `usage` (`prompt_tokens`, `completion_tokens`, `total_tokens`), raw response kept optional for debugging.
- Default model: `ollama/gemma4:e4b-it-qat`, default `api_base`: `http://localhost:11434`.

### `tools.py`
- Tools as plain callables registered by name: `{"echo": echo}`.
- `echo(text)` returns the same text (proves Action → Observation).
- `run_tool(name, args)` looks up and calls; unknown tool → error string (agent continues).

### `agent.py` — ReAct core
- `Agent(model=..., api_base=..., tools=..., system=..., max_steps=...)`.
- Holds `messages` as a list of plain dicts (`role` / `content`).
- `ask(user_text)`:
  1. Append user message.
  2. Loop up to `max_steps`:
     - Call LLM with a short system prompt that asks for ReAct format (or free-form final answer).
     - Parse a **simple, forgiving** response:
       - If it contains `Final Answer:` → stop, return that text.
       - If it contains `Action:` + `Action Input:` → run tool, append Observation, continue.
       - Else treat whole reply as final answer (so normal chat still works).
     - Collect per-step metrics (tokens, elapsed, thought/action text).
  3. Append assistant final reply to history; return `{answer, steps, metrics}`.
- No graphs, memory, vectorization, or multi-tool planners — leave hooks obvious (e.g. `self.tools` dict) for later.

### Prompt shape (explicit in code, not a config framework)

```
You are a helpful agent. Use this format when you need a tool:
Thought: ...
Action: echo
Action Input: ...
When done:
Thought: ...
Final Answer: ...
Available tools: echo — returns the input unchanged.
```

Parsing: regex / line-based, human-readable, fail soft.

### `metrics.py`
- From LiteLLM `usage` + wall-clock `time.perf_counter()`.
- Aggregate: input tokens, output tokens, total tokens, elapsed ms, message/context count (len of messages or rough char/token estimate from usage).

## CLI (`sallm chat`)

Stack: **Typer** for args, **Rich** for panels/tables/markdown.

Flags:
- `--model` (default `ollama/gemma4:e4b-it-qat`)
- `--api-base` (default `http://localhost:11434`)
- `--system` optional
- `--max-steps` (default ~5)

REPL:
- Read lines until `/quit` or Ctrl-D.
- Slash commands: `/clear`, `/history`, `/help`, `/quit`.
- After each turn, print:
  - Assistant answer (and, if present, Thought / Action / Observation steps in a dim panel).
  - A compact metrics line/table: **in/out/total tokens**, **elapsed**, **context messages** (and total tokens so far if easy).

Example:

```bash
uv run sallm chat
uv run sallm chat --model ollama/gemma4:e4b-it-qat
```

## Packaging

- [`pyproject.toml`](pyproject.toml) with `uv`, Python ≥3.12, deps: `litellm`, `typer`, `rich`.
- Hatchling/setuptools src layout; script entry `sallm = "sallm.cli.chat:app"`.
- Root [`.gitignore`](.gitignore): `.venv/`, `__pycache__/`, `.idea/`, `dist/`, etc.
- README: install, Ollama prerequisite (`ollama pull gemma4:e4b-it-qat`), library one-liner, CLI usage. No long docs.

## Explicit non-goals (v1)

- Pydantic / typing / mypy
- Real tool ecosystem, MCP, graphs, memory, RAG
- Streaming (can add later; v1 = sync completion)
- Config YAML / proxy server

## Smoke check after scaffold

- Import `from sallm import Agent` and construct with defaults.
- CLI starts and `/help` works (full Ollama round-trip only if the model is already pulled locally).
