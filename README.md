# sallm-agent

Minimal tool-calling agent library (`sallm`) over [LiteLLM](https://github.com/BerriAI/litellm), aimed at local models via Ollama.

**Library core:** LLM turns + message transcript + CLI tool execution.  
**Shipped tools** are optional deployable CLIs (`echo`, `calc`, `dig`, `memory`). Storage/retrieval is the `memory` tool — not an invisible RAG pipeline inside the agent.

## Setup

```bash
ollama pull gemma4:e4b-it-qat

uv sync --extra dev
# optional: LanceDB backend for the memory tool
uv sync --extra memory --extra dev
```

## Tool contract

| Rule | Detail |
|------|--------|
| Identity | Tool name = first argv token (`calc`, `dig`, `echo`, `memory`) |
| Help | Every tool supports `--help` |
| Args | CLI flags / positionals only (never JSON blobs) |
| Success | exit 0; result = stdout |
| Failure | non-zero exit; stderr/stdout returned as the observation |
| Intermediate | stdout may start with `[intermediate]` (agent keeps going) |

````markdown
```run
calc --expression "2**10"
memory search --query "IAS service"
```
````

## Library

Register `CliTool` instances, or use shipped tools via `builtin_tools`:

```python
import sys
from sallm import Agent
from sallm.tools import CliTool, builtin_tools

# shipped set
agent = Agent(tools=builtin_tools(("calc", "echo")))

# or hand-roll
calc = CliTool(
    name="calc",
    argv=[sys.executable, "-m", "sallm.tools.calc"],
    summary="Evaluate a math expression. Flags: --expression EXPR.",
)
agent = Agent(tools={"calc": calc})
print(agent.ask("What is 2**10? Use the calc tool.")["answer"])
```

Runner helpers: `parse_run_blocks`, `run_tool`, `run_many`, `help_text` in `sallm.tools`.

### Shipped tools

| Tool | Role |
|------|------|
| `echo` / `calc` | demos |
| `dig` | multi-step treasure game (entropy distractor in tests — not for documents) |
| `memory` | `add` / `search` / `clear` over a session store |

```bash
# memory tool (file backend by default)
sallm-memory --path /tmp/mem --session demo add --text "The code is 42"
sallm-memory --path /tmp/mem --session demo search --query "code"
```

Lance backend: `uv sync --extra memory`, then `--backend lance` (embeddings via `$SALLM_EMBEDDING_MODEL` / `$SALLM_API_BASE`).

### Context optimizers (economy only)

`Agent` keeps a full transcript on `agent.messages`. Optional `context=` reshapes the **prompt view** only:

```python
from sallm import Agent
from sallm.context import MaxMessages, SummarizeOverflow
from sallm.llm import complete
from sallm.messages import DEFAULT_MODEL, DEFAULT_API_BASE

agent = Agent(tools=..., context=MaxMessages(40))

def summarize(text: str) -> str:
    result = complete(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": "Summarize briefly.\n\n" + text}],
        api_base=DEFAULT_API_BASE,
    )
    return result.get("content") or ""

agent = Agent(
    tools=...,
    context=SummarizeOverflow(
        threshold=2000,
        keep_last=10,
        summarize_fn=summarize,
    ),
)
```

Metrics report `context_messages` (transcript length) vs `prompt_messages` (view length).

### Prompts (visible templates)

All agent wording lives on `Prompt` (`SYSTEM`, multi-step policy, nudges). `Agent` builds `agent.prompt` from tools / multi-step / optional `--system` extra. After each LLM call, `agent.last_prompt` holds the exact message list sent (after any context optimizer).

```python
from sallm import Agent, Prompt

agent = Agent(tools=..., system="Be terse.")
print(agent.prompt.preview())   # labeled dump of templates + full system
# after ask():
print(agent.last_prompt)        # list[dict] | None
```

CLI: `/prompt`, `/prompt system`, `/prompt last`, and `--show-prompt` (print last LLM view after each turn).

## CLI

```bash
uv run sallm chat
uv run sallm chat --tools echo,calc,dig          # default
uv run sallm chat --tools echo,calc,memory,dig   # + memory tool
uv run sallm chat --tools memory --memory-path .sallm-memory
uv run sallm chat --show-prompt                  # dump last LLM view each turn

uv run sallm chat --context max-messages --max-context-messages 40
uv run sallm chat --context summarize --context-threshold 2000 --context-keep-last 10

uv run sallm chat --script tests/fixtures/sample_conversation.txt
uv run sallm chat --script data/sample_questions.txt --tools calc,echo,memory,dig

# tracing
uv run sallm chat --otlp http://localhost:4318 --metrics-port 9464 --trace-debug
```

`--script`: one non-empty line = one turn. Slash commands: `/help`, `/clear`, `/history`, `/prompt`, `/quit`.

Standalone binaries: `sallm-echo`, `sallm-calc`, `sallm-dig`, `sallm-memory`.

### Tracing

Pass `trace=` to `Agent` or use CLI flags. See [docs/tracing-tempo.md](docs/tracing-tempo.md).

```python
from sallm.trace import Tracer, jsonl_sink
from sallm.prom import SessionMetrics

metrics = SessionMetrics(session_id="demo")
metrics.start_server(port=9464)
trace = Tracer(jsonl_sink("/tmp/sallm.jsonl"), debug=True, truncate=512, metrics=metrics)
agent = Agent(tools=..., trace=trace)
```

## Tests

```bash
uv run pytest tests/ -v
```
