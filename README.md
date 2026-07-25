# sallm-agent

Minimal tool-calling agent library (`sallm`) over [LiteLLM](https://github.com/BerriAI/litellm), aimed at local models via Ollama.

Tools are **small CLI programs** run as subprocesses. The model emits shell-style commands in a fenced `run` block — no JSON tool payloads, no native provider tool APIs.

## Setup

```bash
# pull the default local model
ollama pull gemma4:e4b-it-qat
# optional: embedding model for +retrieve recipes
ollama pull qwen3-embedding:0.6b

# install the package (editable) + test deps
uv sync --extra dev
# optional: LanceDB for long-term memory recipes
uv sync --extra memory --extra dev
```

## Tool contract

| Rule | Detail |
|------|--------|
| Identity | Tool name = first argv token (`calc`, `dig`, `echo`) |
| Help | Every tool supports `--help` |
| Args | CLI flags / positionals only (never JSON blobs) |
| Success | exit 0; result = stdout |
| Failure | non-zero exit; stderr/stdout returned as the observation |
| Intermediate | stdout may start with `[intermediate]` (agent keeps going) |

The model invokes tools like this (multiple lines = concurrent processes):

````markdown
```run
calc --expression "2**10"
echo --text hello
```
````

If unsure of flags, it can run `toolname --help` inside a `run` block.

## Library

Register `CliTool` instances (argv prefix + summary). The library ships example CLI apps under `sallm.toolapps`.

```python
import sys
from sallm import Agent
from sallm.tools import CliTool

calc = CliTool(
    name="calc",
    argv=[sys.executable, "-m", "sallm.toolapps.calc"],
    summary="Evaluate a math expression. Flags: --expression EXPR.",
)
agent = Agent(tools={"calc": calc})
print(agent.ask("What is 2**10? Use the calc tool.")["answer"])
```

Runner helpers live in `sallm.tools`: `parse_run_blocks`, `run_tool`, `run_many`, `help_text`.

### Context optimizers

`Agent` keeps a full append-only transcript on `agent.messages`. Pass an optional `context=` object to decide what the model sees each call — the optimizer returns a *view*; it does not rewrite history.

Built-ins live in `sallm.context`:

```python
from sallm import Agent
from sallm.context import MaxMessages, SummarizeOverflow
from sallm.llm import complete
from sallm.messages import DEFAULT_MODEL, DEFAULT_API_BASE

# Drop older turns from the prompt (transcript unchanged)
agent = Agent(tools=..., context=MaxMessages(40))

# When older text exceeds a token budget, inject one summary + keep recent messages
def summarize(text: str) -> str:
    result = complete(
        model=DEFAULT_MODEL,
        messages=[
            {
                "role": "user",
                "content": "Summarize briefly; keep key facts.\n\n" + text,
            }
        ],
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

`summarize_fn` is yours (same model, a smaller one, or a heuristic). Metrics report both `context_messages` (transcript length) and `prompt_messages` (view length).

### Economy + long-term memory (retrieve)

Two roles, separate files — Agent still only calls `context.prepare()`:

| File | Job |
|------|-----|
| `sallm.context` | Economy (`MaxMessages`, `SummarizeOverflow`) |
| `sallm.chunk` | Slice big text before embed |
| `sallm.store` | In-memory `add` / `query` (tests / baseline) |
| `sallm.lance_store` | Local LanceDB backend |
| `sallm.retrieve` | `CompactAndRetrieve` facade (compact → push → pull) |

Push/pull runs **inside** `prepare()` (not via ```run tools). Default retrieval is **session-scoped**; unlock with `--memory-scope all` or `/memory all`.

```python
from sallm.context import MaxMessages
from sallm.lance_store import LanceStore
from sallm.retrieve import CompactAndRetrieve

store = LanceStore(path=".sallm-lancedb", embed_fn=embed, dimensions=1024)
ctx = CompactAndRetrieve(
    MaxMessages(8),
    store,
    session_id="my-session",
    k=4,
    memory_scope="session",
)
agent = Agent(tools=..., context=ctx)
```

Install LanceDB: `uv sync --extra memory`. Pull an embedding model: `ollama pull qwen3-embedding:0.6b`.

CLI:

```bash
uv run sallm chat --context max-messages --max-context-messages 40
uv run sallm chat --context summarize --context-threshold 2000 --context-keep-last 10
uv run sallm chat --context max-messages+retrieve \
  --max-context-messages 8 \
  --embedding-model ollama/qwen3-embedding:0.6b \
  --embedding-dimensions 1024 \
  --chunk-tokens 512 \
  --memory-scope session \
  --lancedb-path .sallm-lancedb
```

`summarize` / `+retrieve` use the same `--model` / `--api-base` for summarization; embeddings use `--embedding-model`. Compare `context msgs` vs `prompt msgs` in the metrics panel. REPL: `/memory session|all` toggles the retrieval guardrail.

## CLI

The chat app registers `echo`, `calc`, and multi-step `dig` (file-backed state) from `sallm.cli.tools`.

```bash
uv run sallm chat
uv run sallm chat --model ollama/gemma4:e4b-it-qat
uv run sallm chat --context max-messages --max-context-messages 40
uv run sallm chat --context summarize --context-threshold 500 --context-keep-last 4
uv run sallm chat --context max-messages+retrieve --max-context-messages 8

# scripted turns: one non-empty line = one user prompt, then exit
uv run sallm chat --script tests/fixtures/sample_conversation.txt
# local Q&A scripts (e.g. load a transcript, then ask questions line by line)
uv run sallm chat --script data/sample_questions.txt

# optional tracing (off by default — zero overhead when unset)
uv run sallm chat --trace /tmp/sallm.jsonl
uv run sallm chat --otlp http://localhost:4318
uv run sallm chat --otlp http://localhost:4318 --metrics-port 9464
uv run sallm chat --otlp http://localhost:4318 --trace-debug --trace-truncate 2000
uv run sallm chat --trace /tmp/sallm.jsonl --otlp http://localhost:4318
```

`--script` feeds each non-empty line into the same chat loop (history preserved across turns). Blank lines are skipped. Packaged example: `tests/fixtures/sample_conversation.txt`. Put your own load-text-then-ask scripts under `data/` (gitignored). `--context max-messages` limits the LLM prompt to the system message plus the last N transcript messages; `--context summarize` injects a summary once older turns exceed `--context-threshold` tokens (estimated). `+retrieve` recipes also chunk overflow / oversized pastes into LanceDB and inject top-k hits (session-filtered by default). Full history remains on `/history`.

Standalone tool binaries (optional): `sallm-echo`, `sallm-calc`, `sallm-dig`.

Slash commands in the REPL: `/help`, `/clear`, `/history`, `/memory session|all`, `/quit`.

### Tracing

Pass `trace=` to `Agent` (a `sallm.trace.Tracer`) or use the CLI flags above. When unset, the agent never builds events.

Each REPL/Agent run has a stable `session.id`. Each `ask()` is one OTLP trace with a single root `ask` span and nested `chat` / `tool …` children.

| kind | span name | meaning |
|------|-----------|---------|
| `turn` | `ask` | one `ask()` call (root; closed at end) |
| `llm` | `chat` | model completion |
| `tool` | `tool <name>` | subprocess call |
| `nudge` / `rejected` | same | continue / early-answer recovery |

```bash
# correlate turns in Grafana TraceQL: { span.session.id = "<id>" }
uv run sallm chat --otlp http://localhost:4318

# Prometheus token / latency charts (scraped on :9464)
uv run sallm chat --otlp http://localhost:4318 --metrics-port 9464

# store truncated prompt/context/completion on spans (for Grafana)
uv run sallm chat --otlp http://localhost:4318 --trace-debug
uv run sallm chat --otlp http://localhost:4318 --trace-debug --trace-truncate 2000
uv run sallm chat --otlp http://localhost:4318 --trace-debug --trace-truncate 0  # unlimited
```

`--trace-truncate` is per content field (each message, completion, etc.), not the whole joined prompt.

Library:

```python
from sallm.trace import Tracer, jsonl_sink
from sallm.prom import SessionMetrics

metrics = SessionMetrics(session_id="demo")
metrics.start_server(port=9464)
trace = Tracer(jsonl_sink("/tmp/sallm.jsonl"), debug=True, truncate=512, metrics=metrics)
agent = Agent(tools=..., trace=trace)
print(trace.session_id)
```

`truncate` is per content field (each message/completion), not the whole joined prompt.

Local Tempo + Prometheus + Grafana: `docker compose up -d` — see [docs/tracing-tempo.md](docs/tracing-tempo.md) (dashboard **sallm session**).

## Tests

```bash
# runner tests always; e2e needs local Ollama + default model
uv run pytest tests/ -v
```
