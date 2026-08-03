# sallm-agent

Minimal tool-calling agent (`sallm`) for **local small LLMs** (default: Gemma 4 4B via Ollama).

**Focus:** keep context predictable over long sessions — durable SQLite state, LanceDB retrieval, a skill stack, and a visible `ContextReceipt` that explains token spend.

Not an invisible RAG black box: raw messages stay canonical; vectors are a rebuildable index; derived facts must cite source message ids.

## Setup

```bash
ollama pull gemma4:e4b-it-qat
ollama pull qwen3-embedding:0.6b

uv sync --extra dev
```

Core deps include `peewee` (SQLite ORM) and `lancedb` (vector index). There is **no** DSPy/Pydantic dependency.

## Examples

- [`examples/imap_inbox/`](examples/imap_inbox/) — durable IMAP inbox Q&A (CLI tools + long-session recall). See the docstring in `agent.py`.

## Turn pipeline

```
user → persist raw message
  → goal/skill control (small JSON call)
  → vector retrieve (Qwen embed + LanceDB)
  → budgeted prompt + ReAct ```run tools
  → persist answer
  → extract grounded facts + index chunks
```

## Library

```python
from sallm import Agent, RetrievalConfig, Skill, SkillRegistry
from sallm.tools import builtin_tools

agent = Agent(
    tools=builtin_tools(("calc", "echo")),
    state_path="/tmp/sallm/state.db",
    vector_path="/tmp/sallm/vectors",
    session_id="demo",
    retrieval=RetrievalConfig(
        memory_gate=True,
        search_mode="dense",  # or "hybrid"
        use_instruct=True,
        use_rewrite=False,
        use_hyde=False,
    ),
)
result = agent.ask("Remember the code is PURPLE-42.")
print(result["answer"])
print(result["receipt"])  # ContextReceipt as dict
print(result["goal"], result["stack"])
```

Resume by reusing `state_path` + `session_id`.

### VectorStore contract

Implement `upsert` / `search` / `delete_session` / `close` (see `sallm.memory.types.VectorStore`). Default: `LanceVectorStore`. A future **pgvector** adapter can satisfy the same dataclasses (`VectorRecord`, `VectorQuery`, `VectorHit`) without changing the agent.

SQLite stores chunk text + `indexed` flags; LanceDB is rebuilt from those rows after a crash.

### Skills

Default skill is `converse`. Register more with `SkillRegistry` (name, description, prompt fragment, optional tool subset).

### Compiled profiles

Neutral JSON under `sallm/profiles/` (instructions + demos + budgets). Offline:

```bash
uv run sallm optimize --dataset data/cases.jsonl --task controller --out /tmp/profile.json
```

`sallm chat` never optimizes at startup; it only loads a profile.

## CLI

```bash
# Durable long session (recommended)
uv run sallm chat \
  --state-path .sallm/state.db \
  --vector-path .sallm/vectors \
  --session long1 \
  --retrieval-query instruct \
  --search dense \
  --memory-gate \
  --extract waterfall \
  --tools echo,calc

uv run sallm chat --show-prompt
uv run sallm chat --script tests/fixtures/sample_conversation.txt
```

Slash commands: `/help`, `/clear`, `/history`, `/prompt`, `/state`, `/stack`, `/memory`, `/context`, `/quit`.

### Tool contract

| Rule | Detail |
|------|--------|
| Identity | Tool name = first argv token |
| Help | Every tool supports `--help` |
| Args | CLI flags only (no JSON blobs) |
| Intermediate | stdout may start with `[intermediate]` |

````markdown
```run
calc --expression "2**10"
```
````

Shipped tools: `echo`, `calc`, `dig`.

### Legacy context optimizers

Still available without durable state: `--context max-messages|summarize`. Prefer `--state-path` + retrieval for hour-scale sessions.

## Tests

```bash
uv run pytest tests/ -v
# E2E needs Ollama + gemma4:e4b-it-qat (+ qwen3-embedding:0.6b for stack memory)
```

## Limits

Retrieval improves grounding; it does **not** guarantee the model never invents facts. Source-tagged memory and `ContextReceipt` make misses inspectable.

## How it works

Walkthrough with an example, stage-by-stage flow, token-budget simulation, and hypothesis checks: [docs/how-the-agent-works.md](docs/how-the-agent-works.md).

When a script turn (briefing / transcript) is larger than the history budget: [docs/oversized-briefings.md](docs/oversized-briefings.md).

Offline prompt/parameter tuning: [docs/optimize-prompts.md](docs/optimize-prompts.md).

Skills (selection, stack, tools): [docs/skills.md](docs/skills.md).
