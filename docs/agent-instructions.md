# Agent instructions: integrate `sallm` into a project

Copy this file into the consuming project's agent context (e.g. `.cursor/rules/`, `AGENTS.md`, or a skill). Follow it when adding or changing code that uses **sallm-agent**.

---

## What this library is

`sallm` is a **minimal tool-calling agent** for **local small LLMs** (default: Gemma 4 4B via Ollama).

It keeps long sessions predictable with:

- **SQLite** — canonical raw messages, goals, skill stack, chunks
- **LanceDB** — rebuildable vector index (not source of truth)
- **Skills** — session modes (prompt fragment + optional tool subset)
- **ContextReceipt** — visible token budget for each turn

It is **not** an invisible RAG black box. Do not wrap it in layers that hide receipts, invent memory stores, or replace the message transcript with summaries by default.

Python: `>=3.12`. Package name: `sallm-agent`. Import package: `sallm`.

---

## Prerequisites (host)

Before coding against `sallm`, ensure:

```bash
ollama pull gemma4:e4b-it-qat
ollama pull qwen3-embedding:0.6b
# Ollama listening at http://localhost:11434 (default)
```

Without Ollama + these models, `agent.ask(...)` will fail at runtime.

---

## Install into the project

Prefer editable / git dependency (package is not assumed published to PyPI):

```bash
# uv
uv add git+https://github.com/codref/sallm-agent.git
# or path during local development
uv add --editable /path/to/sallm-agent

# pip
pip install "sallm-agent @ git+https://github.com/codref/sallm-agent.git"
```

Core deps pulled in: `litellm`, `typer`, `rich`, `peewee`, `lancedb`.  
There is **no** DSPy or Pydantic dependency — do not add them “for sallm.”

Ignore / do not commit runtime state:

```
.sallm/
*.db
```

---

## Default integration pattern (durable session)

Always prefer durable state for real apps. Resume = same `state_path` + `session_id`.

```python
from pathlib import Path

from sallm import Agent, RetrievalConfig, Skill, SkillRegistry
from sallm.tools import CliTool, builtin_tools

DATA = Path(".sallm")
DATA.mkdir(exist_ok=True)

agent = Agent(
    tools=builtin_tools(("calc", "echo")),  # or "all" / "none" / custom CliTool map
    state_path=DATA / "state.db",
    vector_path=DATA / "vectors",
    session_id="demo",
    retrieval=RetrievalConfig(
        memory_gate=True,
        search_mode="dense",  # or "hybrid"
        use_instruct=True,
        use_rewrite=False,
        use_hyde=False,
    ),
    # optional: model="ollama/gemma4:e4b-it-qat",
    # optional: api_base="http://localhost:11434",
    # optional: max_steps=5,
)

result = agent.ask("Remember the code is PURPLE-42.")
answer = result["answer"]
receipt = result["receipt"]  # ContextReceipt as dict — keep for debugging/UI
goal = result["goal"]
stack = result["stack"]      # list[{"skill", "depth", "note"}]
```

### `ask()` return shape

| Key | Meaning |
|-----|---------|
| `answer` | Final assistant text |
| `steps` | ReAct / tool steps for the turn |
| `metrics` | Token/usage summary |
| `receipt` | Prompt budget breakdown (or `None`) |
| `goal` | Current session goal string |
| `stack` | Active skill frames |
| `stopped` | Present only if the turn stopped early |

### Lifecycle rules

- Construct **one** `Agent` per session (or reuse after restart with same paths/ids).
- Call `agent.ask(text)` for each user turn — do not manually append to `agent.messages` for durable mode.
- `agent.clear()` wipes the session in SQLite + vectors and resets to root skill `converse`.
- Without `state_path`, the agent falls back to an in-memory legacy loop (full transcript + optional trim). Prefer `state_path` for anything longer than a few turns.

---

## Public API (use these)

```python
from sallm import (
    Agent,
    CliTool,
    CompiledProfile,
    ContextReceipt,
    EmbeddingProfile,
    LanceVectorStore,
    ModelProfile,
    Prompt,
    RetrievalConfig,
    Skill,
    SkillRegistry,
    ToolResult,
    VectorHit,
    VectorQuery,
    VectorRecord,
    VectorStore,
    DEFAULT_MODEL,
    DEFAULT_API_BASE,
)
from sallm.tools import builtin_tools
```

Do **not** import private modules (`turn`, `legacy_ask`, `control`, etc.) from app code unless extending the library itself.

---

## Skills

Skills are **modes**, not tools. Default skill is always `converse` (auto-registered).

```python
skills = SkillRegistry([
    Skill(
        name="calc_only",
        description="User wants math; use calc tool only.",
        prompt="Active skill: calc_only.\nSolve with the calc tool; keep answers short.",
        tools=("calc",),  # None = all Agent tools
    ),
    # converse is added automatically if omitted
])

agent = Agent(
    tools=builtin_tools(("calc", "echo")),
    skills=skills,
    state_path=".sallm/state.db",
    session_id="math",
)
```

| Field | Role |
|-------|------|
| `name` | Must match controller routing |
| `description` | Shown to the controller each turn |
| `prompt` | Injected into system prompt while active |
| `tools` | Subset of registered tools (`None` = all) |
| `max_steps` / `budget_overrides` | Declared but not applied yet — set `Agent(max_steps=…)` instead |

Do not confuse skills with tools. Tools are CLI subprocesses invoked via `` ```run `` blocks.

---

## Custom tools

Tools are **CLI processes**, not Python callables.

Contract:

| Rule | Detail |
|------|--------|
| Identity | Tool name = first argv token the model uses |
| Help | Binary must support `--help` |
| Args | CLI flags only (no JSON blobs) |
| Intermediate | stdout may start with `[intermediate]` for multi-round |

```python
from sallm import Agent, CliTool

my_tool = CliTool(
    name="weather",
    argv=["/usr/local/bin/my-weather"],
    summary="Get weather. Flags: --city CITY",
)

agent = Agent(
    tools={"weather": my_tool, **builtin_tools(("echo",))},
    state_path=".sallm/state.db",
    session_id="ops",
)
```

Model invokes tools like:

````markdown
```run
calc --expression "2**10"
```
````

Shipped builtins: `echo`, `calc`, `dig` via `builtin_tools(...)`.

---

## Retrieval knobs

`RetrievalConfig` (or constructor aliases):

| Knob | Typical default | Notes |
|------|-----------------|-------|
| `memory_gate` | `True` | Skip indexing low-value chatter |
| `search_mode` | `"dense"` | or `"hybrid"` |
| `use_instruct` | `True` | Qwen instruct-prefixed queries |
| `use_rewrite` | `False` | Extra LLM rewrite — costlier |
| `use_hyde` | `False` | Hypothetical doc — costlier |

CLI-equivalent labels: `raw` \| `instruct` \| `rewrite` \| `hyde` \| `rewrite+hyde`.

Token budgets live on `ModelProfile` (Gemma defaults):

| Budget | Default | Role |
|--------|---------|------|
| `prompt_budget` | 4096 | Soft cap for compiled prompt |
| `recent_history_tokens` | 1800 | Verbatim recent transcript |
| `retrieval_tokens` | 800 | Cap for injected vector hits |

**Important:** A huge single user turn (briefing/transcript) is stored and indexed, but may **drop out of the recent-history window**. Later answers depend on retrieval under the 800-token cap. Do not assume “it was said once → it stays in the prompt forever.” Prefer short durable facts, or raise budgets / `top_k` deliberately and measure via `receipt`.

---

## Vector store contract (optional swap)

Default: `LanceVectorStore` beside `state_path`.

To plug another backend, implement `VectorStore`: `upsert` / `search` / `delete_session` / `close` using `VectorRecord`, `VectorQuery`, `VectorHit` (`sallm.memory.types`). Pass `vector_store=...` into `Agent`. Do not change agent call sites for a backend swap.

SQLite keeps chunk text + `indexed` flags; Lance can be rebuilt after a crash.

---

## Profiles / offline optimize

Neutral JSON profiles live under `sallm/profiles/`. Chat/runtime **loads** a profile; it does **not** optimize at startup.

```bash
uv run sallm optimize --dataset data/cases.jsonl --task controller --out /tmp/profile.json
```

Load via `compiled_profile=CompiledProfile(...)` only when the project already has a compiled artifact. Do not run optimize in request paths.

---

## CLI (smoke / debug)

```bash
uv run sallm chat \
  --state-path .sallm/state.db \
  --vector-path .sallm/vectors \
  --session long1 \
  --retrieval-query instruct \
  --search dense \
  --memory-gate \
  --extract waterfall \
  --tools echo,calc
```

Slash commands: `/help`, `/clear`, `/history`, `/prompt`, `/state`, `/stack`, `/memory`, `/context`, `/quit`.

Use the CLI to verify Ollama + retrieval before wiring the library into the app.

---

## Turn pipeline (mental model)

```text
user → persist raw message (SQLite)
  → goal/skill control (small JSON LLM call)
  → vector retrieve (embed + LanceDB)
  → budgeted prompt + ReAct ```run tools
  → persist answer
  → extract grounded facts + index chunks
```

Derived facts must cite source message ids. Vectors are a rebuildable index, not canonical history.

---

## Do / don't for implementing agents

**Do**

- Use `state_path` + `session_id` for any multi-turn product surface
- Surface `result["receipt"]` (and optionally `goal` / `stack`) in logs or debug UI
- Register project-specific `Skill`s with clear controller `description`s
- Add tools as CLI binaries/`CliTool` entries with `--help` and flag args
- Keep `.sallm/` out of git
- Keep answers inspectable: prefer retrieval + receipt over silent summarization

**Don't**

- Call `ask()` without Ollama models available in prod/dev docs
- Treat LanceDB as the source of truth
- Stuff entire meeting transcripts into every follow-up and expect them in-prompt
- Introduce Pydantic/DSPy wrappers “because agents usually use them”
- Optimize prompts on every request
- Bypass `Agent` to poke SQLite/Lance unless writing library extensions or migrations
- Use the legacy no-`state_path` path for hour-scale sessions

---

## Minimal checklist for a PR that adds sallm

1. Dependency on `sallm-agent` declared; Python ≥ 3.12
2. Ollama models documented in README / runbook
3. Durable paths + stable `session_id` strategy
4. Tools/skills registered for the product domain
5. `.sallm/` (or chosen data dir) gitignored
6. At least one smoke path: library `ask()` or `sallm chat --script …`
7. Failures / empty answers inspectable via `receipt` + `/memory`-style debugging

---

## Further reading (upstream repo)

- `README.md` — setup, library snippet, CLI
- `docs/how-the-agent-works.md` — stage-by-stage + budgets
- `docs/skills.md` — skill stack and controller actions
- `docs/oversized-briefings.md` — long transcript failure mode
- `docs/optimize-prompts.md` — offline profile tuning
