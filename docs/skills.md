# Skills

Skills are how the agent narrows **how it should behave** for a stretch of the session—without putting every possible mode into every prompt.

A skill is not a tool. Tools are subprocess CLIs the model may call with ```run blocks. A skill chooses the **prompt fragment**, which **tools are visible**, and (when wired) optional step/budget overrides. The active skill sits on a **persisted stack** so long sessions can enter a mode and leave it again.

Shipped default: only **`converse`**. Everything else you register yourself.

---

## Mental model

```text
┌─────────────────────────────────────────┐
│  Skill (mode)                           │
│  - description → controller routing     │
│  - prompt → injected into system text   │
│  - tools → subset of CliTool registry   │
└─────────────────┬───────────────────────┘
                  │ selects / exposes
                  ▼
┌─────────────────────────────────────────┐
│  Tools (actions)                        │
│  calc, echo, dig, your CLIs…            │
│  Invoked only if the model emits ```run │
└─────────────────────────────────────────┘
```

| | Skill | Tool |
|---|--------|------|
| What it is | Session **mode** / policy | **Capability** the model can invoke |
| Who chooses | Controller LLM (JSON) each turn | Main ReAct LLM (```run blocks) |
| Persistence | Stack frames in SQLite | Stateless CLIs (except dig state) |
| In the prompt | Skill `prompt` + goal | Tool names/summaries (possibly filtered) |

Example: a `calc_only` skill might expose only `calc`. The controller switches to that skill when the user asks for math; the ReAct loop then only sees `calc` in “Available tools.”

---

## What a `Skill` contains

```python
from sallm import Skill

Skill(
    name="converse",           # unique id; must match controller output
    description="…",           # shown to the controller for routing
    prompt="…",                # injected into the system prompt when active
    tools=None,                # None = all Agent tools; or ("calc", "echo")
    max_steps=None,            # reserved: prefer Agent(max_steps=…) for now
    budget_overrides={},       # reserved for profile/budget tuning
)
```

| Field | Used today? | Role |
|-------|-------------|------|
| `name` | yes | Stack identity; controller must pick a registered name |
| `description` | yes | Listed in the control prompt as `- name: description` |
| `prompt` | yes | Prepended into system text while this skill is active |
| `tools` | yes | Filters the tool registry for ReAct |
| `max_steps` | not applied yet | Declared for future per-skill step caps |
| `budget_overrides` | not applied yet | Declared for future per-skill token caps |

The registry always ensures **`converse`** exists (auto-registered if you omit it).

---

## The skill stack

With `--state-path`, each session starts with one frame:

```text
depth 0  converse   (root — cannot be popped)
```

Stack operations come from the controller’s `action` field:

| Action | Effect |
|--------|--------|
| `keep` | No stack change; still may update `goal` / `retrieval_query` |
| `push` | Push a new frame if the named skill differs from the active one |
| `replace` | Rewrite the **top** frame’s skill name |
| `pop` | Remove the top frame if depth &gt; 0; root `converse` stays |

Active skill = top of stack. Inspect in chat with `/stack`. Cleared sessions recreate root `converse`.

Without durable state (`state_path=None`), there is no stack: behavior is the legacy full-transcript ReAct agent, effectively always “just chat + all tools you passed in.”

---

## How a skill is selected

Each durable turn, **before** retrieval and ReAct:

1. Controller sees: registered skill descriptions, current goal, active skill, user message.
2. It must reply with JSON only, e.g.:

```json
{
  "goal": "compute powers of two",
  "action": "push",
  "skill": "calc_only",
  "retrieval_query": "math powers of two"
}
```

3. Agent validates `skill` against `SkillRegistry.names()`. Unknown name → **fallback**: keep current skill, mark decision as fallback.
4. `apply_stack_decision` mutates SQLite stack + goal.
5. System prompt is rebuilt with the active skill’s `prompt` and goal.
6. `resolve_tools(active, agent.tools)` builds the tool list for this ReAct round.
7. Retrieval and ReAct run under that mode.

Invalid JSON from the controller also falls back to **keep** (conversation continues).

**Routing policy in the default instruction:** stay on the current skill unless the user clearly changes task; then `push` or `replace`. Prefer short goals and a standalone `retrieval_query` when memory might help.

You can improve routing text offline—see [optimize-prompts.md](optimize-prompts.md) (`--task controller`).

---

## Relation to tools (detail)

```text
Agent(tools=builtin_tools(("calc","echo","dig")))
        │
        │  full registry kept on the Agent
        ▼
active skill.tools is None  ──► ReAct sees calc, echo, dig
active skill.tools = ("calc",) ──► ReAct sees only calc
```

Important consequences:

- **Skills cannot invent tools.** Names in `Skill.tools` must already exist on the Agent. Missing names are skipped.
- **Narrowing tools shortens the system prompt** and reduces wrong-tool calls on small models—that is a main reason to add skills.
- **Tools still opt-in per turn.** Even with `calc` visible, greetings should stay plain text; the base system template still says not to tool-spam.
- **Memory is not a tool.** Long-term recall is the durable pipeline (`--state-path` + retrieve/extract), not a ```run CLI.

---

## Adding a new skill

### 1. Define it

```python
from sallm import Agent, Skill, SkillRegistry
from sallm.tools import builtin_tools

CALC_ONLY = Skill(
    name="calc_only",
    description=(
        "Use when the user needs arithmetic or symbolic math. "
        "Prefer the calc tool; avoid dig/echo."
    ),
    prompt=(
        "Active skill: calc_only.\n"
        "Solve with the calc tool via a ```run block when computation is required.\n"
        "Answer briefly with the numeric result after tool output."
    ),
    tools=("calc",),
)

registry = SkillRegistry([SkillRegistry().get("converse"), CALC_ONLY])
# or: SkillRegistry([CONVERSE, CALC_ONLY])  # from sallm.skills import CONVERSE
```

`description` is for the **controller** (routing). `prompt` is for the **main** model (behavior). Keep both short—Gemma 4 pays for every token every turn.

### 2. Pass the registry into the Agent

```python
agent = Agent(
    tools=builtin_tools(("calc", "echo", "dig")),
    skills=registry,
    state_path=".sallm/state.db",
    vector_path=".sallm/vectors",
    session_id="math1",
)
agent.ask("What is 2**10? Use the calculator.")
# Controller should push/replace → calc_only; ReAct only sees calc.
print(agent.stack)  # frames with converse then calc_only (if pushed)
```

### 3. Teach the controller (optional but recommended)

Add JSONL cases so optimize/search prefers correct routing:

```json
{"id": "m1", "task": "controller", "input": {"user": "What is 2**10?"}, "expected": {"action": "push", "skill": "calc_only"}, "mandatory": true}
{"id": "m2", "task": "controller", "input": {"user": "thanks, enough math"}, "expected": {"action": "pop"}, "mandatory": false}
{"id": "m3", "task": "controller", "input": {"user": "hi"}, "expected": {"action": "keep", "skill": "converse"}, "mandatory": true}
```

Then:

```bash
uv run sallm optimize --dataset data/skills_calc.jsonl --task controller \
  --out .sallm/profiles/controller-calc.json
uv run sallm chat --state-path .sallm/state.db --profile .sallm/profiles/controller-calc.json
```

### 4. Inspect while chatting

```text
/stack     → depths and skill names
/state     → goal + session
/prompt    → confirm skill prompt landed in system text
/context   → token spend (narrower tools → smaller system section)
```

---

## End-to-end turn with skills

```text
User: "Ignore the treasure game. Just compute 3+4."

1. Persist user message
2. Controller JSON → action=push, skill=calc_only, goal="compute 3+4"
3. Stack: converse → calc_only
4. System rebuilt with calc_only.prompt; tools = {calc}
5. Retrieve (optional) under that goal
6. ReAct: model emits ```run / calc --expression "3+4"
7. Answer + extract/index as usual
```

Later: “Thanks, let’s just talk.” → controller `pop` → back to `converse` with echo/dig visible again (if they were on the Agent).

---

## Design guidelines

1. **Few skills.** Each extra name is a routing choice the 4B controller can get wrong.
2. **Descriptions must be distinguishable.** If two skills both say “help the user,” the controller will thrash.
3. **Prefer `keep`.** Push/replace only on clear task changes; pop when the user leaves the task.
4. **Tools subsets should match the description.** Do not advertise “math only” while leaving `dig` visible.
5. **Root stays `converse`.** Treat it as the safe default for chitchat and open-ended work.
6. **Skills are not plugins discovered at runtime.** You construct a `SkillRegistry` in code (or your app’s config) and pass it in—explicit injection, no magic loaders.

---

## Library checklist

```python
from sallm import Agent, Skill, SkillRegistry
from sallm.skills import CONVERSE
from sallm.tools import builtin_tools, CliTool

# 1. Tools first (capabilities)
tools = builtin_tools(("calc", "echo"))
# tools["mine"] = CliTool(name="mine", argv=[...], summary="...")

# 2. Skills second (modes over those capabilities)
skills = SkillRegistry([
    CONVERSE,
    Skill(
        name="echo_lab",
        description="Repeat or format text with the echo tool only.",
        prompt="Active skill: echo_lab.\nUse echo for verbatim repeats.",
        tools=("echo",),
    ),
])

# 3. Durable agent
agent = Agent(
    tools=tools,
    skills=skills,
    state_path=".sallm/state.db",
    vector_path=".sallm/vectors",
    session_id="demo",
)
```

---

## Related

- [How the agent works](how-the-agent-works.md) — full turn pipeline  
- [Optimizing prompts and parameters](optimize-prompts.md) — tuning the controller for better skill choice  
- Code: `sallm/skills.py`, `sallm/control.py`, `sallm/turn.py` (`apply_stack_decision`), `sallm/state/repository.py` (stack CRUD)
