# Optimizing prompts and parameters

This guide covers two related knobs:

1. **Prompt profiles** — instructions (and optional demos) plus **budgets** (`think`, temperature, token caps) in one JSON file.
2. **Retrieval parameters** — mode, top‑k, chunk size (CLI / `EmbeddingProfile`).

`sallm chat` never optimizes at startup. It only **loads** a profile. Search runs via `sallm optimize` (optional `--profile`; default is the packaged Gemma baseline).

There is no DSPy dependency. Search compiles the candidate instruction into the **same program chat runs**, scores it on every dataset case, then rewrites from the **fail traces** until mandatory cases pass (or `--rounds` is exhausted).

---

## What you are optimizing

| Piece | Role | Optimized by |
|-------|------|----------------|
| **Controller** instruction | Goal / skill / `retrieval_query` JSON | `sallm optimize --task controller` |
| **Extractor** instruction | Grounded facts JSON (turn extract) | `--task extractor` |
| **Ingest** instruction | Meaning-first facts from `Agent.remember` blocks | `--task ingest` |
| **Converse** instruction | Extra system guidance for the main ReAct skill | `--task converse` |
| **Rewriter** instruction | Standalone retrieval sentence (when used) | `--task rewriter` |
| **Demonstrations** | Few-shot examples in the profile | Filled by search when present; often empty at first |
| **Budgets** | `think`, `temperature`, token caps | `--search-budgets` / `--budgets-only` (one-at-a-time grid on generation knobs) |
| **Retrieval / embedding** | `raw` \| `instruct` \| `rewrite`, top‑k, chunk size | CLI flags / `EmbeddingProfile` |

Ship empty defaults in `src/sallm/profiles/gemma4-e4b-v1.json`. Non-empty instruction strings override or supplement the built-in baselines for that task.

---

## Quick start

```bash
# 1. Write a small JSONL dataset (see format below)
# 2. Score the current profile instruction (per-case)
uv run sallm optimize \
  --dataset examples/small_chat/opt_cases.jsonl \
  --profile examples/small_chat/profile.json \
  --task converse \
  --evaluate-only

# 3b. Or search generation budgets only (temperature / max tokens / think)
uv run sallm optimize \
  --dataset examples/small_chat/opt_cases.jsonl \
  --profile examples/small_chat/profile.json \
  --task converse \
  --budgets-only \
  --out examples/small_chat/profile.budgets.json

# 3. Rewrite from failures (does not touch --profile unless you pass --out / --in-place)
#    Use a stronger --teacher than the 0.8B student so ```run examples survive
uv run sallm optimize \
  --dataset examples/small_chat/opt_cases.jsonl \
  --profile examples/small_chat/profile.json \
  --task converse \
  --teacher ollama_chat/gemma4:e4b-it-qat \
  --out examples/small_chat/profile.opt.json \
  --rounds 6 \
  --seed 0

# Optional: instruction search then budget search
uv run sallm optimize \
  --dataset examples/small_chat/opt_cases.jsonl \
  --profile examples/small_chat/profile.json \
  --task converse \
  --teacher ollama_chat/gemma4:e4b-it-qat \
  --search-budgets \
  --out examples/small_chat/profile.opt.json

# 4. Use the written profile in chat
uv run sallm chat \
  --state-path .sallm/state.db \
  --vector-path .sallm/vectors \
  --profile examples/small_chat/profile.opt.json
```

`--model` defaults to `target_model` in the profile. Omit `--profile` to continue an existing `--out` file when present, else the packaged empty baseline (`sallm/profiles/gemma4-e4b-v1.json`, same as `sallm chat`). Writing merges into the destination if it already exists, so optimizing `--task controller` does not wipe a prior `instructions.converse`. Search does **not** overwrite the input unless you pass `--in-place`. `--evaluate-only` never writes. Run one `--task` at a time; other instruction keys are left intact.

---

## Dataset format (JSONL)

One JSON object per line:

```json
{"id": "c1", "task": "controller", "input": {"user": "What was the lab code?"}, "expected": {"action": "keep", "skill": "converse"}, "mandatory": true}
{"id": "c2", "task": "controller", "input": {"user": "hi"}, "expected": {"action": "keep"}, "mandatory": false}
{"id": "e1", "task": "extractor", "input": {"transcript": "[1] user: code is ZEBRA-7711"}, "expected": {"contains": ["ZEBRA"]}, "mandatory": false}
{"id": "i1", "task": "ingest", "input": {"transcript": "[1] ingest: docker login then ssh deploy@203.0.113.10"}, "expected": {"contains": ["203.0.113.10"]}, "mandatory": false}
{"id": "a1", "task": "converse", "input": {"user": "What is 17 times 19? Use the calc tool."}, "expected": {"tool": "calc", "argv_contains": ["17", "19"], "observation_contains": ["323"]}, "mandatory": true}
```

| Field | Meaning |
|-------|---------|
| `id` | Stable case id (optional; auto `case-N` if missing) |
| `task` | `controller` \| `extractor` \| `ingest` \| `converse` \| `rewriter` (must match `--task`, unless `--task all`) |
| `input` | Free-form dict. Converse: `user`, optional `history`, optional `retrieved` (injected as `[Retrieved memory]`). Controller: `user`. Extractor/ingest: `transcript`. |
| `expected` | JSON: exact fields (`action`, `skill`, `retrieval_query`) plus `contains`, `retrieval_query_contains`, `retrieval_query_nonempty`, `absent`. Converse: `contains`, `tool` / `argv_contains` / `observation_contains`, `absent`, `no_tool` |
| `mandatory` | If `true`, a miss scores as a hard failure (−∞); average wins cannot hide it |

**Tips**

- Keep cases **local and small**. Ten clear controller cases beat a hundred noisy ones.
- Mark regression guards as `mandatory` (e.g. “never push an unknown skill”, “always keep on greetings”).
- Use the **student** you chat with (`--model` / `target_model`). Use a **stronger `--teacher`** to rewrite instructions — a 0.8B teacher will strip ```run fences.
- Fingerprint the dataset: the artifact records `dataset_fingerprint` so you know which cases produced a profile.

---

## How search works

```text
load profile.json  (instruction[task] + budgets)
        │
        ▼
compile candidate into the runtime program
  converse: Prompt.system() + optional history + user  (then parse/run ```run)
  JSON tasks: instruction + Input dict
        │
        ▼
evaluate ALL cases
        │
        ├── all mandatory pass → write profile, stop
        │
        ▼
collect fail traces (id, expected, got, parsed commands, tool obs)
        │
        ▼
teacher rewrites instruction to fix those fails
        │
        ▼
evaluate rewrite on ALL cases; keep if better
        │
        ▼
repeat up to --rounds; print winner (write only with --out / --in-place)
```

“Better” = fewer mandatory misses, then higher mean quality, then fewer tokens.

`--evaluate-only` is one full-set eval of the profile instruction (no rewrite, no write). It prints a progress bar, per-case ok/FAIL lines, then a summary panel. Search mode also prints live case marks, teacher spinner, and per-round quality / fails.

Student completions use profile `budgets` (`think` for converse/rewriter, `think=False` for JSON tasks, `temperature`, matching max-token cap). The teacher rewrite always uses `think=False` and a separate `--teacher` model when set.

### Scoring

For each case, the scorer builds:

\[
\text{total} = \text{quality} - \frac{\text{tokens}}{10000} - \frac{\text{latency\_ms}}{100000} - \mathbf{1}_{\text{invalid}}\cdot 0.5
\]

- **quality** — JSON: fraction of `expected` fields matched. Converse/text: mean of `contains` hits, parsed `tool` / `argv_contains` / `observation_contains` (after running ```run), plus `absent` and `no_tool` when set.
- **tokens** — usage from the call plus an estimate of instruction+demo size (penalizes bloated prompts).
- **latency** — soft penalty so slow verbose prompts lose ties.
- **mandatory fail** — quality &lt; 1 on a mandatory case → total ≈ −∞.

So the winner should be **correct**, preferably **short**, and not slower than needed. A prose `256` for `2**8` fails if the case requires `tool: calc` and observation `256`.

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--dataset` | required | JSONL path |
| `--profile` | packaged `gemma4-e4b-v1.json` | Input CompiledProfile JSON; omit to optimize the chat default |
| `--out` | (none) | Write merged profile here |
| `--in-place` | off | Write the winner back into `--profile` |
| `--task` | `controller` | Which instruction to rewrite |
| `--rounds` | `6` | Failure-driven rewrite iterations |
| `--seed` | `0` | Teacher prompt diversity |
| `--model` / `--api-base` | profile `target_model` / Ollama | Student (the model that must pass the cases) |
| `--teacher` | student `--model` | Model that rewrites the instruction from fail traces |
| `--tools` | `echo,calc` | Tool registry used when scoring converse ```run blocks |
| `--evaluate-only` | off | Score only; do not rewrite or write |
| `--search-budgets` | off | After instruction search, also tune generation budgets |
| `--budgets-only` | off | Skip instruction rewrite; search budgets only |

---

## Profile artifact

Example shape (schema version 1):

```json
{
  "schema_version": 1,
  "target_model": "ollama/gemma4:e4b-it-qat",
  "instructions": {
    "controller": "You route a long-running local agent. Reply with ONE JSON object…",
    "extractor": "",
    "ingest": ""
  },
  "demonstrations": {
    "controller": "",
    "extractor": "",
    "ingest": ""
  },
  "budgets": {
    "prompt_budget": 2048,
    "max_output_tokens": 512,
    "recent_history_tokens": 900,
    "retrieval_tokens": 400,
    "control_max_tokens": 128,
    "extract_max_tokens": 192,
    "think": false,
    "temperature": 0.2
  },
  "metadata": {
    "dataset_fingerprint": "a1b2c3d4e5f60708",
    "metrics": {
      "c0": {"score": 0.91, "quality": 1.0, "tokens": 1200, "latency_ms": 800}
    },
    "seed": 0,
    "content_digest": "…"
  }
}
```

Runtime load path:

- CLI: `--profile path/to.json`
- Default packaged file: `sallm/profiles/gemma4-e4b-v1.json` (empty instructions = built-in baselines)
- Library: `CompiledProfile.load(path)` passed into `Agent(..., compiled_profile=…)`

Profiles do **not** have to live under `src/sallm/profiles/`. Any path works — including a JSON file next to your app — or build `CompiledProfile(...)` in Python and pass `compiled_profile=` at `Agent` construction. See [`examples/small_chat/`](../examples/small_chat/) for a small-model profile (`think: false` and token caps in `budgets`). For Ollama thinking models prefer LiteLLM’s `ollama_chat/` prefix so `think` hits `/api/chat`.

Empty instruction strings are ignored; non-empty `converse` text is appended **after** the built-in SYSTEM block (so it can override “skip tools if you already know”); controller/extractor/ingest instructions replace the built-in control prompts when provided (`ingest` → `Agent.remember` / `INGEST_INSTRUCTION`).

**Do not** put DSPy modules, pickles, or Pydantic models in this file. Only portable strings and numbers.

---

## Runtime parameters (manual tuning)

These affect token economy and retrieval quality even with a fixed profile.

### Model budgets (`ModelProfile`)

Defaults for Gemma 4:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `prompt_budget` | 4096 | Soft cap for the compiled main prompt |
| `recent_history_tokens` | 1800 | Verbatim tail kept in the prompt |
| `retrieval_tokens` | 800 | Cap for injected memory hits |
| `control_max_tokens` | 256 | Max generation for goal/skill JSON |
| `extract_max_tokens` | 384 | Max generation for fact extraction |
| `max_output_tokens` | 1024 | Main answer/tool-step generation ceiling |
| `think` | `None` | Ollama thinking: `false` / `true` / `"low"` / `"medium"` / `"high"` / `"max"`. Thinking and the reply share `max_output_tokens`. |
| `think_hint` | `None` | Extra system text injected only when thinking is on |
| `temperature` | `None` | Sampling; omit to use the provider default |

Tighter history → more reliance on retrieval. Wider history → fewer retrievals needed, higher tokens per turn. Validate with `/context` (`ContextReceipt`).

In code:

```python
from dataclasses import replace
from sallm import Agent
from sallm.models import resolve_model_profile

profile = replace(
    resolve_model_profile(),
    prompt_budget=3072,
    recent_history_tokens=1200,
    retrieval_tokens=1000,
)
agent = Agent(state_path="…", profile=profile, …)
```

### Embedding / retrieval (`EmbeddingProfile` + CLI)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--embedding-model` | `ollama/qwen3-embedding:0.6b` | Must match how the index was built |
| `--embedding-dimensions` | 1024 | Must match the model (Qwen3-0.6B) |
| `--top-k` | 4 | How many hits enter the retrieval section |
| chunk tokens / overlap | 512 / 64 | Granularity of stored passages |
| `--retrieval-query` | `instruct` | `raw` \| `instruct` \| `rewrite` \| `hyde` \| `rewrite+hyde` |
| `--search` | `dense` | `dense` \| `hybrid` (BM25+dense via Lance) |
| `--memory-gate` / `--no-memory-gate` | on | Skip indexing short interrogatives |

**When to change retrieval mode**

- `instruct` — default; best for Qwen query-side formatting.
- `rewrite` — use when user turns are messy; control’s `retrieval_query` becomes the search sentence (then instruct-wrapped).
- `raw` — debugging or non-instruction embedding models.

Re-index (new session path or rebuild) if you change embedding model or dimensions. Old vectors are not compatible.

### Agent loop knobs

| Flag | Default | Effect |
|------|---------|--------|
| `--max-steps` | 5 | Tool rounds per user turn |
| `--multi-step` / `--no-multi-step` | on | Allow chained tool rounds |
| `--tools` | echo,calc,dig | Smaller tool lists → shorter system prompts |

---

## A practical workflow

1. **Baseline chat** with `--state-path` and `/context`. Note failures: wrong skill, missed facts, tools not fenced.
2. **Write JSONL** that encodes those failures (mandatory where appropriate).
3. **`--evaluate-only --profile …`** and read per-case FAIL lines.
4. **`sallm optimize --profile … --task … --out path.json`** so the teacher rewrites from those fails.
5. **Compare** `metadata.metrics` and remaining mandatory misses.
6. **Chat with `--profile path.json`**. Re-run the scripted session.
7. **Keep** the profile if mandatory cases pass (or misses drop) and tokens/latency do not regress badly.
8. **Optionally `--budgets-only`** to tune temperature / max tokens / think on the fixed instruction.

For parameter-only experiments (no instruction search), change one knob at a time (e.g. `top_k` 2→6) and compare `/context` + answer quality on the same scripted turns (`--script`).

---

## What “good” looks like

| Signal | Good | Bad |
|--------|------|-----|
| Controller JSON | Valid; correct `action`/`skill`; useful `retrieval_query` | Invalid JSON → fallbacks; empty queries when recall is needed |
| Extractor | Few grounded facts; ids exist | Ungrounded facts (runtime drops them) or constant empty `{}` when durable facts were stated |
| Main prompt | `receipt.total_tokens` stable across long sessions | Total climbs with transcript length |
| Profile | Beats baseline on holdout; mandatory cases pass | Higher average score but mandatory miss |
| Search cost | Minutes on a laptop for tens of cases | Huge `--rounds` with no held-out cases |

---

## Limits

- Converse search scores one generation: compiled `Prompt.system()` + optional `input.history` + user → parse/run ```run. Put remember-after-calc cases in the JSONL or search will overfit tools and leak “The result is” into later turns. It does not run controller or retrieval.
- `--teacher` should be stronger than a 0.8B student. Using the student as its own teacher often deletes the ```run example the metric needs.
- `budgets` (`think`, `temperature`, token caps) are applied from the JSON at Agent init and during optimize eval. Generation knobs can be searched with `--search-budgets` / `--budgets-only` (one-at-a-time grid). History/retrieval window sizes still need chat-level measurement — they do not change offline case scoring.
- This is a GEPA-style fail-trace rewrite loop, not MIPRO/GEPA from the DSPy package.

---

## Related

- [How the agent works](how-the-agent-works.md) — turn pipeline and token-budget simulation  
- [Skills](skills.md) — skill stack, routing, and tool subsets  
- [README](../README.md) — CLI overview  
- Code: `sallm/optimization/` (dataset, metrics, search, artifacts), `sallm/cli/optimize_cmd.py`, `sallm/models.py`, `sallm/profiles/`
