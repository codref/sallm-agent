# Optimizing prompts and parameters

This guide covers two related knobs:

1. **Prompt profiles** — instructions (and optional demos) for controller, extractor, converse, and rewriter, searched offline and saved as neutral JSON.
2. **Runtime parameters** — token budgets, retrieval mode, top‑k, chunk size, and related CLI flags you tune by hand (or later bake into profile `budgets`).

`sallm chat` never optimizes at startup. It only **loads** a profile. Search runs via `sallm optimize`.

There is no DSPy dependency. The search borrows the idea of propose → evaluate → keep winners (successive halvings), then exports plain JSON the runtime understands.

---

## What you are optimizing

| Piece | Role | Optimized by |
|-------|------|----------------|
| **Controller** instruction | Goal / skill / `retrieval_query` JSON | `sallm optimize --task controller` |
| **Extractor** instruction | Grounded facts JSON | `--task extractor` |
| **Converse** instruction | Extra system guidance for the main ReAct skill | `--task converse` |
| **Rewriter** instruction | Standalone retrieval sentence (when used) | `--task rewriter` |
| **Demonstrations** | Few-shot examples in the profile | Filled by search when present; often empty at first |
| **Budgets** | `prompt_budget`, history/retrieval caps, etc. | Mostly CLI / `ModelProfile` today; profile may store them |
| **Retrieval / embedding** | `raw` \| `instruct` \| `rewrite`, top‑k, chunk size | CLI flags / `EmbeddingProfile` |

Ship empty defaults in `src/sallm/profiles/gemma4-e4b-v1.json`. Non-empty instruction strings override or supplement the built-in baselines for that task.

---

## Quick start

```bash
# 1. Write a small JSONL dataset (see format below)
# 2. Score the baseline only
uv run sallm optimize \
  --dataset data/controller_cases.jsonl \
  --task controller \
  --evaluate-only \
  --out /tmp/unused.json

# 3. Search for better instructions
uv run sallm optimize \
  --dataset data/controller_cases.jsonl \
  --task controller \
  --candidates 4 \
  --seed 0 \
  --model ollama/gemma4:e4b-it-qat \
  --out .sallm/profiles/controller-v1.json

# 4. Use the profile in chat
uv run sallm chat \
  --state-path .sallm/state.db \
  --vector-path .sallm/vectors \
  --profile .sallm/profiles/controller-v1.json
```

Run one `--task` at a time. Merge winning instructions into one profile file by hand if you optimize several tasks (or re-run and copy fields into a combined JSON).

---

## Dataset format (JSONL)

One JSON object per line:

```json
{"id": "c1", "task": "controller", "input": {"user": "What was the lab code?"}, "expected": {"action": "keep", "skill": "converse"}, "mandatory": true}
{"id": "c2", "task": "controller", "input": {"user": "hi"}, "expected": {"action": "keep"}, "mandatory": false}
{"id": "e1", "task": "extractor", "input": {"transcript": "[1] user: code is ZEBRA-7711"}, "expected": {"contains": ["ZEBRA"]}, "mandatory": false}
{"id": "a1", "task": "converse", "input": {"user": "Say hello briefly"}, "expected": {"contains": ["hello", "hi"]}, "mandatory": false}
```

| Field | Meaning |
|-------|---------|
| `id` | Stable case id (optional; auto `case-N` if missing) |
| `task` | `controller` \| `extractor` \| `converse` \| `rewriter` (must match `--task`, unless `--task all`) |
| `input` | Free-form dict shown to the model as `Input: …` |
| `expected` | For JSON tasks: fields that must match exactly (`action`, `skill`, …). For text tasks: `{"contains": ["needle", …]}` |
| `mandatory` | If `true`, a miss scores as a hard failure (−∞); average wins cannot hide it |

**Tips**

- Keep cases **local and small**. Ten clear controller cases beat a hundred noisy ones.
- Mark regression guards as `mandatory` (e.g. “never push an unknown skill”, “always keep on greetings”).
- Use the **same model** you chat with (`gemma4:e4b-it-qat`). Optimizing with a different teacher can invent instructions the 4B model cannot follow.
- Fingerprint the dataset: the artifact records `dataset_fingerprint` so you know which cases produced a profile.

---

## How search works

```text
baseline instruction
        │
        ▼
propose N variants   (Gemma rewrites: clearer/shorter, keep JSON contract)
        │
        ▼
successive halving   (score on a growing train subset; drop the worse half)
        │
        ▼
full-set finalists   (score remaining candidates on all cases)
        │
        ▼
save winner JSON     (instructions + demos + metrics + seed + fingerprint)
```

### Scoring

For each case, the scorer builds:

\[
\text{total} = \text{quality} - \frac{\text{tokens}}{10000} - \frac{\text{latency\_ms}}{100000} - \mathbf{1}_{\text{invalid}}\cdot 0.5
\]

- **quality** — fraction of `expected` fields matched (dict), or fraction of `contains` needles found (text).
- **tokens** — usage from the call plus an estimate of instruction+demo size (penalizes bloated prompts).
- **latency** — soft penalty so slow verbose prompts lose ties.
- **mandatory fail** — quality &lt; 1 on a mandatory case → total ≈ −∞.

So the winner should be **correct**, preferably **short**, and not slower than needed.

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--dataset` | required | JSONL path |
| `--out` | required | Output profile path |
| `--task` | `controller` | Which instruction family to search |
| `--candidates` | `4` | Baseline + proposed rewrites |
| `--seed` | `0` | Shuffle / proposal reproducibility |
| `--model` / `--api-base` | Gemma / Ollama | Student and teacher (same by default) |
| `--evaluate-only` | off | Score baseline only; do not write a searched profile |

`--evaluate-only` still requires `--out` today; the file is unused in that mode. Use it to get a baseline quality number before a long search.

---

## Profile artifact

Example shape (schema version 1):

```json
{
  "schema_version": 1,
  "target_model": "ollama/gemma4:e4b-it-qat",
  "instructions": {
    "controller": "You route a long-running local agent. Reply with ONE JSON object…"
  },
  "demonstrations": {
    "controller": ""
  },
  "budgets": {},
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

Empty instruction strings are ignored; non-empty `converse` text is prepended into the system prompt; controller/extractor instructions replace the built-in control prompts when provided.

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

1. **Baseline chat** with `--state-path` and `/context`. Note failures: wrong skill, bad retrieval query, missed facts, over-long prompts.
2. **Write JSONL** that encodes those failures (mandatory where appropriate).
3. **`--evaluate-only`** on the baseline instruction for that task.
4. **`sallm optimize --search`** with a small candidate count (4–8). Prefer the same Gemma model.
5. **Compare** artifact `metadata.metrics` quality/tokens/latency to the baseline number.
6. **Chat with `--profile`**. Re-run the long-session recall scenario (or `tests/test_e2e_stack_memory.py`).
7. **Keep** the profile only if quality rises and tokens/latency do not regress badly; keep the dataset fingerprint in git with the profile.
8. **Tune budgets** only after instructions stabilize—otherwise you cannot tell whether a fix came from wording or from a larger window.

For parameter-only experiments (no instruction search), change one knob at a time (e.g. `top_k` 2→6) and compare `/context` + answer quality on the same scripted turns (`--script`).

---

## What “good” looks like

| Signal | Good | Bad |
|--------|------|-----|
| Controller JSON | Valid; correct `action`/`skill`; useful `retrieval_query` | Invalid JSON → fallbacks; empty queries when recall is needed |
| Extractor | Few grounded facts; ids exist | Ungrounded facts (runtime drops them) or constant empty `{}` when durable facts were stated |
| Main prompt | `receipt.total_tokens` stable across long sessions | Total climbs with transcript length |
| Profile | Beats baseline on holdout; mandatory cases pass | Higher average score but mandatory miss |
| Search cost | Minutes on a laptop for tens of cases | Huge candidate counts with no held-out cases |

---

## Limits

- Search optimizes **instruction text** (and scoring of that text), not the full multi-call agent loop end-to-end. A great controller line can still fail if Lance is empty or embeddings are wrong.
- Using Gemma as its own teacher can overfit the train lines. Prefer a held-out slice or separate JSONL for smoke checks after loading the profile.
- Profile `budgets` are reserved in the artifact schema; wiring every budget field from JSON into `ModelProfile` may still require code/CLI overrides—treat CLI/`replace(profile, …)` as the source of truth until you confirm a field is loaded.
- Do not expect MIPRO/GEPA-level magic. This is a small, readable successive-halving loop for local Gemma.

---

## Related

- [How the agent works](how-the-agent-works.md) — turn pipeline and token-budget simulation  
- [Skills](skills.md) — skill stack, routing, and tool subsets  
- [README](../README.md) — CLI overview  
- Code: `sallm/optimization/` (dataset, metrics, search, artifacts), `sallm/cli/optimize_cmd.py`, `sallm/models.py`, `sallm/profiles/`
