# Spec: self-healing through offline optimization

**Status:** plan / design only — **not implemented**.  
**Related:** [optimize-prompts.md](optimize-prompts.md), [how-the-agent-works.md](how-the-agent-works.md), [agent-instructions.md](agent-instructions.md)

This document proposes a product differentiation for SALLM: **profiles stay useful over time by feeding live failures into the existing offline optimize loop**, without putting search inside `Agent.ask`.

---

## Goal

When behavior or budget quality drifts, the system can:

1. Collect a **feed** of labeled cases (same JSONL shape `sallm optimize` already uses).
2. Detect that a **threshold** was hit (eval quality, mandatory misses, user “wrong” rate, receipt stress).
3. Run **offline** optimize (instruction rewrite and/or generation-budget search).
4. **Swap** the compiled profile the agent loads next — chat still only loads; it never optimizes on the request path.

Outcome: prompts (and searchable budgets) stay aligned with real failures, without a complex online learner.

---

## Non-goals (for this spec)

- Running `optimize` inside `ask()` or at chat startup.
- Automatic unlabeled learning (“the model seemed unhappy”).
- Searching window budgets (`prompt_budget`, `recent_history_tokens`, `retrieval_tokens`) via the current offline case scorer — those need receipt / full-turn measurement (see [Limits](#limits-carry-over-from-todays-optimize)).
- Replacing the golden dataset with raw chat logs.

---

## Library vs implementation

| Concern | Owns it | Notes |
|---------|---------|--------|
| `Agent.ask`, profile load, receipts, metrics, traces | **Library** | Emits observations (`got`), not training labels. |
| `Case` JSONL + `sallm optimize` (eval / rewrite / `--budgets-only`) | **Library** | Heal engine + artifact write. |
| Optional helper later: build a `Case` dict from a turn snapshot + correction | **Library (thin)** | Pure data shaping; no threshold, no I/O policy. |
| “This is wrong” UX (`/wrong`, button, API) | **Implementation** | App / CLI example. |
| Append to feed file, privacy, which turns to keep | **Implementation** | |
| Threshold policy and when to call optimize | **Implementation** | |
| Hot-reload or restart with new `--profile` | **Implementation** | |

**One-line split:** the library scores and improves a profile from cases; the app produces those cases and decides when to reload.

---

## What already exists

- Offline optimize never runs in the chat path ([optimize-prompts.md](optimize-prompts.md)).
- Dataset contract (`id`, `task`, `input`, `expected`, `mandatory`) — see `sallm.optimization.dataset.Case`.
- Fail-trace rewrite loop + optional generation-budget grid (`--search-budgets` / `--budgets-only`).
- Runtime outputs useful as **raw material**: `answer`, `steps`, `metrics`, `receipt`; control decision on traces / Prometheus.
- Example golden set: `examples/small_chat/opt_cases.jsonl`.

---

## Proposed loop

```text
Agent.ask / chat
    │
    ▼
observations (answer, steps, receipt, metrics)
    │
    ├─ optional: user “wrong” (+ correction) ──► append Case to feed.jsonl
    │
    ▼
threshold?   e.g. evaluate-only quality drop,
             mandatory misses on golden ∪ feed,
             N new user-labeled fails,
             sustained receipt stress (omitted / over_budget)
    │ yes
    ▼
sallm optimize --dataset (golden + feed) --profile current.json --out next.json
    │
    ▼
load next.json on next session / explicit reload
```

Chat remains load-only. Optimize stays out-of-band (sidecar, cron, or explicit command).

---

## The feed

Optimize already consumes one JSON object per line. The feed is **that format**, not Prometheus or Tempo.

### Sources of cases

1. **Golden / regression set** — hand-authored, mostly `mandatory` (today’s `opt_cases.jsonl` pattern).
2. **User “wrong”** — live labels (see below).
3. **Scripted QA** — same as golden, produced from `--script` sessions when expectations are known.

### User feedback: “this is wrong”

Nothing in the library does this yet. Recommended levels:

| Level | User action | Case strength |
|-------|-------------|---------------|
| Flag only | “Wrong” | Weak: `expected.absent` from bad answer / bad tool pattern; teaches avoidance, not success. |
| Flag + correction | “Wrong — should be X” / “use calc” | Strong: `contains` / `tool` / `observation_contains` / controller fields as appropriate; mark `mandatory: true`. |

Suggested snapshot for a converse case:

- `input.user` — prior user question  
- optional `input.history`, `input.retrieved` — enough context to reproduce the fail  
- `expected` — from correction (preferred) or weak `absent` from `got`  
- store `got` only in metadata or a side log for humans; the scorer needs `expected`

One “wrong” usually becomes **one** `task: converse` case. Add `controller` / `extractor` / `ingest` cases only when the failure stage is known.

| Complaint | Likely `task` | Typical `expected` |
|-----------|---------------|-------------------|
| Bad answer / skipped tool | `converse` | `contains`, `tool`, `absent`, … |
| Wrong routing / no recall query | `controller` | `action`, `skill`, `retrieval_query_*` |
| Fact not stored | `extractor` / `ingest` | `contains` on transcript |

---

## Thresholds (implementation policy)

Examples (pick one or combine; not prescribed by the library):

- `sallm optimize --evaluate-only` on golden ∪ feed: any **mandatory** miss, or mean quality below a floor.
- Count of new user-labeled mandatory cases since last heal ≥ N.
- Optional ops signals: high tool-fail rate, rising `receipt.omitted_messages` / `over_budget` — these may trigger **budget or retrieval** review, not necessarily instruction rewrite.

After a heal: write `--out` (or `--in-place` only when the app owns that path), fingerprint the dataset in profile metadata (already recorded), reload profile.

---

## Budgets: two classes

| Class | Examples | How to keep “optimal” |
|-------|----------|------------------------|
| **Generation knobs** (offline-searchable today) | `temperature`, `max_output_tokens`, `control_max_tokens`, `extract_max_tokens`, `think` | `--search-budgets` / `--budgets-only` on the same case feed. |
| **Window / context knobs** (not offline-scored today) | `prompt_budget`, `recent_history_tokens`, `retrieval_tokens` | Measure via `ContextReceipt` and full-turn chat; out of scope for the current optimize grid unless a later full-agent eval is added. |

“Always optimal” means **optimal on the current feed fingerprint**, not on all future traffic. The feed must track real regressions.

---

## Limits (carry over from today’s optimize)

- Converse search scores one compiled generation path (system + optional history + user → parse/run ```run); it does not run the full controller + retrieve + extract stack.
- Teacher should be stronger than a tiny student, or ```run contracts get stripped.
- Bare “wrong” without correction is a weak signal; heal quality tracks correction quality.
- Do not put DSPy modules or non-portable objects in the profile artifact.

---

## Suggested phasing (when implementing later)

1. **Spec + docs only** (this file) — no code.
2. **Example-only**: `examples/small_chat` appends Cases from a `/wrong` (or API) and documents “run optimize when N fails.”
3. **Optional library helper**: `case_from_turn(snapshot, *, expected, task=...)` — no I/O, no thresholds.
4. **Ops**: evaluate-only gate in CI or a sidecar; profile swap convention.

Until phase 2+, operators keep the manual workflow in [optimize-prompts.md](optimize-prompts.md): note fails → edit JSONL → optimize → chat with `--profile`.

---

## Success criteria (for a future implementation)

- A labeled “wrong” becomes a Case that `--evaluate-only` can fail under the current profile.
- Crossing a documented threshold produces a new profile that passes golden + new fails (or clearly reports remaining mandatory misses).
- `Agent.ask` path unchanged: no optimize, no teacher calls, no feed writes required for normal chat.
- Differentiation remains: **measurable degradation → offline fail-trace heal → new compiled profile**.

---

## Related code (today)

- `sallm/optimization/` — dataset, metrics, search, budgets, artifacts  
- `sallm/cli/optimize_cmd.py` — CLI  
- `sallm/agent.py` — `_finish` / `ask` return shape  
- `sallm/receipt.py` — `ContextReceipt`  
- `examples/small_chat/opt_cases.jsonl` — golden feed shape  
