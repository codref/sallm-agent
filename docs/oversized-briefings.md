# Oversized briefings and long script turns

What happens when a single user turn (a meeting transcript, a “briefing”, or any dump) is **larger than the recent-history budget** — and why follow-up answers can look identical and empty (“the transcripts do not contain…”).

This is the failure mode hit by `data/David_questions.txt` and the e2e fixture `tests/fixtures/long_briefing_qa.txt`.

---

## Short answer

If the briefing is oversized relative to `recent_history_tokens` (default **1800**):

1. It is **stored** in SQLite and **chunked/embedded** into LanceDB.
2. It is **not kept** in the main ReAct prompt history once any newer short Q/A is present.
3. Later questions see only a **small recent window** + up to **`retrieval_tokens` (800)** of vector hits.
4. If retrieval misses the right chunks, Gemma answers with a **generic refusal** — and those refusals then dominate the history window (**denial cascade**).

Durable memory does **not** mean “the whole briefing stays in context.” It means “we can retrieve pieces of it under a hard budget.”

---

## Budgets that matter

From `ModelProfile` (Gemma 4 defaults):

| Knob | Default | Role |
|------|---------|------|
| `prompt_budget` | 4096 | Soft cap for the whole compiled prompt |
| `recent_history_tokens` | **1800** | Verbatim transcript tail in the prompt |
| `retrieval_tokens` | **800** | Cap for injected Lance hits |
| embedding `top_k` | 4 (CLI often 4–6) | How many chunks compete for that 800 |

A David-style transcript is often **~13k tokens** in one line. That is already ~7× the history budget.

---

## How history assembly drops the briefing

`compile_prompt_messages` walks messages **newest → oldest** and stops when the next message would exceed the history budget (`sallm/receipt.py`):

```text
… older … | [HUGE BRIEFING] | "ready" | Q1 | A1 | Q2 | A2 | … | Qn   (newest)
                                      ▲
                      fill 1800 tok from here backward
```

- Recent short Q/A lines fit first.
- When the walker reaches the huge briefing, `hist_tokens + briefing > budget` and **`kept` is already non-empty** → it **breaks**.
- The briefing is **omitted entirely** (not truncated into history). Truncation of an oversized message only happens when it is the **sole** candidate (`kept` empty).

`ContextReceipt` then shows:

- `sections.history.note = "omitted=N"`
- `omitted_messages = N` (often ≥ 1 from the first real question after the dump)

In traces / Prometheus: `sallm.receipt.omitted_messages`, `sallm.receipt.history_tokens`.

**Even `briefing + one question` alone** can omit the briefing: the question is kept first; then the briefing does not fit → omitted.

---

## What still works: indexing + retrieval

On each turn (with `--state-path` / `--vector-path`):

1. The raw briefing message is persisted.
2. The indexer chunks it (~512 tokens, overlap 64) and embeds into Lance.
3. Control proposes a `retrieval_query`; the agent embeds that query and pulls `top_k` hits.
4. Hits are packed into the prompt under the **800**-token retrieval section.

So the model’s only evidence for early facts is **whatever ranked into those hits**. Abstract questions (“fundamental agreement”, “motivation for platform”) often retrieve the wrong slices even when the right text exists in the store.

---

## Why answers look the same (denial cascade)

Typical late-turn prompt contents:

| Section | What the 4B model actually sees |
|---------|----------------------------------|
| system + skill + goal | short |
| retrieval | ≤800 tok of mixed chunks |
| history | last few Q/A, often including prior **“I don’t know / not in the transcript”** replies |

Once a few refusals sit in the 1800-token window:

- They crowd out other recent answers.
- The model pattern-matches to the same hedge.
- Quality collapses even if better chunks are in Lance.

Fresh sessions (`--session David2`) hit the **same** budget math; “second call” is not a separate bug — it is the same eviction, sometimes with worse luck on retrieval.

---

## Script format makes this easy to trigger

`sallm chat --script file.txt` sends **one non-empty line per turn**. Files like `David_questions.txt` are:

1. short preamble  
2. **one giant transcript line**  
3. many `?` questions  

That is the worst shape for the history window: one megamessage that never fits beside the Q/A tail.

---

## How to see it

```bash
uv run sallm chat \
  --script tests/fixtures/long_briefing_qa.txt \
  --state-path .sallm/state.db \
  --vector-path .sallm/vectors \
  --session brief1 \
  --trace /tmp/brief.jsonl \
  --trace-debug
```

During or after:

- `/context` — receipt: `omitted_msgs`, section tokens vs budget  
- Grafana **sallm session** — `receipt / budget`, omitted gauge, ask-trace tables  
- Trace JSONL — `sallm.receipt.omitted_messages`, `sallm.retrieval.hits` on each `ask` span  

Automated check (runs the script twice, scores needles from traces):

```bash
uv run pytest tests/test_e2e_qa_script_trace.py -s
```

The e2e asserts `max(omitted_messages) ≥ 1` after the oversized line, then reports completeness / coherence across two runs.

---

## Mitigations

| Approach | Effect |
|----------|--------|
| **Split the briefing** into many script lines / turns under ~1–1.5k tokens each | Pieces can remain in the recent window longer; indexing still runs per turn |
| **Raise `recent_history_tokens`** (and possibly `prompt_budget`) | Fits more verbatim history; costs more per turn on a 4B |
| **Raise `retrieval_tokens` / `top_k`** | More Lance evidence when history has dropped the dump |
| **`--memory-gate` (default on)** | Keeps short follow-up questions out of the vector index so they do not crowd out briefing chunks |
| **`--search hybrid`** | BM25 + dense (Lance RRF) helps exact tokens / names that dense alone misses |
| **`--retrieval-query rewrite`** / **`hyde`** | Control rewrite and/or HyDE for a better embedding query |
| **Pin or special-case “document” messages** (future) | Keep a truncated briefing slice in history on purpose |
| **Do not rely on one-shot ingest** for exam-style QA | Prefer chunked ingest + explicit “answer only from retrieved memory” skill |

There is no free lunch: keeping a 13k-token dump in every prompt defeats the stack-optimized design for small local models.

---

## Mental model

```text
oversized briefing turn
        │
        ├─────────────► SQLite (full text) ──► chunks ──► Lance
        │
        └─────────────► prompt history?
                              │
                              ├─ alone & > 1800 tok → truncated once, then gone
                              └─ after any newer Q/A → omitted (break); rely on retrieval ≤ 800 tok
```

**Stored ≠ in the prompt.** Oversized means: durable yes, verbatim in context no — unless you change budgets or how you feed the script.

---

## Related

- [How the agent works](how-the-agent-works.md) — turn pipeline and receipt  
- [Optimize prompts](optimize-prompts.md) — budgets and offline search  
- [Tracing + Grafana](tracing-tempo.md) — session dashboard for omitted / retrieval / stack  
- Code: `sallm/receipt.py` (`compile_prompt_messages`), `sallm/models.py` (`ModelProfile`), `tests/test_e2e_qa_script_trace.py`
