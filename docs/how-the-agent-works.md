# How the stack-optimized agent works

This guide walks through one realistic session. Start with the example, then follow what the agent does at each stage, how the token budget is spent, and what that implies for small local models.

## The hypothesis

A 4B model (here: Gemma 4) can stay useful for long Q/A or analysis **if**:

1. we stop stuffing the entire transcript into every prompt,
2. we keep a **bounded** recent window,
3. we **retrieve** older facts from a durable store instead of hoping the model remembered them,
4. we make that budget visible (`ContextReceipt`) so we can measure and tune it.

Retrieval improves grounding. It does **not** guarantee the model never invents facts. Raw messages remain the source of truth.

---

## A concrete example

Imagine a long lab notebook session. Early on you say:

> Please remember this fact for later: **The unique lab code is ZEBRA-7711.**

Then you chat for a while about unrelated topics (clouds, padding, filler turns). Much later:

> What is the unique lab code? Reply with the code only.

**Without** durable memory + retrieval, a truncated prompt often loses `ZEBRA-7711` once it falls out of the recent-history window. The model may guess or say it does not know.

**With** the stack agent (`--state-path` + LanceDB), that early turn is chunked, embedded (Qwen3-Embedding), and searchable. The late question retrieves the chunk, injects it under a retrieval budget, and Gemma answers from that block.

Try the same idea locally:

```bash
uv run sallm chat \
  --state-path .sallm/state.db \
  --vector-path .sallm/vectors \
  --session lab1 \
  --retrieval-query instruct \
  --search dense \
  --memory-gate \
  --tools none
```

After a few turns, inspect with `/state`, `/memory`, `/context`.

---

## Big picture

When you call `agent.ask(...)` with a durable session, one turn looks like this:

```text
user message
    │
    ▼
1. Persist raw message          (SQLite)
    │
    ▼
2. Goal / skill control         (small Gemma JSON call)
    │
    ▼
3. Vector retrieve              (Qwen embed → LanceDB)
    │
    ▼
4. Budgeted prompt + ReAct      (Gemma + optional ```run tools)
    │
    ▼
5. Persist answer               (SQLite)
    │
    ▼
6. Extract + index              (facts → SQLite; chunks → Lance)
```

SQLite is canonical. LanceDB is a **rebuildable index**. If indexing crashes mid-way, unindexed rows are retried on resume.

Without `--state-path`, the agent falls back to the older in-memory ReAct loop (full transcript + optional trim/summarize). The rest of this doc focuses on the durable path.

---

## Stage by stage

### 0. Session load

On construction with `state_path` + `session_id`, the agent:

- opens (or creates) the SQLite DB,
- ensures a root skill frame (`converse`),
- reconnects LanceDB beside the state file,
- flushes any chunks that were saved but never indexed,
- reloads the transcript into `agent.messages`.

Reusing the same `--session` after restart continues the same goal, stack, and memory.

### 1. Persist the user message

The new user text is appended to SQLite **and** to the in-memory transcript. Nothing is summarized away yet. Durability comes first; the prompt view is shaped later.

### 2. Goal / skill control

A **small** completion asks Gemma for one JSON object, for example:

```json
{
  "goal": "recall the lab code",
  "action": "keep",
  "skill": "converse",
  "retrieval_query": "unique lab code ZEBRA"
}
```

| Field | Meaning |
|-------|---------|
| `goal` | Short statement of current intent (steering) |
| `action` | `keep` / `push` / `pop` / `replace` on the skill stack |
| `skill` | Must be a registered skill (default: `converse`) |
| `retrieval_query` | Standalone sentence for vector search |

If the JSON is invalid or names an unknown skill, the agent **falls back**: keep the current skill, keep/derive a simple goal, still try retrieval from the user text. Conversation is not blocked.

Stack effects are persisted. Today only `converse` ships; custom skills add routing text, prompt fragments, and optional tool subsets.

### 3. Retrieve memory

Query pipeline (stackable flags on `RetrievalConfig`):

1. Start from the user text  
2. If `--retrieval-query rewrite` (or `rewrite+hyde`) and control emitted `retrieval_query` → use that sentence  
3. If `hyde` / `rewrite+hyde` → one short LLM call writes a hypothetical answer passage; that passage is what gets embedded (classic HyDE)  
4. If instruct (default) → wrap with Qwen’s `Instruct: …\nQuery: …` template  
5. Search Lance as **dense** (default) or **hybrid** BM25+dense (`--search hybrid`)

| `--retrieval-query` | Behavior |
|---------------------|----------|
| `raw` | Embed query text as-is |
| `instruct` (default) | Qwen instruct template |
| `rewrite` | Prefer control’s `retrieval_query`, then instruct |
| `hyde` | HyDE passage → instruct |
| `rewrite+hyde` | Rewrite, then HyDE, then instruct |

| `--search` | Behavior |
|------------|----------|
| `dense` (default) | Vector similarity only |
| `hybrid` | LanceBM25 FTS + dense, fused with RRF |

Documents in the index are stored **without** the instruction prefix.

LanceDB returns top‑k hits (default 4). Each hit keeps `id`, text, score, and optional `source_id` (message id). Those hits become the retrieval section of the next prompt.

### 3b. Write-time memory gate

After the answer, user text is chunked for indexing. With `--memory-gate` (default **on**), short interrogatives (e.g. “What is Dale’s birthday?”) are **not** written to SQLite/Lance—they pollute retrieval with near-duplicate questions. Long dumps and extractor `derived` facts always pass. Disable with `--no-memory-gate`. Gated chunk counts appear on turn traces as `sallm.memory.gated_chunks`.

### 4. Budgeted prompt + ReAct

The prompt is **not** “everything we ever said.” The compiler packs sections by priority until the profile budget is exhausted:

1. **System** — tool contract, active skill text, current goal  
2. **Retrieval** — source-tagged memory hits  
3. **Recent history** — newest complete turns first; older ones are omitted  

Defaults for Gemma 4 (`ModelProfile`):

| Budget | Tokens (estimate) |
|--------|-------------------|
| Total prompt | 4096 |
| Recent history | 1800 |
| Retrieval | 800 |
| Control max output | 256 |
| Extract max output | 384 |

Estimates use ~4 characters ≈ 1 token (`estimate_tokens`). Good enough for budgets; not a tokenizer.

Then the usual ReAct loop runs:

- Gemma may answer in plain text, or emit a ```run block for CLI tools,
- tool results are appended (and persisted),
- intermediate tool stdout keeps the loop going with nudges,
- `max_steps` caps tool rounds.

Every main call updates `agent.last_prompt` and a **`ContextReceipt`**: section token spends, retrieved ids/scores, omitted message count, fallbacks. In chat: `/context`.

### 5. Persist the answer

The final assistant text is written to SQLite and to `agent.messages`. Metrics still distinguish transcript length (`context_messages`) from what was actually sent (`prompt_messages` / receipt totals).

### 6. Extract and index

After the answer:

1. **Chunk** the latest user text through the memory gate (short questions may be skipped).  
2. **Upsert** accepted chunks into LanceDB; mark rows indexed in SQLite.  
3. **Extract** durable facts (small LLM call): only facts with valid `source_message_ids` are kept. Ungrounded inventions are dropped. Derived facts always pass the gate.

Derived facts never replace raw messages; they are extra searchable notes with provenance.

#### `--extract waterfall` (default)

Extract runs synchronously after raw indexing, before `ask()` returns. Simple and predictable; the user waits for extract latency on every turn.

#### `--extract queue`

Extract is **deferred**: the turn enqueues a SQLite `PendingExtract` job and returns. On later turns, after the first retrieve:

- **Miss-flush** — non-empty `retrieval_query`, zero hits, and pending jobs → drain the queue, re-retrieve **once**, then ReAct.
- **Lazy drain** — otherwise, if jobs are still pending, drain them before ReAct so the backlog clears.

Raw indexing stays eager either way. Use the Grafana **Extract / queue lens** (with `--metrics-port`) to decide:

| Signal | Prefer |
|--------|--------|
| Extract ms is a large share of turn time, miss-flush rare, depth drains on `lazy` | **queue** |
| Miss-flush frequent, or queue depth climbs across turns | **waterfall** |

Compare the same session under both flags before committing to queue in day-to-day use.

---

## Token-budget simulation

Walk through the lab-code session with simplified numbers. Assume the system block is ~600 tokens, and each “large filler” turn pair is ~400 tokens of history.

### Early turn: store the fact

User: *Remember … ZEBRA-7711.*

| Section | ≈ tokens | In prompt? |
|---------|----------|------------|
| System + goal/skill | 600 | yes |
| Retrieval | 0 (empty index) | no hits |
| Recent history | ~50 | yes |
| **Total** | **~650** | under 4096 |

After the turn, a chunk containing `ZEBRA-7711` is indexed.

### Middle turns: filler

Three oversized unrelated turns grow the SQLite transcript. The **prompt** still only keeps the tail within ~1800 history tokens. Older filler (and eventually the ZEBRA turn) leave the recent window.

A history-only compile with a tight budget can look like:

| Section | ≈ tokens | Note |
|---------|----------|------|
| System | 600 | |
| Retrieval | 0 | not asked yet |
| History tail | 1800 | ZEBRA turn **omitted** |
| **Total** | **~2400** | `omitted_messages` ≥ 1 |

Hypothesis check: the secret is gone from verbatim context. A plain truncated agent would fail here.

### Late turn: ask for the code

Control returns something like `retrieval_query: "unique lab code"`. Lance returns the early chunk.

| Section | ≈ tokens | Note |
|---------|----------|------|
| System + goal | 650 | goal ≈ “recall the lab code” |
| Retrieval | 120 | `[mem id=… source=1] … ZEBRA-7711` |
| History tail | 1800 | recent filler only |
| **Total** | **~2570** | still under 4096 |

Gemma sees the code **in the retrieval block**, not because the full chat was replayed.

Rough **extra** cost of the durable path on this turn (order of magnitude):

| Call | Role | Extra tokens |
|------|------|--------------|
| Control | JSON route + query | small prompt + ≤256 out |
| Embedding | query vector | embedding model, not Gemma context |
| Main ReAct | answer | bounded prompt as above |
| Extract | optional facts | small prompt + ≤384 out |

You spend more **calls**, but you avoid unbounded **context growth**. That is the token-economy trade: predictable main prompts, paid for with control/retrieve/extract side work.

---

## What a `ContextReceipt` looks like

After a turn, `/context` (or `result["receipt"]`) resembles:

```text
profile: ollama/gemma4:e4b-it-qat (gemma4-e4b-v1)
budget: 4096  used≈2570  omitted_msgs=8
  ✓ system: 650 tok
  ✓ retrieval: 120 tok
  ✓ history: 1800 tok (omitted=8)
retrieved:
  - a1b2c3d4e5f6… score=-0.12 src=1
```

Use it to validate:

- used ≤ budget (soft estimate),
- retrieval actually carried the fact when history omitted it,
- omitted count rises as sessions lengthen while used stays flat.

---

## Validating the hypothesis

| Claim | How to check |
|-------|----------------|
| Long sessions stay within a predictable prompt size | Watch `/context` `total_tokens` across many turns; it should hover near system + retrieval + history caps, not grow with transcript length. |
| Early facts survive eviction | Store a unique code early, flood with filler, ask later; answer should contain the code when retrieval is on. Automated: `tests/test_e2e_stack_memory.py`. |
| Transcript survives restart | Same `--state-path` + `--session` in a new process; `/history` and `/stack` resume. |
| Bad control JSON does not brick the chat | Invalid controller output → `fallback` keep + raw indexing; user still gets a ReAct answer. |
| Ungrounded “memories” are rejected | Extractor drops facts whose `source_message_ids` are not in the recent message set. |

**Counter-hypothesis (keep honest):** if retrieval misses (bad query, embedding drift, empty index), the model still fails like a truncated agent. The receipt shows empty `retrieved`—that is the debugging signal. A single **oversized** user turn (e.g. a full meeting dump in `--script`) is omitted from history as soon as any newer Q/A exists; see [Oversized briefings](oversized-briefings.md).

Offline prompt search (`sallm optimize`) can improve controller/extractor/converse instructions against JSONL cases; chat never optimizes at startup. It loads a neutral profile JSON when present.

---

## Mental model in one paragraph

Think of the agent as a **small working set** plus an **external notebook**. Gemma reasons over the working set (system, goal, skill, top memory hits, recent turns). SQLite is the notebook; LanceDB is the notebook’s index. Control decides what the notebook query should be and whether the skill stack changes. The receipt is the receipt for the working set—what you paid for in tokens, and what you left on the shelf.

---

## Related reading

- [README](../README.md) — setup, CLI flags, `VectorStore` outline  
- [Oversized briefings](oversized-briefings.md) — what happens when one turn exceeds `recent_history_tokens`  
- [Optimizing prompts and parameters](optimize-prompts.md) — `sallm optimize`, JSONL cases, budgets, retrieval knobs  
- [Skills](skills.md) — selection, stack, adding skills, relation to tools  
- [Tracing with Tempo](tracing-tempo.md) — spans for LLM and tool steps  
- Code entrypoints: `agent.py` (facade), `turn.py` (ReAct), `receipt.py` (budget), `control.py` (route/extract), `memory/` (chunk → embed → Lance)
