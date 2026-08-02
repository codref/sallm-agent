# Tracing + metrics (Tempo / Prometheus / Grafana)

Minimal local stack:

- **Tempo** — OTLP spans (`--otlp`)
- **Prometheus** — scrapes sallm `:9464/metrics` (`--metrics-port`)
- **Grafana** — Explore + provisioned **sallm session** dashboard

No OpenTelemetry SDK.

## Start

From the repo root:

```bash
docker compose up -d
```

| Service | URL |
|---------|-----|
| Grafana | http://localhost:3000 (anonymous admin) |
| Prometheus | http://localhost:9090 |
| OTLP/HTTP | http://localhost:4318 |
| Tempo API | http://localhost:3200 |

## Run the agent

```bash
# traces + Prometheus metrics (default scrape port 9464)
uv run sallm chat --otlp http://localhost:4318 --metrics-port 9464

# durable session (stack + retrieval visible in the dashboard)
uv run sallm chat \
  --otlp http://localhost:4318 --metrics-port 9464 \
  --state-path .sallm/state.db --vector-path .sallm/vectors \
  --session demo1

# debug span payloads (truncated per message/field)
uv run sallm chat --otlp http://localhost:4318 --metrics-port 9464 --trace-debug
```

On startup the CLI prints `session=<name-or-hex>` and `metrics=:9464/metrics`.
`--session myname` is the id used for **Agent state, Prometheus labels, and Tempo
`session.id`** (they used to diverge: metrics/traces got a random hash).

Confirm scrape:

```bash
curl -s localhost:9464/metrics | head
# Prometheus targets: http://localhost:9090/targets  (sallm should be UP)
```

## Session dashboard

1. Open Grafana → **Dashboards** → folder **sallm** → **sallm session**
2. Set the **session_id** variable to the same value as `--session` (or the printed session id)
3. Rows:

| Row | What you see |
|-----|----------------|
| **Session overview** | Turns, **active skill**, **stack depth**, receipt/budget, retrieval hits, omitted history, tool call count |
| **Token economy** | Prompt composition (`system` / `retrieval` / `history`), tokens per turn, cumulative tokens, LLM vs tool latency, avg tokens/turn |
| **Stack, control, tools** | Stack depth over time, control actions (`keep`/`push`/`pop`/`replace`), tool calls + runtime by name, active skill gauge |
| **Extract / queue lens** | Extract mode, queue depth, extract vs turn latency, miss flushes, drain mix (`lazy`/`miss`), facts vs extract calls |
| **Traces (Tempo)** | `ask` traces for the session; separate lists for **tool** and **control** spans |

#### Choosing `--extract waterfall` vs `queue`

Panel descriptions on **Extract / queue lens** encode the rule of thumb:

- Prefer **queue** when extract latency is a large share of turn time **and** miss-flush rate stays low (depth clears on lazy drains).
- Prefer **waterfall** when miss-flush is frequent (deferred extract hurts recall turns) or queue depth climbs.

Run with `--metrics-port 9464` (and Prometheus scraping) to populate these panels.

Extract-related Prometheus series (session_id label):

| Metric | Meaning |
|--------|---------|
| `sallm_extract_mode{mode=…}` | Active mode gauge (`waterfall` / `queue`) |
| `sallm_extract_queue_depth` | Pending deferred extract jobs |
| `sallm_extract_enqueued_total` | Jobs deferred |
| `sallm_extract_drained_total{reason=lazy\|miss}` | Jobs drained |
| `sallm_extract_miss_flush_total` | Miss-driven drain + re-retrieve events |
| `sallm_extract_calls_total` / `sallm_extract_elapsed_ms_sum` | Extract LLM count / wall ms |
| `sallm_extract_last_elapsed_ms` | Latest extract duration |
| `sallm_extract_facts_total` | Grounded derived facts written |

Ask-span attr: `sallm.extract.miss_flush` (bool) when that turn forced a miss flush.

If Tempo panels are empty but `curl localhost:3200/api/search` shows traces: Grafana 11+
stubs TraceQL search on `/api/ds/query` (`backend TraceQL search queries are not supported`).
The provisioned dashboard uses the **TempoHTTP** Infinity datasource against Tempo
`/api/search` instead. Click a Trace ID to open the waterfall in Explore (stack / goal /
receipt attrs live on the `ask` span).

After changing provisioned dashboard / datasource / Tempo / compose:

```bash
docker compose up -d --force-recreate tempo grafana
```

First Grafana start installs `yesoreyeram-infinity-datasource` (see `GF_INSTALL_PLUGINS`).

Open an `ask` span in Explore to read attributes:

- `sallm.stack.path` — e.g. `converse > analyze`
- `sallm.goal`, `sallm.active_skill`, `sallm.stack.depth`
- `sallm.receipt.*_tokens`, `sallm.receipt.budget`
- `sallm.control.action` / `sallm.control.skill`

Child spans under each turn: `control`, `chat` (ReAct), `extract`, `tool <name>`.

PromQL examples:

```promql
sum(sallm_tokens_input_total{session_id="YOUR_SESSION"})
max(sallm_stack_depth{session_id="YOUR_SESSION"})
sum by (section) (sallm_receipt_section_tokens{session_id="YOUR_SESSION"})
sum by (action) (sallm_control_actions_total{session_id="YOUR_SESSION"})
sum by (tool, status) (sallm_tool_calls_total{session_id="YOUR_SESSION"})
max(sallm_extract_queue_depth{session_id="YOUR_SESSION"})
sum(increase(sallm_extract_miss_flush_total{session_id="YOUR_SESSION"}[1h]))
sum by (reason) (sallm_extract_drained_total{session_id="YOUR_SESSION"})
```

TraceQL (Explore → Tempo, or Infinity panels on the dashboard):

```traceql
{ resource.service.name = "sallm" && name = "ask" && span.session.id = "<paste from CLI>" }
{ resource.service.name = "sallm" && name = "control" && span.session.id = "<id>" }
{ resource.service.name = "sallm" && span.gen_ai.operation.name = "execute_tool" && span.session.id = "<id>" }
```

## Correlation

| Id | Scope |
|----|--------|
| `session.id` | whole Agent / chat process |
| `trace_id` | one `ask()` turn |
| `turn.index` | 1, 2, 3… within the session |

## Library

```python
from sallm.prom import SessionMetrics
from sallm.trace import Tracer, jsonl_sink

metrics = SessionMetrics(session_id="demo")
metrics.start_server(port=9464)
trace = Tracer(jsonl_sink("/tmp/sallm.jsonl"), metrics=metrics)
```

## Troubleshoot: "datasource prometheus was not found"

Provisioned name/uid is `prometheus`. Recreate Grafana so provisioning reloads:

```bash
docker compose up -d --force-recreate grafana
```

Then check **Connections → Data sources** — you should see `prometheus` and `Tempo`. Remove any empty manual `prometheus-1` entry.

Confirm: http://localhost:3000/api/datasources

## Stop

```bash
docker compose down
```

Traces are ephemeral inside Tempo unless you add a volume. Metrics exist only while the chat process is running (in-memory counters on `/metrics`).
