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

# metrics only (no OTLP)
uv run sallm chat --metrics-port 9464

# debug span payloads (truncated per message/field)
uv run sallm chat --otlp http://localhost:4318 --metrics-port 9464 --trace-debug
```

On startup the CLI prints `session=<hex>` and `metrics=:9464/metrics`.

Confirm scrape:

```bash
curl -s localhost:9464/metrics | head
# Prometheus targets: http://localhost:9090/targets  (sallm should be UP)
```

## Session dashboard (tokens, etc.)

1. Open Grafana → **Dashboards** → folder **sallm** → **sallm session**
2. Set the **session_id** variable (from the CLI banner), or leave `.*` for all
3. Panels: input/output tokens, turns, context size, cumulative token / LLM latency charts, tool calls, Tempo trace list

PromQL examples:

```promql
sum(sallm_tokens_input_total{session_id="YOUR_SESSION"})
sum(sallm_tokens_output_total{session_id="YOUR_SESSION"})
sum(sallm_llm_elapsed_ms_sum{session_id="YOUR_SESSION"})

# tokens added per scrape interval (≈ per ask when turns are spaced)
sum(increase(sallm_tokens_input_total{session_id="YOUR_SESSION"}[1m]))

# histogram of tokens per ask() turn
sum by (le) (sallm_turn_total_tokens_bucket{session_id="YOUR_SESSION"})
```

## Correlation (traces)

| Id | Scope |
|----|--------|
| `session.id` | whole Agent / chat process |
| `trace_id` | one `ask()` turn |
| `turn.index` | 1, 2, 3… within the session |

```traceql
{ resource.service.name = "sallm" && span.session.id = "<paste from CLI>" }
```

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
