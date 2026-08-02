"""Optional Prometheus metrics for session token / latency charts.

Stdlib only: in-process /metrics HTTP server scraped by Prometheus.
When unset, nothing runs (no threads, no ports).
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Token counts per ask() turn (and per LLM call) — classic Prometheus histogram.
TOKEN_BUCKETS = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, float("inf"))


def _label_value(v):
    return str(v).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _labels(**kwargs):
    parts = [f'{k}="{_label_value(v)}"' for k, v in kwargs.items()]
    return "{" + ",".join(parts) + "}"


class _Histogram:
    """Cumulative-bucket histogram (Prometheus classic)."""

    def __init__(self, buckets=TOKEN_BUCKETS):
        self.buckets = list(buckets)
        self.counts = [0] * len(self.buckets)  # cumulative counts per le
        self.sum = 0.0
        self.count = 0

    def observe(self, value):
        value = float(value or 0)
        if value < 0:
            value = 0.0
        self.sum += value
        self.count += 1
        for i, bound in enumerate(self.buckets):
            if value <= bound:
                self.counts[i] += 1

    def render_lines(self, name, labels):
        """Emit _bucket / _sum / _count lines (HELP/TYPE added by caller)."""
        lines = []
        for bound, count in zip(self.buckets, self.counts):
            le = "+Inf" if bound == float("inf") else str(int(bound) if bound == int(bound) else bound)
            lab = labels.copy()
            lab["le"] = le
            lines.append(f"{name}_bucket{_labels(**lab)} {count}")
        lines.append(f"{name}_sum{_labels(**labels)} {self.sum}")
        lines.append(f"{name}_count{_labels(**labels)} {self.count}")
        return lines


class SessionMetrics:
    """Cumulative counters + per-turn token histograms for one session."""

    def __init__(self, session_id):
        self.session_id = session_id
        self._lock = threading.Lock()
        # (model,) -> totals
        self._llm = {}  # model -> dict
        # (tool, status) -> totals
        self._tools = {}
        self.turns = 0
        self.turn_elapsed_ms = 0.0
        self.context_messages = 0
        self.last_turn_input = 0
        self.last_turn_output = 0
        self.last_turn_total = 0
        self.last_turn_index = 0
        self._turn_input = _Histogram()
        self._turn_output = _Histogram()
        self._turn_total = _Histogram()
        self._llm_total = _Histogram()
        # Stack / control / receipt (gauges + counters for Grafana).
        self.stack_depth = 0
        self.active_skill = ""
        self.goal_chars = 0
        self._skills = {}  # skill -> 1 active / 0 inactive
        self._control = {}  # action -> count
        self.receipt_budget = 0
        self.receipt_total = 0
        self.receipt_omitted = 0
        self.retrieval_hits = 0
        self._receipt_sections = {}  # section -> tokens
        # Extract / queue lens (waterfall vs deferred extract).
        self.extract_mode = "waterfall"
        self.extract_queue_depth = 0
        self.extract_enqueued_total = 0
        self.extract_drained_total = {}  # reason -> count
        self.extract_miss_flush_total = 0
        self.extract_calls_total = 0
        self.extract_elapsed_ms_sum = 0.0
        self.extract_last_elapsed_ms = 0.0
        self.extract_facts_total = 0
        self._httpd = None
        self._thread = None

    def observe_llm(self, model, prompt_tokens, completion_tokens, total_tokens, elapsed_ms):
        model = model or ""
        prompt_tokens = int(prompt_tokens or 0)
        completion_tokens = int(completion_tokens or 0)
        total_tokens = int(total_tokens or 0)
        with self._lock:
            row = self._llm.setdefault(
                model,
                {
                    "input": 0,
                    "output": 0,
                    "total": 0,
                    "elapsed_ms": 0.0,
                    "calls": 0,
                },
            )
            row["input"] += prompt_tokens
            row["output"] += completion_tokens
            row["total"] += total_tokens
            row["elapsed_ms"] += float(elapsed_ms or 0.0)
            row["calls"] += 1
            self._llm_total.observe(total_tokens)

    def observe_tool(self, name, elapsed_ms, returncode=0):
        name = name or ""
        status = "ok" if returncode == 0 else "error"
        key = (name, status)
        with self._lock:
            row = self._tools.setdefault(
                key, {"calls": 0, "elapsed_ms": 0.0}
            )
            row["calls"] += 1
            row["elapsed_ms"] += float(elapsed_ms or 0.0)

    def observe_turn(self, metrics, context_messages=0):
        metrics = metrics or {}
        inp = int(metrics.get("prompt_tokens") or 0)
        out = int(metrics.get("completion_tokens") or 0)
        total = int(metrics.get("total_tokens") or (inp + out))
        with self._lock:
            self.turns += 1
            self.turn_elapsed_ms += float(metrics.get("elapsed_ms") or 0.0)
            self.context_messages = int(context_messages or 0)
            self.last_turn_index = self.turns
            self.last_turn_input = inp
            self.last_turn_output = out
            self.last_turn_total = total
            self._turn_input.observe(inp)
            self._turn_output.observe(out)
            self._turn_total.observe(total)

    def observe_control(self, action, skill=""):
        action = (action or "keep").strip() or "keep"
        with self._lock:
            self._control[action] = self._control.get(action, 0) + 1
            if skill:
                # Keep skill label set even before stack observe.
                self._skills.setdefault(skill, 0)

    def observe_stack(self, depth, active_skill="", goal=""):
        active_skill = (active_skill or "").strip()
        with self._lock:
            self.stack_depth = int(depth or 0)
            self.active_skill = active_skill
            self.goal_chars = len(goal or "")
            for name in list(self._skills):
                self._skills[name] = 0
            if active_skill:
                self._skills[active_skill] = 1

    def observe_receipt(self, receipt):
        if receipt is None:
            return
        if hasattr(receipt, "as_dict"):
            receipt = receipt.as_dict()
        if not isinstance(receipt, dict):
            return
        sections = {
            s.get("name"): int(s.get("tokens") or 0)
            for s in (receipt.get("sections") or [])
            if isinstance(s, dict) and s.get("name")
        }
        with self._lock:
            self.receipt_budget = int(receipt.get("budget") or 0)
            self.receipt_total = int(receipt.get("total_tokens") or 0)
            self.receipt_omitted = int(receipt.get("omitted_messages") or 0)
            self.retrieval_hits = len(receipt.get("retrieved") or [])
            self._receipt_sections = {
                "system": sections.get("system", 0),
                "retrieval": sections.get("retrieval", 0),
                "history": sections.get("history", 0),
            }

    def set_extract_mode(self, mode: str):
        mode = (mode or "waterfall").strip().lower() or "waterfall"
        with self._lock:
            self.extract_mode = mode

    def observe_extract(self, elapsed_ms, facts=0):
        with self._lock:
            self.extract_calls_total += 1
            ms = float(elapsed_ms or 0.0)
            self.extract_elapsed_ms_sum += ms
            self.extract_last_elapsed_ms = ms
            self.extract_facts_total += int(facts or 0)

    def observe_extract_queue(
        self,
        depth,
        *,
        enqueued=0,
        drained=0,
        reason=None,
        miss_flush=False,
    ):
        with self._lock:
            self.extract_queue_depth = int(depth or 0)
            self.extract_enqueued_total += int(enqueued or 0)
            n = int(drained or 0)
            if n:
                key = (reason or "lazy").strip() or "lazy"
                self.extract_drained_total[key] = (
                    self.extract_drained_total.get(key, 0) + n
                )
            if miss_flush:
                self.extract_miss_flush_total += 1

    def render(self):
        sid = self.session_id
        lines = [
            "# HELP sallm_tokens_input_total Prompt tokens sent to the model.",
            "# TYPE sallm_tokens_input_total counter",
            "# HELP sallm_tokens_output_total Completion tokens from the model.",
            "# TYPE sallm_tokens_output_total counter",
            "# HELP sallm_tokens_total Total tokens (prompt + completion).",
            "# TYPE sallm_tokens_total counter",
            "# HELP sallm_llm_elapsed_ms_sum LLM call latency (ms).",
            "# TYPE sallm_llm_elapsed_ms_sum counter",
            "# HELP sallm_llm_calls_total LLM completions.",
            "# TYPE sallm_llm_calls_total counter",
            "# HELP sallm_tool_calls_total Tool subprocess invocations.",
            "# TYPE sallm_tool_calls_total counter",
            "# HELP sallm_tool_elapsed_ms_sum Tool runtime (ms).",
            "# TYPE sallm_tool_elapsed_ms_sum counter",
            "# HELP sallm_turns_total Agent ask() turns.",
            "# TYPE sallm_turns_total counter",
            "# HELP sallm_turn_elapsed_ms_sum Turn wall time from metrics (ms).",
            "# TYPE sallm_turn_elapsed_ms_sum counter",
            "# HELP sallm_context_messages Messages in the agent transcript.",
            "# TYPE sallm_context_messages gauge",
            "# HELP sallm_last_turn_input_tokens Prompt tokens on the latest ask().",
            "# TYPE sallm_last_turn_input_tokens gauge",
            "# HELP sallm_last_turn_output_tokens Completion tokens on the latest ask().",
            "# TYPE sallm_last_turn_output_tokens gauge",
            "# HELP sallm_last_turn_total_tokens Total tokens on the latest ask().",
            "# TYPE sallm_last_turn_total_tokens gauge",
            "# HELP sallm_last_turn_index Latest turn.index (1-based).",
            "# TYPE sallm_last_turn_index gauge",
            "# HELP sallm_turn_input_tokens Prompt tokens per ask() turn.",
            "# TYPE sallm_turn_input_tokens histogram",
            "# HELP sallm_turn_output_tokens Completion tokens per ask() turn.",
            "# TYPE sallm_turn_output_tokens histogram",
            "# HELP sallm_turn_total_tokens Total tokens per ask() turn.",
            "# TYPE sallm_turn_total_tokens histogram",
            "# HELP sallm_llm_total_tokens Total tokens per LLM completion.",
            "# TYPE sallm_llm_total_tokens histogram",
            "# HELP sallm_stack_depth Skill stack depth after the latest turn.",
            "# TYPE sallm_stack_depth gauge",
            "# HELP sallm_skill_active 1 for the active skill, 0 for others seen.",
            "# TYPE sallm_skill_active gauge",
            "# HELP sallm_goal_chars Characters in the current goal string.",
            "# TYPE sallm_goal_chars gauge",
            "# HELP sallm_control_actions_total Controller routing actions.",
            "# TYPE sallm_control_actions_total counter",
            "# HELP sallm_receipt_budget Prompt token budget from ModelProfile.",
            "# TYPE sallm_receipt_budget gauge",
            "# HELP sallm_receipt_total_tokens Estimated prompt tokens (ContextReceipt).",
            "# TYPE sallm_receipt_total_tokens gauge",
            "# HELP sallm_receipt_omitted_messages History messages omitted by budget.",
            "# TYPE sallm_receipt_omitted_messages gauge",
            "# HELP sallm_receipt_section_tokens Tokens per receipt section.",
            "# TYPE sallm_receipt_section_tokens gauge",
            "# HELP sallm_retrieval_hits Vector hits injected into the latest prompt.",
            "# TYPE sallm_retrieval_hits gauge",
            "# HELP sallm_extract_mode 1 for the active extract mode label.",
            "# TYPE sallm_extract_mode gauge",
            "# HELP sallm_extract_queue_depth Pending deferred extract jobs.",
            "# TYPE sallm_extract_queue_depth gauge",
            "# HELP sallm_extract_enqueued_total Extract jobs deferred (queue mode).",
            "# TYPE sallm_extract_enqueued_total counter",
            "# HELP sallm_extract_drained_total Extract jobs drained from the queue.",
            "# TYPE sallm_extract_drained_total counter",
            "# HELP sallm_extract_miss_flush_total Miss-driven drain + re-retrieve events.",
            "# TYPE sallm_extract_miss_flush_total counter",
            "# HELP sallm_extract_calls_total Memory-extract LLM invocations.",
            "# TYPE sallm_extract_calls_total counter",
            "# HELP sallm_extract_elapsed_ms_sum Wall ms spent in extract LLM calls.",
            "# TYPE sallm_extract_elapsed_ms_sum counter",
            "# HELP sallm_extract_last_elapsed_ms Latest extract call duration (ms).",
            "# TYPE sallm_extract_last_elapsed_ms gauge",
            "# HELP sallm_extract_facts_total Grounded derived facts written.",
            "# TYPE sallm_extract_facts_total counter",
        ]
        with self._lock:
            for model, row in self._llm.items():
                lab = _labels(session_id=sid, model=model)
                lines.append(f"sallm_tokens_input_total{lab} {row['input']}")
                lines.append(f"sallm_tokens_output_total{lab} {row['output']}")
                lines.append(f"sallm_tokens_total{lab} {row['total']}")
                lines.append(f"sallm_llm_elapsed_ms_sum{lab} {row['elapsed_ms']}")
                lines.append(f"sallm_llm_calls_total{lab} {row['calls']}")
            for (tool, status), row in self._tools.items():
                lab = _labels(session_id=sid, tool=tool, status=status)
                lines.append(f"sallm_tool_calls_total{lab} {row['calls']}")
                lines.append(f"sallm_tool_elapsed_ms_sum{lab} {row['elapsed_ms']}")
            slab = {"session_id": sid}
            lines.append(f"sallm_turns_total{_labels(**slab)} {self.turns}")
            lines.append(
                f"sallm_turn_elapsed_ms_sum{_labels(**slab)} {self.turn_elapsed_ms}"
            )
            lines.append(
                f"sallm_context_messages{_labels(**slab)} {self.context_messages}"
            )
            lines.append(
                f"sallm_last_turn_input_tokens{_labels(**slab)} {self.last_turn_input}"
            )
            lines.append(
                f"sallm_last_turn_output_tokens{_labels(**slab)} {self.last_turn_output}"
            )
            lines.append(
                f"sallm_last_turn_total_tokens{_labels(**slab)} {self.last_turn_total}"
            )
            lines.append(
                f"sallm_last_turn_index{_labels(**slab)} {self.last_turn_index}"
            )
            lines.append(f"sallm_stack_depth{_labels(**slab)} {self.stack_depth}")
            lines.append(f"sallm_goal_chars{_labels(**slab)} {self.goal_chars}")
            lines.append(f"sallm_receipt_budget{_labels(**slab)} {self.receipt_budget}")
            lines.append(
                f"sallm_receipt_total_tokens{_labels(**slab)} {self.receipt_total}"
            )
            lines.append(
                f"sallm_receipt_omitted_messages{_labels(**slab)} {self.receipt_omitted}"
            )
            lines.append(
                f"sallm_retrieval_hits{_labels(**slab)} {self.retrieval_hits}"
            )
            for mode_name in ("waterfall", "queue"):
                lab = _labels(session_id=sid, mode=mode_name)
                active = 1 if self.extract_mode == mode_name else 0
                lines.append(f"sallm_extract_mode{lab} {active}")
            lines.append(
                f"sallm_extract_queue_depth{_labels(**slab)} {self.extract_queue_depth}"
            )
            lines.append(
                f"sallm_extract_enqueued_total{_labels(**slab)} "
                f"{self.extract_enqueued_total}"
            )
            for reason, count in self.extract_drained_total.items():
                lab = _labels(session_id=sid, reason=reason)
                lines.append(f"sallm_extract_drained_total{lab} {count}")
            lines.append(
                f"sallm_extract_miss_flush_total{_labels(**slab)} "
                f"{self.extract_miss_flush_total}"
            )
            lines.append(
                f"sallm_extract_calls_total{_labels(**slab)} {self.extract_calls_total}"
            )
            lines.append(
                f"sallm_extract_elapsed_ms_sum{_labels(**slab)} "
                f"{self.extract_elapsed_ms_sum}"
            )
            lines.append(
                f"sallm_extract_last_elapsed_ms{_labels(**slab)} "
                f"{self.extract_last_elapsed_ms}"
            )
            lines.append(
                f"sallm_extract_facts_total{_labels(**slab)} {self.extract_facts_total}"
            )
            for skill, active in self._skills.items():
                lab = _labels(session_id=sid, skill=skill)
                lines.append(f"sallm_skill_active{lab} {active}")
            for action, count in self._control.items():
                lab = _labels(session_id=sid, action=action)
                lines.append(f"sallm_control_actions_total{lab} {count}")
            for section, tokens in self._receipt_sections.items():
                lab = _labels(session_id=sid, section=section)
                lines.append(f"sallm_receipt_section_tokens{lab} {tokens}")
            lines.extend(self._turn_input.render_lines("sallm_turn_input_tokens", slab))
            lines.extend(self._turn_output.render_lines("sallm_turn_output_tokens", slab))
            lines.extend(self._turn_total.render_lines("sallm_turn_total_tokens", slab))
            lines.extend(self._llm_total.render_lines("sallm_llm_total_tokens", slab))
        return "\n".join(lines) + "\n"

    def start_server(self, host="0.0.0.0", port=9464):
        """Serve Prometheus text on /metrics (daemon thread)."""
        if self._httpd is not None:
            return self._httpd.server_address
        metrics = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.split("?", 1)[0] != "/metrics":
                    self.send_response(404)
                    self.end_headers()
                    return
                body = metrics.render().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                return

        httpd = ThreadingHTTPServer((host, port), Handler)
        self._httpd = httpd
        self._thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        self._thread.start()
        return httpd.server_address

    def stop_server(self):
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        self._httpd = None
        self._thread = None
