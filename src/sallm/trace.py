"""Optional OpenTelemetry-shaped tracing — plain dicts, zero deps, no SDK.

When Agent.trace is None the agent never calls into this module.
Events follow GenAI-ish attribute names so JSONL and OTLP stay aligned.

Correlation: one session.id per Tracer, one trace_id per ask() turn,
one root `ask` span closed at turn end (children nest under it).
"""

from __future__ import annotations

import json
import secrets
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_TRUNCATE = 512


def _trace_id():
    return secrets.token_hex(16)


def _span_id():
    return secrets.token_hex(8)


def _session_id():
    return secrets.token_hex(8)


def _now_ns():
    return time.time_ns()


def truncate(text, limit=DEFAULT_TRUNCATE):
    """Truncate one content field; limit<=0 means no truncation.

    Callers must truncate each message/field separately — never the joined prompt.
    """
    if text is None:
        return ""
    text = str(text)
    if limit is None or limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"…(+{len(text) - limit})"


def multi_sink(*sinks):
    """Fan-out to several emit callables."""

    def emit(event):
        for sink in sinks:
            sink(event)

    return emit


def jsonl_sink(path):
    """Append one JSON object per line to path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def emit(event):
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    return emit


def _attr_value(v):
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int) and not isinstance(v, bool):
        return {"intValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    if isinstance(v, (list, dict)):
        return {"stringValue": json.dumps(v, ensure_ascii=False, default=str)}
    return {"stringValue": str(v)}


def _otlp_attributes(attrs):
    return [{"key": k, "value": _attr_value(v)} for k, v in (attrs or {}).items()]


def otlp_http_sink(endpoint, service_name="sallm"):
    """POST each event as one OTLP/HTTP JSON span. Failures are swallowed."""
    endpoint = endpoint.rstrip("/")
    if not endpoint.endswith("/v1/traces"):
        endpoint = endpoint + "/v1/traces"
    warned = {"done": False}

    def emit(event):
        if event.get("otlp") is False:
            return
        start_ns = event.get("start_ns") or event.get("ts_ns") or _now_ns()
        end_ns = event.get("end_ns") or start_ns
        attrs = dict(event.get("attrs") or {})
        resource_attrs = [
            {"key": "service.name", "value": {"stringValue": service_name}},
        ]
        # Mirror session onto the resource so Tempo/Grafana tag search finds it.
        sid = attrs.get("session.id")
        if sid:
            resource_attrs.append(
                {"key": "session.id", "value": {"stringValue": str(sid)}}
            )
        body = {
            "resourceSpans": [
                {
                    "resource": {"attributes": resource_attrs},
                    "scopeSpans": [
                        {
                            "scope": {"name": "sallm", "version": "0.1.0"},
                            "spans": [
                                {
                                    "traceId": event["trace_id"],
                                    "spanId": event["span_id"],
                                    "parentSpanId": event.get("parent_id") or "",
                                    "name": event.get("name")
                                    or event.get("kind")
                                    or "event",
                                    "kind": 1,  # INTERNAL
                                    "startTimeUnixNano": str(start_ns),
                                    "endTimeUnixNano": str(end_ns),
                                    "attributes": _otlp_attributes(attrs),
                                    "status": {"code": 1},  # OK
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=2) as resp:
                resp.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if not warned["done"]:
                print(f"[sallm trace] otlp emit failed: {exc}", file=sys.stderr)
                warned["done"] = True

    return emit


class Tracer:
    """Owns session + turn ids; emit is any callable(event_dict).

    debug: store prompt / context / completion text on spans (truncated).
    truncate: max chars per content field (each message, completion, etc.);
    <=0 means unlimited. Joined views are built from already-truncated fields
    and are not truncated again as one blob.
    """

    def __init__(
        self,
        emit,
        debug=False,
        truncate=DEFAULT_TRUNCATE,
        session_id=None,
        include_messages=None,
        metrics=None,
    ):
        self.emit = emit
        self.debug = bool(debug)
        # Back-compat: include_messages=True implies debug.
        if include_messages is not None:
            self.debug = self.debug or bool(include_messages)
        self.truncate = DEFAULT_TRUNCATE if truncate is None else int(truncate)
        self.session_id = session_id or _session_id()
        self.metrics = metrics  # SessionMetrics | None
        self.turn_index = 0
        self.trace_id = None
        self.turn_span_id = None
        self._turn_start_ns = None
        self._llm_span_id = None

    def _t(self, text):
        """Truncate a single content field (not a joined multi-message string)."""
        return truncate(text, self.truncate)

    def _base_attrs(self):
        return {
            "session.id": self.session_id,
            "gen_ai.conversation.id": self.session_id,
            "turn.index": self.turn_index,
        }

    def _event(
        self,
        kind,
        attrs,
        parent_id=None,
        span_id=None,
        start_ns=None,
        end_ns=None,
        name=None,
        otlp=True,
    ):
        now = _now_ns()
        start_ns = start_ns if start_ns is not None else now
        end_ns = end_ns if end_ns is not None else now
        merged = {**self._base_attrs(), **(attrs or {})}
        event = {
            "ts": time.time(),
            "ts_ns": now,
            "start_ns": start_ns,
            "end_ns": end_ns,
            "trace_id": self.trace_id,
            "span_id": span_id or _span_id(),
            "parent_id": parent_id,
            "kind": kind,
            "name": name or kind,
            "attrs": merged,
            "otlp": otlp,
        }
        self.emit(event)
        return event

    def _context_attrs(self, messages, *, full=False):
        msgs = messages or []
        attrs = {"context.messages": len(msgs)}
        snapshot = []
        for m in msgs:
            role = m.get("role") or "?"
            content = m.get("content") or ""
            entry = {"role": role, "chars": len(content)}
            if full or self.debug:
                # Per-message truncate (limit applies to each content, not the list).
                entry["content"] = self._t(content)
            snapshot.append(entry)
        attrs["context.snapshot"] = snapshot
        if self.debug:
            # Join already-truncated fields; do not truncate the joined string.
            lines = []
            for m in msgs:
                role = m.get("role") or "?"
                content = self._t(m.get("content") or "")
                lines.append(f"{role}: {content}")
            attrs["gen_ai.input.messages"] = "\n---\n".join(lines)
        return attrs

    def turn_start(self, user_text, messages, model=None):
        self.turn_index += 1
        self.trace_id = _trace_id()
        self.turn_span_id = _span_id()
        self._turn_start_ns = _now_ns()
        self._llm_span_id = None
        attrs = {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.request.model": model or "",
            "user.text.chars": len(user_text or ""),
            **self._context_attrs(messages),
        }
        if self.debug:
            attrs["user.text"] = self._t(user_text or "")
        # JSONL marker only — OTLP gets one closed `ask` span at turn_end.
        self._event(
            "turn.start",
            attrs,
            parent_id=None,
            span_id=self.turn_span_id,
            name="ask",
            start_ns=self._turn_start_ns,
            otlp=False,
        )

    def llm(
        self,
        model,
        metrics,
        content=None,
        reasoning=None,
        messages=None,
        *,
        name="chat",
        operation="chat",
    ):
        metrics = metrics or {}
        content = content or ""
        reasoning = reasoning or ""
        end_ns = _now_ns()
        start_ns = end_ns - int((metrics.get("elapsed_ms") or 0) * 1_000_000)
        span_name = name or "chat"
        attrs = {
            "gen_ai.operation.name": operation or "chat",
            "gen_ai.request.model": model or "",
            "gen_ai.usage.input_tokens": metrics.get("prompt_tokens", 0),
            "gen_ai.usage.output_tokens": metrics.get("completion_tokens", 0),
            "gen_ai.usage.total_tokens": metrics.get("total_tokens", 0),
            "elapsed_ms": metrics.get("elapsed_ms", 0.0),
            "content.chars": len(content),
            "reasoning.chars": len(reasoning) if reasoning else 0,
        }
        if self.debug:
            if messages is not None:
                attrs.update(self._context_attrs(messages, full=True))
            attrs["gen_ai.completion"] = self._t(content)
            if reasoning:
                attrs["gen_ai.reasoning"] = self._t(reasoning)
        span_id = _span_id()
        self._llm_span_id = span_id
        self._event(
            "llm",
            attrs,
            parent_id=self.turn_span_id,
            span_id=span_id,
            name=span_name,
            start_ns=start_ns,
            end_ns=end_ns,
        )
        if self.metrics is not None:
            self.metrics.observe_llm(
                model=model,
                prompt_tokens=metrics.get("prompt_tokens", 0),
                completion_tokens=metrics.get("completion_tokens", 0),
                total_tokens=metrics.get("total_tokens", 0),
                elapsed_ms=metrics.get("elapsed_ms", 0.0),
            )
        return span_id

    def tool(
        self,
        name,
        command,
        observation=None,
        stdout=None,
        stderr=None,
        returncode=None,
        intermediate=False,
        elapsed_ms=None,
        parent_id=None,
    ):
        end_ns = _now_ns()
        if elapsed_ms is not None:
            start_ns = end_ns - int(elapsed_ms * 1_000_000)
        else:
            start_ns = end_ns
        attrs = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": name or "",
            "tool.command": list(command or []),
            "tool.returncode": returncode if returncode is not None else 0,
            "tool.intermediate": bool(intermediate),
        }
        if elapsed_ms is not None:
            attrs["elapsed_ms"] = elapsed_ms
        if observation is not None:
            attrs["tool.observation"] = self._t(observation)
        if self.debug:
            if stdout is not None:
                attrs["tool.stdout"] = self._t(stdout)
            if stderr is not None:
                attrs["tool.stderr"] = self._t(stderr)
        span_name = f"tool {name}" if name else "tool"
        self._event(
            "tool",
            attrs,
            parent_id=parent_id or self._llm_span_id or self.turn_span_id,
            name=span_name,
            start_ns=start_ns,
            end_ns=end_ns,
        )
        if self.metrics is not None:
            self.metrics.observe_tool(
                name=name,
                elapsed_ms=elapsed_ms or 0.0,
                returncode=returncode if returncode is not None else 0,
            )

    def nudge(self, text):
        attrs = {"nudge.text.chars": len(text or "")}
        if self.debug:
            attrs["nudge.text"] = self._t(text or "")
        self._event(
            "nudge",
            attrs,
            parent_id=self.turn_span_id,
            name="nudge",
        )

    def rejected(self, raw, nudge=None):
        attrs = {"rejected.raw.chars": len(raw or "")}
        if self.debug:
            attrs["rejected.raw"] = self._t(raw or "")
            if nudge:
                attrs["nudge.text"] = self._t(nudge)
        self._event("rejected", attrs, parent_id=self.turn_span_id, name="rejected")

    def control(self, decision):
        """Emit a control/routing span (goal, skill action, retrieval query)."""
        decision = decision or {}
        if hasattr(decision, "__dict__") and not isinstance(decision, dict):
            decision = {
                "goal": getattr(decision, "goal", ""),
                "action": getattr(decision, "action", ""),
                "skill": getattr(decision, "skill", ""),
                "retrieval_query": getattr(decision, "retrieval_query", ""),
                "fallback": bool(getattr(decision, "fallback", False)),
            }
        attrs = {
            "gen_ai.operation.name": "control",
            "sallm.control.action": str(decision.get("action") or ""),
            "sallm.control.skill": str(decision.get("skill") or ""),
            "sallm.control.fallback": bool(decision.get("fallback")),
            "sallm.goal": str(decision.get("goal") or ""),
            "sallm.retrieval_query": str(decision.get("retrieval_query") or ""),
        }
        self._event(
            "control",
            attrs,
            parent_id=self.turn_span_id,
            name="control",
        )
        if self.metrics is not None:
            self.metrics.observe_control(
                action=attrs["sallm.control.action"],
                skill=attrs["sallm.control.skill"],
            )

    @staticmethod
    def _receipt_attrs(receipt) -> dict:
        if receipt is None:
            return {}
        if hasattr(receipt, "as_dict"):
            receipt = receipt.as_dict()
        if not isinstance(receipt, dict):
            return {}
        sections = {s.get("name"): s for s in (receipt.get("sections") or []) if isinstance(s, dict)}
        attrs = {
            "sallm.receipt.budget": int(receipt.get("budget") or 0),
            "sallm.receipt.total_tokens": int(receipt.get("total_tokens") or 0),
            "sallm.receipt.omitted_messages": int(receipt.get("omitted_messages") or 0),
            "sallm.receipt.fallbacks": list(receipt.get("fallbacks") or []),
            "sallm.retrieval.hits": len(receipt.get("retrieved") or []),
        }
        for name in ("system", "retrieval", "history"):
            sec = sections.get(name) or {}
            attrs[f"sallm.receipt.{name}_tokens"] = int(sec.get("tokens") or 0)
            if sec.get("note"):
                attrs[f"sallm.receipt.{name}_note"] = str(sec.get("note"))
        return attrs

    @staticmethod
    def _stack_attrs(stack, goal="") -> dict:
        frames = []
        for f in stack or []:
            if hasattr(f, "skill"):
                frames.append(
                    {
                        "skill": f.skill,
                        "depth": getattr(f, "depth", len(frames)),
                        "note": getattr(f, "note", "") or "",
                    }
                )
            elif isinstance(f, dict):
                frames.append(
                    {
                        "skill": f.get("skill") or "",
                        "depth": f.get("depth", len(frames)),
                        "note": f.get("note") or "",
                    }
                )
        active = frames[-1]["skill"] if frames else ""
        return {
            "sallm.goal": goal or "",
            "sallm.active_skill": active,
            "sallm.stack.depth": len(frames),
            "sallm.stack": frames,
            "sallm.stack.path": " > ".join(f["skill"] for f in frames),
        }

    def turn_end(
        self,
        answer,
        metrics,
        messages,
        stopped=None,
        *,
        stack=None,
        goal="",
        receipt=None,
        control_decision=None,
        gated_chunks: int = 0,
        extract_miss_flush: bool = False,
    ):
        metrics = metrics or {}
        end_ns = _now_ns()
        start_ns = self._turn_start_ns or end_ns
        attrs = {
            "gen_ai.operation.name": "invoke_agent",
            "answer.chars": len(answer or ""),
            "gen_ai.usage.input_tokens": metrics.get("prompt_tokens", 0),
            "gen_ai.usage.output_tokens": metrics.get("completion_tokens", 0),
            "gen_ai.usage.total_tokens": metrics.get("total_tokens", 0),
            "elapsed_ms": metrics.get("elapsed_ms", 0.0),
            "sallm.memory.gated_chunks": int(gated_chunks or 0),
            "sallm.extract.miss_flush": bool(extract_miss_flush),
            **self._context_attrs(messages),
            **self._stack_attrs(stack, goal=goal),
            **self._receipt_attrs(receipt),
        }
        if control_decision is not None:
            if hasattr(control_decision, "action"):
                attrs["sallm.control.action"] = str(control_decision.action or "")
                attrs["sallm.control.skill"] = str(control_decision.skill or "")
            elif isinstance(control_decision, dict):
                attrs["sallm.control.action"] = str(control_decision.get("action") or "")
                attrs["sallm.control.skill"] = str(control_decision.get("skill") or "")
        if stopped:
            attrs["stopped"] = stopped
        if self.debug:
            attrs["gen_ai.output"] = self._t(answer or "")
        # Single root span for the whole turn (OTLP + JSONL).
        self._event(
            "turn",
            attrs,
            parent_id=None,
            span_id=self.turn_span_id,
            name="ask",
            start_ns=start_ns,
            end_ns=end_ns,
            otlp=True,
        )
        if self.metrics is not None:
            self.metrics.observe_turn(
                metrics,
                context_messages=len(messages or []),
            )
            self.metrics.observe_stack(
                depth=int(attrs.get("sallm.stack.depth") or 0),
                active_skill=str(attrs.get("sallm.active_skill") or ""),
                goal=str(attrs.get("sallm.goal") or ""),
            )
            self.metrics.observe_receipt(receipt)
        self.trace_id = None
        self.turn_span_id = None
        self._turn_start_ns = None
        self._llm_span_id = None
