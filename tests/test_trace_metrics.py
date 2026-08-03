"""Tracing + Prometheus session gauges for stack / receipt / control."""

from __future__ import annotations

from sallm.prom import SessionMetrics
from sallm.receipt import ContextReceipt, SectionSpend
from sallm.trace import Tracer


def test_turn_end_emits_stack_and_receipt_attrs():
    events = []
    tr = Tracer(events.append, session_id="sess-1")
    tr.turn_start("hi", [], model="m")
    receipt = ContextReceipt(
        profile="m",
        profile_version="1",
        budget=4096,
        sections=[
            SectionSpend("system", 100),
            SectionSpend("retrieval", 40),
            SectionSpend("history", 200),
        ],
        retrieved=[{"id": "h1"}],
        omitted_messages=2,
        total_tokens=340,
    )
    tr.turn_end(
        answer="ok",
        metrics={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "elapsed_ms": 1},
        messages=[{"role": "user", "content": "hi"}],
        stack=[{"skill": "converse", "depth": 0}, {"skill": "analyze", "depth": 1}],
        goal="find the code",
        receipt=receipt,
        control_decision={"action": "push", "skill": "analyze"},
        gated_chunks=3,
    )
    ask = [e for e in events if e.get("kind") == "turn"][-1]
    attrs = ask["attrs"]
    assert attrs["session.id"] == "sess-1"
    assert attrs["sallm.stack.depth"] == 2
    assert attrs["sallm.active_skill"] == "analyze"
    assert attrs["sallm.stack.path"] == "converse > analyze"
    assert attrs["sallm.goal"] == "find the code"
    assert attrs["sallm.receipt.total_tokens"] == 340
    assert attrs["sallm.receipt.history_tokens"] == 200
    assert attrs["sallm.retrieval.hits"] == 1
    assert attrs["sallm.control.action"] == "push"
    assert attrs["sallm.memory.gated_chunks"] == 3


def test_control_span_and_metrics_render():
    events = []
    metrics = SessionMetrics(session_id="s2")
    tr = Tracer(events.append, session_id="s2", metrics=metrics)
    tr.turn_start("x", [], model="m")
    tr.control(
        {
            "goal": "g",
            "action": "keep",
            "skill": "converse",
            "retrieval_query": "code",
            "fallback": False,
        }
    )
    tr.turn_end(
        answer="a",
        metrics={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "elapsed_ms": 1},
        messages=[],
        stack=[{"skill": "converse", "depth": 0}],
        goal="g",
        receipt={
            "budget": 4096,
            "total_tokens": 50,
            "omitted_messages": 0,
            "sections": [
                {"name": "system", "tokens": 20},
                {"name": "retrieval", "tokens": 0},
                {"name": "history", "tokens": 30},
            ],
            "retrieved": [],
            "fallbacks": [],
        },
    )
    ctrl = [e for e in events if e.get("kind") == "control"]
    assert len(ctrl) == 1
    assert ctrl[0]["attrs"]["sallm.control.action"] == "keep"
    text = metrics.render()
    assert 'sallm_stack_depth{session_id="s2"} 1' in text
    assert 'sallm_skill_active{session_id="s2",skill="converse"} 1' in text
    assert 'sallm_control_actions_total{session_id="s2",action="keep"} 1' in text
    assert 'sallm_receipt_section_tokens{session_id="s2",section="history"} 30' in text
    assert 'sallm_receipt_total_tokens{session_id="s2"} 50' in text
