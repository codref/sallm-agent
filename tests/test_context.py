"""Unit tests for context optimizers."""

from __future__ import annotations

from sallm.context import (
    SUMMARY_PREFIX,
    MaxMessages,
    SummarizeOverflow,
    estimate_tokens,
)
from sallm.messages import assistant, system, user


def _msgs(n_pairs: int, *, with_system: bool = True):
    out = []
    if with_system:
        out.append(system("sys"))
    for i in range(n_pairs):
        out.append(user(f"u{i}"))
        out.append(assistant(f"a{i}"))
    return out


def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1


def test_max_messages_compact_returns_overflow():
    msgs = _msgs(5)
    view, overflow = MaxMessages(4).compact(msgs)
    assert len(view) == 5
    assert [m["content"] for m in overflow] == ["u0", "a0", "u1", "a1", "u2", "a2"]
    assert MaxMessages(4).prepare(msgs) == view
    assert view[0]["role"] == "system"
    assert [m["content"] for m in view[1:]] == ["u3", "a3", "u4", "a4"]
    assert len(msgs) == 11


def test_summarize_compact_returns_overflow():
    overflow = [user("x" * 80), assistant("y" * 80)]
    recent = [user("late-q"), assistant("late-a")]
    msgs = [system("sys"), *overflow, *recent]
    opt = SummarizeOverflow(threshold=1, keep_last=2, summarize_fn=lambda t: "S")
    view, aged = opt.compact(msgs)
    assert aged == overflow
    assert view[1]["content"].startswith(SUMMARY_PREFIX)


def test_max_messages_no_system():
    msgs = _msgs(3, with_system=False)
    view = MaxMessages(2).prepare(msgs)
    assert len(view) == 2
    assert [m["content"] for m in view] == ["u2", "a2"]


def test_max_messages_under_limit_is_copy():
    msgs = _msgs(1)
    view = MaxMessages(40).prepare(msgs)
    assert view == msgs
    assert view is not msgs


def test_max_messages_empty():
    assert MaxMessages(3).prepare([]) == []


def test_summarize_under_threshold_passthrough():
    calls = []

    def summarize(text):
        calls.append(text)
        return "SUM"

    msgs = _msgs(2)
    opt = SummarizeOverflow(threshold=10_000, keep_last=2, summarize_fn=summarize)
    view = opt.prepare(msgs)
    assert view == msgs
    assert calls == []


def test_summarize_overflow_injects_summary_keeps_tail():
    calls = []

    def summarize(text):
        calls.append(text)
        return "earlier stuff"

    # Long overflow to exceed low threshold
    overflow = [user("x" * 80), assistant("y" * 80)]
    recent = [user("late-q"), assistant("late-a")]
    msgs = [system("sys"), *overflow, *recent]
    opt = SummarizeOverflow(threshold=1, keep_last=2, summarize_fn=summarize)
    view = opt.prepare(msgs)

    assert view[0] == msgs[0]
    assert view[1]["role"] == "user"
    assert view[1]["content"].startswith(SUMMARY_PREFIX)
    assert "earlier stuff" in view[1]["content"]
    assert view[2:] == recent
    assert len(msgs) == 5  # canonical unchanged
    assert len(calls) == 1

    # Cache: second prepare with same overflow does not re-call
    view2 = opt.prepare(msgs)
    assert len(calls) == 1
    assert view2[1]["content"] == view[1]["content"]


def test_summarize_on_clear_drops_cache():
    n = {"calls": 0}

    def summarize(text):
        n["calls"] += 1
        return f"sum-{n['calls']}"

    overflow = [user("x" * 80), assistant("y" * 80)]
    recent = [user("q"), assistant("a")]
    msgs = [system("sys"), *overflow, *recent]
    opt = SummarizeOverflow(threshold=1, keep_last=2, summarize_fn=summarize)
    opt.prepare(msgs)
    assert n["calls"] == 1
    opt.on_clear()
    opt.prepare(msgs)
    assert n["calls"] == 2


def test_agent_clear_invokes_optimizer_on_clear():
    from sallm import Agent

    n = {"clears": 0}

    class Tracking:
        def prepare(self, messages):
            return list(messages)

        def on_clear(self):
            n["clears"] += 1

    agent = Agent(tools={}, context=Tracking())
    agent.clear()
    assert n["clears"] == 1


def test_agent_none_context_metrics_match():
    """Without an optimizer, prompt_messages equals context_messages."""
    from unittest.mock import patch

    from sallm import Agent

    fake = {
        "content": "hello",
        "reasoning": None,
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "elapsed_ms": 1.0,
    }
    with patch("sallm.agent.complete", return_value=fake):
        agent = Agent(tools={})
        result = agent.ask("hi")
    m = result["metrics"]
    assert m["context_messages"] == m["prompt_messages"]
    assert m["context_messages"] >= 2


def test_cli_build_context_selectors():
    from sallm.cli.chat import _build_context
    from sallm.context import MaxMessages, SummarizeOverflow

    none, label = _build_context(
        "none",
        model="m",
        api_base="http://x",
        max_context_messages=None,
        context_threshold=None,
        context_keep_last=None,
    )
    assert none is None and label == ""

    mm, label = _build_context(
        "max-messages",
        model="m",
        api_base="http://x",
        max_context_messages=12,
        context_threshold=None,
        context_keep_last=None,
    )
    assert isinstance(mm, MaxMessages)
    assert mm.max_messages == 12
    assert "MaxMessages(12)" == label

    so, label = _build_context(
        "summarize",
        model="m",
        api_base="http://x",
        max_context_messages=None,
        context_threshold=100,
        context_keep_last=3,
    )
    assert isinstance(so, SummarizeOverflow)
    assert so.threshold == 100
    assert so.keep_last == 3
    assert "SummarizeOverflow" in label
