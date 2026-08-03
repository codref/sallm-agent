"""Pluggable context optimizers — economy only (trim / summarize).

Owns: MaxMessages, SummarizeOverflow, estimate_tokens.
Does not own: vector stores, chunking, retrieval, embeddings, Agent loop.

Optimizers implement prepare(messages) -> view and optionally
compact(messages) -> (view, overflow). They must not mutate the input list.

Invariants:
- Preserve the system message at index 0 when present.
- Prefer keeping the tail intact (tool results / continue nudges).
- Return a new list (or a shallow copy); never mutate Agent.messages.
"""

from __future__ import annotations

import hashlib

from .messages import user

SUMMARY_PREFIX = "[Context summary of earlier conversation]\n"


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token). Good enough for budgets."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _split_system(messages: list[dict]):
    if messages and messages[0].get("role") == "system":
        return messages[0], messages[1:]
    return None, messages


def _join_messages(msgs: list[dict]) -> str:
    parts = []
    for m in msgs:
        role = m.get("role") or "?"
        content = m.get("content") or ""
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


class MaxMessages:
    """Keep system message + last `max_messages` non-system messages."""

    def __init__(self, max_messages: int = 40):
        if max_messages < 1:
            raise ValueError("max_messages must be >= 1")
        self.max_messages = max_messages

    def compact(self, messages: list[dict]) -> tuple[list[dict], list[dict]]:
        """Return (view, overflow). overflow = messages dropped from the view."""
        if not messages:
            return [], []
        system, rest = _split_system(messages)
        if len(rest) <= self.max_messages:
            view = [system, *rest] if system is not None else list(rest)
            return view, []
        overflow = list(rest[: -self.max_messages])
        trimmed = rest[-self.max_messages :]
        view = [system, *trimmed] if system is not None else list(trimmed)
        return view, overflow

    def prepare(self, messages: list[dict]) -> list[dict]:
        return self.compact(messages)[0]


class SummarizeOverflow:
    """When older turns exceed a token budget, inject one summary + keep the tail.

    Full transcript stays on Agent.messages. `summarize_fn(text) -> str` is
    caller-owned (LLM call, smaller model, or heuristic).
    """

    def __init__(
        self,
        threshold: int = 2000,
        keep_last: int = 10,
        summarize_fn=None,
    ):
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        if keep_last < 1:
            raise ValueError("keep_last must be >= 1")
        if summarize_fn is None:
            raise ValueError("summarize_fn is required")
        self.threshold = threshold
        self.keep_last = keep_last
        self.summarize_fn = summarize_fn
        self._cache_key = None
        self._cache_summary = None

    def on_clear(self):
        self._cache_key = None
        self._cache_summary = None

    def compact(self, messages: list[dict]) -> tuple[list[dict], list[dict]]:
        """Return (view, overflow). overflow = aged messages replaced by summary."""
        if not messages:
            return [], []
        system, rest = _split_system(messages)
        if len(rest) <= self.keep_last:
            return list(messages), []

        overflow = list(rest[: -self.keep_last])
        recent = rest[-self.keep_last :]
        overflow_text = _join_messages(overflow)
        if estimate_tokens(overflow_text) <= self.threshold:
            return list(messages), []

        key = hashlib.sha256(overflow_text.encode("utf-8")).hexdigest()
        if key != self._cache_key:
            summary = self.summarize_fn(overflow_text)
            self._cache_key = key
            self._cache_summary = summary if summary is not None else ""

        summary_msg = user(SUMMARY_PREFIX + (self._cache_summary or ""))
        if system is not None:
            view = [system, summary_msg, *recent]
        else:
            view = [summary_msg, *recent]
        return view, overflow

    def prepare(self, messages: list[dict]) -> list[dict]:
        return self.compact(messages)[0]
