"""Consciousness loops — read the transcript, return system-prompt addenda.

Do not embed, index, or run tools. Advice only steers the main agent.
"""

from __future__ import annotations

from typing import Protocol

from .messages import DEFAULT_API_BASE, DEFAULT_MODEL
from .tools import tool_descriptions

ADVICE_MARKER = "[Tool advice]"

TOOL_ADVISOR_PROMPT = """You are a tool advisor for another assistant.
You do NOT answer the user. You do NOT invent tool output.
Given available CLI tools and the recent conversation, suggest which tools
(if any) the assistant should consider on this turn.

Rules:
- Name only tools from the available list.
- Prefer "none" for greetings, chit-chat, or when tools are irrelevant.
- For questions about a long document/transcript that was pasted earlier,
  prefer memory (add then search) when memory is available; never dig.
- dig is a toy treasure game only — never for documents or Q&A.
- Reply in plain text only: either "none" or 1-3 short directive sentences.
- No markdown fences, no JSON.

Available tools:
{tools}

Recent conversation:
{transcript}
"""


class Consciousness(Protocol):
    """Analyze messages; return a system addendum (may be empty)."""

    def advise(self, messages: list[dict], tools: dict) -> str: ...


def normalize_consciousness(value) -> list:
    """Accept None, one Consciousness, or a list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def join_addenda(parts: list[str]) -> str:
    """Join non-empty addenda under [Tool advice] markers."""
    cleaned = [p.strip() for p in parts if p and str(p).strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return f"{ADVICE_MARKER}\n{cleaned[0]}"
    blocks = [f"{ADVICE_MARKER} ({i})\n{p}" for i, p in enumerate(cleaned, 1)]
    return "\n\n".join(blocks)


def _format_transcript(
    messages: list[dict],
    *,
    max_messages: int = 6,
    max_chars: int = 2000,
    per_message_chars: int = 400,
) -> str:
    """Compact recent non-system messages for the advisor (keep this small)."""
    rest = [m for m in messages if (m.get("role") or "") != "system"]
    if max_messages > 0:
        rest = rest[-max_messages:]
    lines = []
    total = 0
    for m in rest:
        role = m.get("role") or "?"
        content = (m.get("content") or "").strip()
        if len(content) > per_message_chars:
            content = content[: per_message_chars - 3] + "..."
        line = f"{role}: {content}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines) if lines else "(empty)"


class ToolAdvisor:
    """LLM consciousness: suggest which tools the main agent should consider."""

    # Keep advisor cheap: short context + hard cap on generation.
    DEFAULT_MAX_TOKENS = 64

    def __init__(
        self,
        complete_fn=None,
        *,
        model=None,
        api_base=None,
        max_messages: int = 6,
        max_chars: int = 2000,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.complete_fn = complete_fn
        self.model = model or DEFAULT_MODEL
        self.api_base = api_base or DEFAULT_API_BASE
        self.max_messages = max_messages
        self.max_chars = max_chars
        self.max_tokens = max_tokens

    def advise(self, messages: list[dict], tools: dict) -> str:
        if not tools:
            return ""
        transcript = _format_transcript(
            messages,
            max_messages=self.max_messages,
            max_chars=self.max_chars,
        )
        prompt = TOOL_ADVISOR_PROMPT.format(
            tools=tool_descriptions(tools),
            transcript=transcript,
        )
        if self.complete_fn is not None:
            raw = self.complete_fn(prompt)
            text = raw if isinstance(raw, str) else (raw or "")
        else:
            from .llm import complete

            result = complete(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                api_base=self.api_base,
                max_tokens=self.max_tokens,
            )
            text = result.get("content") or ""

        text = (text or "").strip()
        if not text:
            return ""
        # Normalize soft "none" answers to empty (no system noise).
        if text.lower().rstrip(".") in ("none", "n/a", "no tools", "no tool"):
            return ""
        # Hard trim runaway generations.
        if len(text) > 500:
            text = text[:497] + "..."
        return text
