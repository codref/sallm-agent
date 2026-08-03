"""Agent prompt templates — render and inspect; do not call the LLM or run tools."""

from __future__ import annotations

import json
from pathlib import Path

SYSTEM = """You are a helpful assistant.

Default behavior: answer the user directly in plain text.
Do not run tools for greetings, identity questions, opinions, explanations,
or anything you can answer from your own knowledge.

Tools are small command-line programs. To use them, reply with a fenced run block
containing one command per line (shell-style flags, no JSON):

```run
toolname --flag value
another --flag value
```

Multiple lines run as concurrent processes in one step.
If you are unsure of a tool's flags, run `toolname --help` inside a ```run block first.
Never invent tool output. After finished tool results, reply in short plain text.

{multi_step_policy}

Available tools:
{tools}
"""

MULTI_STEP_ON = """Multi-step mode is ON.
If the user asked for several sequential operations, run one tool (or one batch) at a time.
After each tool result, either emit another ```run block or answer in plain text when finished.
If a tool result starts with [intermediate], the work is not done — run that tool again.
Do not invent tool output."""

MULTI_STEP_OFF = """Multi-step mode is OFF.
Prefer a single ```run block when possible.
Exception: if a tool result starts with [intermediate], you must run that tool again
until you get a final (non-intermediate) result — then answer in plain text.
Do not invent tool output."""

CONTINUE_NUDGE = (
    "The previous tool result was intermediate (not finished). "
    "Emit another ```run block to call the same or next required tool. "
    "Do not give a final answer yet."
)

EARLY_ANSWER_NUDGE = (
    "You replied with text before the tool work finished. "
    "An intermediate tool result is still pending. "
    "Emit a ```run block now. Do not answer the user yet."
)

RESULTS_PREFIX = "Tool results:\n"


class CompiledProfile:
    """Neutral JSON artifact from the offline optimizer (no DSPy objects)."""

    def __init__(
        self,
        target_model: str,
        instructions: dict,
        demonstrations: dict,
        budgets: dict,
        metadata: dict,
    ):
        self.target_model = target_model
        self.instructions = instructions
        self.demonstrations = demonstrations
        self.budgets = budgets
        self.metadata = metadata

    @classmethod
    def load(cls, path: str | Path) -> "CompiledProfile":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            target_model=data.get("target_model") or "",
            instructions=dict(data.get("instructions") or {}),
            demonstrations=dict(data.get("demonstrations") or {}),
            budgets=dict(data.get("budgets") or {}),
            metadata=dict(data.get("metadata") or {}),
        )


class Prompt:
    """Visible agent prompt templates — build system text and named nudges."""

    SYSTEM = SYSTEM
    MULTI_STEP_ON = MULTI_STEP_ON
    MULTI_STEP_OFF = MULTI_STEP_OFF
    CONTINUE_NUDGE = CONTINUE_NUDGE
    EARLY_ANSWER_NUDGE = EARLY_ANSWER_NUDGE
    RESULTS_PREFIX = RESULTS_PREFIX

    def __init__(
        self,
        *,
        tools_text: str = "",
        multi_step: bool = True,
        extra: str | None = None,
        skill_prompt: str = "",
        goal: str = "",
        compiled: CompiledProfile | None = None,
    ):
        self.tools_text = tools_text or "(none)"
        self.multi_step = multi_step
        self.extra = (extra or "").strip() or None
        self.skill_prompt = (skill_prompt or "").strip()
        self.goal = (goal or "").strip()
        self.compiled = compiled

    def policy(self) -> str:
        return self.MULTI_STEP_ON if self.multi_step else self.MULTI_STEP_OFF

    def system(self) -> str:
        converse_extra = ""
        if self.compiled and self.compiled.instructions.get("converse"):
            converse_extra = str(self.compiled.instructions["converse"]).strip()
        base = self.SYSTEM.format(
            tools=self.tools_text,
            multi_step_policy=self.policy(),
        )
        parts = []
        if self.extra:
            parts.append(self.extra.rstrip())
        if converse_extra:
            parts.append(converse_extra)
        if self.skill_prompt:
            parts.append(self.skill_prompt)
        if self.goal:
            parts.append(f"Current goal: {self.goal}")
        parts.append(base)
        return "\n\n".join(parts)

    def as_dict(self) -> dict:
        return {
            "extra": self.extra,
            "multi_step": self.multi_step,
            "policy": self.policy(),
            "tools_text": self.tools_text,
            "skill_prompt": self.skill_prompt,
            "goal": self.goal,
            "system": self.system(),
            "continue_nudge": self.CONTINUE_NUDGE,
            "early_answer_nudge": self.EARLY_ANSWER_NUDGE,
            "results_prefix": self.RESULTS_PREFIX,
            "chars": {
                "system": len(self.system()),
                "policy": len(self.policy()),
                "tools_text": len(self.tools_text),
                "extra": len(self.extra or ""),
            },
        }

    def preview(self) -> str:
        d = self.as_dict()
        lines = [
            f"=== Prompt preview ({d['chars']['system']} chars system) ===",
            f"multi_step: {d['multi_step']}",
            f"goal: {self.goal or '(none)'}",
            "",
            f"--- full system ({d['chars']['system']} chars) ---",
            d["system"],
            "",
            "--- nudges ---",
            f"CONTINUE_NUDGE ({len(self.CONTINUE_NUDGE)} chars):",
            self.CONTINUE_NUDGE,
            "",
            f"EARLY_ANSWER_NUDGE ({len(self.EARLY_ANSWER_NUDGE)} chars):",
            self.EARLY_ANSWER_NUDGE,
            "",
            f"RESULTS_PREFIX: {self.RESULTS_PREFIX!r}",
        ]
        return "\n".join(lines)
