"""ContextReceipt and budgeted prompt assembly."""

from __future__ import annotations

from dataclasses import dataclass, field

from sallm.context import estimate_tokens
from sallm.memory.types import VectorHit
from sallm.messages import assistant, system, user
from sallm.models import ModelProfile


@dataclass
class SectionSpend:
    name: str
    tokens: int
    included: bool = True
    note: str = ""


@dataclass
class ContextReceipt:
    """Public explanation of what entered the prompt and why."""

    profile: str
    profile_version: str
    budget: int
    sections: list[SectionSpend] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)
    omitted_messages: int = 0
    fallbacks: list[str] = field(default_factory=list)
    total_tokens: int = 0

    def as_dict(self) -> dict:
        return {
            "profile": self.profile,
            "profile_version": self.profile_version,
            "budget": self.budget,
            "total_tokens": self.total_tokens,
            "omitted_messages": self.omitted_messages,
            "fallbacks": list(self.fallbacks),
            "sections": [
                {
                    "name": s.name,
                    "tokens": s.tokens,
                    "included": s.included,
                    "note": s.note,
                }
                for s in self.sections
            ],
            "retrieved": list(self.retrieved),
        }


def _fit(text: str, budget: int) -> tuple[str, int, bool]:
    if budget <= 0 or not text:
        return "", 0, bool(text)
    tokens = estimate_tokens(text)
    if tokens <= budget:
        return text, tokens, False
    cut = max(1, budget * 4)
    trimmed = text[:cut].rstrip() + "…"
    return trimmed, estimate_tokens(trimmed), True


def compile_prompt_messages(
    *,
    profile: ModelProfile,
    prompt,  # Prompt-like: .system()
    recent_messages: list[dict],
    hits: list[VectorHit],
    retrieval_budget: int | None = None,
    history_budget: int | None = None,
) -> tuple[list[dict], ContextReceipt]:
    """Assemble system + memory + recent history under the profile budget."""
    receipt = ContextReceipt(
        profile=profile.model,
        profile_version=profile.version,
        budget=profile.prompt_budget,
    )
    messages: list[dict] = []

    sys_text = prompt.system()
    sys_text, sys_tok, sys_trim = _fit(sys_text, profile.prompt_budget)
    messages.append(system(sys_text))
    receipt.sections.append(
        SectionSpend("system", sys_tok, True, "trimmed" if sys_trim else "")
    )
    used = sys_tok

    r_budget = (
        retrieval_budget
        if retrieval_budget is not None
        else profile.retrieval_tokens
    )
    r_budget = min(r_budget, max(0, profile.prompt_budget - used))
    mem_parts = []
    for h in hits:
        mem_parts.append(
            f"[mem id={h.id} source={h.source_id or '-'} score={h.score:.3f}]\n{h.text}"
        )
        receipt.retrieved.append(
            {
                "id": h.id,
                "source_id": h.source_id,
                "score": h.score,
                "chars": len(h.text),
            }
        )
    if mem_parts:
        mem_text = "[Retrieved memory]\n" + "\n---\n".join(mem_parts)
        mem_text, mem_tok, mem_trim = _fit(mem_text, r_budget)
        if mem_text:
            messages.append(user(mem_text))
            used += mem_tok
            receipt.sections.append(
                SectionSpend(
                    "retrieval", mem_tok, True, "trimmed" if mem_trim else ""
                )
            )
        else:
            receipt.sections.append(SectionSpend("retrieval", 0, False, "no budget"))
    else:
        receipt.sections.append(SectionSpend("retrieval", 0, False, "no hits"))

    h_budget = (
        history_budget
        if history_budget is not None
        else profile.recent_history_tokens
    )
    h_budget = min(h_budget, max(0, profile.prompt_budget - used))
    rest = list(recent_messages)
    if rest and rest[0].get("role") == "system":
        rest = rest[1:]
    kept: list[dict] = []
    hist_tokens = 0
    for msg in reversed(rest):
        content = msg.get("content") or ""
        t = estimate_tokens(content) + 2
        if hist_tokens + t > h_budget and kept:
            break
        if hist_tokens + t > h_budget and not kept:
            content, t, _ = _fit(content, h_budget)
            role = msg.get("role") or "user"
            kept.append(
                assistant(content)
                if role == "assistant"
                else user(content)
            )
            hist_tokens += t
            break
        role = msg.get("role") or "user"
        if role == "assistant":
            kept.append(assistant(content))
        elif role == "system":
            kept.append(system(content))
        else:
            kept.append(user(content))
        hist_tokens += t
    kept.reverse()
    omitted = max(0, len(rest) - len(kept))
    receipt.omitted_messages = omitted
    messages.extend(kept)
    used += hist_tokens
    receipt.sections.append(
        SectionSpend(
            "history",
            hist_tokens,
            True,
            f"omitted={omitted}" if omitted else "",
        )
    )
    receipt.total_tokens = used
    if used > profile.prompt_budget:
        receipt.fallbacks.append("over_budget_estimate")
    return messages, receipt
