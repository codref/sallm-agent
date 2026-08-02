"""Write-time memory gates — keep short questions out of the vector index."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

# Kinds that always index (extractor output, etc.).
_ALWAYS = frozenset({"fact", "derived"})


@runtime_checkable
class MemoryGate(Protocol):
    def accept(self, text: str, *, kind: str = "raw") -> bool: ...


class HeuristicMemoryGate:
    """Deterministic gate: long dumps and facts in; short interrogatives out.

    Token estimate is chars/4 (same rough scale used elsewhere in the stack).
    No LLM call — cheap and testable.
    """

    SHORT_TOKEN_LIMIT = 80

    def accept(self, text: str, *, kind: str = "raw") -> bool:
        k = (kind or "raw").strip().lower()
        if k in _ALWAYS:
            return True
        t = (text or "").strip()
        if not t:
            return False
        tokens = max(1, len(t) // 4)
        # Long briefing dumps always index regardless of punctuation.
        if tokens >= self.SHORT_TOKEN_LIMIT:
            return True
        # Short questions pollute dense retrieval with near-duplicate queries.
        if t.endswith("?"):
            return False
        if "?" in t and tokens < 12:
            return False
        return True


class PassThroughGate:
    """Accept everything (opt-out of gating)."""

    def accept(self, text: str, *, kind: str = "raw") -> bool:
        return bool((text or "").strip())
