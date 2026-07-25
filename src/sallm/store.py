"""In-memory text store — reference backend for tests.

Duck-typed store surface (any backend can replace this file):
  add(text, *, id=None, session_id=None) -> None
  query(text, k=5, *, session_id=None) -> list[str]
  clear(*, session_id=None) -> None

Owns: InMemoryStore only (token-overlap ranking).
Does not own: LanceDB, compactors, chunking, Agent.
"""

from __future__ import annotations

import hashlib
import re


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t}


class InMemoryStore:
    """Bag-of-words overlap store. No embeddings, no disk."""

    def __init__(self):
        # id -> {text, session_id, tokens}
        self._rows: dict[str, dict] = {}

    def add(self, text: str, *, id: str | None = None, session_id: str | None = None):
        text = text or ""
        if not text.strip():
            return
        row_id = id or hashlib.sha256(
            f"{session_id or ''}\n{text}".encode("utf-8")
        ).hexdigest()
        self._rows[row_id] = {
            "text": text,
            "session_id": session_id,
            "tokens": _tokens(text),
        }

    def query(
        self, text: str, k: int = 5, *, session_id: str | None = None
    ) -> list[str]:
        if k < 1:
            return []
        q = _tokens(text)
        scored = []
        for row in self._rows.values():
            if session_id is not None and row.get("session_id") != session_id:
                continue
            overlap = len(q & row["tokens"]) if q else 0
            scored.append((overlap, row["text"]))
        scored.sort(key=lambda x: (-x[0], x[1]))
        # Prefer positive overlap; still return something if all zero.
        out = [t for score, t in scored if score > 0][:k]
        if out:
            return out
        return [t for _, t in scored[:k]]

    def clear(self, *, session_id: str | None = None):
        if session_id is None:
            self._rows.clear()
            return
        drop = [
            rid
            for rid, row in self._rows.items()
            if row.get("session_id") == session_id
        ]
        for rid in drop:
            del self._rows[rid]
