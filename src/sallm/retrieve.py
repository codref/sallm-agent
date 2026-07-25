"""Compose economy + storage into one context.prepare() facade.

Owns: CompactAndRetrieve orchestration only.
Does not own: LanceDB details, embeddings, CLI, Agent loop.

Order: compact → chunk+push overflow → offload oversized in-view →
session-scoped pull → inject retrieved block.
Push/pull is automatic here — not a CliTool / ```run call.
"""

from __future__ import annotations

import hashlib

from .chunk import chunk_text
from .context import estimate_tokens
from .messages import user

RETRIEVED_PREFIX = "[Retrieved memory]\n"
OFFLOAD_STUB = (
    "[Large message stored in memory as chunks; "
    "ask questions to retrieve relevant parts.]"
)
SCOPE_SESSION = "session"
SCOPE_ALL = "all"


def _latest_user_text(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content") or ""
    return ""


def _chunk_id(session_id: str, chunk: str) -> str:
    return hashlib.sha256(f"{session_id}\n{chunk}".encode("utf-8")).hexdigest()


class CompactAndRetrieve:
    """Facade: one prepare() that Agent already understands."""

    def __init__(
        self,
        compactor,
        memory,
        *,
        session_id: str,
        k: int = 4,
        chunk_tokens: int = 512,
        chunk_overlap_tokens: int = 64,
        memory_scope: str = SCOPE_SESSION,
    ):
        if compactor is None:
            raise ValueError("compactor is required")
        if memory is None:
            raise ValueError("memory is required")
        if not session_id:
            raise ValueError("session_id is required")
        if k < 1:
            raise ValueError("k must be >= 1")
        scope = (memory_scope or SCOPE_SESSION).strip().lower()
        if scope not in (SCOPE_SESSION, SCOPE_ALL):
            raise ValueError("memory_scope must be 'session' or 'all'")
        self.compactor = compactor
        self.memory = memory
        self.session_id = session_id
        self.k = k
        self.chunk_tokens = chunk_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.memory_scope = scope

    def on_clear(self):
        on_clear = getattr(self.compactor, "on_clear", None)
        if on_clear is not None:
            on_clear()
        clear = getattr(self.memory, "clear", None)
        if clear is not None:
            clear(session_id=self.session_id)

    def _push_text(self, text: str):
        for chunk in chunk_text(
            text,
            max_tokens=self.chunk_tokens,
            overlap_tokens=self.chunk_overlap_tokens,
        ):
            self.memory.add(
                chunk,
                id=_chunk_id(self.session_id, chunk),
                session_id=self.session_id,
            )

    def _push_messages(self, messages: list[dict]):
        for msg in messages:
            content = msg.get("content") or ""
            if content.strip():
                self._push_text(content)

    def _offload_oversized(self, view: list[dict]) -> list[dict]:
        out = []
        for msg in view:
            role = msg.get("role")
            content = msg.get("content") or ""
            if role == "system" or estimate_tokens(content) <= self.chunk_tokens:
                out.append(msg)
                continue
            self._push_text(content)
            preview = content[:200].replace("\n", " ")
            stub = OFFLOAD_STUB
            if preview:
                stub = f"{OFFLOAD_STUB}\nPreview: {preview}…"
            out.append({"role": role, "content": stub})
        return out

    def _query_filter_session_id(self) -> str | None:
        if self.memory_scope == SCOPE_ALL:
            return None
        return self.session_id

    def prepare(self, messages: list[dict]) -> list[dict]:
        compact = getattr(self.compactor, "compact", None)
        if compact is not None:
            view, overflow = compact(messages)
        else:
            view = self.compactor.prepare(messages)
            overflow = []

        if overflow:
            self._push_messages(overflow)

        view = self._offload_oversized(view)

        query_text = _latest_user_text(messages)
        if estimate_tokens(query_text) > self.chunk_tokens:
            parts = chunk_text(
                query_text,
                max_tokens=self.chunk_tokens,
                overlap_tokens=self.chunk_overlap_tokens,
            )
            query_text = parts[-1] if parts else query_text

        hits = []
        if query_text.strip():
            hits = self.memory.query(
                query_text,
                k=self.k,
                session_id=self._query_filter_session_id(),
            )

        if not hits:
            return view

        retrieved = user(RETRIEVED_PREFIX + "\n---\n".join(hits))
        if view and view[0].get("role") == "system":
            return [view[0], retrieved, *view[1:]]
        return [retrieved, *view]
