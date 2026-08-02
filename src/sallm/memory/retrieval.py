"""Query string preparation for embedding-side retrieval."""

from __future__ import annotations

from sallm.models import EmbeddingProfile, resolve_embedding_profile


class DefaultQueryComposer:
    """Modes: raw | instruct | rewrite.

    ``rewrite`` expects the caller to pass an already-rewritten sentence as
    ``query`` (produced by the control LLM). This composer then applies the
    Qwen instruct template so the embedding model sees Instruct+Query.
    """

    def __init__(self, profile: EmbeddingProfile | None = None):
        self.profile = profile or resolve_embedding_profile()

    def compose(self, query: str, *, goal: str = "", mode: str = "instruct") -> str:
        q = (query or "").strip()
        kind = (mode or "instruct").strip().lower()
        if kind == "raw":
            return q
        # instruct and rewrite both use the Qwen query instruction format.
        # For rewrite, ``query`` is already a standalone retrieval sentence.
        if not q and goal:
            q = goal.strip()
        return self.profile.instruct_template.format(query=q)
