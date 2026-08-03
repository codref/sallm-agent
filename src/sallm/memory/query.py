"""Query pipeline: rewrite → HyDE → instruct → embed → search."""

from __future__ import annotations

from dataclasses import dataclass

from sallm.models import ModelProfile

from .config import RetrievalConfig
from .retrieval import DefaultQueryComposer
from .types import VectorHit, VectorQuery


HYDE_INSTRUCTION = """Write a short factual passage that would answer the question.
Do not say you are hypothesizing. Passage only, 2-4 sentences.
Question: {query}
"""


class HyDE:
    """Hypothetical Document Embeddings — one short LLM call, then embed the passage."""

    def __init__(self, profile: ModelProfile, *, max_tokens: int = 128):
        self.profile = profile
        self.max_tokens = max_tokens

    def expand(self, query: str) -> tuple[str, dict]:
        from sallm.llm import complete
        from sallm.messages import user

        q = (query or "").strip()
        if not q:
            return "", {"content": ""}
        prompt = HYDE_INSTRUCTION.format(query=q)
        result = complete(
            model=self.profile.model,
            messages=[user(prompt)],
            api_base=self.profile.api_base,
            max_tokens=self.max_tokens,
        )
        passage = (result.get("content") or "").strip()
        return passage, result


@dataclass(frozen=True)
class RetrieveResult:
    hits: list[VectorHit]
    query_text: str
    embed_text: str
    search_text: str
    hyde_result: dict | None = None


def retrieve_hits(
    *,
    store,
    embedder,
    composer: DefaultQueryComposer,
    session_id: str,
    user_text: str,
    goal: str,
    config: RetrievalConfig,
    top_k: int,
    retrieval_query: str = "",
    hyde: HyDE | None = None,
) -> RetrieveResult:
    """Rewrite → optional HyDE → instruct wrap → embed → dense/hybrid search."""
    q = (user_text or "").strip()
    if config.use_rewrite and (retrieval_query or "").strip():
        q = retrieval_query.strip()
    if not q and (goal or "").strip():
        q = goal.strip()
    if not q:
        return RetrieveResult(
            hits=[], query_text="", embed_text="", search_text="", hyde_result=None
        )

    search_text = q
    embed_source = q
    hyde_result = None
    if config.use_hyde and hyde is not None:
        passage, hyde_result = hyde.expand(q)
        if passage.strip():
            embed_source = passage.strip()

    mode = "instruct" if config.use_instruct else "raw"
    composed = composer.compose(embed_source, goal=goal, mode=mode)
    vector = embedder.embed(composed)
    hits = store.search(
        VectorQuery(
            vector=vector,
            k=top_k,
            session_id=session_id,
            text=search_text if config.search_mode == "hybrid" else None,
            mode=config.search_mode,
        )
    )
    return RetrieveResult(
        hits=hits,
        query_text=q,
        embed_text=composed,
        search_text=search_text,
        hyde_result=hyde_result,
    )
