"""Query pipeline: rewrite, HyDE, instruct, search mode."""

from __future__ import annotations

from sallm.memory import (
    DefaultQueryComposer,
    RetrievalConfig,
    VectorRecord,
    retrieve_hits,
)
from sallm.memory.config import resolve_retrieval_config
from sallm.memory.query import HyDE, RetrieveResult
from sallm.models import ModelProfile

from tests.test_memory_vector import FakeEmbedder, InMemoryVectorStore


class FakeHyDE:
    def expand(self, query: str):
        return f"Hypothetical answer about {query}", {"content": "ok"}


def test_resolve_retrieval_config_modes():
    c = resolve_retrieval_config(retrieval_query="rewrite+hyde", search_mode="hybrid")
    assert c.use_rewrite and c.use_hyde and c.use_instruct
    assert c.search_mode == "hybrid"
    assert c.label == "rewrite+hyde"
    raw = resolve_retrieval_config(retrieval_query="raw")
    assert not raw.use_instruct and raw.label == "raw"


def test_retrieve_rewrite_prefers_control_query():
    store = InMemoryVectorStore(8)
    emb = FakeEmbedder()
    store.upsert(
        [
            VectorRecord(
                id="1",
                text="purple vault code ZEBRA",
                vector=emb.embed("purple vault code ZEBRA"),
                session_id="s1",
            )
        ]
    )
    result = retrieve_hits(
        store=store,
        embedder=emb,
        composer=DefaultQueryComposer(),
        session_id="s1",
        user_text="what was that thing?",
        goal="",
        config=RetrievalConfig(use_rewrite=True, use_instruct=True),
        top_k=2,
        retrieval_query="purple vault",
    )
    assert isinstance(result, RetrieveResult)
    assert result.query_text == "purple vault"
    assert result.hits
    assert "purple" in result.hits[0].text.lower()


def test_retrieve_hyde_embeds_passage_not_question():
    store = InMemoryVectorStore(8)
    emb = FakeEmbedder()
    store.upsert(
        [
            VectorRecord(
                id="1",
                text="purple vault code",
                vector=emb.embed("purple vault code"),
                session_id="s1",
            )
        ]
    )
    result = retrieve_hits(
        store=store,
        embedder=emb,
        composer=DefaultQueryComposer(),
        session_id="s1",
        user_text="vault colour?",
        goal="",
        config=RetrievalConfig(use_hyde=True, use_instruct=False),
        top_k=2,
        hyde=FakeHyDE(),
    )
    assert "Hypothetical answer" in result.embed_text
    assert result.search_text == "vault colour?"


def test_hyde_expand_empty():
    h = HyDE(ModelProfile())
    passage, result = h.expand("  ")
    assert passage == ""
    assert result.get("content") == ""
