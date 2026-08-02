"""VectorStore contract + LanceDB + in-memory fake."""

from __future__ import annotations

import math

import pytest

from sallm.memory import (
    DefaultQueryComposer,
    LanceVectorStore,
    TokenChunker,
    VectorHit,
    VectorQuery,
    VectorRecord,
)
from sallm.memory.index import MemoryIndexer, chunk_id
from sallm.state import SessionRepository


class FakeEmbedder:
    dimensions = 8

    def embed(self, text: str) -> list[float]:
        # Put mass on dimensions keyed by distinctive tokens.
        vec = [0.01] * 8
        t = (text or "").lower()
        if "purple" in t or "vault" in t:
            vec[0] = 5.0
            vec[1] = 5.0
        if "weather" in t or "unrelated" in t:
            vec[6] = 5.0
            vec[7] = 5.0
        for i, ch in enumerate(t[:32]):
            vec[i % 8] += 0.01
        n = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / n for x in vec]


class InMemoryVectorStore:
    """pgvector-shaped fake for contract tests."""

    def __init__(self, dimensions: int = 8):
        self.dimensions = dimensions
        self._rows: dict[str, VectorRecord] = {}

    def upsert(self, records: list[VectorRecord]) -> None:
        for r in records:
            if len(r.vector) != self.dimensions:
                raise ValueError("bad dims")
            self._rows[r.id] = r

    def search(self, query: VectorQuery) -> list[VectorHit]:
        scored = []
        for r in self._rows.values():
            if query.session_id is not None and r.session_id != query.session_id:
                continue
            # cosine
            dot = sum(a * b for a, b in zip(query.vector, r.vector))
            scored.append((dot, r))
        scored.sort(key=lambda x: -x[0])
        return [
            VectorHit(
                id=r.id,
                text=r.text,
                score=s,
                session_id=r.session_id,
                source_id=r.source_id,
            )
            for s, r in scored[: query.k]
        ]

    def delete_session(self, session_id: str) -> None:
        self._rows = {
            k: v for k, v in self._rows.items() if v.session_id != session_id
        }

    def close(self) -> None:
        pass


def _contract_suite(store, embedder, tmp_path=None):
    store.upsert(
        [
            VectorRecord(
                id="1",
                text="purple vault code",
                vector=embedder.embed("purple vault code"),
                session_id="s1",
                source_id="10",
            ),
            VectorRecord(
                id="2",
                text="unrelated weather",
                vector=embedder.embed("unrelated weather"),
                session_id="s1",
            ),
            VectorRecord(
                id="3",
                text="purple vault code",
                vector=embedder.embed("purple vault code"),
                session_id="s2",
            ),
        ]
    )
    # idempotent
    store.upsert(
        [
            VectorRecord(
                id="1",
                text="purple vault code",
                vector=embedder.embed("purple vault code"),
                session_id="s1",
                source_id="10",
            )
        ]
    )
    hits = store.search(
        VectorQuery(vector=embedder.embed("vault purple"), k=2, session_id="s1")
    )
    assert hits
    assert hits[0].text == "purple vault code"
    store.delete_session("s1")
    assert store.search(
        VectorQuery(vector=embedder.embed("vault"), k=2, session_id="s1")
    ) == []
    assert store.search(
        VectorQuery(vector=embedder.embed("vault"), k=2, session_id="s2")
    )


def test_inmemory_vector_contract():
    _contract_suite(InMemoryVectorStore(8), FakeEmbedder())


def test_lance_vector_contract(tmp_path):
    pytest.importorskip("lancedb")
    emb = FakeEmbedder()
    store = LanceVectorStore(tmp_path / "lance", dimensions=8)
    _contract_suite(store, emb)
    store.close()


def test_lance_dimension_error(tmp_path):
    pytest.importorskip("lancedb")
    store = LanceVectorStore(tmp_path / "lance", dimensions=8)
    with pytest.raises(ValueError):
        store.upsert(
            [
                VectorRecord(
                    id="x", text="t", vector=[0.1, 0.2], session_id="s"
                )
            ]
        )


def test_lance_hybrid_search(tmp_path):
    """Hybrid BM25+dense should surface an exact-token document."""
    pytest.importorskip("lancedb")
    emb = FakeEmbedder()
    store = LanceVectorStore(tmp_path / "lance-hybrid", dimensions=8)
    store.upsert(
        [
            VectorRecord(
                id="1",
                text="the ZEBRACODE-9911 lab credential is sealed",
                vector=emb.embed("generic filler prose about weather"),
                session_id="s1",
            ),
            VectorRecord(
                id="2",
                text="unrelated weather notes for tomorrow",
                vector=emb.embed("unrelated weather notes for tomorrow"),
                session_id="s1",
            ),
        ]
    )
    # Dense alone may prefer weather; hybrid text query mentions the unique token.
    hits = store.search(
        VectorQuery(
            vector=emb.embed("what is the lab credential"),
            k=2,
            session_id="s1",
            text="ZEBRACODE-9911",
            mode="hybrid",
        )
    )
    assert hits
    assert any("ZEBRACODE" in h.text for h in hits)
    store.close()


def test_query_composer_modes():
    c = DefaultQueryComposer()
    assert c.compose("hello", mode="raw") == "hello"
    instructed = c.compose("hello", mode="instruct")
    assert instructed.startswith("Instruct:")
    assert "Query: hello" in instructed
    rewritten = c.compose("standalone sentence", mode="rewrite")
    assert "Query: standalone sentence" in rewritten


def test_chunker_and_indexer_recovery(tmp_path):
    repo = SessionRepository(tmp_path / "s.db")
    repo.ensure_session("s1")
    store = InMemoryVectorStore(8)
    emb = FakeEmbedder()
    indexer = MemoryIndexer(repo, store, emb)
    chunks = TokenChunker(32, 4).chunk("alpha " * 40)
    assert len(chunks) >= 1
    indexer.add_text("s1", "alpha " * 40, chunks=chunks, source_message_id=1)
    assert all(c.indexed for c in repo.list_chunks("s1"))
    # Simulate crash: add chunk marked unindexed
    cid = chunk_id("s1", "orphan fact")
    repo.add_chunk("s1", chunk_id=cid, text="orphan fact", source_message_id=2)
    assert repo.unindexed_chunks("s1")
    n = indexer.flush_unindexed("s1")
    assert n == 1
    assert not repo.unindexed_chunks("s1")
    rebuilt = indexer.rebuild("s1")
    assert rebuilt >= 1
    repo.close()
