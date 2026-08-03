"""Write-time memory gate heuristics."""

from __future__ import annotations

from sallm.memory.gates import HeuristicMemoryGate, PassThroughGate
from sallm.memory.index import MemoryIndexer
from sallm.state import SessionRepository

from tests.test_memory_vector import FakeEmbedder, InMemoryVectorStore


def test_heuristic_rejects_short_questions():
    gate = HeuristicMemoryGate()
    assert not gate.accept("What is Dale's birthday?", kind="raw")
    assert not gate.accept("Who?", kind="raw")
    assert gate.accept("Dale was born in 1962 in Oregon.", kind="raw")
    assert gate.accept("x" * 400, kind="raw")  # long dump
    assert gate.accept("Dale birthday is March 3", kind="derived")
    assert gate.accept("fact line?", kind="fact")  # always kinds
    assert not gate.accept("", kind="raw")


def test_pass_through_accepts_questions():
    gate = PassThroughGate()
    assert gate.accept("What is Dale's birthday?")


def test_indexer_skips_gated_chunks(tmp_path):
    repo = SessionRepository(tmp_path / "s.db")
    repo.ensure_session("s1")
    store = InMemoryVectorStore(8)
    emb = FakeEmbedder()
    indexer = MemoryIndexer(
        repo, store, emb, gate=HeuristicMemoryGate()
    )
    indexer.add_text(
        "s1",
        "What colour is the vault?",
        chunks=["What colour is the vault?"],
        source_message_id=1,
        kind="raw",
    )
    assert indexer.last_gated == 1
    assert repo.list_chunks("s1") == []
    assert store._rows == {}

    indexer.add_text(
        "s1",
        "The vault is purple.",
        chunks=["The vault is purple."],
        source_message_id=2,
        kind="raw",
    )
    assert indexer.last_gated == 0
    assert len(repo.list_chunks("s1")) == 1
    repo.close()
