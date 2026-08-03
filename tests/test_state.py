"""SessionRepository — durable SQLite state."""

from __future__ import annotations

from sallm.state import SessionRepository


def test_session_resume_and_messages(tmp_path):
    db = tmp_path / "s.db"
    repo = SessionRepository(db)
    repo.ensure_session("s1")
    assert repo.active_skill("s1") == "converse"
    m = repo.append_message("s1", role="user", content="hello")
    assert m.seq == 0
    repo.append_message("s1", role="assistant", content="hi")
    repo.set_goal("s1", "greet the user")
    repo.close()

    repo2 = SessionRepository(db)
    msgs = repo2.list_messages("s1")
    assert len(msgs) == 2
    assert msgs[0].content == "hello"
    assert repo2.get_goal("s1") == "greet the user"
    assert repo2.active_skill("s1") == "converse"
    repo2.close()


def test_stack_push_pop_replace(tmp_path):
    repo = SessionRepository(tmp_path / "s.db")
    repo.ensure_session("s1")
    repo.push_skill("s1", "analyze", note="deep dive")
    stack = repo.stack("s1")
    assert [f.skill for f in stack] == ["converse", "analyze"]
    assert repo.active_skill("s1") == "analyze"
    repo.replace_skill("s1", "research")
    assert repo.active_skill("s1") == "research"
    assert repo.pop_skill("s1") == "research"
    assert repo.pop_skill("s1") is None  # cannot pop root
    assert repo.active_skill("s1") == "converse"
    repo.close()


def test_clear_session_isolates(tmp_path):
    repo = SessionRepository(tmp_path / "s.db")
    repo.ensure_session("a")
    repo.ensure_session("b")
    repo.append_message("a", role="user", content="only-a")
    repo.append_message("b", role="user", content="only-b")
    repo.clear_session("a")
    assert repo.list_messages("a") == []
    assert repo.list_messages("b")[0].content == "only-b"
    assert repo.active_skill("a") == "converse"
    repo.close()


def test_chunks_and_derived(tmp_path):
    repo = SessionRepository(tmp_path / "s.db")
    repo.ensure_session("s1")
    m = repo.append_message("s1", role="user", content="code is 42")
    c = repo.add_chunk(
        "s1", chunk_id="c1", text="code is 42", source_message_id=m.id
    )
    assert c is not None
    assert c.indexed is False
    repo.mark_indexed("c1")
    assert repo.unindexed_chunks("s1") == []
    repo.add_derived("s1", "code=42", [m.id])
    assert repo.list_derived("s1")[0][0] == "code=42"
    repo.close()
