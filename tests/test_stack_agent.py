"""Control, skills, receipt, stacked agent (mocked LLM)."""

from __future__ import annotations

from unittest.mock import patch

from sallm import Agent, Skill, SkillRegistry
from sallm.control import Controller, MemoryExtractor
from sallm.memory.types import VectorHit
from sallm.models import ModelProfile, resolve_model_profile
from sallm.prompt import Prompt
from sallm.receipt import compile_prompt_messages
from sallm.skills import CONVERSE


def _llm(content, **extra):
    return {
        "content": content,
        "reasoning": None,
        "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
        "elapsed_ms": 1.0,
        **extra,
    }


def test_controller_fallback_on_bad_json():
    profile = resolve_model_profile()
    ctrl = Controller(profile)
    with patch("sallm.control.complete", return_value=_llm("not json")):
        decision, _ = ctrl.decide(
            user_text="hi",
            goal="",
            active_skill="converse",
            skill_descriptions="- converse: chat",
        )
    assert decision.fallback is True
    assert decision.action == "keep"
    assert decision.skill == "converse"


def test_controller_valid_json():
    profile = resolve_model_profile()
    ctrl = Controller(profile)
    body = (
        '{"goal":"find code","action":"keep","skill":"converse",'
        '"retrieval_query":"what is the code"}'
    )
    with patch("sallm.control.complete", return_value=_llm(body)):
        decision, _ = ctrl.decide(
            user_text="what is the code?",
            goal="",
            active_skill="converse",
            skill_descriptions="- converse: chat",
        )
    assert decision.fallback is False
    assert decision.goal == "find code"
    assert decision.retrieval_query == "what is the code"


def test_extractor_rejects_ungrounded():
    ext = MemoryExtractor(resolve_model_profile())
    body = '{"facts":[{"text":"x","source_message_ids":[99]}]}'
    with patch("sallm.control.complete", return_value=_llm(body)):
        facts, _ = ext.extract(
            transcript_snippet="[1] user: hi",
            valid_message_ids={1},
        )
    assert facts == []


def test_extractor_keeps_grounded():
    ext = MemoryExtractor(resolve_model_profile())
    body = '{"facts":[{"text":"code is 42","source_message_ids":[1]}]}'
    with patch("sallm.control.complete", return_value=_llm(body)):
        facts, _ = ext.extract(
            transcript_snippet="[1] user: code is 42",
            valid_message_ids={1},
        )
    assert len(facts) == 1
    assert facts[0].text == "code is 42"


def test_skill_registry_tool_subset():
    reg = SkillRegistry(
        [
            CONVERSE,
            Skill(
                name="calc_only",
                description="math",
                prompt="Use calc.",
                tools=("calc",),
            ),
        ]
    )
    tools = {"calc": object(), "echo": object()}
    assert set(reg.resolve_tools("calc_only", tools)) == {"calc"}
    assert set(reg.resolve_tools("converse", tools)) == {"calc", "echo"}


def test_context_receipt_budget(tmp_path):
    profile = ModelProfile(prompt_budget=200, retrieval_tokens=40, recent_history_tokens=80)
    prompt = Prompt(tools_text="(none)")
    hits = [
        VectorHit(id="h1", text="secret PURPLE-42 " * 20, score=0.9, source_id="1")
    ]
    history = [{"role": "user", "content": "x" * 400} for _ in range(5)]
    history += [{"role": "assistant", "content": "y" * 400} for _ in range(5)]
    msgs, receipt = compile_prompt_messages(
        profile=profile,
        prompt=prompt,
        recent_messages=history,
        hits=hits,
    )
    assert receipt.budget == 200
    assert receipt.total_tokens <= 200 + 5  # small estimate slack ok via fit
    assert receipt.retrieved
    assert receipt.omitted_messages >= 1
    assert msgs[0]["role"] == "system"
    d = receipt.as_dict()
    assert "sections" in d


class _Emb:
    dimensions = 8

    def embed(self, text: str):
        return [0.1] * 8


def test_stacked_agent_resume(tmp_path):
    from tests.test_memory_vector import InMemoryVectorStore

    store = InMemoryVectorStore(8)
    queue = [
        _llm(
            '{"goal":"remember code","action":"keep","skill":"converse",'
            '"retrieval_query":"secret code"}'
        ),
        _llm("Got it, I will remember PURPLE-42."),
        _llm('{"facts":[{"text":"code PURPLE-42","source_message_ids":[1]}]}'),
    ]

    def take(**kwargs):
        return queue.pop(0) if queue else _llm("done")

    with patch("sallm.control.complete", side_effect=take), patch(
        "sallm.turn.complete", side_effect=take
    ):
        agent = Agent(
            tools={},
            state_path=tmp_path / "s.db",
            vector_store=store,
            embedder=_Emb(),
            session_id="s1",
            max_steps=2,
        )
        result = agent.ask("The secret code is PURPLE-42.")
        assert result["answer"]
        assert result["goal"]
        assert result["receipt"] is not None
        sid = agent.session_id
        agent_path = tmp_path / "s.db"

    queue2 = [
        _llm(
            '{"goal":"recall code","action":"keep","skill":"converse",'
            '"retrieval_query":"secret code PURPLE"}'
        ),
        _llm("The code is PURPLE-42."),
        _llm('{"facts":[]}'),
    ]

    def take2(**kwargs):
        return queue2.pop(0) if queue2 else _llm("ok")

    with patch("sallm.control.complete", side_effect=take2), patch(
        "sallm.turn.complete", side_effect=take2
    ):
        agent2 = Agent(
            tools={},
            state_path=agent_path,
            vector_store=store,
            embedder=_Emb(),
            session_id=sid,
            max_steps=2,
        )
        assert any("PURPLE" in (m.get("content") or "") for m in agent2.messages)
        r2 = agent2.ask("What was the secret code?")
        assert "PURPLE" in (r2.get("answer") or "")


def test_schema_version_2(tmp_path):
    from sallm.state import SessionRepository
    from sallm.state.models import SCHEMA_VERSION, PendingExtract, SchemaMeta

    repo = SessionRepository(tmp_path / "v2.db")
    assert SCHEMA_VERSION == 2
    row = SchemaMeta.get(SchemaMeta.key == "version")
    assert row.value == "2"
    repo.ensure_session("s")
    job = repo.enqueue_extract("s", 1)
    assert job.status == "pending"
    assert repo.count_pending_extracts("s") == 1
    assert PendingExtract.select().count() == 1
    repo.mark_extract_done(job.id)
    assert repo.count_pending_extracts("s") == 0


def test_schema_migrate_1_to_2(tmp_path):
    from sallm.state.models import SCHEMA_VERSION, SchemaMeta, db
    from sallm.state import SessionRepository

    path = tmp_path / "old.db"
    # Simulate a v1 DB: create tables with version marker 1.
    repo = SessionRepository(path)
    SchemaMeta.update(value="1").where(SchemaMeta.key == "version").execute()
    repo.close()
    db.close()

    repo2 = SessionRepository(path)
    assert SchemaMeta.get(SchemaMeta.key == "version").value == str(SCHEMA_VERSION)
    repo2.ensure_session("s")
    repo2.enqueue_extract("s", 7)
    assert repo2.count_pending_extracts("s") == 1


def test_extract_waterfall_runs_before_return(tmp_path):
    from tests.test_memory_vector import InMemoryVectorStore

    store = InMemoryVectorStore(8)
    extract_calls = []

    queue = [
        _llm(
            '{"goal":"g","action":"keep","skill":"converse","retrieval_query":""}'
        ),
        _llm("ok"),
        _llm('{"facts":[]}'),
    ]

    def take(**kwargs):
        # detect extract by prompt content
        msgs = kwargs.get("messages") or []
        text = ""
        if msgs:
            text = msgs[0].get("content") or ""
        if "Extract durable facts" in text or "durable facts" in text.lower():
            extract_calls.append(True)
        return queue.pop(0) if queue else _llm("done")

    with patch("sallm.control.complete", side_effect=take), patch(
        "sallm.turn.complete", side_effect=take
    ):
        agent = Agent(
            tools={},
            state_path=tmp_path / "w.db",
            vector_store=store,
            embedder=_Emb(),
            session_id="w",
            max_steps=2,
            extract_mode="waterfall",
        )
        agent.ask("hello there")
        assert extract_calls, "extract should run before ask returns in waterfall"
        assert agent.repo.count_pending_extracts("w") == 0


def test_extract_queue_defers_until_next_ask(tmp_path):
    from tests.test_memory_vector import InMemoryVectorStore
    from sallm.prom import SessionMetrics
    from sallm.trace import Tracer

    store = InMemoryVectorStore(8)
    metrics = SessionMetrics("q1")
    tracer = Tracer(lambda _e: None, metrics=metrics)

    # Turn A: control + react (enqueue, no extract). Turn B: control + lazy drain extract + react.
    queue = [
        _llm(
            '{"goal":"g","action":"keep","skill":"converse","retrieval_query":""}'
        ),
        _llm("stored"),
        _llm(
            '{"goal":"g","action":"keep","skill":"converse","retrieval_query":""}'
        ),
        _llm('{"facts":[{"text":"note","source_message_ids":[1]}]}'),
        _llm("next"),
    ]

    def take(**kwargs):
        return queue.pop(0) if queue else _llm("done")

    with patch("sallm.control.complete", side_effect=take), patch(
        "sallm.turn.complete", side_effect=take
    ):
        agent = Agent(
            tools={},
            state_path=tmp_path / "q.db",
            vector_store=store,
            embedder=_Emb(),
            session_id="q1",
            max_steps=2,
            extract_mode="queue",
            trace=tracer,
        )
        agent.ask("Please remember the code is ZEBRA.")
        assert agent.repo.count_pending_extracts("q1") == 1
        assert metrics.extract_enqueued_total >= 1
        assert metrics.extract_calls_total == 0

        agent.ask("thanks")
        # Previous job drained; this turn enqueued a new pending extract.
        assert metrics.extract_calls_total >= 1
        assert "lazy" in metrics.extract_drained_total
        assert metrics.extract_enqueued_total >= 2
        assert agent.repo.count_pending_extracts("q1") == 1
        text = metrics.render()
        assert "sallm_extract_queue_depth" in text
        assert "sallm_extract_enqueued_total" in text
        assert 'mode="queue"' in text


def test_extract_miss_flush_retries_retrieve(tmp_path):
    from tests.test_memory_vector import InMemoryVectorStore
    from sallm.prom import SessionMetrics
    from sallm.trace import Tracer
    from sallm.memory.types import VectorHit

    store = InMemoryVectorStore(8)
    metrics = SessionMetrics("m1")
    tracer = Tracer(lambda _e: None, metrics=metrics)

    queue = [
        # control wants retrieval
        _llm(
            '{"goal":"recall","action":"keep","skill":"converse",'
            '"retrieval_query":"unique lab code"}'
        ),
        # miss-flush drain extract
        _llm('{"facts":[{"text":"code is ZEBRA-1","source_message_ids":[1]}]}'),
        # react answer
        _llm("ZEBRA-1"),
    ]

    def take(**kwargs):
        return queue.pop(0) if queue else _llm("done")

    retrieve_calls = {"n": 0}
    real_hits_after = [
        VectorHit(id="h1", text="code is ZEBRA-1", score=0.1, source_id="1")
    ]

    def fake_retrieve(**kwargs):
        retrieve_calls["n"] += 1
        from types import SimpleNamespace

        if retrieve_calls["n"] == 1:
            return SimpleNamespace(hits=[], hyde_result=None)
        return SimpleNamespace(hits=real_hits_after, hyde_result=None)

    with patch("sallm.control.complete", side_effect=take), patch(
        "sallm.turn.complete", side_effect=take
    ), patch("sallm.agent.retrieve_hits", side_effect=fake_retrieve):
        agent = Agent(
            tools={},
            state_path=tmp_path / "m.db",
            vector_store=store,
            embedder=_Emb(),
            session_id="m1",
            max_steps=2,
            extract_mode="queue",
            trace=tracer,
        )
        # Prior turn left a pending job.
        stored = agent.repo.append_message(
            "m1", role="user", content="code is ZEBRA-1", kind="chat"
        )
        agent.repo.enqueue_extract("m1", stored.id)
        assert agent.repo.count_pending_extracts("m1") == 1

        result = agent.ask("What is the lab code?")
        assert retrieve_calls["n"] >= 2, "miss-flush should re-retrieve"
        assert metrics.extract_miss_flush_total >= 1
        assert agent._last_miss_flush is True
        assert result["answer"]
        assert "sallm_extract_miss_flush_total" in metrics.render()
        assert "miss" in metrics.extract_drained_total


def test_should_miss_flush_predicate():
    from sallm.extract_queue import should_miss_flush
    from sallm.memory.types import VectorHit

    assert should_miss_flush("lab code", [], 1) is True
    assert should_miss_flush("", [], 1) is False
    assert should_miss_flush("lab code", [], 0) is False
    assert (
        should_miss_flush(
            "lab code",
            [VectorHit(id="a", text="x", score=1.0)],
            1,
        )
        is False
    )
