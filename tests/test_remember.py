"""Agent.remember — meaning-first ingest (mocked LLM)."""

from __future__ import annotations

from unittest.mock import patch

from sallm import Agent
from sallm.control import INGEST_INSTRUCTION, MemoryExtractor
from sallm.models import resolve_model_profile
from tests.test_memory_vector import InMemoryVectorStore


class _Emb:
    dimensions = 8

    def embed(self, text: str):
        return [0.1] * 8


def _llm(content, **extra):
    return {
        "content": content,
        "reasoning": None,
        "usage": {"prompt_tokens": 5, "completion_tokens": 8, "total_tokens": 13},
        "elapsed_ms": 2.0,
        **extra,
    }


def _agent(tmp_path, session_id="rem1", **kwargs):
    return Agent(
        tools={},
        state_path=tmp_path / "state.db",
        vector_store=InMemoryVectorStore(8),
        embedder=_Emb(),
        session_id=session_id,
        memory_gate=False,
        **kwargs,
    )


def test_remember_requires_durable_state():
    agent = Agent(tools={})
    try:
        agent.remember("docker login")
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "state_path" in str(exc)


def test_remember_indexes_facts_not_chat_history(tmp_path):
    agent = _agent(tmp_path, session_id="rem1")
    before = len(agent.messages)
    block = (
        "docker login registry.example.com\n"
        "ssh deploy@203.0.113.10\n"
        "kubectl get pods -n prod\n"
    )

    def _complete(*_a, **_k):
        mid = agent.repo.list_messages(agent.session_id)[-1].id
        payload = (
            '{"facts":[{"text":"User authenticated to Docker at '
            'registry.example.com then SSHed to 203.0.113.10",'
            '"source_message_ids":[%d]}]}' % mid
        )
        return _llm(payload)

    with patch("sallm.control.complete", side_effect=_complete):
        result = agent.remember(block, source="zsh_history:demo")

    assert result["facts"] == 1
    assert result["raw_chunks"] == 0
    assert "203.0.113.10" in result["fact_texts"][0]
    assert len(agent.messages) == before

    chunks = agent.repo.list_chunks(agent.session_id)
    assert any("203.0.113.10" in c.text for c in chunks)

    agent2 = Agent(
        tools={},
        state_path=tmp_path / "state.db",
        vector_store=InMemoryVectorStore(8),
        embedder=_Emb(),
        session_id="rem1",
        memory_gate=False,
    )
    kinds = [m.kind for m in agent2.repo.list_messages("rem1")]
    assert "ingest" in kinds
    joined = "\n".join(m.get("content") or "" for m in agent2.messages)
    assert "docker login registry.example.com" not in joined


def test_remember_with_index_raw(tmp_path):
    agent = _agent(tmp_path, session_id="raw1")

    def _complete(*_a, **_k):
        mid = agent.repo.list_messages(agent.session_id)[-1].id
        return _llm(
            f'{{"facts":[{{"text":"note about alpha","source_message_ids":[{mid}]}}]}}'
        )

    with patch("sallm.control.complete", side_effect=_complete):
        result = agent.remember(
            "alpha beta gamma " * 40,
            source="note",
            index_raw=True,
        )
    assert result["facts"] >= 1
    assert result["raw_chunks"] >= 1


def test_remember_records_prometheus_metrics(tmp_path):
    from sallm.prom import SessionMetrics
    from sallm.trace import Tracer

    metrics = SessionMetrics("rem-metrics")
    tracer = Tracer(lambda _e: None, session_id="rem-metrics")
    tracer.metrics = metrics
    agent = _agent(tmp_path, session_id="rem-metrics", trace=tracer)

    def _complete(*_a, **_k):
        mid = agent.repo.list_messages(agent.session_id)[-1].id
        return _llm(
            f'{{"facts":[{{"text":"remembered host","source_message_ids":[{mid}]}}]}}'
        )

    with patch("sallm.control.complete", side_effect=_complete):
        agent.remember("ssh host.example", source="hist")

    assert metrics.remember_calls_total == 1
    assert metrics.remember_facts_total == 1
    assert metrics.remember_chars_total > 0
    assert metrics.remember_tokens_input_total == 5
    assert metrics.remember_tokens_output_total == 8
    text = metrics.render()
    assert "sallm_remember_calls_total" in text
    assert "sallm_remember_tokens_total" in text
    assert "sallm_remember_last_facts" in text


def test_ingest_instruction_from_compiled_profile(tmp_path):
    from sallm.prompt import CompiledProfile

    custom = "INGEST PROFILE OVERRIDE: emit English host facts only."
    profile = CompiledProfile(
        target_model="test",
        instructions={"ingest": custom},
        demonstrations={"ingest": ""},
        budgets={},
        metadata={},
    )
    agent = Agent(
        tools={},
        state_path=tmp_path / "s.db",
        vector_store=InMemoryVectorStore(8),
        embedder=_Emb(),
        session_id="prof",
        memory_gate=False,
        compiled_profile=profile,
    )
    assert agent.ingest_interpreter.instruction == custom


def test_ingest_instruction_distinct_from_turn_extract():
    assert "ingested" in INGEST_INSTRUCTION.lower() or "ingest" in INGEST_INSTRUCTION.lower()
    ext = MemoryExtractor(resolve_model_profile(), instruction=INGEST_INSTRUCTION)
    assert ext.instruction is INGEST_INSTRUCTION


def test_remember_many(tmp_path):
    agent = _agent(tmp_path, session_id="many")

    def _complete(*_a, **_k):
        mid = agent.repo.list_messages(agent.session_id)[-1].id
        return _llm(
            f'{{"facts":[{{"text":"fact for message {mid}","source_message_ids":[{mid}]}}]}}'
        )

    with patch("sallm.control.complete", side_effect=_complete):
        out = agent.remember_many(
            [
                {"text": "ssh host-a.example", "source": "block-1"},
                "curl https://api.example/health",
            ]
        )
    assert len(out) == 2
    assert all(r["facts"] == 1 for r in out)
