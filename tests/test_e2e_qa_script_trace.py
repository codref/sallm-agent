"""E2E: run a dale-style QA script twice; score coherence/completeness from traces.

Default fixture: ``tests/fixtures/long_briefing_qa.txt`` (oversized briefing + 6 Qs).

Override with::

    SALLM_E2E_QA_SCRIPT=data/dale_questions.txt pytest tests/test_e2e_qa_script_trace.py -s

Expectations load from ``*.expect.jsonl`` or sibling ``*.jsonl`` (optimize format).
Requires local Ollama with gemma4 + qwen3-embedding.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from sallm import Agent
from sallm.context import estimate_tokens
from sallm.messages import DEFAULT_API_BASE, DEFAULT_MODEL
from sallm.models import resolve_embedding_profile
from sallm.trace import Tracer, jsonl_sink
from tests.qa_trace_score import (
    answers_from_trace,
    coherence,
    load_expectations,
    load_jsonl_events,
    pair_script_questions,
    resolve_qa_paths,
    score_run,
)

CHAT = DEFAULT_MODEL
EMBED = "ollama/qwen3-embedding:0.6b"


def _ollama_up() -> bool:
    try:
        with urllib.request.urlopen(DEFAULT_API_BASE, timeout=2) as resp:
            return 200 <= resp.status < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _ollama_has(model: str) -> bool:
    try:
        with urllib.request.urlopen(
            DEFAULT_API_BASE.rstrip("/") + "/api/tags", timeout=5
        ) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return False
    names = []
    for m in data.get("models") or []:
        names.append(m.get("name") or "")
        names.append(m.get("model") or "")
    needle = model.split("/", 1)[-1]
    return any(needle in n or n.startswith(needle.split(":")[0]) for n in names)


pytestmark = pytest.mark.skipif(
    not _ollama_up()
    or not _ollama_has("gemma4:e4b-it-qat")
    or not _ollama_has("qwen3-embedding:0.6b"),
    reason="Need Ollama with gemma4:e4b-it-qat and qwen3-embedding:0.6b",
)


def _run_script_once(
    *,
    tmp_path: Path,
    prompts: list[str],
    questions: list[str],
    expectations,
    run_name: str,
) -> tuple[object, Path]:
    state = tmp_path / f"{run_name}-state.db"
    vectors = tmp_path / f"{run_name}-vectors"
    trace_path = tmp_path / f"{run_name}.jsonl"
    sid = f"e2e-qa-{run_name}"

    tracer = Tracer(
        jsonl_sink(trace_path),
        debug=True,
        truncate=2000,
        session_id=sid,
    )
    emb = resolve_embedding_profile(EMBED, api_base=DEFAULT_API_BASE, top_k=6)
    agent = Agent(
        model=CHAT,
        api_base=DEFAULT_API_BASE,
        tools={},
        state_path=state,
        vector_path=vectors,
        session_id=sid,
        embedding_profile=emb,
        retrieval_mode="instruct",
        max_steps=2,
        trace=tracer,
    )

    answers: dict[str, str] = {}
    question_turn_index: dict[str, int] = {}
    for line in prompts:
        result = agent.ask(line)
        if line in questions or line.rstrip().endswith("?"):
            answers[line] = result.get("answer") or ""
            question_turn_index[line] = tracer.turn_index
            # Attach receipt snapshot onto the just-closed turn via re-read of file
            # (turn_end already wrote receipt attrs when Agent called trace.turn_end).

    events = load_jsonl_events(trace_path)
    by_turn = answers_from_trace(events)
    # Prefer live answers; fill from debug trace if missing.
    for q, ti in question_turn_index.items():
        if not answers.get(q):
            attrs = by_turn.get(ti) or {}
            answers[q] = str(attrs.get("gen_ai.output") or "")

    report = score_run(
        expectations=expectations,
        question_prompts=questions,
        answers_by_question=answers,
        trace_attrs_by_turn=by_turn,
        question_turn_index=question_turn_index,
        session_id=sid,
        trace_path=trace_path,
    )
    if hasattr(agent, "close"):
        try:
            agent.close()
        except Exception:
            pass
    return report, trace_path


def test_e2e_qa_script_twice_trace_coherence(tmp_path, capsys):
    script, expect_path = resolve_qa_paths()
    assert script.is_file(), script
    assert expect_path.is_file(), (
        f"missing expectations at {expect_path}; "
        "add *.expect.jsonl or optimize-style *.jsonl"
    )

    expectations = load_expectations(expect_path)
    prompts, aligned = pair_script_questions(script, expectations)
    questions = [p for p in prompts if p.rstrip().endswith("?")]
    assert questions, "script has no question lines (ending with ?)"
    assert aligned, "no expectations aligned to questions"

    # Limit very large scripts (full dale) unless explicitly unbounded.
    import os

    max_q = int(os.environ.get("SALLM_E2E_QA_MAX_QUESTIONS") or "0")
    if max_q <= 0 and script.name == "dale_questions.txt":
        max_q = 8  # keep default dale runs tractable
    if max_q > 0 and len(questions) > max_q:
        keep = set(questions[:max_q])
        # Keep preamble + long context + first max_q questions.
        trimmed = []
        q_seen = 0
        for p in prompts:
            if p.rstrip().endswith("?"):
                if q_seen >= max_q:
                    continue
                q_seen += 1
            trimmed.append(p)
        prompts = trimmed
        questions = [p for p in prompts if p.rstrip().endswith("?")]
        aligned = aligned[: len(questions)]

    long_lines = [p for p in prompts if estimate_tokens(p) > 1800]
    assert long_lines, "fixture should include an oversized context line (>1800 tok)"

    report_a, trace_a = _run_script_once(
        tmp_path=tmp_path,
        prompts=prompts,
        questions=questions,
        expectations=aligned,
        run_name="run1",
    )
    report_b, trace_b = _run_script_once(
        tmp_path=tmp_path,
        prompts=prompts,
        questions=questions,
        expectations=aligned,
        run_name="run2",
    )

    coh = coherence(report_a, report_b)

    # --- Diagnostics from traces (what is going on) ---
    def _diag(report) -> dict:
        omitted = [
            s.omitted_messages
            for s in report.scores
            if s.omitted_messages is not None
        ]
        hits = [
            s.retrieval_hits for s in report.scores if s.retrieval_hits is not None
        ]
        return {
            "mean_quality": round(report.mean_quality, 3),
            "denial_rate": round(report.denial_rate, 3),
            "mandatory_failures": [s.id for s in report.mandatory_failures],
            "max_omitted": max(omitted) if omitted else None,
            "mean_retrieval_hits": (sum(hits) / len(hits)) if hits else None,
            "per_question": [
                {
                    "id": s.id,
                    "quality": round(s.quality, 2),
                    "denial": s.denial,
                    "omitted": s.omitted_messages,
                    "hits": s.retrieval_hits,
                    "answer": (s.answer or "")[:120],
                }
                for s in report.scores
            ],
        }

    summary = {
        "script": str(script),
        "n_prompts": len(prompts),
        "n_questions": len(questions),
        "long_context_tokens": [estimate_tokens(p) for p in long_lines],
        "run1": _diag(report_a),
        "run2": _diag(report_b),
        "coherence": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in coh.items()},
        "trace_a": str(trace_a),
        "trace_b": str(trace_b),
    }
    # Always print so ``pytest -s`` shows the story.
    print("\n=== QA script e2e summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # Persist summary next to traces for later inspection.
    (tmp_path / "qa_e2e_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Structural: both runs scored the same questions; traces exist and have ask turns.
    assert len(report_a.scores) == len(questions)
    assert len(report_b.scores) == len(questions)
    assert trace_a.is_file() and trace_a.stat().st_size > 0
    assert trace_b.is_file() and trace_b.stat().st_size > 0
    assert any(
        (e.get("kind") == "turn" or e.get("name") == "ask")
        for e in load_jsonl_events(trace_a)
    )

    # Budget pathology: after the oversized briefing is ingested, later question
    # turns should show history omission (transcript evicted from the 1800-tok window).
    post_context_omitted = [
        s.omitted_messages
        for s in report_a.scores
        if s.omitted_messages is not None
    ]
    assert post_context_omitted, "expected receipt.omitted_messages on traced question turns"
    assert max(post_context_omitted) >= 1, (
        "expected omitted history after oversized context; "
        f"got {post_context_omitted}. summary={summary}"
    )

    # Completeness: retrieval should recover planted facts often enough on the
    # compact fixture. Full dale runs are noisier — loosen via env.
    min_quality = float(
        __import__("os").environ.get("SALLM_E2E_QA_MIN_QUALITY") or "0.35"
    )
    assert report_a.mean_quality >= min_quality, (
        f"run1 completeness {report_a.mean_quality:.3f} < {min_quality}; "
        f"denials={report_a.denial_rate:.2f} details={summary['run1']}"
    )
    assert report_b.mean_quality >= min_quality, (
        f"run2 completeness {report_b.mean_quality:.3f} < {min_quality}; "
        f"details={summary['run2']}"
    )

    # Coherence across two fresh sessions: quality should not swing wildly.
    max_delta = float(
        __import__("os").environ.get("SALLM_E2E_QA_MAX_DELTA") or "0.45"
    )
    assert coh["mean_abs_delta"] <= max_delta, (
        f"runs diverged: mean_abs_delta={coh['mean_abs_delta']:.3f} > {max_delta}; "
        f"coherence={coh}"
    )

    # Fresh sessions should not both collapse into refusal mode.
    assert report_a.denial_rate < 0.75 and report_b.denial_rate < 0.75, summary
