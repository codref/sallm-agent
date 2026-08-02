"""Unit tests for QA trace scoring helpers (no Ollama)."""

from __future__ import annotations

import json
from pathlib import Path

from tests.qa_trace_score import (
    ExpectCase,
    answers_from_trace,
    coherence,
    load_expectations,
    pair_script_questions,
    score_run,
)


def test_load_fixture_expectations():
    expect = Path(__file__).parent / "fixtures" / "long_briefing_qa.expect.jsonl"
    cases = load_expectations(expect)
    assert len(cases) == 6
    assert cases[0].contains
    assert cases[0].mandatory


def test_pair_script_questions_aligns():
    script = Path(__file__).parent / "fixtures" / "long_briefing_qa.txt"
    expect = Path(__file__).parent / "fixtures" / "long_briefing_qa.expect.jsonl"
    prompts, aligned = pair_script_questions(script, load_expectations(expect))
    assert any(len(p) > 5000 for p in prompts)
    assert len(aligned) == 6
    assert aligned[0].question.endswith("?")


def test_score_run_and_coherence(tmp_path: Path):
    expectations = [
        ExpectCase(id="q1", question="What is the code?", contains=["ORION"], mandatory=True),
        ExpectCase(id="q2", question="Who is liaison?", contains=["Carlos"], mandatory=False),
    ]
    questions = [e.question for e in expectations]
    answers_a = {
        questions[0]: "The code is ORION-7.",
        questions[1]: "The transcripts do not contain that information.",
    }
    answers_b = {
        questions[0]: "ORION-7",
        questions[1]: "Carlos is the liaison.",
    }
    report_a = score_run(
        expectations=expectations,
        question_prompts=questions,
        answers_by_question=answers_a,
        session_id="a",
    )
    report_b = score_run(
        expectations=expectations,
        question_prompts=questions,
        answers_by_question=answers_b,
        session_id="b",
    )
    assert report_a.scores[0].quality == 1.0
    assert report_a.scores[1].denial is True
    assert report_a.scores[1].quality == 0.0
    assert report_b.mean_quality == 1.0
    coh = coherence(report_a, report_b)
    assert coh["mean_abs_delta"] > 0
    assert coh["denial_rate_a"] == 0.5


def test_answers_from_trace_prefers_turn_end(tmp_path: Path):
    events = [
        {
            "kind": "turn.start",
            "name": "ask",
            "attrs": {"turn.index": 1},
        },
        {
            "kind": "turn",
            "name": "ask",
            "attrs": {
                "turn.index": 1,
                "gen_ai.output": "ORION-7",
                "sallm.receipt.omitted_messages": 2,
                "sallm.retrieval.hits": 3,
            },
        },
    ]
    by_turn = answers_from_trace(events)
    assert by_turn[1]["gen_ai.output"] == "ORION-7"
    assert by_turn[1]["sallm.receipt.omitted_messages"] == 2


def test_load_optimize_style_jsonl(tmp_path: Path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "dale-q01",
                "task": "converse",
                "input": {"user": "What is Dale's primary concern?"},
                "expected": {"contains": ["defend", "data"]},
                "mandatory": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cases = load_expectations(path)
    assert cases[0].id == "dale-q01"
    assert "defend" in cases[0].contains
