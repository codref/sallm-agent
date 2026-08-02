"""Offline optimizer — successive halving without DSPy."""

from __future__ import annotations

from sallm.optimization import (
    Candidate,
    Case,
    load_artifact,
    propose_instructions,
    save_artifact,
    score_case,
    successive_halving,
)
from sallm.optimization.metrics import Score


def test_propose_with_fake_teacher():
    def teacher(prompt, i):
        return f"Instruction variant {i}: be clear."

    texts = propose_instructions(
        baseline="Base instruction.",
        task="controller",
        model="x",
        api_base="http://localhost",
        n=3,
        seed=1,
        teacher_fn=teacher,
    )
    assert len(texts) == 3
    assert texts[0] == "Base instruction."


def test_successive_halving_and_mandatory():
    cases = [
        Case(
            id="1",
            task="controller",
            input={"user": "hi"},
            expected={"action": "keep"},
            mandatory=True,
        ),
        Case(
            id="2",
            task="controller",
            input={"user": "bye"},
            expected={"action": "keep"},
        ),
    ]

    def predict(case, instruction, demos):
        # Candidate "good" always returns keep; "bad" fails mandatory.
        if "BAD" in instruction:
            return {"action": "push"}, {"total_tokens": 10, "elapsed_ms": 1}
        return {"action": "keep"}, {"total_tokens": 5, "elapsed_ms": 1}

    cands = [
        Candidate(name="good", instruction="GOOD keep skill"),
        Candidate(name="bad", instruction="BAD always push"),
        Candidate(name="good2", instruction="GOOD2 keep"),
        Candidate(name="noisy", instruction="GOOD noisy keep"),
    ]
    winner, report = successive_halving(
        cands, cases, predict_fn=predict, seed=0, min_keep=1
    )
    assert winner.name != "bad"
    assert "final" in report
    # bad should be heavily penalized on full eval if it survives
    assert report["winner"] == winner.name


def test_reject_avg_win_with_mandatory_fail():
    case = Case(
        id="m",
        task="t",
        input={},
        expected={"x": 1},
        mandatory=True,
    )

    def predict(case, instruction, demos):
        return {"x": 0}, {"total_tokens": 1, "elapsed_ms": 1}

    s = score_case(case, instruction="i", demos="", predict_fn=predict)
    assert s.mandatory_fail
    assert s.total < -1e8


def test_artifact_roundtrip(tmp_path):
    path = tmp_path / "p.json"
    save_artifact(
        path,
        target_model="ollama/gemma4:e4b-it-qat",
        instructions={"controller": "Do X"},
        demonstrations={"controller": ""},
        budgets={"prompt_budget": 4096},
        dataset_fingerprint="abc",
        metrics={"good": {"score": 1.0}},
        seed=7,
    )
    data = load_artifact(path)
    assert data["instructions"]["controller"] == "Do X"
    assert data["metadata"]["seed"] == 7
    assert data["metadata"]["content_digest"]
