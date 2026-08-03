"""Score long-context QA script runs from sallm JSONL traces + expect needles."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from sallm.cli.chat import iter_prompts
from sallm.optimization.metrics import contains_all


_DENIAL_RE = re.compile(
    r"(do not contain|does not contain|no (specific|explicit) mention|"
    r"retrieved memor(?:y|ies) do not|provided transcripts? do not|"
    r"impossible to (state|determine)|not (mentioned|stated) in|"
    r"i do not have (any )?(specific )?information|"
    r"not named at the end|not have any mention)",
    re.I,
)


@dataclass
class ExpectCase:
    id: str
    question: str
    contains: list[str]
    mandatory: bool = False


@dataclass
class TurnScore:
    id: str
    question: str
    answer: str
    quality: float
    mandatory_fail: bool
    denial: bool
    omitted_messages: int | None = None
    retrieval_hits: int | None = None
    receipt_total: int | None = None
    receipt_budget: int | None = None
    turn_index: int | None = None


@dataclass
class RunReport:
    session_id: str
    trace_path: Path
    scores: list[TurnScore] = field(default_factory=list)

    @property
    def mean_quality(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.quality for s in self.scores) / len(self.scores)

    @property
    def denial_rate(self) -> float:
        if not self.scores:
            return 0.0
        return sum(1 for s in self.scores if s.denial) / len(self.scores)

    @property
    def mandatory_failures(self) -> list[TurnScore]:
        return [s for s in self.scores if s.mandatory_fail]


def load_expectations(path: Path) -> list[ExpectCase]:
    """Load expect JSONL (optimize-style or simplified question/contains)."""
    cases: list[ExpectCase] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        # Optimize converse cases: input.user + expected.contains
        if "expected" in data or (data.get("task") == "converse"):
            q = str((data.get("input") or {}).get("user") or data.get("question") or "")
            exp = data.get("expected") or {}
            needles = list(exp.get("contains") or data.get("contains") or [])
        else:
            q = str(data.get("question") or "")
            needles = list(data.get("contains") or [])
        cases.append(
            ExpectCase(
                id=str(data.get("id") or f"q{i+1:02d}"),
                question=q,
                contains=needles,
                mandatory=bool(data.get("mandatory")),
            )
        )
    return cases


def pair_script_questions(
    script_path: Path, expectations: list[ExpectCase]
) -> tuple[list[str], list[ExpectCase]]:
    """Return (all prompts, expectations aligned to question lines ending with ?)."""
    prompts = list(iter_prompts(script_path))
    questions = [p for p in prompts if p.rstrip().endswith("?")]
    if not expectations:
        return prompts, []
    if len(expectations) != len(questions):
        # Align by normalized question text when counts differ (e.g. dale jsonl).
        by_q = {_norm(e.question): e for e in expectations if e.question}
        aligned: list[ExpectCase] = []
        for i, q in enumerate(questions):
            hit = by_q.get(_norm(q))
            if hit is None:
                aligned.append(
                    ExpectCase(id=f"q{i+1:02d}", question=q, contains=[], mandatory=False)
                )
            else:
                aligned.append(hit)
        return prompts, aligned
    # Same count: zip in order, prefer expect question text if present.
    aligned = []
    for i, (q, exp) in enumerate(zip(questions, expectations)):
        aligned.append(
            ExpectCase(
                id=exp.id or f"q{i+1:02d}",
                question=exp.question or q,
                contains=list(exp.contains),
                mandatory=exp.mandatory,
            )
        )
    return prompts, aligned


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def load_jsonl_events(path: Path) -> list[dict]:
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        events.append(json.loads(line))
    return events


def answers_from_trace(events: list[dict]) -> dict[int, dict]:
    """Map turn.index -> attrs from closed ask/turn spans (prefer debug output)."""
    out: dict[int, dict] = {}
    for ev in events:
        if ev.get("kind") not in ("turn",) and ev.get("name") not in ("ask",):
            continue
        if ev.get("kind") == "turn.start":
            continue
        attrs = ev.get("attrs") or {}
        idx = int(attrs.get("turn.index") or 0)
        if idx <= 0:
            continue
        # Later turn event for same index overwrites (turn_end is authoritative).
        out[idx] = attrs
    return out


def score_run(
    *,
    expectations: list[ExpectCase],
    question_prompts: list[str],
    answers_by_question: dict[str, str],
    trace_attrs_by_turn: dict[int, dict] | None = None,
    question_turn_index: dict[str, int] | None = None,
    session_id: str = "",
    trace_path: Path | None = None,
) -> RunReport:
    report = RunReport(
        session_id=session_id,
        trace_path=trace_path or Path("."),
    )
    for exp, q in zip(expectations, question_prompts):
        answer = answers_by_question.get(q) or answers_by_question.get(exp.question) or ""
        quality = contains_all(answer, exp.contains) if exp.contains else 1.0
        mandatory_fail = bool(exp.mandatory and quality < 1.0)
        denial = bool(_DENIAL_RE.search(answer))
        omitted = hits = total = budget = turn_index = None
        if question_turn_index and trace_attrs_by_turn:
            ti = question_turn_index.get(q)
            if ti is not None:
                turn_index = ti
                attrs = trace_attrs_by_turn.get(ti) or {}
                omitted = attrs.get("sallm.receipt.omitted_messages")
                hits = attrs.get("sallm.retrieval.hits")
                total = attrs.get("sallm.receipt.total_tokens")
                budget = attrs.get("sallm.receipt.budget")
        report.scores.append(
            TurnScore(
                id=exp.id,
                question=q,
                answer=answer,
                quality=quality,
                mandatory_fail=mandatory_fail,
                denial=denial,
                omitted_messages=int(omitted) if omitted is not None else None,
                retrieval_hits=int(hits) if hits is not None else None,
                receipt_total=int(total) if total is not None else None,
                receipt_budget=int(budget) if budget is not None else None,
                turn_index=turn_index,
            )
        )
    return report


def coherence(a: RunReport, b: RunReport) -> dict:
    """Compare two runs on the same questions."""
    by_id_a = {s.id: s for s in a.scores}
    by_id_b = {s.id: s for s in b.scores}
    ids = [s.id for s in a.scores if s.id in by_id_b]
    deltas = [abs(by_id_a[i].quality - by_id_b[i].quality) for i in ids]
    return {
        "n": len(ids),
        "mean_quality_a": a.mean_quality,
        "mean_quality_b": b.mean_quality,
        "mean_abs_delta": sum(deltas) / max(len(deltas), 1),
        "max_abs_delta": max(deltas) if deltas else 0.0,
        "denial_rate_a": a.denial_rate,
        "denial_rate_b": b.denial_rate,
    }


def resolve_qa_paths(script: Path | None = None) -> tuple[Path, Path]:
    """Default fixture, or SALLM_E2E_QA_SCRIPT (+ sibling expect / dale jsonl)."""
    import os

    env = (os.environ.get("SALLM_E2E_QA_SCRIPT") or "").strip()
    if script is None and env:
        script = Path(env)
    if script is None:
        script = Path(__file__).parent / "fixtures" / "long_briefing_qa.txt"
    script = script.resolve()
    candidates = [
        script.with_suffix(".expect.jsonl"),
        script.parent / (script.stem + ".expect.jsonl"),
        script.parent / (script.stem + ".jsonl"),
    ]
    expect = next((p for p in candidates if p.is_file()), candidates[0])
    return script, expect
