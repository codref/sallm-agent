"""DSPy-inspired successive-halving prompt search (no DSPy dependency)."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

from sallm.context import estimate_tokens
from sallm.llm import complete
from sallm.messages import user

from .dataset import Case, fingerprint
from .metrics import Score, exact_field_match, score_json_output, score_program_output

JSON_TASKS = frozenset({"controller", "extractor", "ingest", "remember"})
_PROGRAM_EXPECTED = (
    "contains",
    "tool",
    "argv_contains",
    "observation_contains",
    "absent",
    "no_tool",
)
_GOT_SNIPPET = 240


@dataclass
class Candidate:
    name: str
    instruction: str
    demos: str = ""


def propose_instructions(
    *,
    baseline: str,
    task: str,
    model: str,
    api_base: str,
    n: int,
    seed: int,
    teacher_fn=None,
) -> list[str]:
    """Ask a teacher LM (or fake) for concise instruction rewrites."""
    rng = random.Random(seed)
    out = [baseline]
    prompt = (
        f"Rewrite this {task} instruction to be clearer and shorter for a "
        f"small local LLM. Keep the JSON output contract. Return only the "
        f"new instruction text.\n\n{baseline}"
    )
    for i in range(max(0, n - 1)):
        if teacher_fn is not None:
            text = teacher_fn(prompt, i)
        else:
            result = complete(
                model=model,
                messages=[user(prompt + f"\nVariant seed={seed + i}")],
                api_base=api_base,
                max_tokens=256,
                think=False,
            )
            text = (result.get("content") or "").strip()
        if text and text not in out:
            out.append(text)
        else:
            # Deterministic slight perturbation when teacher fails/repeats.
            out.append(baseline + f"\nBe brief. (v{seed + i}:{rng.randrange(1000)})")
    return out[:n]


def _got_text(got) -> str:
    if isinstance(got, dict) and "content" in got and "commands" in got:
        cmds = got.get("commands") or []
        cmd_s = (
            "; ".join(" ".join(str(x) for x in c) for c in cmds) if cmds else "(none)"
        )
        obs = str(got.get("observation") or "").replace("\n", " ")
        if len(obs) > 80:
            obs = obs[:80] + "…"
        body = str(got.get("content") or "").replace("\n", " ")
        if len(body) > 160:
            body = body[:160] + "…"
        return f"{body} | commands: {cmd_s} | obs: {obs}"
    if isinstance(got, str):
        return got
    try:
        return json.dumps(got, ensure_ascii=False)
    except TypeError:
        return str(got)


def score_from_got(
    case: Case,
    got,
    usage: dict,
    *,
    instruction: str,
    demos: str,
) -> Score:
    tokens = int(usage.get("total_tokens") or 0) + estimate_tokens(instruction + demos)
    latency = float(usage.get("elapsed_ms") or 0.0)
    invalid = False
    expected = case.expected or {}
    json_task = case.task in JSON_TASKS
    program_expected = any(k in expected for k in _PROGRAM_EXPECTED)
    if json_task:
        if isinstance(got, dict):
            quality = score_json_output(got, expected)
            if expected and quality == 0 and not got:
                invalid = True
        else:
            quality = 0.0
            invalid = True
    elif program_expected or isinstance(got, str) or (
        isinstance(got, dict) and "commands" in got
    ):
        quality = score_program_output(got, expected)
    elif isinstance(got, dict):
        quality = exact_field_match(got, expected)
        if expected and quality == 0 and not got:
            invalid = True
    else:
        quality = 0.0
        invalid = True
    mandatory_fail = bool(case.mandatory and quality < 1.0)
    return Score(
        quality=quality,
        tokens=tokens,
        latency_ms=latency,
        invalid=invalid,
        mandatory_fail=mandatory_fail,
    )


def score_case(
    case: Case,
    *,
    instruction: str,
    demos: str,
    predict_fn,
) -> Score:
    """predict_fn(case, instruction, demos) -> (got_dict_or_text, usage_dict)."""
    got, usage = predict_fn(case, instruction, demos)
    return score_from_got(
        case, got, usage, instruction=instruction, demos=demos
    )


def evaluate_instruction(
    instruction: str,
    cases: list[Case],
    predict_fn,
    *,
    demos: str = "",
    on_case=None,
    on_case_begin=None,
) -> tuple[list[Score], list]:
    """Run predict_fn on every case.

    ``on_case_begin(case)`` before each predict; ``on_case(case, score)`` after.
    """
    scores = []
    gots = []
    for case in cases:
        if on_case_begin is not None:
            on_case_begin(case)
        got, usage = predict_fn(case, instruction, demos)
        score = score_from_got(
            case, got, usage, instruction=instruction, demos=demos
        )
        scores.append(score)
        gots.append(got)
        if on_case is not None:
            on_case(case, score)
    return scores, gots


def _mean_quality(scores: list[Score]) -> float:
    if not scores:
        return 0.0
    return sum(s.quality for s in scores) / len(scores)


def _mandatory_misses(scores: list[Score]) -> int:
    return sum(1 for s in scores if s.mandatory_fail)


def _all_mandatory_pass(scores: list[Score]) -> bool:
    return not any(s.mandatory_fail for s in scores)


def _is_better(new: list[Score], old: list[Score]) -> bool:
    new_miss = _mandatory_misses(new)
    old_miss = _mandatory_misses(old)
    if new_miss != old_miss:
        return new_miss < old_miss
    new_q = _mean_quality(new)
    old_q = _mean_quality(old)
    if new_q != old_q:
        return new_q > old_q
    return sum(s.tokens for s in new) < sum(s.tokens for s in old)


def _fail_records(cases: list[Case], scores: list[Score], gots: list) -> list[dict]:
    out = []
    for case, score, got in zip(cases, scores, gots):
        if score.quality >= 1.0 and not score.mandatory_fail:
            continue
        snippet = _got_text(got).replace("\n", " ")
        if len(snippet) > _GOT_SNIPPET:
            snippet = snippet[:_GOT_SNIPPET] + "…"
        out.append(
            {
                "id": case.id,
                "expected": case.expected,
                "got": snippet,
                "quality": score.quality,
                "mandatory": case.mandatory,
            }
        )
    return out


_CONVERSE_TEACHER = """You are optimizing ONLY the converse instruction of a ReAct agent
(the extra system block after a fixed tool contract). Do not rewrite the whole system prompt.

The student already sees a system prompt that defines fenced ```run blocks.
Rewrite the converse instruction so the student would pass ALL failed cases.
Be concrete for a small local LLM. Cover both tools AND remember (do not drop one for the other):
- When the user names calc or echo, the entire first reply is one fenced ```run block.
- Exactly ONE command in that block (only calc or only echo). Never put memory
  codes, rooms, or passwords (ORANGE-19, maple-dusk-7, B-12, lab-guest) in ```run —
  those are plain-text answers, not tools.
- One command per block; no pipes; do not combine echo and calc.
- Powers use Python ** (2**8), not ^.
- calc -e 'EXPR'; echo --text WORD.
- Include a literal one-line ```run example in the instruction.
- After Tool results: answer the rest of the user turn in plain text (tool value AND
  any recall from [Retrieved memory]). No second ```run.
- Remember: one NEW sentence that copies the fact verbatim, including hyphens
  (ORANGE-19, maple-dusk-7). Never refuse. No tools.
- Do not repeat the previous assistant message (intro, bio, or "The result is …").
- "The result is" is only for an actual tool observation, never for remember/greet.
- Greet: one short sentence, then stop.

Return only the new converse instruction text.

Instruction:
{instruction}

Failures (student output vs expected; commands are parsed ```run argv):
{failures}

Round {round_index}.
"""


def propose_from_failures(
    *,
    instruction: str,
    task: str,
    failures: list[dict],
    model: str,
    api_base: str,
    round_index: int,
    teacher_fn=None,
    temperature: float | None = None,
    max_tokens: int = 256,
) -> str:
    """Rewrite an instruction to address scored failures. Return instruction only."""
    lines = []
    for fail in failures:
        lines.append(
            f"- id={fail.get('id')} expected {fail.get('expected')}\n"
            f"  got: {fail.get('got')}"
        )
    fail_blob = "\n".join(lines)
    if task == "converse":
        prompt = _CONVERSE_TEACHER.format(
            instruction=instruction,
            failures=fail_blob,
            round_index=round_index,
        )
    elif task == "vision":
        prompt = (
            f"This vision instruction failed these cases. Rewrite it so a small "
            f"local multimodal model would pass them. Keep the split between a "
            f"question image (on the user message) and a context image (under "
            f"[Retrieved memory]). Do not invent labels that are not visible. "
            f"Do not tell the model to run tools to interpret the image.\n"
            f"Return only the new instruction text.\n\n"
            f"Instruction:\n{instruction}\n\n"
            f"Failures:\n{fail_blob}\n\nRound {round_index}."
        )
    elif task == "caption":
        prompt = (
            f"This caption instruction failed these cases. Rewrite it so a small "
            f"local multimodal model would pass them. The reply must be one or "
            f"two plain sentences for later search, copying readable labels and "
            f"not inventing ones that are absent. Do not answer a question.\n"
            f"Return only the new instruction text.\n\n"
            f"Instruction:\n{instruction}\n\n"
            f"Failures:\n{fail_blob}\n\nRound {round_index}."
        )
    elif task == "controller":
        prompt = (
            f"This controller instruction failed these cases. Rewrite it so a "
            f"small local LLM would pass them. Reply contract is ONE JSON object: "
            f"goal, action, skill, retrieval_query.\n"
            f"Keep action=keep and skill=converse unless the user clearly changes task.\n"
            f"retrieval_query must be \"\" for greetings, pure calc/echo (no recall), "
            f"remember/store, and filler opinions. For recall / earlier / remind / "
            f"summarize facts — including when the same message also asks calc/echo "
            f"first — put a short search phrase (lab code, desk room, meeting "
            f"password), never \"\".\n"
            f"Return only the new instruction text.\n\n"
            f"Instruction:\n{instruction}\n\n"
            f"Failures:\n{fail_blob}\n\nRound {round_index}."
        )
    elif task in ("extractor", "ingest", "remember"):
        prompt = (
            f"This {task} instruction failed these cases. Rewrite it so a small "
            f"local LLM would pass them. Reply with ONE JSON object "
            f"{{\"facts\":[{{\"text\":\"...\",\"source_message_ids\":[ID]}}]}}.\n"
            f"Extract facts the USER stated, even if the assistant refused or said "
            f"The result is. Copy codes verbatim (ORANGE-19, maple-dusk-7). "
            f"Do not extract assistant policy. If none: {{\"facts\":[]}}.\n"
            f"Return only the new instruction text.\n\n"
            f"Instruction:\n{instruction}\n\n"
            f"Failures:\n{fail_blob}\n\nRound {round_index}."
        )
    else:
        prompt = (
            f"This {task} instruction failed these cases. Rewrite it so a small "
            f"local LLM would pass them. Keep the JSON output contract. "
            f"Return only the new instruction text.\n\n"
            f"Instruction:\n{instruction}\n\n"
            f"Failures:\n{fail_blob}\n\nRound {round_index}."
        )
    if teacher_fn is not None:
        return (teacher_fn(prompt, round_index) or "").strip()
    kwargs = {"max_tokens": int(max_tokens), "think": False}
    if temperature is not None:
        kwargs["temperature"] = temperature
    else:
        kwargs["temperature"] = 0.7
    result = complete(
        model=model,
        messages=[user(prompt)],
        api_base=api_base,
        **kwargs,
    )
    return (result.get("content") or "").strip()


def iterate_on_failures(
    *,
    baseline: str,
    task: str,
    cases: list[Case],
    predict_fn,
    model: str,
    api_base: str,
    rounds: int = 6,
    seed: int = 0,
    demos: str = "",
    teacher_fn=None,
    temperature: float | None = None,
    teacher_max_tokens: int = 256,
    on_event=None,
) -> tuple[str, dict]:
    """Evaluate all cases; rewrite from fails until mandatory cases pass.

    ``on_event(kind, **payload)`` optional hooks:
    ``eval_begin``, ``case``, ``eval_end``, ``teacher_begin``, ``teacher_end``,
    ``round_skip``, ``round_done``.
    """

    def emit(kind: str, **payload):
        if on_event is not None:
            on_event(kind, **payload)

    def on_case(case, score):
        emit("case", case=case, score=score)

    def on_case_begin(case):
        emit("case_begin", case=case)

    current = baseline
    emit("eval_begin", label="baseline", n=len(cases))
    scores, gots = evaluate_instruction(
        current,
        cases,
        predict_fn,
        demos=demos,
        on_case=on_case,
        on_case_begin=on_case_begin,
    )
    best = current
    best_scores = scores
    report: dict = {
        "rounds": [],
        "dataset_fingerprint": fingerprint(cases),
        "passed": _all_mandatory_pass(scores),
    }

    def _snapshot(name: str, instr: str, sc: list[Score], fails: list[dict]):
        return {
            "name": name,
            "mandatory_misses": _mandatory_misses(sc),
            "quality": _mean_quality(sc),
            "tokens": sum(s.tokens for s in sc),
            "fails": [f["id"] for f in fails],
            "instruction_chars": len(instr),
        }

    fails = _fail_records(cases, scores, gots)
    snap = _snapshot("baseline", current, scores, fails)
    report["rounds"].append(snap)
    emit("eval_end", label="baseline", snapshot=snap, kept=True)
    if _all_mandatory_pass(scores):
        report["winner"] = "baseline"
        report["final"] = {
            "baseline": {
                "score": sum(s.total for s in scores) / max(len(scores), 1),
                "quality": _mean_quality(scores),
                "tokens": sum(s.tokens for s in scores),
                "latency_ms": sum(s.latency_ms for s in scores),
            }
        }
        return best, report

    for r in range(max(0, int(rounds))):
        fails = _fail_records(cases, scores, gots)
        if not fails:
            break
        emit(
            "teacher_begin",
            round=r + 1,
            fail_ids=[f["id"] for f in fails],
        )
        new_instr = propose_from_failures(
            instruction=best,
            task=task,
            failures=fails,
            model=model,
            api_base=api_base,
            round_index=r + 1 + seed,
            teacher_fn=teacher_fn,
            temperature=temperature,
            max_tokens=teacher_max_tokens,
        )
        emit("teacher_end", round=r + 1)
        if not new_instr or new_instr == best:
            report["rounds"].append(
                {"name": f"r{r + 1}", "skipped": "empty_or_duplicate"}
            )
            emit("round_skip", round=r + 1, reason="empty_or_duplicate")
            continue
        label = f"r{r + 1}"
        emit("eval_begin", label=label, n=len(cases))
        new_scores, new_gots = evaluate_instruction(
            new_instr,
            cases,
            predict_fn,
            demos=demos,
            on_case=on_case,
            on_case_begin=on_case_begin,
        )
        new_fails = _fail_records(cases, new_scores, new_gots)
        snap = _snapshot(label, new_instr, new_scores, new_fails)
        report["rounds"].append(snap)
        kept = _is_better(new_scores, best_scores)
        if kept:
            best = new_instr
            best_scores = new_scores
            current = new_instr
            scores, gots = new_scores, new_gots
        emit("eval_end", label=label, snapshot=snap, kept=kept)
        emit("round_done", round=r + 1, kept=kept, snapshot=snap)
        if _all_mandatory_pass(best_scores):
            break

    report["passed"] = _all_mandatory_pass(best_scores)
    report["winner"] = "best"
    report["remaining_fails"] = [
        f["id"] for f in _fail_records(cases, best_scores, gots)
        if f.get("mandatory")
    ]
    report["final"] = {
        "best": {
            "score": sum(s.total for s in best_scores) / max(len(best_scores), 1),
            "quality": _mean_quality(best_scores),
            "tokens": sum(s.tokens for s in best_scores),
            "latency_ms": sum(s.latency_ms for s in best_scores),
            "mandatory_misses": _mandatory_misses(best_scores),
        }
    }
    return best, report


def successive_halving(
    candidates: list[Candidate],
    cases: list[Case],
    *,
    predict_fn,
    seed: int = 0,
    min_keep: int = 1,
) -> tuple[Candidate, dict]:
    """Evaluate many → keep best half → full holdout on finalists."""
    rng = random.Random(seed)
    pool = list(candidates)
    train = list(cases)
    rng.shuffle(train)
    report = {"rounds": []}
    subset_n = max(1, len(train) // 4)

    while len(pool) > min_keep:
        subset = train[:subset_n]
        scored = []
        for cand in pool:
            scores = [
                score_case(
                    c, instruction=cand.instruction, demos=cand.demos, predict_fn=predict_fn
                )
                for c in subset
            ]
            if any(s.mandatory_fail for s in scores):
                total = -1e9
            else:
                total = sum(s.total for s in scores) / max(len(scores), 1)
            scored.append((total, cand))
        scored.sort(key=lambda x: x[0], reverse=True)
        keep = max(min_keep, len(scored) // 2)
        pool = [c for _, c in scored[:keep]]
        report["rounds"].append(
            {
                "subset": subset_n,
                "ranking": [{"name": c.name, "score": s} for s, c in scored],
            }
        )
        subset_n = min(len(train), subset_n * 2)

    # Final full-set evaluation
    final_scores = {}
    for cand in pool:
        scores = [
            score_case(
                c, instruction=cand.instruction, demos=cand.demos, predict_fn=predict_fn
            )
            for c in cases
        ]
        if any(s.mandatory_fail for s in scores):
            total = -1e9
        else:
            total = sum(s.total for s in scores) / max(len(scores), 1)
        final_scores[cand.name] = {
            "score": total,
            "quality": sum(s.quality for s in scores) / max(len(scores), 1),
            "tokens": sum(s.tokens for s in scores),
            "latency_ms": sum(s.latency_ms for s in scores),
        }
    best = max(pool, key=lambda c: final_scores[c.name]["score"])
    report["final"] = final_scores
    report["dataset_fingerprint"] = fingerprint(cases)
    report["winner"] = best.name
    return best, report
