"""DSPy-inspired successive-halving prompt search (no DSPy dependency)."""

from __future__ import annotations

import random
from dataclasses import dataclass

from sallm.context import estimate_tokens
from sallm.llm import complete
from sallm.messages import user

from .dataset import Case, fingerprint
from .metrics import Score, contains_all, exact_field_match


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
            )
            text = (result.get("content") or "").strip()
        if text and text not in out:
            out.append(text)
        else:
            # Deterministic slight perturbation when teacher fails/repeats.
            out.append(baseline + f"\nBe brief. (v{seed + i}:{rng.randrange(1000)})")
    return out[:n]


def score_case(
    case: Case,
    *,
    instruction: str,
    demos: str,
    predict_fn,
) -> Score:
    """predict_fn(case, instruction, demos) -> (got_dict_or_text, usage_dict)."""
    got, usage = predict_fn(case, instruction, demos)
    tokens = int(usage.get("total_tokens") or 0) + estimate_tokens(instruction + demos)
    latency = float(usage.get("elapsed_ms") or 0.0)
    invalid = False
    if isinstance(got, dict):
        quality = exact_field_match(got, case.expected)
        if case.expected and quality == 0 and not got:
            invalid = True
    else:
        needles = list(case.expected.get("contains") or [])
        quality = contains_all(str(got), needles)
    mandatory_fail = bool(case.mandatory and quality < 1.0)
    return Score(
        quality=quality,
        tokens=tokens,
        latency_ms=latency,
        invalid=invalid,
        mandatory_fail=mandatory_fail,
    )


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
