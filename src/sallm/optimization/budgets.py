"""Offline search over CompiledProfile.budgets (generation knobs)."""

from __future__ import annotations

import copy
from typing import Callable

from .dataset import Case, fingerprint
from .search import (
    _is_better,
    _mandatory_misses,
    _mean_quality,
    evaluate_instruction,
)

# Axes that affect offline student scoring for each task.
# Window sizes (history/retrieval) need full Agent.ask; not searched here.
_TASK_AXES: dict[str, dict[str, list]] = {
    "converse": {
        "temperature": [0.0, 0.1, 0.2, 0.35, 0.5],
        "max_output_tokens": [64, 128, 256, 384, 512],
        "think": [False, True],
    },
    "vision": {
        "temperature": [0.0, 0.1, 0.2, 0.35, 0.5],
        "max_output_tokens": [64, 128, 256, 384, 512],
        "think": [False, True],
    },
    "caption": {
        "temperature": [0.0, 0.1, 0.2],
        "max_output_tokens": [64, 96, 128],
        "think": [False],
    },
    "rewriter": {
        "temperature": [0.0, 0.1, 0.2, 0.35, 0.5],
        "max_output_tokens": [64, 128, 256, 384],
        "think": [False, True],
    },
    "controller": {
        "temperature": [0.0, 0.1, 0.2, 0.35, 0.5],
        "control_max_tokens": [64, 96, 128, 192, 256],
    },
    "extractor": {
        "temperature": [0.0, 0.1, 0.2, 0.35, 0.5],
        "extract_max_tokens": [96, 128, 192, 256, 384],
    },
    "ingest": {
        "temperature": [0.0, 0.1, 0.2, 0.35, 0.5],
        "extract_max_tokens": [128, 256, 384, 512, 1024],
    },
    "remember": {
        "temperature": [0.0, 0.1, 0.2, 0.35, 0.5],
        "extract_max_tokens": [128, 256, 384, 512, 1024],
    },
}


def expand_budget_candidates(
    baseline: dict,
    *,
    task: str,
    seed: int = 0,
) -> list[dict]:
    """One-at-a-time variants around ``baseline`` (baseline first)."""
    axes = _TASK_AXES.get(task) or _TASK_AXES["converse"]
    base = copy.deepcopy(dict(baseline or {}))
    out = [base]
    seen = {_freeze(base)}
    # Deterministic order; seed only shuffles value order slightly.
    for key, values in axes.items():
        vals = list(values)
        if seed:
            # Rotate so different seeds try different first steps.
            rot = int(seed) % max(len(vals), 1)
            vals = vals[rot:] + vals[:rot]
        current = base.get(key)
        for value in vals:
            if value == current:
                continue
            cand = copy.deepcopy(base)
            cand[key] = value
            key_f = _freeze(cand)
            if key_f in seen:
                continue
            seen.add(key_f)
            out.append(cand)
    return out


def _freeze(budgets: dict) -> tuple:
    return tuple(sorted((k, _norm(v)) for k, v in budgets.items()))


def _norm(v):
    if isinstance(v, float):
        return round(v, 4)
    return v


def search_budgets(
    *,
    baseline: dict,
    task: str,
    cases: list[Case],
    instruction: str,
    predict_factory: Callable[[dict], Callable],
    demos: str = "",
    seed: int = 0,
    on_event=None,
) -> tuple[dict, dict]:
    """Evaluate budget candidates with fixed instruction; keep best.

    ``predict_factory(budgets) -> predict_fn(case, instruction, demos)``.
    """

    def emit(kind: str, **payload):
        if on_event is not None:
            on_event(kind, **payload)

    candidates = expand_budget_candidates(baseline, task=task, seed=seed)
    report: dict = {
        "rounds": [],
        "dataset_fingerprint": fingerprint(cases),
        "candidates": len(candidates),
    }

    def snapshot(name: str, budgets: dict, scores):
        return {
            "name": name,
            "budgets": {
                k: budgets.get(k)
                for k in sorted(
                    set(budgets) | set((_TASK_AXES.get(task) or {}))
                )
            },
            "mandatory_misses": _mandatory_misses(scores),
            "quality": _mean_quality(scores),
            "tokens": sum(s.tokens for s in scores),
        }

    best = copy.deepcopy(candidates[0])
    emit("eval_begin", label="budgets:baseline", n=len(cases))

    def on_case(case, score):
        emit("case", case=case, score=score)

    def on_case_begin(case):
        emit("case_begin", case=case)

    scores, _ = evaluate_instruction(
        instruction,
        cases,
        predict_factory(best),
        demos=demos,
        on_case=on_case,
        on_case_begin=on_case_begin,
    )
    best_scores = scores
    snap = snapshot("baseline", best, scores)
    report["rounds"].append(snap)
    emit("eval_end", label="budgets:baseline", snapshot=snap, kept=True)

    for i, cand in enumerate(candidates[1:], start=1):
        label = f"budgets:c{i}"
        emit("eval_begin", label=label, n=len(cases))
        new_scores, _ = evaluate_instruction(
            instruction,
            cases,
            predict_factory(cand),
            demos=demos,
            on_case=on_case,
            on_case_begin=on_case_begin,
        )
        kept = _is_better(new_scores, best_scores)
        snap = snapshot(label, cand, new_scores)
        snap["kept"] = kept
        # Show which key changed vs baseline for the console.
        deltas = {
            k: cand.get(k)
            for k in (_TASK_AXES.get(task) or {})
            if cand.get(k) != baseline.get(k)
        }
        snap["deltas"] = deltas
        report["rounds"].append(snap)
        emit("eval_end", label=label, snapshot=snap, kept=kept)
        if kept:
            best = copy.deepcopy(cand)
            best_scores = new_scores

    report["passed"] = not any(s.mandatory_fail for s in best_scores)
    report["winner"] = "best"
    report["final"] = {
        "best": {
            "budgets": best,
            "quality": _mean_quality(best_scores),
            "tokens": sum(s.tokens for s in best_scores),
            "latency_ms": sum(s.latency_ms for s in best_scores),
            "mandatory_misses": _mandatory_misses(best_scores),
        }
    }
    return best, report
