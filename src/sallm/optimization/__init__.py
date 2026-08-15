"""Offline prompt optimization — DSPy-inspired, dependency-free."""

from .artifacts import (
    load_artifact,
    merge_budgets,
    merge_task_instruction,
    save_artifact,
    write_profile,
)
from .budgets import expand_budget_candidates, search_budgets
from .dataset import Case, fingerprint, load_jsonl
from .metrics import Score, contains_all, exact_field_match, score_json_output, score_program_output
from .search import (
    Candidate,
    iterate_on_failures,
    propose_from_failures,
    propose_instructions,
    score_case,
    successive_halving,
)

__all__ = [
    "Candidate",
    "Case",
    "Score",
    "contains_all",
    "exact_field_match",
    "expand_budget_candidates",
    "fingerprint",
    "iterate_on_failures",
    "load_artifact",
    "load_jsonl",
    "merge_budgets",
    "merge_task_instruction",
    "propose_from_failures",
    "propose_instructions",
    "save_artifact",
    "score_case",
    "score_json_output",
    "score_program_output",
    "search_budgets",
    "successive_halving",
    "write_profile",
]
