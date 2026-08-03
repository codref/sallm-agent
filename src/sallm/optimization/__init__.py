"""Offline prompt optimization — DSPy-inspired, dependency-free."""

from .artifacts import load_artifact, save_artifact
from .dataset import Case, fingerprint, load_jsonl
from .metrics import Score, contains_all, exact_field_match
from .search import Candidate, propose_instructions, score_case, successive_halving

__all__ = [
    "Candidate",
    "Case",
    "Score",
    "contains_all",
    "exact_field_match",
    "fingerprint",
    "load_artifact",
    "load_jsonl",
    "propose_instructions",
    "save_artifact",
    "score_case",
    "successive_halving",
]
