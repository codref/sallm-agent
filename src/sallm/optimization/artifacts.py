"""Neutral compiled-profile artifacts (no DSPy / no Pydantic)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ARTIFACT_SCHEMA = 1


def content_digest(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def save_artifact(
    path: str | Path,
    *,
    target_model: str,
    instructions: dict,
    demonstrations: dict,
    budgets: dict,
    dataset_fingerprint: str,
    metrics: dict,
    seed: int,
) -> dict:
    payload = {
        "schema_version": ARTIFACT_SCHEMA,
        "target_model": target_model,
        "instructions": instructions,
        "demonstrations": demonstrations,
        "budgets": budgets,
        "metadata": {
            "dataset_fingerprint": dataset_fingerprint,
            "metrics": metrics,
            "seed": seed,
            "created_at": time.time(),
        },
    }
    payload["metadata"]["content_digest"] = content_digest(
        {
            "instructions": instructions,
            "demonstrations": demonstrations,
            "budgets": budgets,
        }
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def merge_task_instruction(
    data: dict,
    *,
    task: str,
    instruction: str,
    demos: str | None = None,
    dataset_fingerprint: str = "",
    metrics: dict | None = None,
    seed: int = 0,
) -> dict:
    """Copy a profile dict, replacing one task's instruction (keep budgets)."""
    payload = json.loads(json.dumps(data))
    payload.setdefault("schema_version", ARTIFACT_SCHEMA)
    payload.setdefault("instructions", {})[task] = instruction
    if demos is not None:
        payload.setdefault("demonstrations", {})[task] = demos
    meta = dict(payload.get("metadata") or {})
    meta["dataset_fingerprint"] = dataset_fingerprint
    meta["metrics"] = metrics or {}
    meta["seed"] = seed
    meta["optimize_task"] = task
    meta["created_at"] = time.time()
    meta["content_digest"] = content_digest(
        {
            "instructions": payload.get("instructions") or {},
            "demonstrations": payload.get("demonstrations") or {},
            "budgets": payload.get("budgets") or {},
        }
    )
    payload["metadata"] = meta
    return payload


def merge_budgets(
    data: dict,
    *,
    budgets: dict,
    dataset_fingerprint: str = "",
    metrics: dict | None = None,
    seed: int = 0,
    optimize_task: str = "budgets",
) -> dict:
    """Copy a profile dict, overlaying ``budgets`` keys (keep instructions)."""
    payload = json.loads(json.dumps(data))
    payload.setdefault("schema_version", ARTIFACT_SCHEMA)
    merged = dict(payload.get("budgets") or {})
    for key, value in (budgets or {}).items():
        if value is None:
            continue
        merged[key] = value
    payload["budgets"] = merged
    meta = dict(payload.get("metadata") or {})
    meta["dataset_fingerprint"] = dataset_fingerprint
    meta["metrics"] = metrics or {}
    meta["seed"] = seed
    meta["optimize_task"] = optimize_task
    meta["created_at"] = time.time()
    meta["content_digest"] = content_digest(
        {
            "instructions": payload.get("instructions") or {},
            "demonstrations": payload.get("demonstrations") or {},
            "budgets": payload.get("budgets") or {},
        }
    )
    payload["metadata"] = meta
    return payload


def write_profile(path: str | Path, data: dict) -> dict:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return data


def load_artifact(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(data.get("schema_version") or 0) != ARTIFACT_SCHEMA:
        raise ValueError(f"unsupported artifact schema in {path}")
    return data
