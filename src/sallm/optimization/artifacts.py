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


def load_artifact(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(data.get("schema_version") or 0) != ARTIFACT_SCHEMA:
        raise ValueError(f"unsupported artifact schema in {path}")
    return data
