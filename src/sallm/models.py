"""Model and embedding profiles — budgets and defaults for Gemma / Qwen."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .messages import DEFAULT_API_BASE, DEFAULT_MODEL

# Ollama /api/chat ``think`` values besides true/false.
THINK_LEVELS = ("low", "medium", "high", "max")


def coerce_think(value) -> bool | str | None:
    """Normalize profile JSON / kwargs to False, True, or a level string."""
    if value is None or isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("", "false", "off", "none", "0"):
        return False
    if text in ("true", "on", "1"):
        return True
    if text in THINK_LEVELS:
        return text
    raise ValueError(f"think must be bool or {THINK_LEVELS}, got {value!r}")


def think_on(value) -> bool:
    """True when the provider should emit a separate thinking trace."""
    return coerce_think(value) not in (None, False)


@dataclass(frozen=True)
class EmbeddingProfile:
    """Retrieval embedding defaults (Qwen3-Embedding-0.6B)."""

    model: str = "ollama/qwen3-embedding:0.6b"
    api_base: str = DEFAULT_API_BASE
    dimensions: int = 1024
    chunk_tokens: int = 512
    chunk_overlap: int = 64
    top_k: int = 4
    # Qwen query side: Instruct + Query. Documents stay unprefixed.
    instruct_template: str = (
        "Instruct: Retrieve conversation passages relevant to the "
        "current goal and question\nQuery: {query}"
    )


@dataclass(frozen=True)
class ModelProfile:
    """Token budgets and generation limits for a chat/control model."""

    model: str = DEFAULT_MODEL
    api_base: str = DEFAULT_API_BASE
    # Soft prompt budget for the main ReAct call (estimated tokens).
    prompt_budget: int = 4096
    max_output_tokens: int = 1024
    # Recent transcript kept verbatim in the prompt view.
    recent_history_tokens: int = 1800
    retrieval_tokens: int = 800
    control_max_tokens: int = 256
    extract_max_tokens: int = 384
    # Soft caps for optional compiled instructions / demos.
    instruction_tokens: int = 400
    demo_tokens: int = 400
    # Thinking and the visible reply share max_output_tokens (Ollama has no
    # separate think cap). False disables; True enables; "low"/"medium"/
    # "high"/"max" set effort on models that honor levels (gpt-oss, etc.).
    think: bool | str | None = None
    # Optional extra system text, injected only when thinking is on.
    think_hint: str | None = None
    # Sampling; omit (None) so the provider default is used.
    temperature: float | None = None
    version: str = "gemma4-e4b-v1"

    def complete_kwargs(self, *, max_tokens: int, json_mode: bool = False) -> dict:
        """Kwargs for ``sallm.llm.complete`` besides model/messages/api_base."""
        kw: dict = {"max_tokens": int(max_tokens)}
        if json_mode:
            kw["think"] = False
        elif self.think is not None:
            kw["think"] = self.think
        if self.temperature is not None:
            kw["temperature"] = float(self.temperature)
        return kw


# Built-in profiles keyed by LiteLLM model id (and short aliases).
_PROFILES: dict[str, ModelProfile] = {
    DEFAULT_MODEL: ModelProfile(),
    "gemma4:e4b-it-qat": ModelProfile(),
    "ollama/gemma4:e4b-it-qat": ModelProfile(),
}

_EMBEDDINGS: dict[str, EmbeddingProfile] = {
    "ollama/qwen3-embedding:0.6b": EmbeddingProfile(),
    "qwen3-embedding:0.6b": EmbeddingProfile(),
}


def resolve_model_profile(model: str | None = None, **overrides) -> ModelProfile:
    """Return a profile for ``model``, applying optional field overrides."""
    key = (model or DEFAULT_MODEL).strip()
    base = _PROFILES.get(key) or ModelProfile(model=key)
    if model and base.model != key and key not in _PROFILES:
        base = replace(base, model=key)
    if overrides:
        return replace(base, **{k: v for k, v in overrides.items() if v is not None})
    return base


def resolve_embedding_profile(
    model: str | None = None, **overrides
) -> EmbeddingProfile:
    """Return an embedding profile, applying optional field overrides."""
    key = (model or EmbeddingProfile().model).strip()
    base = _EMBEDDINGS.get(key) or EmbeddingProfile(model=key)
    if overrides:
        return replace(base, **{k: v for k, v in overrides.items() if v is not None})
    return base
