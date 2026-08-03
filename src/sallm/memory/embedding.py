"""LiteLLM / Ollama embedding adapter."""

from __future__ import annotations

from litellm import embedding

from sallm.models import EmbeddingProfile, resolve_embedding_profile


class LiteLLMEmbedder:
    """Embed text via LiteLLM. Documents are unprefixed; callers compose queries."""

    def __init__(self, profile: EmbeddingProfile | None = None, **overrides):
        self.profile = profile or resolve_embedding_profile(**overrides)
        self._dimensions = self.profile.dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        response = embedding(
            model=self.profile.model,
            input=[text or ""],
            api_base=self.profile.api_base,
        )
        data = response.data[0]
        vec = data.get("embedding") if isinstance(data, dict) else data["embedding"]
        vec = [float(x) for x in list(vec)]
        if len(vec) != self._dimensions:
            raise ValueError(
                f"embedding length {len(vec)} != dimensions {self._dimensions}"
            )
        return vec


def make_embed_fn(model: str, api_base: str, dimensions: int):
    """Return a plain ``embed(text) -> list[float]`` callable over LiteLLM."""
    embedder = LiteLLMEmbedder(
        resolve_embedding_profile(
            model=model, api_base=api_base, dimensions=dimensions
        )
    )
    return embedder.embed
