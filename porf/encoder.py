"""Embedding encoder for Co-STORM information routing."""

from __future__ import annotations

import litellm
litellm.suppress_debug_info = True


class Encoder:
    """Thin wrapper around litellm.embedding()."""

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self._available: bool | None = None

    def available(self) -> bool:
        if self._available is None:
            try:
                litellm.embedding(model=self.model, input=["test"])
                self._available = True
            except Exception:
                self._available = False
        return self._available

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = litellm.embedding(model=self.model, input=texts)
        return [item["embedding"] for item in resp.data]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
