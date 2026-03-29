"""Semantic Reward Calculator — scores provider responses using TF-IDF similarity."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class RewardCalculator:
    """Computes a composite reward that balances semantic fidelity, latency,
    and cost:

        Reward = (Similarity * 2) - (Latency * 0.5) - (Cost * 100)

    Semantic similarity is measured as the TF-IDF cosine similarity between
    the original prompt and the (simulated) provider response.
    """

    W_SIMILARITY = 2.0
    W_LATENCY = 0.5
    W_COST = 100.0

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(max_features=300)
        self._fitted = False

    def _ensure_fitted(self, texts: list[str]) -> None:
        if not self._fitted:
            self._vectorizer.fit(texts)
            self._fitted = True

    def semantic_similarity(self, prompt: str, response: str) -> float:
        corpus = [prompt, response]
        try:
            self._ensure_fitted(corpus)
            vecs = self._vectorizer.transform(corpus)
        except ValueError:
            vecs = TfidfVectorizer(max_features=300).fit_transform(corpus)

        sim = cosine_similarity(vecs[0:1], vecs[1:2])[0][0]
        return float(np.clip(sim, 0.0, 1.0))

    def compute(
        self,
        prompt: str,
        response: str,
        latency: float,
        cost: float,
    ) -> float:
        similarity = self.semantic_similarity(prompt, response)
        reward = (
            self.W_SIMILARITY * similarity
            - self.W_LATENCY * latency
            - self.W_COST * cost
        )
        return round(reward, 4)

    def compute_breakdown(
        self,
        prompt: str,
        response: str,
        latency: float,
        cost: float,
    ) -> dict:
        similarity = self.semantic_similarity(prompt, response)
        return {
            "similarity": round(similarity, 4),
            "latency_penalty": round(self.W_LATENCY * latency, 4),
            "cost_penalty": round(self.W_COST * cost, 4),
            "reward": round(
                self.W_SIMILARITY * similarity
                - self.W_LATENCY * latency
                - self.W_COST * cost,
                4,
            ),
        }
