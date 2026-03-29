"""
Embedding-Based Reward Quality Metric.

Replaces synthetic / random reward values with a high-fidelity semantic
evaluation.  On every provider response:

  1. Embed the ``actual_response`` text.
  2. Embed the ``expected_intent`` (or a benchmark response) text.
  3. Compute **Cosine Similarity** between the two embeddings.

The resulting similarity score (∈ [-1, 1], typically [0, 1]) is fed back
into the RL agent as the *quality* component of its reward signal.

The default embedding backend is ``TfidfVectorizer`` (zero external
dependencies beyond scikit-learn).  Swap for a sentence-transformer model
in production by subclassing ``EmbeddingBackend``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Element-wise cosine similarity between two 1-D vectors."""
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


class EmbeddingBackend(ABC):
    """Interface for pluggable embedding strategies."""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray: ...

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> np.ndarray: ...


class TfidfEmbedder(EmbeddingBackend):
    """
    Lightweight TF-IDF embedder that builds an incremental vocabulary
    from all texts it has seen so far (warm-started with a seed corpus).
    """

    def __init__(self, max_features: int = 512) -> None:
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._corpus: List[str] = []
        self._is_fitted = False
        self._seed()

    def _seed(self) -> None:
        seed_texts = [
            "Summarize the key legal risks in this contract.",
            "Write a concise marketing email for a product launch.",
            "Translate this text into French and Spanish.",
            "Generate optimised SQL to find churned users by region.",
            "Design a low-latency API architecture with trade-offs.",
            "Explain quantum computing with a practical code example.",
            "Provide a root cause analysis for this incident report.",
            "Draft unit tests for the authentication module.",
        ]
        self._corpus.extend(seed_texts)
        self._vectorizer.fit(self._corpus)
        self._is_fitted = True

    def _ensure_vocab(self, texts: List[str]) -> None:
        changed = False
        for t in texts:
            if t not in self._corpus:
                self._corpus.append(t)
                changed = True
        if changed or not self._is_fitted:
            self._vectorizer.fit(self._corpus)
            self._is_fitted = True

    def embed(self, text: str) -> np.ndarray:
        self._ensure_vocab([text])
        return self._vectorizer.transform([text]).toarray().flatten()

    def embed_pair(self, text_a: str, text_b: str) -> tuple[np.ndarray, np.ndarray]:
        """Embed two texts with a consistent vocabulary (same dimensionality)."""
        self._ensure_vocab([text_a, text_b])
        matrix = self._vectorizer.transform([text_a, text_b]).toarray()
        return matrix[0], matrix[1]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        self._ensure_vocab(texts)
        return self._vectorizer.transform(texts).toarray()


class RewardCalculator:
    """
    Computes a semantic-similarity reward by comparing the actual provider
    response against the original prompt intent (or a benchmark answer).

    Parameters
    ----------
    embedder : EmbeddingBackend, optional
        Defaults to ``TfidfEmbedder``.
    quality_weight : float
        Multiplier on the cosine-similarity score before it is blended into
        the final reward.
    latency_penalty : float
        Penalty coefficient applied to response latency.
    cost_penalty : float
        Penalty coefficient applied to monetary cost.
    """

    def __init__(
        self,
        embedder: Optional[EmbeddingBackend] = None,
        *,
        quality_weight: float = 1.0,
        latency_penalty: float = 0.6,
        cost_penalty: float = 120.0,
    ) -> None:
        self.embedder: EmbeddingBackend = embedder or TfidfEmbedder()
        self.quality_weight = quality_weight
        self.latency_penalty = latency_penalty
        self.cost_penalty = cost_penalty

    def semantic_quality(
        self,
        actual_response: str,
        expected_intent: str,
    ) -> float:
        """
        Cosine similarity between the embeddings of ``actual_response``
        and ``expected_intent``.  Returns a float in [0, 1].
        """
        if hasattr(self.embedder, "embed_pair"):
            emb_actual, emb_intent = self.embedder.embed_pair(
                actual_response, expected_intent
            )
        else:
            emb_actual = self.embedder.embed(actual_response)
            emb_intent = self.embedder.embed(expected_intent)
        sim = cosine_similarity(emb_actual, emb_intent)
        return float(np.clip(sim, 0.0, 1.0))

    def compute(
        self,
        actual_response: str,
        expected_intent: str,
        latency: float,
        cost: float,
    ) -> float:
        """
        Full reward = quality_weight * cosine_sim
                    - latency_penalty * latency
                    - cost_penalty * cost
        """
        quality = self.semantic_quality(actual_response, expected_intent)
        reward = (
            self.quality_weight * quality
            - self.latency_penalty * latency
            - self.cost_penalty * cost
        )
        logger.debug(
            "Reward %.4f (quality=%.3f, latency=%.4f, cost=%.5f)",
            reward, quality, latency, cost,
        )
        return float(reward)
