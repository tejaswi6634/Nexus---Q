"""
Semantic Cache — Survival Layer Level 3.

When all local compute is exhausted the cache returns the closest
historical response using TF-IDF cosine similarity.  It also accelerates
warm-path requests: if a semantically identical prompt was recently
answered, the cached response is served in < 5 ms with zero cost.

Implementation notes:
    * Thread-safe via ``threading.Lock``.
    * Uses ``TfidfVectorizer`` from scikit-learn (zero external deps
      beyond the project's existing requirements).
    * Entries expire after ``ttl_seconds`` (default 1 hour).
    * For production, swap the TF-IDF backend for a sentence-transformer
      + FAISS index to scale to millions of entries.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    prompt: str
    response: str
    provider: str
    quality: float
    timestamp: float
    embedding: np.ndarray


class SemanticCache:
    """
    Vector-similarity response cache backed by TF-IDF embeddings.

    Parameters
    ----------
    max_entries : int
        Ring-buffer capacity; oldest entries are evicted first.
    similarity_threshold : float
        Minimum cosine similarity required to return a cache hit.
    ttl_seconds : float
        Time-to-live for cache entries; stale entries are skipped.
    """

    def __init__(
        self,
        max_entries: int = 2000,
        similarity_threshold: float = 0.82,
        ttl_seconds: float = 3600.0,
    ) -> None:
        self._max_entries = max_entries
        self._similarity_threshold = similarity_threshold
        self._ttl_seconds = ttl_seconds

        self._entries: List[_CacheEntry] = []
        self._vectorizer = TfidfVectorizer(
            max_features=512, ngram_range=(1, 2), sublinear_tf=True,
        )
        self._corpus: List[str] = []
        self._vocab_version: int = 0
        self._lock = threading.Lock()

        self._seed_vectorizer()

    def _seed_vectorizer(self) -> None:
        seed = [
            "Summarize this contract and list legal risks.",
            "Write a marketing email for a product launch.",
            "Generate Python code with tests.",
            "Explain quantum computing with examples.",
            "Design a low-latency API architecture.",
            "Draft unit tests for authentication.",
            "Translate this text into French.",
            "Create an incident postmortem from logs.",
            "Rewrite this paragraph for a non-technical audience.",
            "Convert meeting notes into action items and owners.",
            "Draft a SQL query to find top churned users by region.",
            "Generate optimised SQL to find churned users by region.",
            "Provide a root cause analysis for this incident report.",
            "Design a fault-tolerant event-driven architecture.",
            "Optimize this distributed system for cost and throughput.",
            "Benchmark ML models and recommend a deployment strategy.",
        ]
        self._corpus.extend(seed)
        self._vectorizer.fit(self._corpus)
        self._vocab_version = 1

    def _refit_if_needed(self, texts: List[str]) -> bool:
        """Add new texts to corpus and refit. Returns True if vocabulary changed."""
        new_texts = [t for t in texts if t not in self._corpus]
        if not new_texts:
            return False
        self._corpus.extend(new_texts)
        self._vectorizer.fit(self._corpus)
        self._vocab_version += 1
        for entry in self._entries:
            entry.embedding = self._vectorizer.transform(
                [entry.prompt]
            ).toarray().flatten()
        return True

    def _embed(self, text: str) -> np.ndarray:
        self._refit_if_needed([text])
        return self._vectorizer.transform([text]).toarray().flatten()

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        dot = float(np.dot(a, b))
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)

    # ── Public API ───────────────────────────────────────────────────

    def store(
        self,
        prompt: str,
        response: str,
        provider: str = "unknown",
        quality: float = 0.5,
    ) -> None:
        with self._lock:
            emb = self._embed(prompt)
            self._entries.append(
                _CacheEntry(prompt, response, provider, quality, time.time(), emb)
            )
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries:]

    def lookup(self, prompt: str) -> Optional[Tuple[str, float, str]]:
        """
        Find the closest cached response.

        Returns
        -------
        tuple[str, float, str] | None
            ``(response_text, similarity_score, original_provider)``
            or ``None`` if no sufficiently similar entry exists.
        """
        with self._lock:
            if not self._entries:
                return None

            now = time.time()
            query_emb = self._embed(prompt)

            best_sim = -1.0
            best_entry: Optional[_CacheEntry] = None

            for entry in self._entries:
                if now - entry.timestamp > self._ttl_seconds:
                    continue
                sim = self._cosine_sim(query_emb, entry.embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry

            if best_entry is not None and best_sim >= self._similarity_threshold:
                logger.info(
                    "SemanticCache HIT  sim=%.3f  orig_provider=%s",
                    best_sim, best_entry.provider,
                )
                return best_entry.response, best_sim, best_entry.provider

        return None

    @property
    def size(self) -> int:
        return len(self._entries)
