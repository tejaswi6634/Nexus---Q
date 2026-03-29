"""
ML-Driven Prompt Complexity Analyzer & Token Estimator.

Replaces heuristic keyword scoring with a trained TF-IDF + LogisticRegression
classifier that tags every incoming prompt as 'simple', 'medium', or 'complex'.
"""

from __future__ import annotations

import re
import logging
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

COMPLEXITY_LABELS: List[str] = ["simple", "medium", "complex"]

LABEL_TO_SCORE = {"simple": 0.15, "medium": 0.50, "complex": 0.90}


def _build_training_corpus() -> Tuple[List[str], List[str]]:
    """Curated micro-dataset used to bootstrap the classifier."""
    samples = [
        # ---- simple ----
        ("Translate this sentence to French.", "simple"),
        ("What is 2 + 2?", "simple"),
        ("Summarize this paragraph in one line.", "simple"),
        ("Say hello in Japanese.", "simple"),
        ("Convert 5 miles to kilometres.", "simple"),
        ("List the days of the week.", "simple"),
        ("What color is the sky?", "simple"),
        ("Rewrite this sentence in passive voice.", "simple"),
        ("Define the word serendipity.", "simple"),
        ("Give me a one-line joke.", "simple"),
        ("What is the capital of France?", "simple"),
        ("Spell-check the following text.", "simple"),
        ("Write a short greeting email.", "simple"),
        ("How do you say goodbye in German?", "simple"),
        ("Correct the grammar in this sentence.", "simple"),
        # ---- medium ----
        ("Write a marketing email for a new mobile app launch.", "medium"),
        ("Draft a SQL query to find top churned users by region.", "medium"),
        ("Convert meeting notes into action items and owners.", "medium"),
        ("Generate unit tests for this Python function.", "medium"),
        ("Rewrite this paragraph for a non-technical audience.", "medium"),
        ("Create a concise incident postmortem from raw logs.", "medium"),
        ("Explain the differences between REST and GraphQL.", "medium"),
        ("Build a regex to validate email addresses.", "medium"),
        ("Write a product requirements document for a chat feature.", "medium"),
        ("Summarize this contract and highlight key obligations.", "medium"),
        ("Describe how a load balancer works with an example.", "medium"),
        ("Design database tables for a booking system.", "medium"),
        ("Write a comparative review of React vs Vue.", "medium"),
        ("Generate a data-migration script from CSV to Postgres.", "medium"),
        ("Draft release notes for version 3.2.", "medium"),
        # ---- complex ----
        ("Design a low-latency API architecture with tradeoffs.", "complex"),
        ("Explain quantum angle encoding with a practical example.", "complex"),
        ("Optimize this distributed system for cost and throughput.", "complex"),
        ("Summarize this contract and list all legal risks with mitigations.", "complex"),
        ("Benchmark these three ML models and recommend a deployment strategy.", "complex"),
        ("Refactor this monolith into micro-services with a migration plan.", "complex"),
        ("Perform a multi-step root-cause analysis of this production incident.", "complex"),
        ("Design a fault-tolerant event-driven architecture for payments.", "complex"),
        ("Write a research proposal on federated learning for healthcare.", "complex"),
        ("Create an end-to-end MLOps pipeline with CI/CD and monitoring.", "complex"),
        ("Analyse trade-offs between CQRS and traditional CRUD.", "complex"),
        ("Architect a real-time fraud detection system with sub-100ms latency.", "complex"),
        ("Propose an optimization strategy for LLM inference on edge devices.", "complex"),
        ("Evaluate security vulnerabilities in this microservice topology.", "complex"),
        ("Design a multi-tenant SaaS platform with isolation guarantees.", "complex"),
    ]
    texts, labels = zip(*samples)
    return list(texts), list(labels)


class ComplexityClassifier:
    """
    ML-driven prompt complexity classifier.

    Uses TF-IDF vectorisation fed into Logistic Regression to map raw prompt
    text to one of ``['simple', 'medium', 'complex']``.  A compact training
    corpus is bundled so the model can be warm-started without external data.
    """

    def __init__(self, *, retrain: bool = True) -> None:
        self.pipeline: Pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=300,
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(
                max_iter=400,
                solver="lbfgs",
                C=1.0,
            )),
        ])
        self._is_fitted: bool = False
        if retrain:
            self._warm_start()

    def _warm_start(self) -> None:
        texts, labels = _build_training_corpus()
        self.pipeline.fit(texts, labels)
        self._is_fitted = True
        logger.info("ComplexityClassifier warm-started on %d samples", len(texts))

    def fit(self, texts: List[str], labels: List[str]) -> None:
        """Train (or re-train) the classifier on an external dataset."""
        self.pipeline.fit(texts, labels)
        self._is_fitted = True

    def classify(self, prompt_text: str) -> str:
        """Return the predicted label: 'simple' | 'medium' | 'complex'."""
        if not self._is_fitted:
            self._warm_start()
        return self.pipeline.predict([prompt_text])[0]

    def classify_proba(self, prompt_text: str) -> dict[str, float]:
        """Return class probabilities keyed by label."""
        if not self._is_fitted:
            self._warm_start()
        probs = self.pipeline.predict_proba([prompt_text])[0]
        return dict(zip(self.pipeline.classes_, probs))

    def score(self, prompt_text: str, token_count: int = 0) -> float:
        """
        Backwards-compatible numeric score in [0, 1].

        Maps the predicted label through ``LABEL_TO_SCORE`` so downstream
        modules that relied on a float complexity value keep working.
        """
        label = self.classify(prompt_text)
        return LABEL_TO_SCORE[label]


class TokenEstimator:
    """Estimates token count from prompt text (lightweight BPE-inspired fallback)."""

    def estimate(self, prompt_text: str, provided_count: int | None = None) -> int:
        if provided_count is not None and provided_count > 0:
            return int(provided_count)
        chunks = re.findall(r"\w+|[^\w\s]", prompt_text, flags=re.UNICODE)
        return max(1, int(len(chunks) * 1.15))
