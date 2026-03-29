"""
Confidence-Based Fallback — the 'Quality' Layer.

After receiving a provider response, the ``ConfidenceScorer`` evaluates
whether the output is genuinely useful or should be re-routed to a
secondary verifier model.

Scoring dimensions:
    * **Length adequacy** — very short responses signal refusal or truncation.
    * **Uncertainty markers** — phrases like "I don't know", "I cannot help",
      "as an AI" reduce confidence.
    * **Hedging density** — excessive use of "probably", "I think", "it seems"
      dilutes authority.
    * **Log-probability integration** — if the provider returns token-level
      log-probs, low average logprob further penalises confidence.

When ``confidence < verification_threshold``, the engine triggers a
*quality fallback* to a verifier model chosen by the Digital Twin.
"""

from __future__ import annotations

import logging
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)

# (compiled pattern, penalty weight)
_UNCERTAINTY_PATTERNS: List[Tuple[re.Pattern, float]] = [
    (re.compile(r"\bi\s+don'?t\s+know\b", re.I), 0.35),
    (re.compile(r"\bi'?m\s+not\s+sure\b", re.I), 0.30),
    (re.compile(r"\bi\s+cannot\s+(?:help|answer|provide)\b", re.I), 0.40),
    (re.compile(r"\bi\s+can'?t\s+(?:help|answer|provide)\b", re.I), 0.40),
    (re.compile(r"\bthis\s+is\s+(?:beyond|outside)\s+(?:my|the)\b", re.I), 0.35),
    (re.compile(r"\bI\s+apologize\b", re.I), 0.15),
    (re.compile(r"\bAs\s+an\s+AI\b", re.I), 0.10),
    (re.compile(r"\bmight\s+be\s+incorrect\b", re.I), 0.25),
    (re.compile(r"\bnot\s+(?:entirely\s+)?certain\b", re.I), 0.25),
    (re.compile(r"\bpossibly\b|\bperhaps\b|\bmaybe\b", re.I), 0.08),
]

_HEDGING_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bI\s+think\b", re.I),
    re.compile(r"\bIt\s+seems\b", re.I),
    re.compile(r"\bprobably\b", re.I),
    re.compile(r"\bgenerally\b", re.I),
]


class ConfidenceScorer:
    """
    Evaluates post-response quality confidence on a [0, 1] scale.

    Parameters
    ----------
    min_response_words : int
        Responses shorter than this are penalised heavily.
    optimal_response_words : int
        Target length used to normalise the length score component.
    verification_threshold : float
        Confidence below this triggers a quality-fallback re-route.
    """

    def __init__(
        self,
        min_response_words: int = 8,
        optimal_response_words: int = 50,
        verification_threshold: float = 0.40,
    ) -> None:
        self._min_words = min_response_words
        self._optimal_words = optimal_response_words
        self.verification_threshold = verification_threshold

    def score(
        self,
        response_text: str,
        expected_length_hint: int = 0,
        log_probs: float | None = None,
    ) -> float:
        """Return a confidence value in [0.0, 1.0]."""
        if not response_text or not response_text.strip():
            return 0.0

        words = response_text.split()
        word_count = len(words)

        # ── Length component ──────────────────────────────────────────
        if word_count < self._min_words:
            length_score = 0.2
        elif expected_length_hint > 0:
            ratio = min(word_count / expected_length_hint, 2.0)
            length_score = min(1.0, ratio) if ratio <= 1.0 else max(0.5, 2.0 - ratio)
        else:
            length_score = min(1.0, word_count / self._optimal_words)

        # ── Uncertainty penalty ───────────────────────────────────────
        uncertainty_penalty = 0.0
        for pattern, weight in _UNCERTAINTY_PATTERNS:
            if pattern.search(response_text):
                uncertainty_penalty += weight
        uncertainty_penalty = min(uncertainty_penalty, 0.80)

        # ── Hedging penalty ───────────────────────────────────────────
        hedge_hits = sum(1 for p in _HEDGING_PATTERNS if p.search(response_text))
        hedge_penalty = min(0.15, hedge_hits * 0.04)

        # ── Log-prob integration ──────────────────────────────────────
        logprob_score = 1.0
        if log_probs is not None:
            logprob_score = max(0.0, min(1.0, (log_probs + 5.0) / 5.0))

        raw = (
            0.35 * length_score
            + 0.30 * (1.0 - uncertainty_penalty)
            + 0.15 * (1.0 - hedge_penalty)
            + 0.20 * logprob_score
        )
        return max(0.0, min(1.0, raw))

    def needs_verification(self, confidence: float) -> bool:
        return confidence < self.verification_threshold

    def evaluate(
        self,
        response_text: str,
        expected_length_hint: int = 0,
        log_probs: float | None = None,
    ) -> Tuple[float, bool]:
        """Return ``(confidence_score, needs_verification)``."""
        sc = self.score(response_text, expected_length_hint, log_probs)
        return sc, self.needs_verification(sc)
