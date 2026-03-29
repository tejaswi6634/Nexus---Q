"""
Digital Twin / Fallback Simulator — the 'Intelligence' Layer.

Before switching to a fallback provider the simulator *predicts* the
expected outcome (cost, latency, quality) using rolling historical
telemetry and provider baselines.

The ``composite_score`` is weighted differently depending on the
``UserTier`` attached to the request:

    * **Premium** → 60 % quality, 15 % latency, 25 % cost
    * **Standard** → 25 % quality, 25 % latency, 50 % cost

This ensures Premium users are steered towards GPT-4-class models
while Standard users are directed to the cheapest adequate option.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

from .models import SimulationOutcome, UnifiedInferenceRequest, UserTier

logger = logging.getLogger(__name__)

_PROVIDER_BASELINES: Dict[str, Dict[str, float]] = {
    "OpenAI-GPT4":     {"latency": 0.75, "cost_mult": 1.00, "quality": 0.93},
    "Anthropic-Claude": {"latency": 0.42, "cost_mult": 0.75, "quality": 0.89},
    "Google-Gemini":   {"latency": 0.35, "cost_mult": 0.50, "quality": 0.84},
    "Local-Llama3-8B": {"latency": 0.24, "cost_mult": 0.15, "quality": 0.64},
    "Local-Phi3-Mini": {"latency": 0.12, "cost_mult": 0.08, "quality": 0.52},
}


@dataclass(slots=True)
class _HistoricalRecord:
    latency: float
    cost: float
    quality: float


class FallbackSimulator:
    """
    Simulates expected outcomes for candidate providers before a
    fallback decision is committed.

    Maintains a per-provider deque of recent observations and blends
    them with static baselines to produce estimates.
    """

    def __init__(self, history_window: int = 100) -> None:
        self._history: Dict[str, Deque[_HistoricalRecord]] = {}
        self._history_window = history_window

    # ── Telemetry Ingestion ──────────────────────────────────────────

    def record_outcome(
        self, provider: str, latency: float, cost: float, quality: float,
    ) -> None:
        if provider not in self._history:
            self._history[provider] = deque(maxlen=self._history_window)
        self._history[provider].append(
            _HistoricalRecord(latency=latency, cost=cost, quality=quality)
        )

    # ── Estimation ───────────────────────────────────────────────────

    def _estimate(self, provider: str) -> Dict[str, float]:
        hist = self._history.get(provider)
        if hist and len(hist) >= 3:
            return {
                "latency": sum(r.latency for r in hist) / len(hist),
                "cost": sum(r.cost for r in hist) / len(hist),
                "quality": sum(r.quality for r in hist) / len(hist),
            }
        bl = _PROVIDER_BASELINES.get(
            provider, {"latency": 0.5, "cost_mult": 0.50, "quality": 0.70}
        )
        return {
            "latency": bl["latency"],
            "cost": bl.get("cost_mult", 0.50) * 0.005,
            "quality": bl["quality"],
        }

    # ── Simulation ───────────────────────────────────────────────────

    def simulate(
        self, provider: str, request: UnifiedInferenceRequest,
    ) -> SimulationOutcome:
        est = self._estimate(provider)

        if request.user_tier == UserTier.PREMIUM:
            composite = (
                0.60 * est["quality"]
                - 0.15 * est["latency"]
                - 0.25 * (est["cost"] * 100)
            )
        else:
            composite = (
                0.25 * est["quality"]
                - 0.25 * est["latency"]
                - 0.50 * (est["cost"] * 100)
            )

        return SimulationOutcome(
            provider=provider,
            estimated_latency=est["latency"],
            estimated_cost=est["cost"],
            estimated_quality=est["quality"],
            composite_score=composite,
            recommendation="suitable" if composite > 0 else "marginal",
        )

    def rank_fallbacks(
        self,
        request: UnifiedInferenceRequest,
        available_providers: List[str],
    ) -> List[SimulationOutcome]:
        outcomes = [self.simulate(p, request) for p in available_providers]
        outcomes.sort(key=lambda o: o.composite_score, reverse=True)
        return outcomes

    def recommend_fallback(
        self,
        request: UnifiedInferenceRequest,
        available_providers: List[str],
    ) -> Optional[str]:
        ranked = self.rank_fallbacks(request, available_providers)
        if not ranked:
            return None
        best = ranked[0]
        logger.info(
            "Digital Twin recommends %s  "
            "(score=%.3f  est_lat=%.3fs  est_cost=$%.5f  est_qual=%.2f)",
            best.provider, best.composite_score,
            best.estimated_latency, best.estimated_cost, best.estimated_quality,
        )
        return best.provider
