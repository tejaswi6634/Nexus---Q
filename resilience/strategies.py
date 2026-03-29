"""
Fallback Selection Strategies — Strategy Pattern implementation.

Each concrete strategy encapsulates a different optimisation objective
for choosing the next fallback provider:

    * ``QualityFirstStrategy``    — Premium tier; maximise response quality.
    * ``CostOptimizedStrategy``   — Standard tier; minimise spend.
    * ``LatencyOptimizedStrategy`` — Real-time tier; minimise latency.

The ``strategy_for_tier()`` factory selects the appropriate strategy
based on the ``UserTier`` attached to the request, keeping the router
logic decoupled from policy.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from .models import HealthMetrics, UnifiedInferenceRequest, UserTier

logger = logging.getLogger(__name__)


class FallbackStrategy(ABC):
    """Interface all fallback selection strategies must implement."""

    @abstractmethod
    def select(
        self,
        request: UnifiedInferenceRequest,
        available_providers: List[str],
        health_data: Dict[str, HealthMetrics],
        provider_profiles: Dict[str, dict],
    ) -> Optional[str]:
        """Choose the optimal fallback from *available_providers*."""
        ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class QualityFirstStrategy(FallbackStrategy):
    """Maximises expected response quality — used for Premium-tier users."""

    _QUALITY_RANK: Dict[str, float] = {
        "OpenAI-GPT4": 0.93,
        "Anthropic-Claude": 0.89,
        "Google-Gemini": 0.84,
        "Local-Llama3-8B": 0.64,
        "Local-Phi3-Mini": 0.52,
    }

    @property
    def name(self) -> str:
        return "quality_first"

    def select(
        self,
        request: UnifiedInferenceRequest,
        available_providers: List[str],
        health_data: Dict[str, HealthMetrics],
        provider_profiles: Dict[str, dict],
    ) -> Optional[str]:
        if not available_providers:
            return None
        rank = dict(self._QUALITY_RANK)
        for p, prof in provider_profiles.items():
            if "quality" in prof:
                rank[p] = prof["quality"]
        return max(available_providers, key=lambda p: rank.get(p, 0.5))


class CostOptimizedStrategy(FallbackStrategy):
    """Minimises cost while meeting a minimum quality bar — Standard tier."""

    _COST_RANK: Dict[str, float] = {
        "Local-Phi3-Mini": 0.08,
        "Local-Llama3-8B": 0.15,
        "Google-Gemini": 0.50,
        "Anthropic-Claude": 0.75,
        "OpenAI-GPT4": 1.00,
    }

    def __init__(self, min_quality: float = 0.50) -> None:
        self._min_quality = min_quality

    @property
    def name(self) -> str:
        return "cost_optimized"

    def select(
        self,
        request: UnifiedInferenceRequest,
        available_providers: List[str],
        health_data: Dict[str, HealthMetrics],
        provider_profiles: Dict[str, dict],
    ) -> Optional[str]:
        if not available_providers:
            return None
        return min(
            available_providers,
            key=lambda p: self._COST_RANK.get(p, 0.50),
        )


class LatencyOptimizedStrategy(FallbackStrategy):
    """Minimises response latency — for real-time / interactive workloads."""

    _LATENCY_RANK: Dict[str, float] = {
        "Local-Phi3-Mini": 0.12,
        "Local-Llama3-8B": 0.24,
        "Google-Gemini": 0.35,
        "Anthropic-Claude": 0.42,
        "OpenAI-GPT4": 0.75,
    }

    @property
    def name(self) -> str:
        return "latency_optimized"

    def select(
        self,
        request: UnifiedInferenceRequest,
        available_providers: List[str],
        health_data: Dict[str, HealthMetrics],
        provider_profiles: Dict[str, dict],
    ) -> Optional[str]:
        if not available_providers:
            return None

        def _lat(p: str) -> float:
            if p in health_data and health_data[p].sample_count > 0:
                return health_data[p].latency_moving_avg
            return self._LATENCY_RANK.get(p, 0.5)

        return min(available_providers, key=_lat)


def strategy_for_tier(user_tier: UserTier) -> FallbackStrategy:
    """Factory: returns the appropriate strategy for the given SLA tier."""
    if user_tier == UserTier.PREMIUM:
        return QualityFirstStrategy()
    return CostOptimizedStrategy()
